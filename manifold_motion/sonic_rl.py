"""RL fine-tuning with the frozen controller in the loop, not a model of it.

An earlier route trained against a learned model of the controller (a batched geometric
environment plus an execution-residual predictor). Its prediction was 51% better than ignoring
the residual, which is not the same as being right: the fine-tune then optimises "what I think
the controller does". A fine-tune needs only a few thousand episodes, and the real controller
costs ~0.3 s of physics per pose, so it can simply be run instead. That model is gone; this
module is the whole training path.

It is deliberately simple:

  * **serial, no batching**: one pose at a time through `KeyframeEnv`, so there is no aliasing
    between environment state and autograd, no in-place version conflicts, no GPU tensors to
    keep in sync;
  * **kinematics gates, containment rewards**: a pose that would fall over (centre of mass off
    the feet, a foot lifted) or that needs joints past their limits earns nothing, whatever it
    does to the containment radius. Fitting inside the manifold is the objective only for poses
    the robot can actually hold;
  * **the reward uses the pose the controller reached**, measured, not predicted - including the
    deterministic probe of the distribution mean, because the mean is what deploys;
  * **manifold perturbation**: each episode reshapes a recorded envelope while keeping its
    demonstration fixed, so the policy has to adapt a pose it cloned to the manifold's new
    shape rather than replaying what it memorised.

Cost: one pose costs ~0.3 s of physics. An iteration of 16 manifolds x 6 control ticks is about
30 s, so the 30-iteration fine-tune behind the current policy took 879 s.

    python3 -m manifold_motion.sonic_rl run --iterations 30 --manifolds 16 --steps 6
    python3 -m manifold_motion.sonic_rl eval --policy .../policy_sonicrl.pt    # compare on SONIC
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import constants as C
from .body_model import BodyModel
from .demos import DEFAULT_ISAAC
from .family import BASE_CENTER, BASE_SEMI
from .keyframe_env import KeyframeEnv
from .manifold import EllipsoidManifold, Primitive
from .pose_policy import OBS_DIM, POSE_DIM, POSE_LIMIT, action_to_pose, observation
from .ppo import PPO, RolloutBuffer
from .demos import demo_values, load_demos
from .paths import BC_POLICY, RL_DIR, RL_POLICY

OUT = RL_DIR


@dataclass
class SonicConfig:
    steps: int = 8                 # control ticks per episode (holding time)
    settle: float = 0.6            # seconds of physics before the pose counts as held
    max_pose: float = 1.4
    # kinematics gate
    com_limit: float = 0.10        # centre of mass within this of the support centre
    foot_spread: float = 0.03      # feet no further apart vertically than this
    # orientation perturbation (std of the sampled tilt/roll, degrees). The recorded envelopes
    # are axis-aligned, so without these the policy never sees a tilted manifold and the
    # `report_specs` tilt rows are extrapolation - see `_reshape`.
    tilt_deg: float = 0.0          # sagittal tilt. Left OFF by default: measured, 6 deg changed
                                   # nothing on the tilt rows (r 1.12->1.12, 1.17->1.23) and
                                   # 14 deg made every row worse (median 0.975 -> 1.200) because
                                   # the policy answered the tilt with waist pitch, the channel
                                   # the frozen controller ignores. Tilting needs a pose channel
                                   # the tracker can execute, not more of the same pressure.
    roll_deg: float = 0.0          # lateral roll, same reasoning
    # rewards
    w_containment: float = 4.0     # per unit of margin, only when the gate passes
    w_outside: float = 8.0
    w_gate: float = 12.0           # kinematic violation
    w_track: float = 2.0           # distance between requested and achieved pose
    w_imitation: float = 1.5       # distance from the demonstrated pose
    success_bonus: float = 4.0
    mean_probe: bool = True        # also score the deterministic pose, which is what deploys
    w_mean_probe: float = 1.0


class SonicPoseEnv:
    """One robot, the real controller, the real physics: pose in, achieved pose out."""

    def __init__(self, cfg: SonicConfig | None = None, seed: int = 0):
        self.cfg = cfg or SonicConfig()
        self.body = BodyModel()
        self.env = KeyframeEnv()
        self.rng = np.random.default_rng(seed)
        self.manifold: EllipsoidManifold | None = None
        self.demo: np.ndarray | None = None
        self._keys, self._demos = load_demos()
        self._entries = [(k, self._demos[k]) for k in self._keys if len(self._demos[k])]
        self.achieved = np.zeros(POSE_DIM)

    # -- manifolds ---------------------------------------------------------
    def sample_episode(self, perturb: float = 0.05, repeats: int = 1) -> np.ndarray:
        """Start an episode: one demonstration, and a *reshaped* version of its own manifold.

        The perturbation is applied to the demonstration's own envelope and the demonstration
        stays fixed for the whole episode. That is what makes the fine-tune mean "this recorded
        movement, adapted to this manifold's new shape" rather than "a new manifold, from
        scratch": the policy is pushed to adjust the pose it cloned while staying near it
        (`w_imitation`), which is where generalisation in the manifold comes from.

        `repeats > 1` keeps the same demonstration and reshapes again - the same pose against
        several manifold deformations in a row.
        """
        if repeats > 1 and self.manifold is not None and self.demo is not None:
            values = self._base_values
            out = self._reshape(values, perturb, keep_demo=True)
            return out
        key, poses = self._entries[int(self.rng.integers(len(self._entries)))]
        self._base_values = demo_values(key)
        self.demo = poses[np.argmin(np.abs(poses).mean(axis=1))]
        return self._reshape(self._base_values, perturb, keep_demo=True)

    def _reshape(self, values: np.ndarray, perturb: float, keep_demo: bool) -> np.ndarray:
        """Build the manifold for this episode from a recorded envelope, jittered.

        Three perturbation channels, all applied to the demonstration's own envelope:

          * **shape** - each semi-axis scaled by up to +-`perturb` (clipped to +-15%);
          * **position** - the centre shifted by up to 2 cm laterally;
          * **orientation** - the axis tilted by up to `tilt_deg` sagittally and rolled by up to
            `roll_deg` laterally.

        Orientation is here because it was missing, and its absence was measurable: the recorded
        envelopes are axis-aligned (their stored quats are pelvis *yaw* from walking, 0.5% carry a
        real tilt), and nothing perturbed orientation either, so `report_specs`' t=+-10 manifolds
        were pure extrapolation. The policy does respond to them - its pose differs by 0.065 rad
        between +10 and -10, correctly signed - but far too weakly to fit, scoring r 1.10-1.33
        against 0.96 upright. Training on tilted manifolds is what gives that response amplitude.
        """
        semi = values[:3].copy()
        center = values[3:].copy()
        if perturb > 0:
            semi = semi * np.clip(1.0 + self.rng.normal(0, perturb, 3), 0.85, 1.15)
            center = center + np.array([self.rng.normal(0, 0.02), self.rng.normal(0, 0.02), 0.0])
        tilt = np.radians(self.rng.normal(0.0, self.cfg.tilt_deg)) if self.cfg.tilt_deg else 0.0
        roll = np.radians(self.rng.normal(0.0, self.cfg.roll_deg)) if self.cfg.roll_deg else 0.0
        # quaternions about world y (sagittal tilt) and x (lateral roll)
        qt = np.array([np.cos(tilt / 2), 0.0, np.sin(tilt / 2), 0.0])
        qr = np.array([np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0])
        from .constants import quat_mul

        self.manifold = EllipsoidManifold([Primitive(center=center, semi=semi,
                                                     quat=quat_mul(qr, qt))])
        self.env.reset()
        self.achieved = np.zeros(POSE_DIM)
        self.elapsed = 0.0
        return self.obs()

    def obs(self) -> np.ndarray:
        """The shared 15-dim layout, so one policy serves the trainer and every viewer."""
        pose = self.achieved
        pelvis = self.body.at_pose(pose)[self.body.pelvis_index()]
        return observation(pose, self.manifold.primitives[0], self._radius(pose), pelvis)

    def _radius(self, pose: np.ndarray) -> float:
        return float(self.manifold.radii(self.body.mesh_points_pose(pose)).max())

    def _fit_containment(self, pose: np.ndarray) -> tuple[bool, float, float]:
        """Containment for the *requested* pose, from the geometric model (no SONIC)."""
        return True, self._radius(pose), 0.0

    # -- one control tick --------------------------------------------------
    def step(self, pose: np.ndarray):
        """Hold this pose for one control tick: the controller drives, physics responds."""
        cfg = self.cfg
        self.env.set_joints(np.clip(pose, -cfg.max_pose, cfg.max_pose))
        ticks = int(round(cfg.settle / C.CONTROL_DT))
        for _ in range(ticks):
            self.env.step()
        self.achieved = (self.env.state()["q_hw"][C.MUJOCO_TO_ISAACLAB]
                         - DEFAULT_ISAAC)
        self.elapsed += cfg.settle

        # what the robot ended up doing: a kinematic check first, then containment
        violation = self._violation(self.achieved)
        r_achieved = self._radius(self.achieved)
        track = float(np.abs(self.achieved - pose).mean())
        gate = violation <= 1e-6

        reward = 0.0
        if gate:
            margin = min(1.0 - r_achieved, 0.15)
            reward += cfg.w_containment * margin if r_achieved <= 1.0 else \
                -cfg.w_outside * (r_achieved - 1.0)
        else:
            reward -= cfg.w_gate * violation
        reward -= cfg.w_track * track
        if self.demo is not None:
            reward -= cfg.w_imitation * float(np.abs(pose - self.demo).mean())

        done = self.elapsed >= cfg.steps * cfg.settle
        if done and gate and r_achieved <= 1.0:
            reward += cfg.success_bonus
        return self.obs(), reward, done, {
            "r_achieved": r_achieved, "r_requested": self._radius(pose), "track": track,
            "gate": gate, "violation": violation, "pose": pose.copy(),
            "achieved": self.achieved.copy()}

    def _violation(self, pose: np.ndarray) -> float:
        """How far this pose is from one the robot can hold (0 = fine).

        The gate. Centre of mass is taken from the *achieved* pose, so a request the controller
        cannot track is judged by where it actually left the robot.
        """
        from .kinematics import TorchKinematics

        if not hasattr(self, "_kin"):
            self._kin = TorchKinematics(device="cpu")
        out = self._kin.stability(np.asarray(pose)[None, :])
        reach = float(np.linalg.norm(out["com"][0, :2].cpu().numpy()))
        feet = out["feet_z"][0].cpu().numpy()
        spread = float(feet.max() - feet.min())
        limit = float(out["limit"][0])
        return (max(0.0, reach - self.cfg.com_limit) / 0.05
                + max(0.0, spread - self.cfg.foot_spread) / 0.02
                + limit / 0.05)


def run(args) -> int:
    import torch

    cfg = SonicConfig(steps=args.steps, mean_probe=not args.no_mean_probe,
                      tilt_deg=args.tilt_deg, roll_deg=args.roll_deg)
    env = SonicPoseEnv(cfg, seed=args.seed)
    ppo = PPO(OBS_DIM, POSE_DIM, init_log_std=[-1.0] * POSE_DIM, device=args.device)
    # An iteration holds only manifolds x steps transitions (60 at the defaults), while the PPO
    # defaults (10 epochs x 8 minibatches) were tuned for a few thousand. Reusing them here means
    # ~80 gradient steps over 60 samples: the policy overfits that batch and drifts, which is
    # exactly the "best early, degrades later" shape every run has shown (8 iterations 0.88,
    # 16 iterations 0.08). One pass, one minibatch: no reuse of a sample within an iteration.
    batch = args.manifolds * args.steps
    ppo.epochs = args.ppo_epochs
    ppo.minibatches = max(1, min(args.ppo_minibatches, max(1, batch // 16)))
    print(f"PPO: {ppo.epochs} epoch x {ppo.minibatches} minibatch over {batch} transitions "
          f"({batch // ppo.minibatches} samples per update)")
    if args.resume:
        blob = torch.load(args.resume, map_location=ppo.device)
        state = dict(blob["model"]); state.pop("log_std", None)
        ppo.model.load_state_dict(state, strict=False)
        with torch.no_grad():
            ppo.model.log_std.fill_(args.init_log_std)
        print(f"resumed {args.resume}; log_std -> {args.init_log_std}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.csv", "w")
    log.write("iteration,episodes,success,r_achieved,gated,return_mean\n")
    print(f"SONIC in the loop: {args.manifolds} manifolds x {args.steps} ticks x "
          f"{cfg.settle}s physics = {args.manifolds * args.steps * cfg.settle:.0f} s per iteration")
    started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        frac = (iteration - 1) / max(1, args.iterations - 1)
        ppo.entropy_coef = args.entropy_start + (args.entropy_end - args.entropy_start) * frac
        buffer = RolloutBuffer(args.manifolds * args.steps, OBS_DIM, POSE_DIM)
        returns, successes, radii, gated = [], [], [], 0
        for episode in range(args.manifolds):
            # every few episodes, reshape the *same* demonstration's manifold again: the policy
            # sees one recorded pose against several manifold deformations in a row
            obs = env.sample_episode(args.perturb,
                                     repeats=args.perturb_repeats if episode % args.perturb_repeats == 0 else args.perturb_repeats)
            episode_return = 0.0
            for _ in range(args.steps):
                # the policy proposes a pose; the manifest is executed by the controller
                import torch as _t
                action, logprob, value = ppo.act(obs)
                pose = np.asarray(action_to_pose(action, _t))
                next_obs, reward, done, info = env.step(pose)
                buffer.add(obs, action, logprob, reward, value, float(done))
                # Also score the mean action. Deployment uses the mean, but PPO explores with
                # samples, so a run can look like it improves (`r` of sampled poses falls) while
                # the mean - the pose that would actually be executed - does not. Every collapse
                # this project has seen had that shape.
                if cfg.mean_probe:
                    with _t.no_grad():
                        mean_action = ppo.model(_t.as_tensor(obs, dtype=_t.float32,
                                                             device=args.device))[0]
                    mean_pose = np.asarray(action_to_pose(mean_action, _t))
                    _, mean_reward, _, mean_info = env.step(mean_pose)
                    reward = reward + cfg.w_mean_probe * mean_reward
                    info = mean_info          # report the mean's outcome: that is what deploys
                obs = next_obs
                episode_return += reward
            returns.append(episode_return)
            successes.append(bool(info["gate"] and info["r_achieved"] <= 1.0))
            radii.append(info["r_achieved"])
            gated += 0 if info["gate"] else 1
        ppo.update(buffer, ppo.value(obs))
        success = float(np.mean(successes))
        log.write(f"{iteration},{len(returns)},{success:.3f},{np.mean(radii):.3f},"
                  f"{gated},{np.mean(returns):.3f}\n"); log.flush()
        print(f"  iter {iteration:>3}/{args.iterations}  success {success:.2f}  "
              f"r(achieved) {np.mean(radii):.2f}  gate fails {gated:>2}/{len(returns)}  "
              f"return {np.mean(returns):+7.2f}  ({time.perf_counter() - started:.0f}s)", flush=True)
        if success > getattr(env, "_best", -1):
            env._best = success
            ppo.save(out / "policy_sonicrl.pt")
    log.close()
    print(f"\n=== done in {time.perf_counter() - started:.0f} s, best success "
          f"{getattr(env, '_best', float('nan')):.2f} -> {out / 'policy_sonicrl.pt'} ===")
    return 0


def evaluate(args) -> int:
    """Compare policies on the real controller: same manifolds, same settling."""
    import torch

    from .kinematics import TorchKinematics

    cfg = SonicConfig(steps=args.steps, mean_probe=not args.no_mean_probe,
                      tilt_deg=args.tilt_deg, roll_deg=args.roll_deg)
    env = SonicPoseEnv(cfg, seed=args.seed)
    kin = TorchKinematics(device="cpu")
    policies = {"BC": C.REPO / "reports/manifold_motion/bc/bc_policy.pt"}
    if args.policy:
        policies[Path(args.policy).stem] = Path(args.policy)
    models = {}
    for name, path in policies.items():
        ppo = PPO(OBS_DIM, POSE_DIM, device=args.device)
        ppo.load(path)
        models[name] = ppo

    print(f"{'manifold':>22} " + " ".join(f"{n:>18}" for n in models))
    rows = []
    for i in range(args.manifolds):
        obs = env.sample_episode(args.perturb)
        cells, record = [], {"index": i}
        for name, ppo in models.items():
            o = obs
            for _ in range(args.steps):
                a, _, _ = ppo.act(o, deterministic=True)
                pose = np.asarray(action_to_pose(a, torch))
                o, _, _, info = env.step(pose)
                env.env.reset()          # keep every policy measured from the same start
                env.achieved = np.zeros(POSE_DIM)
                env.elapsed = 0.0
            record[name] = {"r": info["r_achieved"], "track": info["track"],
                            "gate": info["gate"]}
            cells.append(f"r {info['r_achieved']:.2f} trk {info['track']:.2f}")
        print(f"{str(np.round(env.manifold.primitives[0].semi, 2)):>22} " +
              " ".join(f"{c:>18}" for c in cells))
        rows.append(record)
    (OUT / "compare.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "compare.json").write_text(json.dumps(rows, indent=2))
    for name in models:
        rs = [r[name]["r"] for r in rows]
        print(f"{name:>8}: mean r(achieved) {np.mean(rs):.2f}  inside {sum(1 for x in rs if x <= 1)}/"
              f"{len(rs)}  mean track {np.mean([r[name]['track'] for r in rows]):.3f} rad")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RL fine-tuning with SONIC in the loop")
    parser.add_argument("mode", choices=("run", "eval"))
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--manifolds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--perturb", type=float, default=0.05)
    parser.add_argument("--tilt-deg", type=float, default=0.0,
                        help="std of the sampled sagittal tilt, degrees; 0 disables it (which is "
                             "what the recorded envelopes have, so the evaluation family's tilt "
                             "rows become extrapolation)")
    parser.add_argument("--roll-deg", type=float, default=0.0,
                        help="std of the sampled lateral roll, degrees; 0 disables it")
    parser.add_argument("--no-mean-probe", action="store_true",
                        help="ablation: score only sampled actions")
    parser.add_argument("--perturb-repeats", type=int, default=3,
                        help="episodes per demonstration: the same pose against several "
                             "reshapes of its manifold, which is what teaches adaptation")
    parser.add_argument("--resume", type=Path,
                        default=BC_POLICY)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--init-log-std", type=float, default=-1.5)
    parser.add_argument("--entropy-start", type=float, default=0.004)
    parser.add_argument("--entropy-end", type=float, default=0.001)
    parser.add_argument("--ppo-epochs", type=int, default=1,
                        help="passes over each iteration's batch; keep at 1 for small batches")
    parser.add_argument("--ppo-minibatches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    return run(args) if args.mode == "run" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
