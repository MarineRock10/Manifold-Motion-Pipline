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

    ./scripts/python.sh -m manifold_motion.stage1.sonic_rl run --iterations 30 --manifolds 16 --steps 6
    ./scripts/python.sh -m manifold_motion.stage1.sonic_rl eval --policy .../policy_sonicrl.pt    # compare on SONIC
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manifold_motion.core import constants as C
from manifold_motion.core.body_model import BodyModel
from manifold_motion.dataio.demos import DEFAULT_ISAAC
from manifold_motion.core.family import BASE_CENTER, BASE_SEMI
from manifold_motion.simulation.keyframe_env import KeyframeEnv
from manifold_motion.core.manifold import EllipsoidManifold, Primitive
from manifold_motion.stage1.pose_policy import OBS_DIM, POSE_DIM, POSE_LIMIT, action_to_pose, observation
from manifold_motion.stage1.ppo import PPO, RolloutBuffer
from manifold_motion.dataio.demos import demo_values, load_demos
from manifold_motion.core.paths import BC_POLICY, RL_DIR, RL_POLICY

OUT = RL_DIR


def _warm_start_actor(torch, checkpoint: dict[str, object], device, init_log_std: float):
    """Build an actor whose deterministic pose is exactly the Stage-1 imitation output.

    The imitation checkpoint predicts a joint delta, while PPO samples the pre-``tanh`` action
    used by :func:`action_to_pose`.  Loading the imitation MLP directly into PPO would therefore
    change the pose scale.  This wrapper keeps the same encoder/head weights and maps the delta
    through ``atanh(delta / POSE_LIMIT)`` before constructing the Gaussian policy.  The value head
    is intentionally new: it is learned from the SONIC/MuJoCo reward during fine-tuning.
    """
    import torch.nn as nn

    input_dim = int(checkpoint["input_dim"])
    output_dim = int(checkpoint["output_dim"])
    hidden = int(checkpoint.get("hidden", 256))
    if output_dim != POSE_DIM:
        raise ValueError(f"imitation checkpoint output_dim={output_dim}, expected {POSE_DIM}")

    class WarmActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
            )
            self.pose = nn.Linear(hidden, output_dim)
            self.value = nn.Linear(hidden, 1)
            self.log_std = nn.Parameter(torch.full((output_dim,), float(init_log_std)))

        def forward(self, obs):
            features = self.body(obs)
            pose_delta = self.pose(features)
            scaled = torch.clamp(pose_delta / POSE_LIMIT, -0.999, 0.999)
            raw_action = torch.atanh(scaled)
            return raw_action, self.value(features).squeeze(-1)

        def distribution(self, obs):
            mu, value = self(obs)
            return torch.distributions.Normal(mu, self.log_std.exp()), value

    model = WarmActorCritic().to(device)
    source = checkpoint["model"]
    state = {
        "body.0.weight": source["0.weight"], "body.0.bias": source["0.bias"],
        "body.1.weight": source["1.weight"], "body.1.bias": source["1.bias"],
        "body.3.weight": source["3.weight"], "body.3.bias": source["3.bias"],
        "pose.weight": source["5.weight"], "pose.bias": source["5.bias"],
    }
    model.load_state_dict(state, strict=False)
    return model


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

    def __init__(self, cfg: SonicConfig | None = None, seed: int = 0,
                 catalog: str | Path | None = None, catalog_split: int = 0,
                 catalog_target_field: str = "target_exec", catalog_target_frame: int = -1):
        self.cfg = cfg or SonicConfig()
        self.body = BodyModel()
        self.env = KeyframeEnv()
        self.rng = np.random.default_rng(seed)
        self.manifold: EllipsoidManifold | None = None
        self.demo: np.ndarray | None = None
        self._keys, self._demos = load_demos()
        self._entries = [(k, self._demos[k]) for k in self._keys if len(self._demos[k])]
        self.achieved = np.zeros(POSE_DIM)
        self._sample_count = 0
        self._catalog = None
        self._catalog_features = None
        self._catalog_row = None
        self._catalog_ids = None
        self._catalog_target = None
        self._catalog_target_frame = None
        if catalog is not None:
            with np.load(Path(catalog), allow_pickle=False) as archive:
                corridor = np.asarray(archive["corridor"], dtype=np.float32)
                target = np.asarray(archive[catalog_target_field], dtype=np.float32)
                split = np.asarray(archive["split"], dtype=np.uint8)
            if corridor.ndim != 3 or corridor.shape[2] != 7:
                raise ValueError(f"catalog corridor must be [N,T,7], got {corridor.shape}")
            if target.ndim != 3 or target.shape[0] != len(corridor) or target.shape[2] < POSE_DIM:
                raise ValueError(f"catalog target must be [N,T,>={POSE_DIM}], got {target.shape}")
            if catalog_split not in (0, 1, 2):
                raise ValueError("catalog_split must be 0 (train), 1 (validation) or 2 (test)")
            ids = np.flatnonzero(split == catalog_split)
            if not len(ids):
                raise ValueError(f"catalog split {catalog_split} is empty")
            frame = target.shape[1] // 2 if catalog_target_frame < 0 else int(catalog_target_frame)
            if frame < 0 or frame >= target.shape[1]:
                raise ValueError(f"catalog target frame {frame} is outside horizon {target.shape[1]}")
            self._catalog = corridor
            from manifold_motion.stage1.manifold_imitation import manifold_features
            self._catalog_features = manifold_features(corridor)
            self._catalog_ids = ids
            self._catalog_target = target
            self._catalog_target_frame = frame
            self._catalog_source = str(catalog)

    def policy_observation(self, imitation: bool = False,
                           feature_mean: np.ndarray | None = None,
                           feature_std: np.ndarray | None = None) -> np.ndarray:
        """Return the observation contract for the selected policy.

        The legacy PPO policy sees the 15-dim closed-loop state observation.  A warm-started
        Stage-1 imitation policy sees the same normalized 22-dim static ``M_e`` descriptor used
        during supervised training; keeping this normalization in one method prevents the
        common error of loading an imitation checkpoint with raw, unnormalized corridor values.
        """
        if not imitation:
            return self.obs()
        if self._catalog is None or self._catalog_features is None or self._catalog_row is None:
            raise RuntimeError("imitation observation requires a catalog episode")
        if feature_mean is None or feature_std is None:
            raise ValueError("imitation observation requires checkpoint feature statistics")
        value = (self._catalog_features[self._catalog_row] - feature_mean) / feature_std
        return np.asarray(value, dtype=np.float32)

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
        reuse = (repeats > 1 and self.manifold is not None and self.demo is not None
                 and self._sample_count % repeats != 0)
        self._sample_count += 1
        if self._catalog is not None:
            if reuse:
                row = self._catalog_row
            else:
                row = int(self._catalog_ids[int(self.rng.integers(len(self._catalog_ids)))])
                self._catalog_row = row
            frame = self._catalog_target_frame
            frame_data = self._catalog[row, frame]
            self._base_values = np.concatenate([frame_data[3:6], frame_data[:3]])
            default = C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB]
            self.demo = self._catalog_target[row, frame, :POSE_DIM] - default
            return self._reshape(self._base_values, perturb, keep_demo=True)
        if reuse:
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
        from manifold_motion.core.constants import quat_mul

        self.manifold = EllipsoidManifold([Primitive(center=center, semi=semi,
                                                     quat=quat_mul(qr, qt))])
        self.env.reset()
        self.achieved = np.zeros(POSE_DIM)
        self.elapsed = 0.0
        return self.obs()

    def _snapshot(self) -> dict[str, object]:
        """Capture MuJoCo + SONIC recurrent state for a same-state mean probe."""
        key = self.env
        sim = key.env
        controller = key.controller
        reference = key.reference
        if reference is None:
            raise RuntimeError("cannot snapshot before reset")
        history = [{name: value.copy() for name, value in entry.items()}
                   for entry in controller.history]
        return {
            "qpos": sim.data.qpos.copy(), "qvel": sim.data.qvel.copy(),
            "ctrl": sim.data.ctrl.copy(), "q_des": sim.q_des.copy(), "time": sim.time,
            "reference_joint_pos": reference.joint_pos.copy(),
            "reference_joint_vel": reference.joint_vel.copy(), "reference_cursor": reference.cursor,
            "reference_play": reference.play, "history": history,
            "last_action": controller.last_action.copy(),
            "delta_heading": None if controller.delta_heading is None else controller.delta_heading.copy(),
            "achieved": self.achieved.copy(), "elapsed": self.elapsed,
        }

    def _restore(self, snapshot: dict[str, object]) -> None:
        """Restore a snapshot captured by :meth:`_snapshot` before another probe."""
        import mujoco

        key = self.env
        sim = key.env
        controller = key.controller
        reference = key.reference
        if reference is None:
            raise RuntimeError("cannot restore before reset")
        sim.data.qpos[:] = snapshot["qpos"]
        sim.data.qvel[:] = snapshot["qvel"]
        sim.data.ctrl[:] = snapshot["ctrl"]
        sim.q_des[:] = snapshot["q_des"]
        sim.time = float(snapshot["time"])
        reference.joint_pos = np.asarray(snapshot["reference_joint_pos"]).copy()
        reference.joint_vel = np.asarray(snapshot["reference_joint_vel"]).copy()
        reference.cursor = int(snapshot["reference_cursor"])
        reference.play = bool(snapshot["reference_play"])
        controller.history.clear()
        for entry in snapshot["history"]:
            controller.history.append({name: value.copy() for name, value in entry.items()})
        controller.last_action = np.asarray(snapshot["last_action"]).copy()
        heading = snapshot["delta_heading"]
        controller.delta_heading = None if heading is None else np.asarray(heading).copy()
        self.achieved = np.asarray(snapshot["achieved"]).copy()
        self.elapsed = float(snapshot["elapsed"])
        mujoco.mj_forward(sim.model, sim.data)

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
        from manifold_motion.core.kinematics import TorchKinematics

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
                      tilt_deg=args.tilt_deg, roll_deg=args.roll_deg,
                      w_containment=args.w_containment,
                      w_outside=args.w_outside,
                      w_track=args.w_track,
                      w_imitation=args.w_imitation)
    env = SonicPoseEnv(cfg, seed=args.seed, catalog=args.catalog,
                       catalog_split=args.catalog_split,
                       catalog_target_field=args.catalog_target_field,
                       catalog_target_frame=args.catalog_target_frame)
    imitation_checkpoint = None
    imitation_obs = False
    feature_mean = feature_std = None
    if args.imitation_checkpoint is not None:
        if args.catalog is None:
            raise ValueError("--imitation-checkpoint requires --catalog")
        imitation_checkpoint = torch.load(args.imitation_checkpoint, map_location="cpu",
                                          weights_only=False)
        feature_mean = np.asarray(imitation_checkpoint["input_mean"], dtype=np.float32)
        feature_std = np.asarray(imitation_checkpoint["input_std"], dtype=np.float32)
        obs_dim = int(imitation_checkpoint["input_dim"])
        ppo = PPO(obs_dim, POSE_DIM, lr=args.learning_rate,
                  init_log_std=[args.init_log_std] * POSE_DIM, device=args.device)
        ppo.model = _warm_start_actor(torch, imitation_checkpoint, ppo.device,
                                      args.init_log_std)
        ppo.optimizer = torch.optim.Adam(ppo.model.parameters(), lr=3e-4, eps=1e-5)
        imitation_obs = True
        print(f"warm-started SONIC policy from {args.imitation_checkpoint}; "
              f"pose head {obs_dim}->{POSE_DIM} with atanh action adapter")
    else:
        obs_dim = OBS_DIM
        ppo = PPO(obs_dim, POSE_DIM, lr=args.learning_rate,
                  init_log_std=[-1.0] * POSE_DIM, device=args.device)
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
    resume_path = args.resume
    if imitation_checkpoint is not None and resume_path == BC_POLICY:
        # The legacy BC checkpoint has the 15-dim actor contract and cannot be loaded into the
        # 22-dim catalog actor.  An explicit --resume remains supported for warm-policy resumes.
        resume_path = None
    if resume_path and Path(resume_path).exists():
        blob = torch.load(resume_path, map_location=ppo.device, weights_only=False)
        state = dict(blob["model"]); state.pop("log_std", None)
        ppo.model.load_state_dict(state, strict=False)
        with torch.no_grad():
            ppo.model.log_std.fill_(args.init_log_std)
        print(f"resumed {resume_path}; log_std -> {args.init_log_std}")
    elif resume_path:
        print(f"resume checkpoint not found ({resume_path}); starting from a fresh policy")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.csv", "w")
    log.write("iteration,episodes,success,r_achieved,gated,return_mean\n")
    print(f"SONIC in the loop: {args.manifolds} manifolds x {args.steps} ticks x "
          f"{cfg.settle}s physics = {args.manifolds * args.steps * cfg.settle:.0f} s per iteration")
    started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        frac = (iteration - 1) / max(1, args.iterations - 1)
        ppo.entropy_coef = args.entropy_start + (args.entropy_end - args.entropy_start) * frac
        buffer = RolloutBuffer(args.manifolds * args.steps, obs_dim, POSE_DIM)
        returns, successes, radii, gated = [], [], [], 0
        for episode in range(args.manifolds):
            # every few episodes, reshape the *same* demonstration's manifold again: the policy
            # sees one recorded pose against several manifold deformations in a row
            env.sample_episode(args.perturb, repeats=args.perturb_repeats)
            obs = env.policy_observation(imitation_obs, feature_mean, feature_std)
            episode_return = 0.0
            for _ in range(args.steps):
                # the policy proposes a pose; the manifest is executed by the controller
                import torch as _t
                action, logprob, value = ppo.act(obs)
                pose = np.asarray(action_to_pose(action, _t))
                probe_snapshot = env._snapshot() if cfg.mean_probe else None
                _, reward, done, info = env.step(pose)
                next_obs = env.policy_observation(imitation_obs, feature_mean, feature_std)
                # Also score the mean action. Deployment uses the mean, but PPO explores with
                # samples, so a run can look like it improves (`r` of sampled poses falls) while
                # the mean - the pose that would actually be executed - does not. Every collapse
                # this project has seen had that shape.
                if cfg.mean_probe:
                    with _t.no_grad():
                        mean_action = ppo.model(_t.as_tensor(obs, dtype=_t.float32,
                                                             device=args.device))[0]
                    mean_pose = np.asarray(action_to_pose(mean_action, _t))
                    env._restore(probe_snapshot)
                    _, mean_reward, mean_done, mean_info = env.step(mean_pose)
                    reward = reward + cfg.w_mean_probe * mean_reward
                    info = mean_info          # report the mean's outcome: that is what deploys
                    done = mean_done
                    next_obs = env.policy_observation(imitation_obs, feature_mean, feature_std)
                buffer.add(obs, action, logprob, reward, value, float(done))
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

    from manifold_motion.core.kinematics import TorchKinematics

    cfg = SonicConfig(steps=args.steps, mean_probe=not args.no_mean_probe,
                      tilt_deg=args.tilt_deg, roll_deg=args.roll_deg)
    env = SonicPoseEnv(cfg, seed=args.seed, catalog=args.catalog,
                       catalog_split=args.catalog_split,
                       catalog_target_field=args.catalog_target_field,
                       catalog_target_frame=args.catalog_target_frame)
    kin = TorchKinematics(device="cpu")
    imitation_checkpoint = None
    imitation_obs = False
    feature_mean = feature_std = None
    if args.imitation_checkpoint is not None:
        if args.catalog is None:
            raise ValueError("--imitation-checkpoint requires --catalog")
        imitation_checkpoint = torch.load(args.imitation_checkpoint, map_location="cpu",
                                          weights_only=False)
        feature_mean = np.asarray(imitation_checkpoint["input_mean"], dtype=np.float32)
        feature_std = np.asarray(imitation_checkpoint["input_std"], dtype=np.float32)
        imitation_obs = True
    policies = {}
    if not args.skip_bc and BC_POLICY.exists():
        policies["BC"] = BC_POLICY
    if args.policy:
        policies[Path(args.policy).stem] = Path(args.policy)
    if not policies:
        raise FileNotFoundError("no policy to evaluate; pass --policy or build the BC policy")
    models = {}
    for name, path in policies.items():
        obs_dim = int(imitation_checkpoint["input_dim"]) if imitation_checkpoint is not None else OBS_DIM
        ppo = PPO(obs_dim, POSE_DIM, lr=args.learning_rate, device=args.device)
        if imitation_checkpoint is not None:
            ppo.model = _warm_start_actor(torch, imitation_checkpoint, ppo.device, -1.0)
            ppo.optimizer = torch.optim.Adam(ppo.model.parameters(), lr=3e-4, eps=1e-5)
        ppo.load(path)
        models[name] = (ppo, imitation_obs)

    print(f"{'manifold':>22} " + " ".join(f"{n:>18}" for n in models))
    rows = []
    for i in range(args.manifolds):
        env.sample_episode(args.perturb)
        base_obs = env.policy_observation(imitation_obs, feature_mean, feature_std)
        cells, record = [], {"index": i}
        for name, (ppo, use_imitation_obs) in models.items():
            o = base_obs.copy()
            for _ in range(args.steps):
                a, _, _ = ppo.act(o, deterministic=True)
                pose = np.asarray(action_to_pose(a, torch))
                _, _, _, info = env.step(pose)
                o = env.policy_observation(use_imitation_obs, feature_mean, feature_std)
            record[name] = {"r": info["r_achieved"], "track": info["track"],
                            "gate": info["gate"]}
            cells.append(f"r {info['r_achieved']:.2f} trk {info['track']:.2f}")
            env.env.reset()          # keep every policy measured from the same start
            env.env.controller.reset()  # do not leak SONIC's recurrent state across policies
            env.achieved = np.zeros(POSE_DIM)
            env.elapsed = 0.0
        print(f"{str(np.round(env.manifold.primitives[0].semi, 2)):>22} " +
              " ".join(f"{c:>18}" for c in cells))
        rows.append(record)
    out = Path(args.out)
    (out / "compare.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(rows, indent=2))
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
    parser.add_argument("--imitation-checkpoint", type=Path, default=None,
                        help="Stage-1 imitation checkpoint used to warm-start the catalog actor")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--w-containment", type=float, default=4.0,
                        help="reward for staying inside the corridor")
    parser.add_argument("--w-outside", type=float, default=8.0,
                        help="penalty once the achieved mesh leaves the corridor")
    parser.add_argument("--w-track", type=float, default=2.0,
                        help="SONIC requested/achieved pose tracking penalty")
    parser.add_argument("--w-imitation", type=float, default=1.5,
                        help="anchor to the accepted imitation pose")
    parser.add_argument("--catalog", type=Path, default=None,
                        help="catalog NPZ; when set, sample accepted reverse-synthesized windows")
    parser.add_argument("--catalog-split", type=int, default=0,
                        help="catalog split: 0 train, 1 validation, 2 test")
    parser.add_argument("--catalog-target-field", choices=("target_ref", "target_exec"),
                        default="target_exec")
    parser.add_argument("--catalog-target-frame", type=int, default=-1,
                        help="catalog target frame; -1 selects the middle frame")
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--skip-bc", action="store_true",
                        help="do not require the legacy BC checkpoint when evaluating --policy")
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
