"""Static manifold fitting: given an ellipsoid, choose a pose that fits inside it.

Level 0 of the manifold curriculum. The simplest manifold is the robot's own standing
envelope (see `body_envelope.py`); here the ellipsoid is scaled down and the policy learns
how much to crouch so the whole body stays inside. No locomotion, one action dimension.

The body model is calibrated from the MJCF: landmark positions (pelvis, head, wrists,
knees, ankles) as a function of the crouch amount, written by `body_envelope.py`.

    python3 -m manifold_g1.static_fit train  --scale 0.75 --iterations 200
    python3 -m manifold_g1.static_fit verify --scale 0.75 --policy reports/.../policy.pt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import constants as C
from .manifold import EllipsoidManifold, Primitive
from .ppo import PPO, RolloutBuffer

ENVELOPE_JSON = C.REPO / "reports" / "manifold_g1" / "body_envelope.json"
DEFAULT_POLICY = C.REPO / "reports" / "manifold_g1" / "static_fit" / "policy.pt"


class BodyModel:
    """Landmark positions as a function of the crouch amount (linear interpolation)."""

    def __init__(self, path: Path = ENVELOPE_JSON):
        data = json.loads(Path(path).read_text())
        envelopes = sorted(data["envelopes"], key=lambda e: e["crouch_amount"])
        self.crouch = np.array([e["crouch_amount"] for e in envelopes])
        self.names = list(envelopes[0]["landmarks"].keys())
        self.points = np.array([[e["landmarks"][n] for n in self.names] for e in envelopes])  # [C, N, 3]
        self.standing_semi = np.array(envelopes[0]["semi"])
        self.standing_center = np.array(envelopes[0]["center"])

    def at(self, crouch: float) -> np.ndarray:
        """Landmark world positions at a crouch amount (pelvis at x=y=0)."""
        crouch = float(np.clip(crouch, self.crouch[0], self.crouch[-1]))
        return np.stack([np.interp(crouch, self.crouch, self.points[:, i, axis])
                         for i in range(len(self.names)) for axis in range(3)]).reshape(-1, 3)


def manifold_for_scale(body: BodyModel, scale: float, center_x: float = 0.0) -> EllipsoidManifold:
    """A static ellipsoid: the standing envelope squashed in height by `scale`.

    Only the vertical axis is scaled - the body's lateral extent (arm span) cannot be changed
    by crouching, so a laterally narrower manifold would be infeasible by construction.
    """
    semi = body.standing_semi.copy()
    semi[2] = body.standing_semi[2] * scale
    return EllipsoidManifold([Primitive(
        center=np.array([center_x, 0.0, body.standing_center[2] * scale]), semi=semi)])


@dataclass
class StaticConfig:
    dt: float = 0.1
    max_episode_steps: int = 60           # 6 s to settle and hold
    crouch_max: float = 1.3
    tau_h: float = 0.45
    w_inside: float = 1.0                 # per second inside the manifold
    w_outside: float = 5.0                # per second and unit of violation
    w_effort: float = 0.5                 # per second, discourages unnecessary crouching
    success_hold: int = 10                # steps inside before the episode counts as success
    success_margin: float = 0.98


class StaticFitEnv:
    """One pose dimension (crouch) vs a static ellipsoid manifold."""

    action_dim = 1

    def __init__(self, body: BodyModel, manifold: EllipsoidManifold, cfg: StaticConfig | None = None):
        self.body = body
        self.manifold = manifold
        self.cfg = cfg or StaticConfig()
        self.reset()

    def _radius(self, crouch: float) -> float:
        return float(self.manifold.radii(self.body.at(crouch)).max())

    def obs(self) -> np.ndarray:
        primitive = self.manifold.primitives[0]
        points = self.body.at(self.crouch)
        pelvis = points[self.body.names.index("pelvis")]
        return np.concatenate([
            [self.crouch / self.cfg.crouch_max, self._radius(self.crouch)],
            (primitive.center - pelvis) / 0.5,
            primitive.semi,
            [primitive.axis()[2]],
        ]).astype(np.float32)

    def reset(self, **_) -> tuple[np.ndarray, dict]:
        self.crouch = 0.0                     # start standing, outside a squashed manifold
        self.steps = 0
        self.inside_ticks = 0
        self.episode_return = 0.0
        return self.obs(), {"radius": self._radius(self.crouch)}

    def step(self, action: np.ndarray, tick_callback=None):
        cfg = self.cfg
        target = float(np.clip(float(np.asarray(action).reshape(-1)[0]), 0.0, 1.0)) * cfg.crouch_max
        reward = 0.0
        terminated = truncated = False
        outcome = "running"
        for _ in range(5):                    # 50 Hz inner loop, like the other envs
            self.crouch += (target - self.crouch) * min(1.0, C.CONTROL_DT / cfg.tau_h)
            radius = self._radius(self.crouch)
            if radius <= 1.0:
                reward += cfg.w_inside * C.CONTROL_DT
                self.inside_ticks += 1
            else:
                reward -= cfg.w_outside * (radius - 1.0) * C.CONTROL_DT
                self.inside_ticks = 0
            reward -= cfg.w_effort * (self.crouch / cfg.crouch_max) * C.CONTROL_DT
            if tick_callback is not None:
                tick_callback()
        self.steps += 1
        radius = self._radius(self.crouch)
        if self.inside_ticks >= cfg.success_hold:
            reward += 5.0
            terminated, outcome = True, "success"
        elif self.steps >= cfg.max_episode_steps:
            truncated, outcome = True, "timeout"
        self.episode_return += reward
        return self.obs(), float(reward), terminated, truncated, {
            "outcome": outcome, "episode_step": self.steps, "episode_return": self.episode_return,
            "crouch_amount": float(self.crouch), "manifold_radius": float(radius),
            "manifold_radius_max": float(radius), "spine_alignment": 1.0,
            "base_z": float(self.body.at(self.crouch)[self.body.names.index("pelvis")][2]),
            "distance": 0.0, "tunnel_semi_z": float(self.manifold.primitives[0].semi[2]),
        }


def train(args) -> int:
    body = BodyModel()
    rng = np.random.default_rng(args.seed)
    env = StaticFitEnv(body, manifold_for_scale(body, args.scale_max), StaticConfig())
    obs, _ = env.reset()
    ppo = PPO(int(obs.shape[0]), env.action_dim, init_log_std=[-0.5])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"static fit: height scale sampled in [{args.scale_min:.2f}, {args.scale_max:.2f}]")
    for scale in (args.scale_max, args.scale_min):
        env.manifold = manifold_for_scale(body, scale)
        print(f"  scale {scale:.2f}: standing r={env._radius(0.0):.2f}  "
              f"full-crouch r={env._radius(env.cfg.crouch_max):.2f}  "
              f"(the manifold only binds if standing r > 1)")
    for iteration in range(1, args.iterations + 1):
        buffer = RolloutBuffer(args.rollout_steps, obs.shape[0], env.action_dim)
        scale = float(rng.uniform(args.scale_min, args.scale_max))
        env.manifold = manifold_for_scale(body, scale)
        obs, _ = env.reset()
        returns, outcomes, crouches = [], [], []
        episode_return = 0.0
        for _ in range(args.rollout_steps):
            current_obs = obs
            action, logprob, value = ppo.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            buffer.add(current_obs, action, logprob, reward, value, float(done))
            episode_return += reward
            if done or truncated:
                returns.append(episode_return)
                outcomes.append(info["outcome"])
                crouches.append(info["crouch_amount"])
                episode_return = 0.0
                scale = float(rng.uniform(args.scale_min, args.scale_max))
                env.manifold = manifold_for_scale(body, scale)
                obs, _ = env.reset()
        ppo.update(buffer, ppo.value(obs))
        if iteration % max(1, args.iterations // 8) == 0 or iteration == 1:
            success = np.mean([o == "success" for o in outcomes]) if outcomes else float("nan")
            print(f"[iter {iteration:3d}] episodes={len(outcomes):2d} success={success:.2f} "
                  f"crouch={np.mean(crouches) if crouches else float('nan'):.2f} "
                  f"return={np.mean(returns) if returns else float('nan'):+.2f}", flush=True)
        ppo.save(out_dir / "policy.pt")

    print("\nfinal greedy behaviour (the learned crouch policy):")
    for scale in (1.0, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6):
        env.manifold = manifold_for_scale(body, scale)
        obs, _ = env.reset()
        for _ in range(env.cfg.max_episode_steps):
            action, _, _ = ppo.act(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        mean, std = ppo.action_stats(obs)
        print(f"  scale {scale:.2f} (semi_z {env.manifold.primitives[0].semi[2]:.3f}): "
              f"{info['outcome']:>8s} crouch={info['crouch_amount']:.3f} r={info['manifold_radius']:.2f} "
              f"| action mean={mean[0]:+.2f} std={std[0]:.2f}")
    (out_dir / "summary.json").write_text(json.dumps(
        {"scale_range": [args.scale_min, args.scale_max],
         "standing_semi": body.standing_semi.tolist(),
         "standing_center": body.standing_center.tolist()}, indent=2))
    return 0


def verify(args) -> int:
    """Run the static policy in MuJoCo + SONIC: command the crouch, measure the real fit.

    Containment is evaluated pelvis-relative (this is a *pose* envelope task), which removes
    the robot's forward drift from the comparison.
    """
    import mujoco

    from .task_env import GoalReachEnv, TaskConfig

    body = BodyModel()
    ppo = PPO(9, 1)
    ppo.load(Path(args.policy))
    env = GoalReachEnv(manifold_for_scale(body, 1.0), TaskConfig())
    pelvis_index = body.names.index("pelvis")

    print("SONIC verification of the static policy (pelvis-relative containment):")
    print(f"{'scale':>6} {'crouch':>7} {'pelvis_z':>9} {'r_model':>8} {'r_real':>7} {'body point (worst)':>20}")
    for scale in (1.0, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6):
        manifold = manifold_for_scale(body, scale)
        env.set_manifold(manifold)
        obs, _ = env.reset()
        env.crouch_amount = 0.0
        env.reference.set_amount(0.0)
        info = {"manifold_radius_max": 0.0}
        radii_real = []
        for _ in range(60):
            st = env.env.state()
            landmark = {}
            for name in body.names:
                if name == "head_virtual":
                    torso = env.env.data.xpos[mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")]
                    landmark[name] = torso + np.array([0.0, 0.0, 0.30])
                else:
                    bid = mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_BODY, name)
                    landmark[name] = env.env.data.xpos[bid].copy()
            pelvis = landmark["pelvis"]
            model_pelvis = body.at(env.crouch_amount)[pelvis_index]
            points = np.array([landmark[n] - pelvis + model_pelvis for n in body.names])
            radii = manifold.radii(points)
            radius_real = float(radii.max())
            radii_real.append(radius_real)
            static_obs = np.concatenate([
                [env.crouch_amount / env.cfg.crouch_max, radius_real],
                (manifold.primitives[0].center - model_pelvis) / 0.5,
                manifold.primitives[0].semi, [manifold.primitives[0].axis()[2]],
            ]).astype(np.float32)
            action, _, _ = ppo.act(static_obs, deterministic=True)
            cmd = float(np.clip(action[0], 0.0, 1.0))
            _, _, done, truncated, info = env.step(np.array([0.0, 0.0, 0.0, cmd]))
            if done or truncated:
                break
        worst = body.names[int(np.argmax(radii))]
        print(f"{scale:>6.2f} {info['crouch_amount']:>7.3f} {info['base_z']:>9.3f} "
              f"{float(manifold.radii(body.at(info['crouch_amount'])).max()):>8.2f} "
              f"{radii_real[-1]:>7.2f} {worst:>20s}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Static manifold fitting (crouch to fit)")
    parser.add_argument("mode", choices=("train", "verify"))
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--scale-min", type=float, default=0.62)
    parser.add_argument("--scale-max", type=float, default=0.78)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_g1/static_fit"))
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    return train(args) if args.mode == "train" else verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
