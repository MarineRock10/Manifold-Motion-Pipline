"""Verify a pure-geometry policy with the real stack: same policy, two worlds.

The geometric model learned the action distribution; here the *same* policy drives
(a) the geometric model and (b) MuJoCo + frozen SONIC, using the identical task-space
observation. The comparison quantifies the executability gap.

    python3 -m manifold_g1.verify_geo --geo-policy reports/manifold_g1/geo_single/policy.pt \
        --manifold single --semi-x 2.6 --semi-y 1.3 --tunnel-height 0.72 --episodes 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .geo_env import GeoConfig, GeometricManifoldEnv
from .manifold import EllipsoidManifold
from .ppo import PPO
from .task_env import GoalReachEnv, TaskConfig


def rollout(env, ppo: PPO, reduced: bool, episodes: int) -> list[dict]:
    results = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = truncated = False
        pelvis: list[float] = []
        radii: list[float] = []
        info = {"outcome": "running"}
        while not (done or truncated):
            action, _, _ = ppo.act(env.reduced_obs() if reduced else obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            pelvis.append(info["base_z"])
            radii.append(info["manifold_radius_max"])
        results.append({"outcome": info["outcome"], "steps": info["episode_step"],
                        "return": info["episode_return"], "pelvis_z": float(np.mean(pelvis)),
                        "radius_max": float(np.max(radii))})
    return results


def summarize(name: str, results: list[dict]) -> dict:
    summary = {
        "world": name,
        "success": float(np.mean([r["outcome"] == "success" for r in results])),
        "out_of_manifold": float(np.mean([r["outcome"] == "out_of_manifold" for r in results])),
        "timeout": float(np.mean([r["outcome"] == "timeout" for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "pelvis_z": float(np.mean([r["pelvis_z"] for r in results])),
        "radius_max": float(np.mean([r["radius_max"] for r in results])),
        "mean_return": float(np.mean([r["return"] for r in results])),
    }
    print(f"  {name:>16}: success {summary['success']:.2f}  steps {summary['mean_steps']:5.1f}  "
          f"pelvis_z {summary['pelvis_z']:.3f}  r_max {summary['radius_max']:.2f}  "
          f"return {summary['mean_return']:+.2f}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Same geometric policy, geometric vs MuJoCo+SONIC")
    parser.add_argument("--geo-policy", type=Path, default=Path("reports/manifold_g1/geo_single/policy.pt"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--manifold", choices=("single", "tunnel"), default="single")
    parser.add_argument("--semi-x", type=float, default=2.6)
    parser.add_argument("--semi-y", type=float, default=1.3)
    parser.add_argument("--tunnel-height", type=float, default=0.72)
    parser.add_argument("--entry-height", type=float, default=1.3)
    parser.add_argument("--length", type=float, default=4.5)
    parser.add_argument("--primitives", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_g1/verify_geo.json"))
    args = parser.parse_args()

    if args.manifold == "single":
        manifold = EllipsoidManifold.single(semi_x=args.semi_x, semi_y=args.semi_y,
                                            semi_z=args.tunnel_height)
    else:
        manifold = EllipsoidManifold.tunnel(length=args.length, semi_y=args.semi_y,
                                            entry_semi_z=args.entry_height,
                                            tunnel_semi_z=args.tunnel_height, count=args.primitives)

    geo_env = GeometricManifoldEnv(manifold, GeoConfig())
    obs, _ = geo_env.reset()
    ppo = PPO(int(obs.shape[0]), geo_env.action_dim)
    ppo.load(args.geo_policy)
    print(f"policy {args.geo_policy}  obs_dim={obs.shape[0]}")

    print(f"manifold: {args.manifold}  start_x={manifold.start_x():.2f}  goal_x={manifold.goal_x():.2f}")
    print("same policy in both worlds:")
    geo_summary = summarize("geo model", rollout(geo_env, ppo, reduced=False, episodes=args.episodes))
    mujoco_env = GoalReachEnv(manifold, TaskConfig())
    mujoco_summary = summarize("MuJoCo+SONIC", rollout(mujoco_env, ppo, reduced=True, episodes=args.episodes))

    first = manifold.primitives[0]
    report = {"manifold": {"center": [float(v) for v in first.center],
                           "semi": [float(v) for v in first.semi],
                           "axis": [float(v) for v in first.axis()],
                           "start_x": manifold.start_x(), "goal_x": manifold.goal_x()},
              "geometric": geo_summary, "mujoco_sonic": mujoco_summary,
              "executability_gap": {
                  "success_delta": mujoco_summary["success"] - geo_summary["success"],
                  "pelvis_delta": mujoco_summary["pelvis_z"] - geo_summary["pelvis_z"],
                  "steps_delta": mujoco_summary["mean_steps"] - geo_summary["mean_steps"],
              }}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"executability gap: success {report['executability_gap']['success_delta']:+.2f}, "
          f"pelvis {report['executability_gap']['pelvis_delta']:+.3f} m, "
          f"steps {report['executability_gap']['steps_delta']:+.1f}")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
