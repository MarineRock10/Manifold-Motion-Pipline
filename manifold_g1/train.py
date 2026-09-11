"""Headless PPO training on the soft-manifold goal task.

The manifold is a chain of 2-3 ellipsoids on flat ground; the far one can be flattened
(a "low tunnel") so the robot has to crouch. Containment is a soft reward term, so no
collision geometry and no model rebuilds are involved.

Example:
  python3 -m manifold_g1.train --iterations 60 --rollout-steps 512 --curriculum
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .geo_env import GeoConfig, GeometricManifoldEnv
from .manifold import EllipsoidManifold
from .ppo import PPO, RolloutBuffer
from .task_env import GoalReachEnv, TaskConfig


def make_manifold(args, tunnel_height: float) -> EllipsoidManifold:
    """One fixed manifold: a single ellipsoid by default, or a short chain."""
    if args.manifold == "single":
        return EllipsoidManifold.single(semi_x=args.semi_x, semi_y=args.semi_y,
                                        semi_z=tunnel_height, tilt_deg=args.tilt_deg)
    return EllipsoidManifold.tunnel(
        length=args.length, semi_y=args.semi_y, entry_semi_z=args.entry_height,
        tunnel_semi_z=tunnel_height, count=args.primitives, tilt_deg=args.tilt_deg,
    )


def evaluate(ppo: PPO, env: GoalReachEnv, episodes: int = 5, tunnel_height: float | None = None) -> dict:
    results = []
    for _ in range(episodes):
        obs, _ = env.reset(tunnel_semi_z=tunnel_height)
        done = truncated = False
        steps = 0
        episode_return = 0.0
        pelvis_z: list[float] = []
        radii: list[float] = []
        spines: list[float] = []
        action_means: list[np.ndarray] = []
        action_stds: list[np.ndarray] = []
        info = {"outcome": "running"}
        while not (done or truncated):
            mean, std = ppo.action_stats(obs)
            action_means.append(mean)
            action_stds.append(std)
            action, _, _ = ppo.act(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_return += reward
            steps += 1
            pelvis_z.append(info["base_z"])
            radii.append(info["manifold_radius_max"])
            spines.append(info["spine_alignment"])
        results.append({"outcome": info["outcome"], "steps": steps, "return": episode_return,
                        "action_mean": np.mean(action_means, axis=0).tolist(),
                        "action_std": np.mean(action_stds, axis=0).tolist(),
                        "pelvis_z": float(np.mean(pelvis_z)), "radius_max": float(np.max(radii)),
                        "spine": float(np.mean(spines))})
    return {
        "success": float(np.mean([r["outcome"] == "success" for r in results])),
        "fall": float(np.mean([r["outcome"] == "fall" for r in results])),
        "out_of_manifold": float(np.mean([r["outcome"] == "out_of_manifold" for r in results])),
        "timeout": float(np.mean([r["outcome"] == "timeout" for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_return": float(np.mean([r["return"] for r in results])),
        "pelvis_z": float(np.mean([r["pelvis_z"] for r in results])),
        "radius_max": float(np.mean([r["radius_max"] for r in results])),
        "spine": float(np.mean([r["spine"] for r in results])),
        "tunnel_height": tunnel_height,
        "action_mean": np.mean([r["action_mean"] for r in results], axis=0).tolist(),
        "action_std": np.mean([r["action_std"] for r in results], axis=0).tolist(),
        "episodes": episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PPO on the soft-manifold goal task")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--env", choices=("geo", "mujoco"), default="geo",
                        help="geo = pure-geometry model (fast, default); mujoco = frozen SONIC in MuJoCo")
    parser.add_argument("--manifold", choices=("single", "tunnel"), default="single",
                        help="single ellipsoid (first pipeline) or a short chain")
    parser.add_argument("--semi-x", type=float, default=2.6, help="half-length of a single ellipsoid")
    parser.add_argument("--length", type=float, default=4.5)
    parser.add_argument("--semi-y", type=float, default=1.6)
    parser.add_argument("--entry-height", type=float, default=1.3, help="half-height of the entry ellipsoid")
    parser.add_argument("--tunnel-height", type=float, default=1.3, help="half-height of the far ellipsoid")
    parser.add_argument("--tunnel-height-min", type=float, default=0.62)
    parser.add_argument("--primitives", type=int, default=3)
    parser.add_argument("--tilt-deg", type=float, default=0.0)
    parser.add_argument("--curriculum", action="store_true",
                        help="lower the far ellipsoid as the rolling success rate improves")
    parser.add_argument("--curriculum-success", type=float, default=0.8)
    parser.add_argument("--curriculum-step", type=float, default=0.04)
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_g1/ppo_soft"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-tunnels", default=None,
                        help="comma-separated far-ellipsoid half-heights for an evaluation sweep")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifold = make_manifold(args, args.tunnel_height)
    env = (GeometricManifoldEnv(manifold, GeoConfig())
           if args.env == "geo" else GoalReachEnv(manifold, TaskConfig()))
    obs, _ = env.reset()
    obs_dim = int(obs.shape[0])
    init_log_std = [-1.0] * (env.action_dim - 1) + [0.3]   # extra exploration on the crouch dim
    # start from holding the spawn crouch: a policy that stands up immediately dies inside a
    # low manifold before it can discover the goal, a fatal local optimum
    init_mu = np.zeros(env.action_dim, dtype=np.float32)
    init_mu[-1] = env.cfg.spawn_crouch / env.cfg.crouch_max
    ppo = PPO(obs_dim, env.action_dim, device=args.device, init_log_std=init_log_std,
              init_mu_bias=init_mu)
    if args.resume is not None:
        ppo.load(args.resume)
        print(f"resumed from {args.resume}")
    print(f"env={args.env} obs_dim={obs_dim} action_dim={env.action_dim} "
          f"primitives={len(env.manifold.primitives)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        if args.eval_tunnels:
            results = [evaluate(ppo, env, args.eval_episodes, float(v)) for v in args.eval_tunnels.split(",")]
            print(json.dumps(results, indent=2))
            (out_dir / "eval_tunnels.json").write_text(json.dumps(results, indent=2))
        else:
            print(json.dumps(evaluate(ppo, env, args.eval_episodes), indent=2))
        return 0

    log_path = out_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["iteration", "env_steps", "episodes", "mean_return", "success_rate",
                     "fall_rate", "timeout_rate", "tunnel_floor", "wall_time_s",
                     "policy_loss", "value_loss", "entropy"])
    episode_path = out_dir / "episodes.csv"
    episode_file = open(episode_path, "w", newline="")
    episode_writer = csv.writer(episode_file)
    episode_writer.writerow(["iteration", "tunnel_height", "outcome", "steps", "return",
                             "pelvis_z", "radius_max", "spine"])

    tunnel_floor = args.tunnel_height
    sample = (lambda: float(rng.uniform(tunnel_floor, min(tunnel_floor + 0.2, args.tunnel_height)))
              if args.curriculum else args.tunnel_height)

    total_steps = 0
    wall_start = time.time()
    recent_success: list[float] = []
    tunnel = sample()
    obs, _ = env.reset(tunnel_semi_z=tunnel if args.curriculum else None)

    for iteration in range(1, args.iterations + 1):
        buffer = RolloutBuffer(args.rollout_steps, obs_dim, env.action_dim)
        episode_returns: list[float] = []
        outcomes: list[str] = []
        episode_return = 0.0
        episode_pelvis: list[float] = []
        episode_radius: list[float] = []
        episode_spine: list[float] = []
        for _ in range(args.rollout_steps):
            action, logprob, value = ppo.act(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            buffer.add(obs, action, logprob, reward, value, float(done))
            episode_return += reward
            episode_pelvis.append(info["base_z"])
            episode_radius.append(info["manifold_radius_max"])
            episode_spine.append(info["spine_alignment"])
            obs = next_obs
            total_steps += 1
            if done or truncated:
                episode_returns.append(episode_return)
                outcomes.append(info["outcome"])
                recent_success.append(1.0 if info["outcome"] == "success" else 0.0)
                episode_writer.writerow([iteration, f"{info['tunnel_semi_z']:.3f}", info["outcome"],
                                         info["episode_step"], f"{episode_return:.3f}",
                                         f"{np.mean(episode_pelvis):.3f}",
                                         f"{np.max(episode_radius):.3f}",
                                         f"{np.mean(episode_spine):.3f}"])
                episode_file.flush()
                episode_return = 0.0
                episode_pelvis = []
                episode_radius = []
                episode_spine = []
                tunnel = sample()
                obs, _ = env.reset(tunnel_semi_z=tunnel if args.curriculum else None)

        stats = ppo.update(buffer, ppo.value(obs))

        if args.curriculum and len(recent_success) >= 20 and float(np.mean(recent_success[-20:])) > args.curriculum_success:
            if tunnel_floor > args.tunnel_height_min:
                tunnel_floor = max(args.tunnel_height_min, tunnel_floor - args.curriculum_step)
                print(f"  [curriculum] success {np.mean(recent_success[-20:]):.2f} -> tunnel floor "
                      f"{tunnel_floor:.2f} m", flush=True)

        mean_return = float(np.mean(episode_returns)) if episode_returns else float("nan")
        success_rate = float(np.mean([o == "success" for o in outcomes])) if outcomes else float("nan")
        fall_rate = float(np.mean([o == "fall" for o in outcomes])) if outcomes else float("nan")
        timeout_rate = float(np.mean([o == "timeout" for o in outcomes])) if outcomes else float("nan")
        writer.writerow([iteration, total_steps, len(outcomes), f"{mean_return:.3f}",
                         f"{success_rate:.3f}", f"{fall_rate:.3f}", f"{timeout_rate:.3f}",
                         f"{tunnel_floor:.2f}", f"{time.time() - wall_start:.1f}",
                         f"{stats['policy_loss']:.4f}", f"{stats['value_loss']:.4f}",
                         f"{stats['entropy']:.4f}"])
        log_file.flush()
        ppo.save(out_dir / "policy.pt")
        print(f"[iter {iteration:3d}] steps={total_steps:6d} episodes={len(outcomes):2d} "
              f"return={mean_return:8.2f} success={success_rate:5.2f} fall={fall_rate:5.2f} "
              f"timeout={timeout_rate:5.2f} tunnel>={tunnel_floor:.2f} "
              f"pi_loss={stats['policy_loss']:+.3f} v_loss={stats['value_loss']:.3f} "
              f"H={stats['entropy']:.3f}", flush=True)

        if iteration % args.eval_every == 0:
            heights = [tunnel_floor, args.tunnel_height] if args.curriculum else [args.tunnel_height]
            result = [evaluate(ppo, env, args.eval_episodes, h) for h in heights]
            print(f"  eval: {json.dumps(result)}", flush=True)
            (out_dir / "eval_latest.json").write_text(json.dumps(result, indent=2))

    log_file.close()
    episode_file.close()
    summary = {
        "iterations": args.iterations,
        "env_steps": total_steps,
        "wall_time_s": time.time() - wall_start,
        "tunnel_floor": tunnel_floor,
        "final_eval": evaluate(ppo, env, max(args.eval_episodes, 10)),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
