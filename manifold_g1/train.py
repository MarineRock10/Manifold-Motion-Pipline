"""PPO training on the single-environment manifold goal-reaching task.

Example:
  python3 -m manifold_g1.train --iterations 40 --rollout-steps 512 --eval-every 10
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .manifold import ManifoldSpec
from .ppo import PPO, RolloutBuffer
from .task_env import GoalReachEnv, TaskConfig


def evaluate(ppo: PPO, env: GoalReachEnv, episodes: int = 5) -> dict:
    results = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = truncated = False
        steps = 0
        collisions = 0
        episode_return = 0.0
        info = {"outcome": "running"}
        while not (done or truncated):
            action, _, _ = ppo.act(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_return += reward
            steps += 1
            collisions += int(info["collision"])
        results.append({"outcome": info["outcome"], "steps": steps,
                        "return": episode_return, "collision_steps": collisions})
    return {
        "success": float(np.mean([r["outcome"] == "success" for r in results])),
        "fall": float(np.mean([r["outcome"] == "fall" for r in results])),
        "timeout": float(np.mean([r["outcome"] == "timeout" for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_return": float(np.mean([r["return"] for r in results])),
        "collision_steps": float(np.mean([r["collision_steps"] for r in results])),
        "episodes": episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PPO on the manifold goal-reaching task")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--length", type=float, default=6.0)
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--height", type=float, default=2.0)
    parser.add_argument("--start-x", type=float, default=-2.0)
    parser.add_argument("--goal-x", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_g1/ppo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--viz", action="store_true", help="open the live 3D training dashboard")
    parser.add_argument("--viz-video", type=Path, default=None, help="record the dashboard to mp4")
    parser.add_argument("--viz-every", type=float, default=0.5, help="seconds between dashboard updates")
    parser.add_argument("--viz-fps", type=float, default=2.0, help="dashboard video frame rate")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    spec = ManifoldSpec(length=args.length, width=args.width, height=args.height,
                        start_x=args.start_x, goal_x=args.goal_x)
    env = GoalReachEnv(spec, TaskConfig())
    obs, _ = env.reset()
    obs_dim = int(obs.shape[0])
    ppo = PPO(obs_dim, env.action_dim, device=args.device)
    if args.resume is not None:
        ppo.load(args.resume)
        print(f"resumed from {args.resume}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dashboard = None
    if (args.viz or args.viz_video is not None) and not args.eval_only:
        from .live_viz import LiveDashboard
        dashboard = LiveDashboard(env, spec, show=args.viz, video_path=args.viz_video,
                                  min_interval=args.viz_every, fps=args.viz_fps)
        print(f"[viz] dashboard enabled ({'window' if args.viz else 'video'})")

    if args.eval_only:
        result = evaluate(ppo, env, args.eval_episodes)
        print(json.dumps(result, indent=2))
        return 0

    log_path = out_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["iteration", "env_steps", "episodes", "mean_return", "success_rate",
                     "fall_rate", "timeout_rate", "wall_time_s", "policy_loss", "value_loss", "entropy"])

    total_steps = 0
    wall_start = time.time()
    command_scale = np.array([env.cfg.max_lin_vel, env.cfg.max_lat_vel, env.cfg.max_yaw_rate])
    for iteration in range(1, args.iterations + 1):
        buffer = RolloutBuffer(args.rollout_steps, obs_dim, env.action_dim)
        episode_returns: list[float] = []
        outcomes: list[str] = []
        episode_return = 0.0
        for _ in range(args.rollout_steps):
            action, logprob, value = ppo.act(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            buffer.add(obs, action, logprob, reward, value, float(done))
            episode_return += reward
            obs = next_obs
            total_steps += 1
            if dashboard is not None:
                dashboard.log_step(action * command_scale, env.compliance_radius())
                dashboard.maybe_update()
            if done or truncated:
                episode_returns.append(episode_return)
                outcomes.append(info["outcome"])
                if dashboard is not None:
                    dashboard.log_episode(info["outcome"], episode_return)
                episode_return = 0.0
                obs, _ = env.reset()
        stats = ppo.update(buffer, ppo.value(obs))

        mean_return = float(np.mean(episode_returns)) if episode_returns else float("nan")
        success_rate = float(np.mean([o == "success" for o in outcomes])) if outcomes else float("nan")
        fall_rate = float(np.mean([o == "fall" for o in outcomes])) if outcomes else float("nan")
        timeout_rate = float(np.mean([o == "timeout" for o in outcomes])) if outcomes else float("nan")
        writer.writerow([iteration, total_steps, len(outcomes), f"{mean_return:.3f}",
                         f"{success_rate:.3f}", f"{fall_rate:.3f}", f"{timeout_rate:.3f}",
                         f"{time.time() - wall_start:.1f}", f"{stats['policy_loss']:.4f}",
                         f"{stats['value_loss']:.4f}", f"{stats['entropy']:.4f}"])
        log_file.flush()
        ppo.save(out_dir / "policy.pt")
        print(f"[iter {iteration:3d}] steps={total_steps:6d} episodes={len(outcomes):2d} "
              f"return={mean_return:8.2f} success={success_rate:5.2f} fall={fall_rate:5.2f} "
              f"timeout={timeout_rate:5.2f} pi_loss={stats['policy_loss']:+.3f} "
              f"v_loss={stats['value_loss']:.3f} H={stats['entropy']:.3f}", flush=True)

        if iteration % args.eval_every == 0:
            result = evaluate(ppo, env, args.eval_episodes)
            print(f"  eval: {json.dumps(result)}", flush=True)
            (out_dir / "eval_latest.json").write_text(json.dumps(result, indent=2))

    log_file.close()
    if dashboard is not None:
        dashboard.maybe_update(force=True)
        dashboard.close()
    summary = {
        "iterations": args.iterations,
        "env_steps": total_steps,
        "wall_time_s": time.time() - wall_start,
        "spec": vars(spec),
        "final_eval": evaluate(ppo, env, max(args.eval_episodes, 10)),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
