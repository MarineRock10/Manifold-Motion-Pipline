"""PPO training on the single-environment manifold goal-reaching task.

The manifold is a corridor whose ceiling ramps from `--height-start` at the entrance to
`--height-goal` at the far end. With `--curriculum` the goal-end height is lowered as the
success rate improves, which is the L3 experiment: manifold height -> body height.

Example:
  python3 -m manifold_g1.train --iterations 60 --rollout-steps 512 --eval-every 15 \
      --curriculum
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


def evaluate(ppo: PPO, env: GoalReachEnv, episodes: int = 5, height_goal: float | None = None) -> dict:
    results = []
    for _ in range(episodes):
        obs, _ = env.reset(height_goal=height_goal)
        done = truncated = False
        steps = 0
        collisions = 0
        episode_return = 0.0
        info = {"outcome": "running"}
        pelvis_z: list[float] = []
        margins: list[float] = []
        while not (done or truncated):
            action, _, _ = ppo.act(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_return += reward
            steps += 1
            collisions += int(info["collision"])
            pelvis_z.append(info["base_z"])
            margins.append(info["ceiling_margin"])
        results.append({"outcome": info["outcome"], "steps": steps, "return": episode_return,
                        "collision_steps": collisions, "pelvis_z": float(np.mean(pelvis_z)),
                        "min_margin": float(np.min(margins))})
    return {
        "success": float(np.mean([r["outcome"] == "success" for r in results])),
        "fall": float(np.mean([r["outcome"] == "fall" for r in results])),
        "collision": float(np.mean([r["outcome"] == "collision" for r in results])),
        "timeout": float(np.mean([r["outcome"] == "timeout" for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_return": float(np.mean([r["return"] for r in results])),
        "pelvis_z": float(np.mean([r["pelvis_z"] for r in results])),
        "min_margin": float(np.mean([r["min_margin"] for r in results])),
        "height_goal": height_goal,
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
    parser.add_argument("--height-start", type=float, default=1.5)
    parser.add_argument("--height-goal", type=float, default=1.5)
    parser.add_argument("--curriculum", action="store_true",
                        help="lower the goal-end ceiling as the rolling success rate improves")
    parser.add_argument("--height-goal-min", type=float, default=0.95)
    parser.add_argument("--curriculum-success", type=float, default=0.8)
    parser.add_argument("--curriculum-step", type=float, default=0.05)
    parser.add_argument("--start-x", type=float, default=-2.0)
    parser.add_argument("--goal-x", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_g1/ppo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-heights", default=None,
                        help="comma-separated goal-end heights for an evaluation sweep")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    spec = ManifoldSpec(length=args.length, width=args.width,
                        height_start=args.height_start, height_goal=args.height_goal,
                        start_x=args.start_x, goal_x=args.goal_x)
    env = GoalReachEnv(spec, TaskConfig())
    obs, _ = env.reset(height_start=args.height_start, height_goal=args.height_goal)
    obs_dim = int(obs.shape[0])
    # the crouch channel (last action dim) needs extra exploration: it starts far from
    # the upright behaviour the policy discovers first
    init_log_std = [-1.0] * (env.action_dim - 1) + [0.3]
    ppo = PPO(obs_dim, env.action_dim, device=args.device, init_log_std=init_log_std)
    if args.resume is not None:
        ppo.load(args.resume)
        print(f"resumed from {args.resume}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        if args.eval_heights:
            results = [evaluate(ppo, env, args.eval_episodes, float(h))
                       for h in args.eval_heights.split(",")]
            print(json.dumps(results, indent=2))
            (out_dir / "eval_heights.json").write_text(json.dumps(results, indent=2))
        else:
            result = evaluate(ppo, env, args.eval_episodes)
            print(json.dumps(result, indent=2))
        return 0

    log_path = out_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["iteration", "env_steps", "episodes", "mean_return", "success_rate",
                     "fall_rate", "timeout_rate", "height_goal_min", "wall_time_s",
                     "policy_loss", "value_loss", "entropy"])
    episode_path = out_dir / "episodes.csv"
    episode_file = open(episode_path, "w", newline="")
    episode_writer = csv.writer(episode_file)
    episode_writer.writerow(["iteration", "height_goal", "outcome", "steps", "return",
                             "pelvis_z", "min_margin"])

    # curriculum starts at the easy goal-end height and lowers it toward --height-goal-min
    goal_min = args.height_goal
    sample_goal = (lambda: float(rng.uniform(goal_min, min(goal_min + 0.15, args.height_start)))
                   if args.curriculum else args.height_goal)

    total_steps = 0
    wall_start = time.time()
    recent_success: list[float] = []
    height_goal = sample_goal()
    obs, _ = env.reset(height_start=args.height_start, height_goal=height_goal)

    for iteration in range(1, args.iterations + 1):
        buffer = RolloutBuffer(args.rollout_steps, obs_dim, env.action_dim)
        episode_returns: list[float] = []
        outcomes: list[str] = []
        episode_return = 0.0
        episode_pelvis: list[float] = []
        episode_margin: list[float] = []
        for _ in range(args.rollout_steps):
            action, logprob, value = ppo.act(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            buffer.add(obs, action, logprob, reward, value, float(done))
            episode_return += reward
            episode_pelvis.append(info["base_z"])
            episode_margin.append(info["ceiling_margin"])
            obs = next_obs
            total_steps += 1
            if done or truncated:
                episode_returns.append(episode_return)
                outcomes.append(info["outcome"])
                recent_success.append(1.0 if info["outcome"] == "success" else 0.0)
                episode_writer.writerow([iteration, f"{height_goal:.3f}", info["outcome"],
                                         info["episode_step"], f"{episode_return:.3f}",
                                         f"{np.mean(episode_pelvis):.3f}", f"{np.min(episode_margin):.3f}"])
                episode_file.flush()
                episode_return = 0.0
                episode_pelvis = []
                episode_margin = []
                height_goal = sample_goal()
                obs, _ = env.reset(height_start=args.height_start, height_goal=height_goal)

        stats = ppo.update(buffer, ppo.value(obs))

        if args.curriculum and len(recent_success) >= 20:
            if float(np.mean(recent_success[-20:])) > args.curriculum_success and goal_min > args.height_goal_min:
                goal_min = max(args.height_goal_min, goal_min - args.curriculum_step)
                print(f"  [curriculum] success {np.mean(recent_success[-20:]):.2f} -> "
                      f"goal-end height floor {goal_min:.2f} m", flush=True)

        mean_return = float(np.mean(episode_returns)) if episode_returns else float("nan")
        success_rate = float(np.mean([o == "success" for o in outcomes])) if outcomes else float("nan")
        fall_rate = float(np.mean([o == "fall" for o in outcomes])) if outcomes else float("nan")
        timeout_rate = float(np.mean([o == "timeout" for o in outcomes])) if outcomes else float("nan")
        writer.writerow([iteration, total_steps, len(outcomes), f"{mean_return:.3f}",
                         f"{success_rate:.3f}", f"{fall_rate:.3f}", f"{timeout_rate:.3f}",
                         f"{goal_min:.2f}", f"{time.time() - wall_start:.1f}",
                         f"{stats['policy_loss']:.4f}", f"{stats['value_loss']:.4f}",
                         f"{stats['entropy']:.4f}"])
        log_file.flush()
        ppo.save(out_dir / "policy.pt")
        print(f"[iter {iteration:3d}] steps={total_steps:6d} episodes={len(outcomes):2d} "
              f"return={mean_return:8.2f} success={success_rate:5.2f} fall={fall_rate:5.2f} "
              f"timeout={timeout_rate:5.2f} h_goal>={goal_min:.2f} "
              f"pi_loss={stats['policy_loss']:+.3f} v_loss={stats['value_loss']:.3f} "
              f"H={stats['entropy']:.3f}", flush=True)

        if iteration % args.eval_every == 0:
            heights = [args.height_goal] if not args.curriculum else [goal_min, args.height_start]
            result = [evaluate(ppo, env, args.eval_episodes, h) for h in heights]
            print(f"  eval: {json.dumps(result)}", flush=True)
            (out_dir / "eval_latest.json").write_text(json.dumps(result, indent=2))

    log_file.close()
    episode_file.close()
    summary = {
        "iterations": args.iterations,
        "env_steps": total_steps,
        "wall_time_s": time.time() - wall_start,
        "spec": vars(spec),
        "goal_min": goal_min,
        "final_eval": evaluate(ppo, env, max(args.eval_episodes, 10)),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
