"""Diagnostic figure for the single-pipeline validation runs.

Left column - the generated manifold:
  top:    top-down corridor (walls, start, goal) with the rollout trajectories
  bottom: cross-section with the free-space ellipse inscribed in (W, H) and the
          robot's body points colored by their normalized ellipse radius
          (r < 1 means inside the free-space ellipse; this is the quantity that
          must shrink when L3 sweeps the manifold height)

Right column - the action distribution the policy learned:
  top:    command mean +/- 2 sigma versus distance to goal, with the greedy rollout
  bottom: samples drawn from the policy at different episode phases, in (vx, wz)

Usage:
  python3 -m manifold_g1.visualize --resume reports/manifold_g1/ppo/policy.pt \
      --out reports/manifold_g1/viz/manifold_policy.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from .manifold import ManifoldSpec
from .ppo import PPO
from .task_env import GoalReachEnv

BODY_NAMES = ["pelvis", "torso_link", "left_knee_link", "right_knee_link",
              "left_wrist_yaw_link", "right_wrist_yaw_link"]
BODY_STYLE = {
    "pelvis": ("o", "#1f77b4"),
    "torso_link": ("s", "#ff7f0e"),
    "left_knee_link": ("^", "#2ca02c"),
    "right_knee_link": ("v", "#2ca02c"),
    "left_wrist_yaw_link": ("D", "#d62728"),
    "right_wrist_yaw_link": ("d", "#d62728"),
}


def rollout(env: GoalReachEnv, ppo: PPO, body_ids: list[int], max_steps: int = 200) -> dict:
    """One greedy episode; records states, actions and body points."""
    record: dict[str, list] = {k: [] for k in
                               ("pos", "quat", "dist", "action", "obs", "bodies", "phase")}
    obs, _ = env.reset()
    done = truncated = False
    while not (done or truncated) and len(record["pos"]) < max_steps:
        action, _, _ = ppo.act(obs, deterministic=True)
        obs, _, done, truncated, info = env.step(action)
        state = env.env.state()
        record["pos"].append(state["base_pos"])
        record["quat"].append(state["base_quat"])
        record["dist"].append(info["distance"])
        record["action"].append(action)
        record["obs"].append(obs)
        record["bodies"].append(env.env.data.xpos[body_ids].copy())
    out = {k: np.asarray(v) for k, v in record.items()}
    out["outcome"] = info["outcome"]
    return out


def sample_actions(ppo: PPO, obs: np.ndarray, samples: int, seed: int = 0) -> np.ndarray:
    """[T, samples, 3] draws from the policy at every recorded state."""
    torch.manual_seed(seed)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=ppo.device)
    with torch.no_grad():
        dist, _ = ppo.model.distribution(obs_t)
        draws = dist.sample((samples,)).permute(1, 0, 2)
    return draws.cpu().numpy()


def ellipse_radius(spec: ManifoldSpec, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.sqrt((y / (0.5 * spec.width)) ** 2 + ((z - 0.5 * spec.height) / (0.5 * spec.height)) ** 2)


def draw_manifold_panels(axes, spec: ManifoldSpec, episodes: list[dict], radii: np.ndarray) -> None:
    ax_top, ax_cross = axes

    half_w, half_l = 0.5 * spec.width, 0.5 * spec.length
    ax_top.add_patch(plt.Rectangle((-half_l, -half_w), spec.length, spec.width,
                                   facecolor="#f2f2f2", edgecolor="#888888", zorder=1))
    ax_top.axvline(spec.start_x - 0.8, color="#888888", lw=3)
    for episode, color in zip(episodes, plt.cm.viridis(np.linspace(0.15, 0.85, len(episodes)))):
        pos = episode["pos"]
        ax_top.plot(pos[:, 0], pos[:, 1], color=color, lw=2, label=f"{episode['outcome']}")
    ax_top.plot([spec.start_x], [0.0], marker="o", color="black", ms=7, label="start")
    ax_top.plot([spec.goal_x], [0.0], marker="*", color="gold", ms=18, markeredgecolor="black", label="goal")
    ax_top.set_xlim(spec.start_x - 1.2, spec.goal_x + 1.0)
    ax_top.set_ylim(-half_w - 0.4, half_w + 0.4)
    ax_top.set_xlabel("x [m]")
    ax_top.set_ylabel("y [m]")
    ax_top.set_title(f"Manifold corridor L={spec.length:g} W={spec.width:g} H={spec.height:g} m")
    ax_top.legend(loc="upper left", fontsize=8)
    ax_top.set_aspect("equal", adjustable="box")

    ax_cross.add_patch(plt.Rectangle((-half_w, 0.0), spec.width, spec.height,
                                     facecolor="#f2f2f2", edgecolor="#888888", zorder=1))
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    ax_cross.plot(0.5 * spec.width * np.cos(theta), 0.5 * spec.height + 0.5 * spec.height * np.sin(theta),
                  color="#1f77b4", lw=2, ls="--", label="free-space ellipse")
    for episode in episodes:
        bodies = episode["bodies"]          # [T, N, 3]
        r = radii_of(episode, spec)         # [T, N]
        for index, name in enumerate(BODY_NAMES):
            marker, color = BODY_STYLE[name]
            ax_cross.scatter(bodies[:, index, 1], bodies[:, index, 2], c=r[:, index], cmap="viridis",
                             vmin=0.0, vmax=max(1.0, float(r.max())), marker=marker, s=14,
                             alpha=0.8, label=name if episode is episodes[0] else None)
    ax_cross.axhline(0.0, color="#555555", lw=1)
    ax_cross.set_xlim(-half_w - 0.15, half_w + 0.15)
    ax_cross.set_ylim(-0.05, spec.height + 0.1)
    ax_cross.set_xlabel("y [m]")
    ax_cross.set_ylabel("z [m]")
    ax_cross.set_title("Cross-section: ellipse + body points (color = normalized radius)")
    ax_cross.legend(loc="upper right", fontsize=7, ncol=2)
    ax_cross.set_aspect("equal", adjustable="box")


def radii_of(episode: dict, spec: ManifoldSpec) -> np.ndarray:
    bodies = episode["bodies"]              # [T, N, 3]
    return ellipse_radius(spec, bodies[:, :, 1], bodies[:, :, 2])


def draw_action_panels(axes, spec: ManifoldSpec, episodes: list[dict], command_scale: np.ndarray,
                       sample_cmd: np.ndarray) -> None:
    ax_band, ax_scatter = axes

    ax_band.set_title("Policy action distribution vs distance to goal")
    phases = [(spec.start_x, spec.goal_x)]
    colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(episodes)))
    ax_wz = ax_band.twinx()
    for index, (episode, color) in enumerate(zip(episodes, colors)):
        dist = episode["dist"]
        samples = sample_cmd[index]                     # [T, S, 3]
        mean = samples.mean(axis=1)                     # [T, 3]
        std = samples.std(axis=1)
        ax_band.plot(dist, mean[:, 0], color=color, lw=2, label="vx mean" if index == 0 else None)
        ax_band.fill_between(dist, mean[:, 0] - 2 * std[:, 0], mean[:, 0] + 2 * std[:, 0],
                             color=color, alpha=0.18, label="vx ±2σ" if index == 0 else None)
        ax_band.plot(dist, episode["action"][:, 0] * command_scale[0], color=color, lw=1, ls=":", alpha=0.7,
                     label="greedy vx" if index == 0 else None)
        ax_wz.plot(dist, mean[:, 2], color=color, lw=1, ls="--", alpha=0.7,
                   label="wz mean" if index == 0 else None)
    ax_band.axvline(0.35, color="#999999", lw=1, ls="--")
    ax_band.set_xlabel("distance to goal [m]")
    ax_band.set_ylabel("vx [m/s]")
    ax_wz.set_ylabel("wz [rad/s]")
    ax_band.invert_xaxis()
    handles1, labels1 = ax_band.get_legend_handles_labels()
    handles2, labels2 = ax_wz.get_legend_handles_labels()
    ax_band.legend(handles1 + handles2, labels1 + labels2, fontsize=7, loc="lower left")

    epoch = None
    for episode, samples, color in zip(episodes, sample_cmd, colors):
        dist = episode["dist"]
        for lo, hi, alpha in ((2.5, 99.0, 0.25), (1.0, 2.5, 0.45), (0.0, 1.0, 0.8)):
            mask = (dist >= lo) & (dist < hi)
            if not mask.any():
                continue
            pts = samples[mask][:, :, [0, 2]].reshape(-1, 2) * command_scale[[0, 2]]
            ax_scatter.scatter(pts[:, 0], pts[:, 1], s=4, alpha=alpha, color=color)
        ax_scatter.plot(episode["action"][:, 0] * command_scale[0],
                        episode["action"][:, 2] * command_scale[2],
                        color="black", lw=1.2, marker="o", ms=2.5, alpha=0.8)
    ax_scatter.axhline(0.0, color="#cccccc", lw=1)
    ax_scatter.axvline(0.0, color="#cccccc", lw=1)
    ax_scatter.set_xlabel("vx command [m/s]")
    ax_scatter.set_ylabel("wz command [rad/s]")
    ax_scatter.set_title("Sampled commands (opacity = near start / mid / near goal); black = greedy rollout")


def main() -> int:
    parser = argparse.ArgumentParser(description="Manifold + policy-distribution diagnostic figure")
    parser.add_argument("--resume", type=Path, default=Path("reports/manifold_g1/ppo/policy.pt"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--length", type=float, default=6.0)
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--height", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_g1/viz/manifold_policy.png"))
    args = parser.parse_args()

    spec = ManifoldSpec(length=args.length, width=args.width, height=args.height)
    env = GoalReachEnv(spec)
    body_ids = [mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in BODY_NAMES]
    command_scale = np.array([env.cfg.max_lin_vel, env.cfg.max_lat_vel, env.cfg.max_yaw_rate])

    ppo = PPO(int(env.reset()[0].shape[0]), env.action_dim)
    ppo.load(args.resume)

    episodes = [rollout(env, ppo, body_ids) for _ in range(args.episodes)]
    sample_cmd = [sample_actions(ppo, episode["obs"], args.samples) * command_scale
                  for episode in episodes]
    radii = np.concatenate([radii_of(episode, spec).ravel() for episode in episodes])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    draw_manifold_panels((axes[0, 0], axes[1, 0]), spec, episodes, radii)
    draw_action_panels((axes[0, 1], axes[1, 1]), spec, episodes, command_scale, sample_cmd)
    fig.suptitle("Manifold-conditioned motion pipeline: manifold (left) vs learned action distribution (right)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)

    stats = {
        "outcomes": [episode["outcome"] for episode in episodes],
        "ellipse_radius_max": float(radii.max()),
        "ellipse_radius_mean": float(radii.mean()),
        "action_std_mean": float(np.mean([s.std(axis=1).mean() for s in sample_cmd])),
        "figure": str(args.out),
    }
    args.out.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
