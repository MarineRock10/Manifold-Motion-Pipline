"""Collect (commanded, achieved) pairs for *static* poses - the regime the residual model needs.

The walking clips give thousands of pairs, but walking keeps joints inside the range the
controller tracks well: measured residuals there are ~0.05 rad, while holding an extreme
keyframe (deep crouch, arms out) shows ~0.15 rad. Fitting the execution residual on walking
data and applying it to posed keyframes would be extrapolation exactly where the correction
matters.

This samples poses the way the primitive model uses them - held still - and records what the
controller reaches.

    python3 -m manifold_g1.residual_data --poses 400 --out reports/manifold_g1/residual_pairs.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C
from .demos import DEFAULT_ISAAC

OUT = Path("reports/manifold_g1/residual_pairs.npz")


def sample_poses(n: int, seed: int, settle: float = 0.8, stride: int = 1) -> np.ndarray:
    """Pose deltas in the range the primitive explores: a few pose channels plus joint noise."""
    rng = np.random.default_rng(seed)
    poses = np.zeros((n, 29))
    for i in range(n):
        # legs: crouch-like (hips, knees, ankles), scaled by a per-sample depth
        depth = rng.uniform(0.2, 1.3)
        poses[i, [0, 1]] = -0.6 * depth + rng.normal(0, 0.15, 2)     # hip pitch L/R
        poses[i, [9, 10]] = 1.0 * depth + rng.normal(0, 0.15, 2)     # knee L/R
        poses[i, [13, 14]] = -0.4 * depth + rng.normal(0, 0.1, 2)    # ankle pitch L/R
        # waist
        poses[i, 8] = rng.uniform(-0.3, 0.5)                          # waist pitch
        poses[i, 12] = rng.uniform(-0.6, 0.6)                         # waist yaw
        # arms: the channel the controller ignores most, so sample it widely
        poses[i, [15, 22]] = rng.uniform(-0.6, 0.6, 2)                # shoulder pitch
        poses[i, [16, 23]] = rng.uniform(-0.8, 0.8, 2)                # shoulder roll
        poses[i, [18, 25]] = rng.uniform(0.0, 1.2, 2)                 # elbow
        poses[i] += rng.normal(0, 0.05, 29)
    return np.clip(poses, -1.4, 1.4)


def collect(poses: np.ndarray, settle: float) -> tuple[np.ndarray, np.ndarray]:
    from .keyframe_env import KeyframeEnv

    env = KeyframeEnv()
    commanded, achieved = [], []
    for i, pose in enumerate(poses):
        env.reset()
        env.set_joints(pose)
        for _ in range(int(settle / C.CONTROL_DT)):
            env.step()
        commanded.append(pose)
        achieved.append(env.state()["q_hw"][C.MUJOCO_TO_ISAACLAB] - DEFAULT_ISAAC)
        if (i + 1) % 50 == 0:
            err = np.abs(np.stack(achieved[-50:]) - np.stack(commanded[-50:]))
            print(f"  {i + 1}/{len(poses)} poses, recent |residual| mean {err.mean():.4f} rad",
                  flush=True)
    return np.stack(commanded), np.stack(achieved)


def main() -> int:
    parser = argparse.ArgumentParser(description="Static-pose execution pairs for the residual")
    parser.add_argument("--poses", type=int, default=400)
    parser.add_argument("--settle", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    poses = sample_poses(args.poses, args.seed)
    print(f"collecting {len(poses)} static poses (settle {args.settle}s each, "
          f"~{len(poses) * (args.settle + 0.3):.0f}s wall)")
    commanded, achieved = collect(poses, args.settle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, q_cmd=commanded, q_act=achieved)

    residual = achieved - commanded
    groups = {"legs": slice(0, 12), "waist": slice(12, 15), "arms": slice(15, 29)}
    print(f"\nstatic-pose residuals ({len(poses)} poses):")
    print(f"  overall |residual| {np.abs(residual).mean():.4f} rad  "
          f"(walking data was ~0.05; this is the regime that matters)")
    for name, sl in groups.items():
        print(f"  {name:>6}: commanded {np.abs(commanded[:, sl]).mean():.3f}  "
              f"residual {np.abs(residual[:, sl]).mean():.4f}")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
