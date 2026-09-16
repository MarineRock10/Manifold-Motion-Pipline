"""Behaviour-cloning warm start for the primitive model.

Reinforcement learning from scratch on this task collapses: the reward is sparse (a success
needs the whole body inside the ellipsoid and held there), the action is 29-dimensional, and
most of the manifold pool is hard enough that random exploration almost never succeeds. Four
runs peaked in the first ~100 iterations and then drifted to poses no demonstration covers,
outside every manifold, unable to recover - the signature of a policy with no usable gradient.

The demonstration set fixes that: 10k recorded poses, each paired with the envelope it occupied,
is exactly (M, s) -> q supervision. Training on it first puts the policy in the right region of
pose space - it can already produce plausible, manifold-appropriate poses - and RL then only has
to refine within it, instead of searching from noise.

    python3 -m manifold_g1.bc train --epochs 200      # -> bc_policy.pt
    python3 -m manifold_g1.bc eval                    # agreement with held-out demos
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import constants as C
from .primitive import POSE_DIM, load_demos
from .primitive_torch import Config, TorchPrimitiveEnv

OUT = C.REPO / "reports" / "manifold_g1" / "primitive_torch" / "bc_policy.pt"


def build_dataset(kin, demo_poses: np.ndarray, cfg: Config):
    """Observations from the environment's own obs layout, labels from the recorded poses.

    The observation is built by the real environment (so training and deployment cannot drift
    apart), but instead of rolling the policy the manifold of each demo is loaded directly.
    """
    import torch

    from .manifold import EllipsoidManifold, Primitive
    from .primitive import load_demos as _load

    _, all_demos = _load()
    keys = list(all_demos)
    pose_list = [all_demos[k] for k in keys]
    env = TorchPrimitiveEnv(kin, demo_poses, cfg, n=1, pool=1, seed=0)

    # one (M, q) pair per stored demo pose: the manifold is the envelope it was recorded under
    semi = np.stack([np.array([float(v) for v in k.split(",")][:3]) for k in keys])
    center = np.stack([np.array([float(v) for v in k.split(",")][3:]) for k in keys])
    return env, semi, center, pose_list


def train(args) -> int:
    import torch

    from .kinematics import TorchKinematics
    from .ppo import PPO

    kin = TorchKinematics(device=args.device)
    demo_poses = np.concatenate(list(load_demos()[1].values()))
    cfg = Config()
    env = TorchPrimitiveEnv(kin, demo_poses, cfg, n=args.batch, pool=1, seed=0)
    probe = TorchPrimitiveEnv(kin, demo_poses, cfg, n=1, pool=1, seed=0)
    obs_dim = probe.obs().shape[1]
    ppo = PPO(obs_dim, POSE_DIM, init_log_std=[-0.3] * POSE_DIM, device=args.device)

    from .primitive import load_demos as _load
    _, all_demos = _load()
    keys = list(all_demos)
    semi_all = np.stack([np.array([float(v) for v in k.split(",")][:3]) for k in keys])
    center_all = np.stack([np.array([float(v) for v in k.split(",")][3:]) for k in keys])
    poses = np.concatenate([all_demos[key] for key in keys])
    owner = np.concatenate([np.full(len(all_demos[key]), i) for i, key in enumerate(keys)])

    rng = np.random.default_rng(args.seed)
    n_train = int(0.9 * len(poses))
    perm = rng.permutation(len(poses))
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    print(f"BC data: {len(poses)} poses over {len(keys)} manifolds "
          f"({n_train} train / {len(val_idx)} val)")

    optimizer = torch.optim.Adam(ppo.model.parameters(), lr=args.lr)
    started = time.perf_counter()
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_idx)
        losses = []
        for start in range(0, len(train_idx), args.batch):
            idx = train_idx[start:start + args.batch]
            env.set_manifolds_from_arrays(semi_all[owner[idx]], center_all[owner[idx]], None)
            obs = env.obs()
            target = torch.as_tensor(poses[idx], dtype=kin.dtype, device=kin.device)
            # The action must mean the same thing here and in the RL environment: there
            # `target = clamp(action, -1, 1) * POSE_LIMIT`, i.e. the raw network output is the
            # normalised action. Training the clone through a tanh instead (mse(tanh(mu)*1.4, q))
            # fits a different scale: the environment then receives mu ~ 2.2, clamps it to 1, and
            # the pose is nothing like the one that was cloned - which is what made the RL
            # fine-tune collapse right after a good warm start.
            from .primitive_torch import action_to_pose
            pred = ppo.model(obs)[0]
            loss = torch.nn.functional.mse_loss(action_to_pose(pred, torch), target)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo.model.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())
        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            val = _evaluate(ppo, env, semi_all, center_all, poses[val_idx],
                            owner[val_idx], kin)
            print(f"  epoch {epoch:4d}  train mse {np.mean(losses):.4f}  "
                  f"val |pose err| {val['pose_err']:.3f} rad  "
                  f"r_cloned {val['radius']:.2f} (r_demo {val['radius_demo']:.2f})  "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
            if val["pose_err"] < best:
                best = val["pose_err"]
                ppo.save(args.out)
    print(f"\n=== BC finished in {time.perf_counter() - started:.0f} s, "
          f"best val pose error {best:.3f} rad -> {args.out} ===")
    return 0


def _evaluate(ppo, env, demo_semi: np.ndarray, demo_center: np.ndarray,
              poses: np.ndarray, owners: np.ndarray, kin) -> dict:
    """Held-out agreement: pose error, and containment of the clone vs of the demonstration.

    The evaluation set is `(semi[owners], center[owners], poses)`, all three already sliced to
    the same rows: passing pre-sliced manifolds and unsliced poses (or the reverse) silently
    pairs poses with the wrong ellipsoids, which is how an earlier version reported r = 11 for
    inputs that were actually inside.
    """
    import torch

    semi = demo_semi[owners]
    center = demo_center[owners]
    assert len(semi) == len(poses), f"manifold {len(semi)} vs pose {len(poses)} rows"
    env.set_manifolds_from_arrays(semi, center, None)
    with torch.no_grad():
        from .primitive_torch import action_to_pose
        pred = action_to_pose(ppo.model(env.obs())[0], torch).cpu().numpy()
        r_pred = env.radius_np(torch.as_tensor(pred, dtype=kin.dtype, device=kin.device))
        r_demo = env.radius_np(torch.as_tensor(poses, dtype=kin.dtype, device=kin.device))
    return {"pose_err": float(np.abs(pred - poses).mean()),
            "radius": float(np.mean(r_pred)), "radius_demo": float(np.mean(r_demo))}


def _points(kin, pose_t):
    pts, mask = kin.forward(pose_t, return_mask=True)
    return pts[0].cpu().numpy()


def evaluate(args) -> int:
    import torch

    from .kinematics import TorchKinematics
    from .ppo import PPO

    kin = TorchKinematics(device=args.device)
    demo_poses = np.concatenate(list(load_demos()[1].values()))
    env = TorchPrimitiveEnv(kin, demo_poses, Config(), n=1, pool=1, seed=0)
    ppo = PPO(env.obs().shape[1], POSE_DIM, device=args.device)
    ppo.load(args.policy)
    from .primitive import load_demos as _load
    _, all_demos = _load()
    keys = list(all_demos)
    semi = np.stack([np.array([float(v) for v in k.split(",")][:3]) for k in keys])
    center = np.stack([np.array([float(v) for v in k.split(",")][3:]) for k in keys])
    poses = np.concatenate([all_demos[key] for key in keys])
    owner = np.concatenate([np.full(len(all_demos[key]), i) for i, key in enumerate(keys)])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(poses), min(400, len(poses)), replace=False)
    out = _evaluate(ppo, env, semi, center, poses[idx], owner[idx], kin)
    print(f"{args.policy.name}: val pose error {out['pose_err']:.3f} rad, "
          f"r_cloned {out['radius']:.2f} vs r_demo {out['radius_demo']:.2f} "
          f"(the demo pose is the target: its r shows what this manifold allows)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Behaviour-cloning warm start")
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--policy", type=Path, default=OUT)
    args = parser.parse_args()
    return train(args) if args.mode == "train" else evaluate(args)
