"""Run the Stage-1 imitation output through frozen SONIC and MuJoCo.

This gate is intentionally small and paired: the same held-out catalogue rows are evaluated for
the learned imitation model, a family-mean pose, a global-mean pose and the recorded target.  It
measures what the supervised error misses: executed tracking and exact mesh-to-corridor radius.
It is a flat-ground physical gate, not a claim that a family supports every deployment task.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manifold_motion.core import constants as C
from manifold_motion.planning.corridor import execution_corridor_radius
from manifold_motion.simulation.keyframe_env import KeyframeEnv
from manifold_motion.stage1.pose_policy import POSE_DIM, action_to_pose


def _model(torch, checkpoint: dict[str, object]):
    hidden = int(checkpoint.get("hidden", 256))
    model = torch.nn.Sequential(
        torch.nn.Linear(int(checkpoint["input_dim"]), hidden),
        torch.nn.LayerNorm(hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, int(checkpoint["output_dim"])),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _rollout(pose: np.ndarray, corridor: np.ndarray, steps: int) -> dict[str, object]:
    env = KeyframeEnv()
    env.reset()
    env.set_joints(np.asarray(pose, dtype=np.float64))
    q_rows, pos_rows, quat_rows = [], [], []
    for _ in range(steps):
        env.step()
        state = env.state()
        q_rows.append(state["q_hw"][C.MUJOCO_TO_ISAACLAB].copy())
        pos_rows.append(state["base_pos"].copy())
        quat_rows.append(state["base_quat"].copy())
    q = np.stack(q_rows)
    radius = execution_corridor_radius(q, np.stack(pos_rows), np.stack(quat_rows), corridor)
    target_error = float(np.mean(np.abs(q[-1] - (C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB] + pose))))
    return {"accepted": bool(radius["corridor_radius_spatial_union_max"] <= 1.0),
            "radius": radius, "tracking_mae_rad": target_error,
            "final_q": q[-1].tolist()}


def _rollout_policy(catalog: Path, row_id: int, corridor: np.ndarray, policy,
                    feature_mean: np.ndarray, feature_std: np.ndarray,
                    steps: int, device: str) -> dict[str, object]:
    """Run one deterministic warm-start policy on exactly one catalogue row.

    The environment is restricted to ``row_id`` so the policy and direct-pose methods see the
    same corridor/action window.  A fresh KeyframeEnv is used per row, matching ``_rollout``'s
    reset semantics and preventing SONIC recurrent state from leaking between paired rows.
    """
    import torch

    from manifold_motion.stage1.sonic_rl import SonicConfig, SonicPoseEnv

    env = SonicPoseEnv(SonicConfig(steps=steps, mean_probe=False), seed=0,
                       catalog=catalog, catalog_split=2,
                       catalog_target_field="target_exec", catalog_target_frame=-1)
    env._catalog_ids = np.asarray([int(row_id)], dtype=np.int64)
    env.sample_episode(perturb=0.0, repeats=1)
    obs = env.policy_observation(True, feature_mean, feature_std)
    q_rows, pos_rows, quat_rows = [], [], []
    info = {"gate": False, "track": float("inf")}
    for _ in range(steps):
        action, _, _ = policy.act(obs, deterministic=True)
        pose = np.asarray(action_to_pose(action, torch), dtype=np.float64)
        _, _, _, info = env.step(pose)
        state = env.env.state()
        q_rows.append(state["q_hw"][C.MUJOCO_TO_ISAACLAB].copy())
        pos_rows.append(state["base_pos"].copy())
        quat_rows.append(state["base_quat"].copy())
        obs = env.policy_observation(True, feature_mean, feature_std)
    radius = execution_corridor_radius(np.stack(q_rows), np.stack(pos_rows),
                                        np.stack(quat_rows), corridor)
    accepted = bool(info["gate"] and radius["corridor_radius_spatial_union_max"] <= 1.0)
    return {"accepted": accepted, "radius": radius,
            "tracking_mae_rad": float(info["track"]),
            "final_q": np.asarray(q_rows[-1]).tolist()}


def evaluate(args) -> int:
    import torch

    with np.load(args.catalog, allow_pickle=False) as archive:
        corridor = np.asarray(archive["corridor"], dtype=np.float32)
        target = np.asarray(archive[args.target_field], dtype=np.float32)
        split = np.asarray(archive["split"], dtype=np.uint8)
        family = np.asarray(archive["primitive"], dtype=np.int64)
        names = np.asarray(archive["primitive_names"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _model(torch, checkpoint)
    frame = target.shape[1] // 2 if args.target_frame < 0 else args.target_frame
    default = C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB].astype(np.float32)
    # Keep feature construction in one place with the trainer; importing it avoids a silent
    # change in the model input contract between training and physical evaluation.
    from manifold_motion.stage1.manifold_imitation import manifold_features
    features = manifold_features(corridor)
    features = (features - np.asarray(checkpoint["input_mean"])) / np.asarray(checkpoint["input_std"])
    test_ids = np.flatnonzero(split == 2)
    if args.max_rows > 0:
        test_ids = test_ids[:args.max_rows]
    with torch.no_grad():
        pred_delta = model(torch.as_tensor(features[test_ids], dtype=torch.float32)).numpy()
    target_delta = target[test_ids, frame, :29] - default[None]
    train_ids = np.flatnonzero(split == 0)
    all_train_target = target[train_ids, frame, :29] - default[None]
    global_delta = all_train_target.mean(axis=0)
    family_delta = {int(value): all_train_target[family[train_ids] == value].mean(axis=0)
                    for value in sorted(set(int(v) for v in family[train_ids]))}
    methods = {
        "imitation": pred_delta,
        "family_mean": np.stack([family_delta.get(int(v), global_delta) for v in family[test_ids]]),
        "global_mean": np.repeat(global_delta[None], len(test_ids), axis=0),
        "recorded_target": target_delta,
    }
    report: dict[str, object] = {
        "schema": "manifold-motion.stage1.physical-gate.v1",
        "catalog": str(args.catalog), "checkpoint": str(args.checkpoint),
        "target_field": args.target_field, "target_frame": int(frame),
        "rows": int(len(test_ids)), "steps": int(args.steps), "methods": {},
    }
    started = time.perf_counter()
    for method, deltas in methods.items():
        rows = []
        for row_id, delta in zip(test_ids, deltas):
            pose = delta
            try:
                result = _rollout(pose, corridor[row_id], args.steps)
                rows.append({"row": int(row_id), "family": str(names[int(family[row_id])]), **result})
            except Exception as error:  # keep a failed physical row in the denominator
                rows.append({"row": int(row_id), "family": str(names[int(family[row_id])]),
                             "accepted": False, "error": type(error).__name__ + ": " + str(error)})
        accepted = [row for row in rows if row.get("accepted")]
        radii = [row["radius"]["corridor_radius_spatial_union_max"] for row in accepted if "radius" in row]
        tracking = [row["tracking_mae_rad"] for row in rows if "tracking_mae_rad" in row]
        report["methods"][method] = {
            "rows": len(rows), "accepted": len(accepted),
            "acceptance_rate": len(accepted) / len(rows) if rows else 0.0,
            "radius_mean": float(np.mean(radii)) if radii else None,
            "radius_p95": float(np.quantile(radii, 0.95)) if radii else None,
            "tracking_mae_rad": float(np.mean(tracking)) if tracking else None,
            "failures": [row for row in rows if not row.get("accepted")],
        }
    if args.sonic_policy is not None:
        if args.imitation_checkpoint is None:
            raise ValueError("--sonic-policy requires --imitation-checkpoint")
        from manifold_motion.stage1.ppo import PPO
        from manifold_motion.stage1.sonic_rl import _warm_start_actor

        imitation = torch.load(args.imitation_checkpoint, map_location="cpu", weights_only=False)
        policy = PPO(int(imitation["input_dim"]), POSE_DIM, device=args.device)
        policy.model = _warm_start_actor(torch, imitation, policy.device, -1.0)
        policy.optimizer = torch.optim.Adam(policy.model.parameters(), lr=3e-4, eps=1e-5)
        policy.load(args.sonic_policy)
        feature_mean = np.asarray(imitation["input_mean"], dtype=np.float32)
        feature_std = np.asarray(imitation["input_std"], dtype=np.float32)
        rows = []
        for row_id in test_ids:
            try:
                result = _rollout_policy(args.catalog, int(row_id), corridor[row_id], policy,
                                         feature_mean, feature_std, args.steps, args.device)
                rows.append({"row": int(row_id), "family": str(names[int(family[row_id])]), **result})
            except Exception as error:
                rows.append({"row": int(row_id), "family": str(names[int(family[row_id])]),
                             "accepted": False, "error": type(error).__name__ + ": " + str(error)})
        accepted = [row for row in rows if row.get("accepted")]
        radii = [row["radius"]["corridor_radius_spatial_union_max"] for row in accepted if "radius" in row]
        tracking = [row["tracking_mae_rad"] for row in rows if "tracking_mae_rad" in row]
        report["methods"]["sonic_warm_start"] = {
            "rows": len(rows), "accepted": len(accepted),
            "acceptance_rate": len(accepted) / len(rows) if rows else 0.0,
            "radius_mean": float(np.mean(radii)) if radii else None,
            "radius_p95": float(np.quantile(radii, 0.95)) if radii else None,
            "tracking_mae_rad": float(np.mean(tracking)) if tracking else None,
            "failures": [row for row in rows if not row.get("accepted")],
        }
    report["elapsed_s"] = time.perf_counter() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sonic-policy", type=Path, default=None,
                        help="optional warm-start PPO policy for paired physical evaluation")
    parser.add_argument("--imitation-checkpoint", type=Path, default=None,
                        help="imitation checkpoint paired with --sonic-policy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/stage1_physical_gate_v1.json"))
    parser.add_argument("--target-field", choices=("target_ref", "target_exec"), default="target_exec")
    parser.add_argument("--target-frame", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--max-rows", type=int, default=24)
    args = parser.parse_args()
    if args.steps < 1 or args.max_rows < 1:
        parser.error("steps and max-rows must be positive")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
