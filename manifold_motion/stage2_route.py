"""Execute the semantic Stage-2 chain ``(M, c) -> z_p -> R_ref -> SONIC``.

This is intentionally a thin, auditable router rather than a hidden handoff: it stores the
primitive probabilities, selected per-primitive checkpoint, generated reference, and the final
MuJoCo hard-gate result in one output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import constants as C
from .primitive_router import _features, _load_router, _load_windows, _normalize
from .seed_replay import ReplayConfig
from .seed_replay import SeedReplayRunner
from .seed_windows import PRIMITIVE_NAMES
from . import stage2_flow
from .stage2_validate import validate_trajectory


def _parse_model(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        try:
            primitive, path = value.split("=", 1)
            primitive_id = int(primitive)
        except ValueError as error:
            raise ValueError("--model must use PRIMITIVE_ID=PATH") from error
        if not 0 <= primitive_id < len(PRIMITIVE_NAMES):
            raise ValueError(f"primitive ID {primitive_id} is outside the taxonomy")
        if primitive_id in result:
            raise ValueError(f"duplicate model for primitive ID {primitive_id}")
        result[primitive_id] = Path(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="route a manifold condition into a Stage-2 dynamic model")
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True,
                        help="routable per-primitive checkpoint: e.g. 2=reports/.../conditional_mean.pt")
    parser.add_argument("--split", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--condition-npz", type=Path, default=None,
                        help="online perception condition.npz; replaces corridor/SDF and optional local command")
    parser.add_argument("--scene", type=Path, default=None,
                        help="optional MuJoCo scene containing the perceived obstacles for the final hard gate")
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/stage2_routed"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    raw = _load_windows(args.windows)
    model_map = _parse_model(args.model)
    router, checkpoint = _load_router(args.router)
    candidates = np.flatnonzero(raw["split"] == args.split)
    if not len(candidates):
        parser.error(f"split {args.split} has no windows")
    source_index = int(candidates[args.index % len(candidates)])
    routing_raw = dict(raw)
    condition_provenance = "stored_window"
    perception_command: np.ndarray | None = None
    if args.condition_npz is not None:
        with np.load(args.condition_npz) as condition_archive:
            for key in ("corridor", "sdf", "command"):
                if key not in condition_archive.files:
                    if key == "command":
                        continue
                    parser.error(f"{args.condition_npz} must contain '{key}'")
                value = np.asarray(condition_archive[key], dtype=np.float32)
                if value.shape != raw[key][source_index].shape:
                    parser.error(f"{key} shape {value.shape} does not match expected {raw[key][source_index].shape}")
                routing_raw[key] = raw[key].copy()
                routing_raw[key][source_index] = value
                if key == "command":
                    perception_command = value
        condition_provenance = str(args.condition_npz)
    feature = _features(routing_raw, include_command=bool(checkpoint.get("include_command", True)))[source_index:source_index + 1]
    normalized = _normalize(feature, checkpoint["feature_mean"], checkpoint["feature_std"])
    with torch.no_grad():
        probabilities = torch.softmax(router(torch.as_tensor(normalized)), dim=1)[0].numpy()
    active = np.asarray(checkpoint["active_primitive_ids"], dtype=np.int64)
    order = np.argsort(probabilities)[::-1]
    router_id = int(active[order[0]])
    route_id = router_id
    selection_policy = "trained_router"
    # A perception corridor can be outside the reverse-SEED router distribution.  In that
    # case a raw softmax may call any small vertical semi-axis a crouch even when the actual
    # obstacle is a side wall.  Keep main's primitive checkpoints and SONIC path unchanged,
    # but apply a conservative geometry prior at this deployment boundary.
    if args.condition_npz is not None:
        with np.load(args.condition_npz) as condition_archive:
            perception_corridor = np.asarray(condition_archive["corridor"], dtype=np.float32)
            if "command" in condition_archive.files:
                perception_command = np.asarray(condition_archive["command"], dtype=np.float32)
        vertical_min = float(np.quantile(perception_corridor[:, 5], 0.10))
        lateral_min = float(np.quantile(perception_corridor[:, 4], 0.10))
        lateral_command = (abs(float(perception_command[1]))
                           if perception_command is not None and len(perception_command) >= 2 else 0.0)
        if vertical_min < 0.90:
            preferred = 2  # genuine low/overhead clearance -> crouch
        elif lateral_min < 0.65 or lateral_command >= 0.40:
            # A diagonal/sideways route can require the lateral gait even when the local
            # aperture is not yet narrow.  The command direction is learned from the same
            # local route prefix and avoids selecting nominal walk for a -45 degree detour.
            preferred = 4
        else:
            preferred = 5  # open corridor -> nominal walk
        if preferred in model_map:
            route_id = preferred
            selection_policy = "perception_geometry_guard"
    if route_id not in model_map:
        parser.error(f"router chose {PRIMITIVE_NAMES[route_id]}, but no verified --model was supplied for it")
    # The primitive one-hot and the physical state/history must describe the same motion
    # family.  Previously only the one-hot was overridden: a nominal-walk model could receive
    # a side-gait state from the first held-out window and produce a backwards/unstable phase.
    # Select an actor-disjoint source window carrying the routed primitive; the online corridor,
    # SDF and command are still overridden below.  Primitive 2 has no test windows in the pilot,
    # so prefer validation and finally training as an explicit, auditable fallback.
    source_candidates = np.flatnonzero((raw["split"] == args.split) &
                                       (raw["primitive"] == route_id))
    conditioned_split = args.split
    if not len(source_candidates):
        source_candidates = np.flatnonzero(raw["primitive"] == route_id)
    if not len(source_candidates):
        parser.error(f"no conditioning windows exist for {PRIMITIVE_NAMES[route_id]}")
    source_selection = "requested_index"
    source_command_cosine: float | None = None
    if perception_command is not None and len(perception_command) >= 2:
        target_xy = np.asarray(perception_command[:2], dtype=np.float64)
        target_norm = float(np.linalg.norm(target_xy))
        candidate_xy = np.asarray(raw["command"][source_candidates, :2], dtype=np.float64)
        candidate_norm = np.linalg.norm(candidate_xy, axis=1)
        valid = candidate_norm >= 0.30
        if target_norm > 1.0e-6 and np.any(valid):
            cosines = np.full(len(source_candidates), -np.inf, dtype=np.float64)
            cosines[valid] = (candidate_xy[valid] @ target_xy) / (candidate_norm[valid] * target_norm)
            best = int(np.argmax(cosines))
            source_index = int(source_candidates[best])
            source_command_cosine = float(cosines[best])
            source_selection = "online_command_direction"
        else:
            source_index = int(source_candidates[args.index % len(source_candidates)])
    else:
        source_index = int(source_candidates[args.index % len(source_candidates)])
    conditioned_split = int(raw["split"][source_index])
    conditioned_split_indices = np.flatnonzero(raw["split"] == conditioned_split)
    conditioned_index = int(np.flatnonzero(conditioned_split_indices == source_index)[0])
    args.out.mkdir(parents=True, exist_ok=True)
    sample_args = argparse.Namespace(
        windows=args.windows, autoencoder=Path("reports/manifold_motion/stage2_flow/autoencoder.pt"),
        flow=Path("reports/manifold_motion/stage2_flow/flow.pt"), out=args.out,
        split=conditioned_split, index=conditioned_index, steps=32, num_candidates=1,
        sampler="conditional_mean", mean_model=model_map[route_id],
        residual_flow=Path("reports/manifold_motion/stage2_residual_flow/residual_flow.pt"),
        seed=args.seed, device=args.device, condition_npz=args.condition_npz, primitive_id=route_id)
    stage2_flow.sample(sample_args)
    sample_path = args.out / "sample.npz"
    with np.load(sample_path) as sample:
        trajectory = np.asarray(sample["generated_ref"])
        corridor = np.asarray(sample["condition_corridor"]) if "condition_corridor" in sample.files else None
    runner = SeedReplayRunner(args.scene) if args.scene is not None else None
    executed, summary = validate_trajectory(trajectory, source=sample_path, source_hz=30.0,
                                             config=ReplayConfig(), corridor=corridor,
                                             stratum=PRIMITIVE_NAMES[route_id], runner=runner)
    route_report = {"source_index": source_index, "split": conditioned_split,
                    "requested_split": args.split, "conditioned_index": conditioned_index,
                    "source_selection": source_selection,
                    "source_command_cosine": source_command_cosine,
                    "source_primitive": PRIMITIVE_NAMES[int(raw["primitive"][source_index])],
                    "routed_primitive": PRIMITIVE_NAMES[route_id], "routed_primitive_id": route_id,
                    "router_proposal": PRIMITIVE_NAMES[router_id], "selection_policy": selection_policy,
                    "router_confidence": float(probabilities[order[0]]),
                    "probabilities": [{"primitive": PRIMITIVE_NAMES[int(active[i])],
                                       "probability": float(probabilities[i])} for i in order],
                    "dynamic_checkpoint": str(model_map[route_id]), "condition_provenance": condition_provenance,
                    "scene": str(args.scene) if args.scene is not None else str(C.FLAT_SCENE),
                    "accepted": bool(summary["accepted"]),
                    "failed_checks": summary["failed_checks"], "execution": summary}
    np.savez_compressed(args.out / "executed.npz", **executed)
    (args.out / "route_summary.json").write_text(json.dumps(route_report, indent=2) + "\n")
    print(json.dumps(route_report, indent=2))
    return 0 if summary["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
