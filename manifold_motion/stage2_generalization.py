"""Stage-2 dataset audit and support-preserving environment augmentation.

The pilot windows already use actor-disjoint splits, but they do not expose the geometry and
transition coverage that a generalisation claim needs.  This module keeps that split intact,
adds auditable ``environment_bucket`` and ``transition_phase`` metadata, and can create bounded
condition variants by widening the reverse corridor.  The variants are deliberately marked as
synthetic: they are useful for conditioning robustness, not a replacement for sensor-derived
obstacle data or a physical replay.

Examples::

    python3 -m manifold_motion.stage2_generalization audit \
      --windows reports/manifold_motion/seed_windows_corridor_stage2_v2/seed_stage2_windows.npz

    python3 -m manifold_motion.stage2_generalization augment \
      --windows reports/manifold_motion/seed_windows_corridor_stage2_v2/seed_stage2_windows.npz \
      --out reports/manifold_motion/seed_windows_generalization_v1 \
      --copies 2 --seed 20260921
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .perception_corridor import PerceptionGridConfig, corridor_condition_sdf
from .seed_windows import PRIMITIVE_NAMES


REQUIRED = {
    "state", "history", "primitive", "manifold", "command", "target_ref", "target_exec",
    "split", "clip_index", "source_origin", "corridor", "sdf",
}


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        return {key: np.asarray(archive[key]) for key in archive.files}


def _clip_phase(raw: dict[str, np.ndarray]) -> np.ndarray:
    """Normalised source time within a clip, used only as metadata for transition analysis."""
    clip = raw["clip_index"].astype(np.int64)
    origin = raw["source_origin"].astype(np.float32)
    phase = np.zeros(len(clip), dtype=np.float32)
    for value in np.unique(clip):
        indices = np.flatnonzero(clip == value)
        if len(indices) == 0:
            continue
        span = float(np.max(origin[indices]) - np.min(origin[indices]))
        phase[indices] = (origin[indices] - np.min(origin[indices])) / max(span, 1.0)
    return phase


def _environment_signature(raw: dict[str, np.ndarray]) -> np.ndarray:
    corridor = raw["corridor"].astype(np.float32)
    semi = corridor[:, :, 3:6]
    yaw = corridor[:, :, 6]
    sdf = raw["sdf"].astype(np.float32)
    return np.column_stack([
        semi.mean(axis=1), semi.min(axis=1), semi.max(axis=1),
        np.quantile(semi, 0.1, axis=1), np.quantile(semi, 0.9, axis=1),
        np.mean(np.cos(yaw), axis=1), np.mean(np.sin(yaw), axis=1),
        sdf.mean(axis=(1, 2, 3)), (sdf > 0).mean(axis=(1, 2, 3)),
    ]).astype(np.float32)


def _bucket(signature: np.ndarray) -> np.ndarray:
    """Stable coarse geometry bucket; no labels or primitive IDs are used."""
    # Use only coarse aperture statistics and free-space fraction.  Including every time-aligned
    # corridor frame would make almost every window its own bucket and would not be a useful
    # environment split diagnostic.
    selected = np.column_stack([signature[:, :3], signature[:, 3:6], signature[:, -1]])
    quantised = np.round(selected, 2)
    quantised[:, :6] = np.round(quantised[:, :6] / 0.10) * 0.10
    quantised[:, 6:] = np.round(quantised[:, 6:] / 0.10) * 0.10
    values = []
    for row in quantised:
        digest = hashlib.sha1(row.tobytes()).hexdigest()[:10]
        values.append(int(digest, 16) % 10_000_000)
    return np.asarray(values, dtype=np.int64)


def audit(raw: dict[str, np.ndarray]) -> dict[str, Any]:
    primitive = raw["primitive"].astype(np.int64)
    split = raw["split"].astype(np.int64)
    signature = _environment_signature(raw)
    bucket = _bucket(signature)
    phase = _clip_phase(raw)
    report: dict[str, Any] = {
        "windows": int(len(primitive)),
        "split_counts": {name: int(np.sum(split == value))
                         for value, name in enumerate(("train", "validation", "test"))},
        "primitive_counts": {
            PRIMITIVE_NAMES[int(value)] if 0 <= value < len(PRIMITIVE_NAMES) else str(int(value)):
            int(np.sum(primitive == value)) for value in np.unique(primitive)
        },
        "actor_disjoint_split_present": bool("split" in raw and "clip_index" in raw),
        "clip_count": int(len(np.unique(raw["clip_index"]))),
        "environment_bucket_count": int(len(np.unique(bucket))),
        "environment_bucket_counts": {str(int(value)): int(np.sum(bucket == value))
                                       for value in np.unique(bucket)},
        "transition_phase": {
            "low_transition_windows": int(np.sum(primitive == 3)),
            "phase_p10": float(np.quantile(phase[primitive == 3], 0.1)) if np.any(primitive == 3) else None,
            "phase_p50": float(np.quantile(phase[primitive == 3], 0.5)) if np.any(primitive == 3) else None,
            "phase_p90": float(np.quantile(phase[primitive == 3], 0.9)) if np.any(primitive == 3) else None,
        },
        "environment_signature": {
            "semi_xyz_min": signature[:, :3].min(axis=0).astype(float).tolist(),
            "semi_xyz_max": signature[:, :3].max(axis=0).astype(float).tolist(),
            "sdf_free_fraction_min": float(signature[:, -1].min()),
            "sdf_free_fraction_max": float(signature[:, -1].max()),
        },
        "warning": "corridor/SDF labels are reverse-synthesized; this is not a perception result",
    }
    return report


def _support_scale(primitive: int, rng: np.random.Generator) -> np.ndarray:
    """Only widen the safe corridor, preserving the recorded motion's support semantics."""
    scale = rng.uniform(1.0, 1.12, size=3).astype(np.float32)
    if primitive == 2:      # crouch: vary height context, but never make it tighter
        scale[2] = rng.uniform(1.0, 1.16)
    elif primitive == 4:    # side-on: vary lateral clearance context
        scale[1] = rng.uniform(1.0, 1.16)
    elif primitive == 5:    # nominal walk: keep a meaningful wide context
        scale[:] = rng.uniform(1.02, 1.18, size=3)
    return scale


def _augment_one(raw: dict[str, np.ndarray], copy_index: int,
                 rng: np.random.Generator) -> dict[str, np.ndarray]:
    # Augment training actors only.  Validation/test actors remain pristine so reported
    # generalisation cannot be improved by evaluating synthetic copies of the same windows.
    train = raw["split"] == 0
    result = {key: value[train].copy() if np.asarray(value).ndim > 0 and len(value) == len(train)
              else value.copy() for key, value in raw.items()
              if key not in {"environment_bucket", "transition_phase", "augmentation_id"}}
    corridor = raw["corridor"][train].copy().astype(np.float32)
    for index, primitive in enumerate(raw["primitive"][train].astype(np.int64)):
        scale = _support_scale(int(primitive), rng)
        corridor[index, :, 3:6] *= scale[None, :]
    sdf = np.stack([corridor_condition_sdf(item, PerceptionGridConfig()) for item in corridor])
    result["corridor"] = corridor.astype(np.float32)
    result["sdf"] = sdf.astype(np.float32)
    # Small initial-state perturbations make the conditional model see realistic tracking
    # deviations.  Contact bits remain exact and target_ref remains the physically verified
    # source motion; this is a robustness augmentation, not a claim that the perturbed state
    # was separately replayed.
    for key in ("state", "history"):
        value = result[key].astype(np.float32)
        value[..., :29] += rng.normal(0.0, 0.015, size=value[..., :29].shape).astype(np.float32)
        value[..., 29:58] += rng.normal(0.0, 0.05, size=value[..., 29:58].shape).astype(np.float32)
        value[..., 58:61] += rng.normal(0.0, 0.005, size=value[..., 58:61].shape).astype(np.float32)
        value[..., 61:64] += rng.normal(0.0, 0.03, size=value[..., 61:64].shape).astype(np.float32)
        result[key] = value
    # A copied window belongs to the same actor split and clip, so no information leaks across
    # train/validation/test.  The synthetic ID is only for audit and can be ignored by loaders.
    result["augmentation_id"] = np.full(len(corridor), copy_index, dtype=np.int16)
    return result


def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = sorted(set().union(*(part.keys() for part in parts)))
    output: dict[str, np.ndarray] = {}
    for key in keys:
        arrays = [part[key] for part in parts if key in part]
        if len(arrays) != len(parts):
            continue
        output[key] = np.concatenate(arrays, axis=0)
    return output


def command_audit(args: argparse.Namespace) -> int:
    raw = _load(args.windows)
    report = audit(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def command_augment(args: argparse.Namespace) -> int:
    if args.copies < 1:
        raise ValueError("--copies must be >= 1")
    raw = _load(args.windows)
    rng = np.random.default_rng(args.seed)
    parts = [raw]
    parts.extend(_augment_one(raw, copy_index, rng) for copy_index in range(1, args.copies + 1))
    output = _concat(parts)
    output["transition_phase"] = _clip_phase(output)
    signature = _environment_signature(output)
    output["environment_bucket"] = _bucket(signature)
    output["augmentation_id"] = np.concatenate([
        np.zeros(len(raw["state"]), dtype=np.int16),
        *[np.full(int(np.sum(raw["split"] == 0)), i, dtype=np.int16) for i in range(1, args.copies + 1)],
    ])
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "seed_stage2_windows.npz", **output)
    metadata = audit(output)
    metadata.update({
        "source": str(args.windows), "copies": int(args.copies), "seed": int(args.seed),
        "augmentation": "support-preserving corridor widening + regenerated local SDF",
        "synthetic_condition_warning": "variants are not sensor-derived and were not physically replayed",
    })
    (args.out / "seed_stage2_windows_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "windows": int(len(output["state"])),
                      "split_counts": metadata["split_counts"],
                      "environment_bucket_count": metadata["environment_bucket_count"]}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit/augment Stage-2 dynamic windows")
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--windows", type=Path, required=True)
    audit_parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/stage2_data_audit.json"))
    audit_parser.set_defaults(handler=command_audit)
    augment = sub.add_parser("augment")
    augment.add_argument("--windows", type=Path, required=True)
    augment.add_argument("--out", type=Path, required=True)
    augment.add_argument("--copies", type=int, default=2)
    augment.add_argument("--seed", type=int, default=20260921)
    augment.set_defaults(handler=command_augment)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
