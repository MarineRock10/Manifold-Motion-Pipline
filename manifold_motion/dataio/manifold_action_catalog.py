"""Build a reproducible catalogue of reverse-synthesized manifolds and actions.

The accepted SEED replay windows already contain the complete Stage-2 condition: an environment
corridor ``M_e(t)``, the measured robot envelope ``M_self(t)``, state/history, and the reference
that frozen SONIC executed.  This command turns that tensor archive into a human-auditable list
with one row per manifold/action window.  It intentionally does not invent extra labels or copy
an action onto an untested corridor: every row keeps its actor-disjoint split and accepted replay
provenance.

The CSV/JSON are small enough to version with the repository.  The tensor NPZ is written beside
them for local training and can be regenerated from the source archive at any time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "manifold-motion.manifold-action-catalog.v1"
SPLIT_NAMES = ("train", "validation", "test")
REQUIRED = (
    "state", "history", "primitive", "target_ref", "target_exec", "split", "clip_index",
    "source_origin", "corridor", "sdf", "self_manifold",
)


def _metadata(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("clips"), list):
        raise ValueError(f"{path} is not seed_windows metadata with a clips list")
    return value


def _hash_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:16]


def _clip_row(clips: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if 0 <= index < len(clips):
        return clips[index]
    return {"motion_id": f"clip_{index:06d}", "actor_uid": "unknown",
            "primitive": "unknown", "source_primitive": "unknown", "windows": "0"}


def _row(*, index: int, arrays: dict[str, np.ndarray], clips: list[dict[str, Any]],
         primitive_names: list[str]) -> dict[str, Any]:
    corridor = np.asarray(arrays["corridor"][index], dtype=np.float32)
    self_manifold = np.asarray(arrays["self_manifold"][index], dtype=np.float32)
    primitive_id = int(arrays["primitive"][index])
    clip_index = int(arrays["clip_index"][index])
    clip = _clip_row(clips, clip_index)
    split_id = int(arrays["split"][index])
    if split_id < 0 or split_id >= len(SPLIT_NAMES):
        raise ValueError(f"invalid split id {split_id} at row {index}")
    if primitive_id < 0 or primitive_id >= len(primitive_names):
        raise ValueError(f"invalid primitive id {primitive_id} at row {index}")
    semi = corridor[:, 3:6]
    self_mean = np.mean(self_manifold, axis=0)
    return {
        "catalog_id": f"me_action_{index:06d}",
        "manifold_hash": _hash_array(corridor),
        "action_id": f"exec_{index:06d}",
        "split": SPLIT_NAMES[split_id],
        "family_id": primitive_id,
        "skill_family": primitive_names[primitive_id],
        "source_motion_id": str(clip.get("motion_id", "")),
        "actor_uid": str(clip.get("actor_uid", "")),
        "clip_index": clip_index,
        "source_origin_30hz": int(arrays["source_origin"][index]),
        "window_frames": int(corridor.shape[0]),
        "corridor_min_semi_x_m": float(np.min(semi[:, 0])),
        "corridor_min_semi_y_m": float(np.min(semi[:, 1])),
        "corridor_min_semi_z_m": float(np.min(semi[:, 2])),
        "corridor_mean_semi_x_m": float(np.mean(semi[:, 0])),
        "corridor_mean_semi_y_m": float(np.mean(semi[:, 1])),
        "corridor_mean_semi_z_m": float(np.mean(semi[:, 2])),
        "self_mean_semi_x_m": float(self_mean[0]),
        "self_mean_semi_y_m": float(self_mean[1]),
        "self_mean_semi_z_m": float(self_mean[2]),
        "source": "reverse_synthesized_from_accepted_R_exec",
        "physical_status": "accepted_clip_window",
    }


def build_catalog(windows: Path, metadata: Path, out: Path, *, max_rows: int = 0) -> dict[str, Any]:
    meta = _metadata(metadata)
    with np.load(windows, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED if key not in archive.files]
        if missing:
            raise ValueError(f"{windows} is missing required arrays: {missing}")
        arrays = {key: np.asarray(archive[key]) for key in archive.files}

    count = int(arrays["state"].shape[0])
    if any(int(arrays[key].shape[0]) != count for key in REQUIRED):
        raise ValueError("catalog arrays do not have the same leading dimension")
    selected = count if max_rows <= 0 else min(count, int(max_rows))
    primitive_names = [str(value) for value in arrays.get("primitive_names", [])]
    if not primitive_names:
        raise ValueError("source archive has no primitive_names array")
    rows = [_row(index=i, arrays=arrays, clips=meta["clips"], primitive_names=primitive_names)
            for i in range(selected)]

    by_actor: dict[str, set[str]] = defaultdict(set)
    for value in rows:
        by_actor[value["actor_uid"]].add(value["split"])
    leakage = {actor: sorted(splits) for actor, splits in by_actor.items() if len(splits) > 1}
    if leakage:
        raise ValueError(f"actor-disjoint split violation for {len(leakage)} actors")

    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "manifold_action_catalog_v1.csv"
    json_path = out / "manifold_action_catalog_v1.json"
    npz_path = out / "manifold_action_catalog_v1.npz"
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    catalog_arrays = {
        key: (value[:selected] if value.ndim else value)
        for key, value in arrays.items()
    }
    catalog_arrays["catalog_id"] = np.asarray([row["catalog_id"] for row in rows])
    catalog_arrays["manifold_hash"] = np.asarray([row["manifold_hash"] for row in rows])
    np.savez_compressed(npz_path, **catalog_arrays)

    family_counts = Counter(row["skill_family"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "source_windows": str(windows),
        "source_metadata": str(metadata),
        "source_provenance": meta.get("environment", {}).get("provenance"),
        "rows": len(rows),
        "source_rows": count,
        "actor_count": len(by_actor),
        "family_count": len(family_counts),
        "family_counts": dict(sorted(family_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "actor_split_leakage": leakage,
        "condition_arrays": {
            "M_e": list(arrays["corridor"].shape[1:]),
            "M_self": list(arrays["self_manifold"].shape[1:]),
            "state": list(arrays["state"].shape[1:]),
            "history": list(arrays["history"].shape[1:]),
            "target_ref": list(arrays["target_ref"].shape[1:]),
            "target_exec": list(arrays["target_exec"].shape[1:]),
        },
        "physical_status": "all rows inherit accepted SONIC/MuJoCo replay admission; no synthetic corridor variants are marked accepted",
        "files": {"csv": str(csv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/manifold_action_catalog_v1"))
    parser.add_argument("--max-rows", type=int, default=0, help="debug cap; 0 keeps every window")
    args = parser.parse_args()
    if args.max_rows < 0:
        parser.error("--max-rows must be non-negative")
    report = build_catalog(args.windows, args.metadata, args.out, max_rows=args.max_rows)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
