"""Build an auditable primitive-specific Stage-2 dataset from compatible window archives.

The original pilot contains the crouch training actors but no accepted crouch validation/test
actors.  The extension archive adds only independently held-out A129/A247 actors.  This command
selects one primitive ID from each archive and joins arrays without re-splitting, so actors never
leak across train/validation/test just because windows are concatenated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED = ("state", "history", "primitive", "manifold", "command", "target_ref", "target_exec", "split")
OPTIONAL = ("corridor", "sdf", "clip_index", "source_origin")


def _load(path: Path, primitive: int) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        missing = set(REQUIRED) - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        indices = np.flatnonzero(np.asarray(archive["primitive"]) == primitive)
        if not len(indices):
            raise ValueError(f"{path} contains no primitive {primitive}")
        keys = list(REQUIRED) + [key for key in OPTIONAL if key in archive.files]
        return {key: np.asarray(archive[key])[indices] for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description="merge actor-disjoint primitive windows without re-splitting")
    parser.add_argument("--base", type=Path, required=True, help="archive providing training actors")
    parser.add_argument("--heldout", type=Path, required=True, help="archive providing validation/test actors")
    parser.add_argument("--primitive-id", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base, heldout = _load(args.base, args.primitive_id), _load(args.heldout, args.primitive_id)
    shared = sorted(set(base) & set(heldout))
    missing_optional = sorted((set(base) | set(heldout)) - set(shared) - set(REQUIRED))
    if missing_optional:
        raise ValueError(f"window archives do not share optional fields {missing_optional}")
    arrays: dict[str, np.ndarray] = {}
    for key in shared:
        if base[key].shape[1:] != heldout[key].shape[1:]:
            raise ValueError(f"{key} shape differs: {base[key].shape[1:]} vs {heldout[key].shape[1:]}")
        arrays[key] = np.concatenate([base[key], heldout[key]], axis=0)
    # Clip indices are only tracing fields.  Offset held-out clips to make the combined archive
    # unambiguous in reports and retain source provenance in a separate row-aligned vector.
    if "clip_index" in arrays:
        base_count = len(base["clip_index"])
        arrays["clip_index"] = np.concatenate([base["clip_index"], heldout["clip_index"] + base_count])
    arrays["dataset_source"] = np.concatenate([
        np.zeros(len(base["split"]), dtype=np.uint8), np.ones(len(heldout["split"]), dtype=np.uint8)])
    counts = {name: int((arrays["split"] == code).sum())
              for code, name in enumerate(("train", "validation", "test"))}
    if not counts["train"] or not counts["validation"] or not counts["test"]:
        raise ValueError(f"merged dataset must contain every actor split; got {counts}")
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "seed_stage2_windows.npz", **arrays)
    report = {"primitive_id": args.primitive_id, "base": str(args.base), "heldout": str(args.heldout),
              "windows": int(len(arrays["split"])), "split_counts": counts,
              "base_rows": int(len(base["split"])), "heldout_rows": int(len(heldout["split"])),
              "source_encoding": {"0": "base training archive", "1": "new held-out actor archive"},
              "warning": "No random re-split occurred; split IDs are preserved from actor UIDs."}
    (args.out / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
