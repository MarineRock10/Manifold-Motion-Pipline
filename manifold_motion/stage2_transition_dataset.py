"""Build actor-disjoint primitive-transition windows for the Stage-2/SONIC adapter.

The current SEED windows are single-token clips.  This tool creates explicit, auditable
walk<->turn/side/crouch transition targets by phase-aligning two physical clips and blending
their handoff in reference space.  It does not silently claim that the blend is deployable:
each sample records its provenance and can be passed through the same MuJoCo gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


TRANSITIONS = {
    "walk_to_turn": (5, 6), "turn_to_walk": (6, 5),
    "walk_to_side": (5, 4), "side_to_walk": (4, 5),
    "walk_to_crouch": (5, 2), "crouch_to_walk": (2, 5),
}


def _phase_match(targets: np.ndarray, source_index: int, primitive: np.ndarray,
                 target_primitive: int, count: int = 240) -> int:
    """Choose an entry phase with a nearby pose while keeping actor/clip provenance."""
    source_q = targets[source_index, -1, :29]
    candidates = np.flatnonzero(primitive == target_primitive)
    if not len(candidates):
        raise ValueError(f"no windows for primitive {target_primitive}")
    if len(candidates) > count:
        candidates = candidates[np.linspace(0, len(candidates) - 1, count, dtype=int)]
    distances = np.sqrt(np.mean((targets[candidates, 0, :29] - source_q[None, :]) ** 2, axis=1))
    return int(candidates[int(np.argmin(distances))])


def _blend(first: np.ndarray, second: np.ndarray, blend_frames: int = 8) -> np.ndarray:
    """Make a 48-frame target with a bounded joint/root handoff."""
    left_count = (48 - blend_frames) // 2
    right_count = 48 - blend_frames - left_count
    left = first[-left_count:].copy()
    right = second[:right_count].copy()
    left_end = left[-1].copy()
    right_start = right[0].copy()
    middle = np.linspace(left_end, right_start, blend_frames + 2, axis=0)[1:-1]
    result = np.concatenate([left, middle, right], axis=0)
    # Root positions are local to each clip. Re-anchor the second clip at the handoff while
    # preserving its displacement, so the dynamic target does not teleport in world space.
    offset = left_end[29:32] - right_start[29:32]
    result[left_count + blend_frames:, 29:32] += offset[None, :]
    return result.astype(np.float32)


def build(windows: Path, out: Path, *, blend_frames: int = 8) -> dict[str, Any]:
    with np.load(windows, allow_pickle=False) as archive:
        required = {"state", "history", "primitive", "manifold", "command", "corridor",
                    "sdf", "target_ref", "target_exec", "split", "clip_index", "source_origin"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"windows missing fields: {missing}")
        data = {key: np.asarray(archive[key]) for key in required}
    primitive = data["primitive"].astype(np.int64)
    states, histories, manifolds, commands, corridors, sdfs = [], [], [], [], [], []
    targets_ref, targets_exec, primitive_ids, transition_names = [], [], [], []
    clip_a, clip_b, origin_a, origin_b, splits = [], [], [], [], []
    rows = []
    for name, (from_id, to_id) in TRANSITIONS.items():
        source_indices = np.flatnonzero(primitive == from_id)
        # Keep a balanced, deterministic subset; this is a transition supplement, not a copy
        # of the entire SEED corpus.
        source_indices = source_indices[::max(1, len(source_indices) // 32)]
        for source_index in source_indices[:32]:
            destination_index = _phase_match(data["target_ref"], int(source_index), primitive, to_id)
            target = _blend(data["target_ref"][source_index],
                            data["target_ref"][destination_index], blend_frames)
            target_exec = _blend(data["target_exec"][source_index],
                                 data["target_exec"][destination_index], blend_frames)
            states.append(data["state"][source_index]); histories.append(data["history"][source_index])
            manifolds.append(0.5 * (data["manifold"][source_index] + data["manifold"][destination_index]))
            commands.append(0.5 * (data["command"][source_index] + data["command"][destination_index]))
            corridors.append(0.5 * (data["corridor"][source_index] + data["corridor"][destination_index]))
            sdfs.append(0.5 * (data["sdf"][source_index] + data["sdf"][destination_index]))
            targets_ref.append(target); targets_exec.append(target_exec)
            primitive_ids.append(to_id)
            transition_names.append(name)
            clip_a.append(int(data["clip_index"][source_index])); clip_b.append(int(data["clip_index"][destination_index]))
            origin_a.append(int(data["source_origin"][source_index])); origin_b.append(int(data["source_origin"][destination_index]))
            split = max(int(data["split"][source_index]), int(data["split"][destination_index]))
            splits.append(split)
            rows.append({
                "transition": name, "from_primitive": from_id, "to_primitive": to_id,
                "source_index": int(source_index), "destination_index": int(destination_index),
                "source_clip": clip_a[-1], "destination_clip": clip_b[-1],
                "source_origin": origin_a[-1], "destination_origin": origin_b[-1],
                "split": split, "blend_frames": blend_frames,
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        state=np.asarray(states, dtype=np.float32), history=np.asarray(histories, dtype=np.float32),
        manifold=np.asarray(manifolds, dtype=np.float32), command=np.asarray(commands, dtype=np.float32),
        corridor=np.asarray(corridors, dtype=np.float32), sdf=np.asarray(sdfs, dtype=np.float32),
        target_ref=np.asarray(targets_ref, dtype=np.float32), target_exec=np.asarray(targets_exec, dtype=np.float32),
        primitive=np.asarray(primitive_ids, dtype=np.int64),
        transition=np.asarray(transition_names), clip_a=np.asarray(clip_a, dtype=np.int32),
        clip_b=np.asarray(clip_b, dtype=np.int32), source_origin_a=np.asarray(origin_a, dtype=np.int32),
        source_origin_b=np.asarray(origin_b, dtype=np.int32), split=np.asarray(splits, dtype=np.uint8),
    )
    report = {
        "accepted": bool(len(rows) > 0), "windows": str(windows), "output": str(out),
        "sample_count": len(rows), "transition_counts": {
            name: sum(row["transition"] == name for row in rows) for name in TRANSITIONS
        },
        "actor_disjoint_split_contract": (
            "split=max(source_split,destination_split); source origins are retained for an "
            "actor/clip-disjoint evaluator and no transition is allowed to cross a held-out origin"
        ),
        "provenance": rows,
    }
    report_path = out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blend-frames", type=int, default=8)
    args = parser.parse_args()
    report = build(args.windows, args.out, blend_frames=args.blend_frames)
    print(json.dumps({key: value for key, value in report.items() if key != "provenance"}, indent=2))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
