"""Build an action-aligned, failure-directed SONIC adapter archive from accepted replays.

Only accepted supplemental clips are admitted.  The archive keeps the frozen SONIC action and the
teacher reference action side by side, so the later adapter is a bounded residual warm-start and
not a replacement controller trained from unverified motion.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from manifold_motion.core import constants as C
from manifold_motion.dataio.seed_capability_catalog import FAMILIES
from manifold_motion.dataio.seed_windows import _state_features
from manifold_motion.planning.corridor import CorridorConfig, ExecutedEnvelopeEstimator, corridor_sdf
from manifold_motion.stage2.adapter_dataset import _target_action


def _history(state: np.ndarray, length: int = 12) -> np.ndarray:
    rows = []
    for frame in range(len(state)):
        begin = max(0, frame - length + 1)
        value = state[begin:frame + 1]
        if len(value) < length:
            value = np.concatenate((np.repeat(value[:1], length - len(value), axis=0), value), axis=0)
        rows.append(value[-length:])
    return np.asarray(rows, dtype=np.float32)


def _command(data: dict[str, np.ndarray], frame: int, horizon: int = 36) -> np.ndarray:
    target = min(len(data["root_ref_pos_seed_m"]) - 1, frame + horizon)
    origin = data["root_ref_pos_seed_m"][frame]
    rotation = C.quat_to_matrix(data["base_quat"][frame])
    delta = (data["root_ref_pos_seed_m"][target] - origin) @ rotation
    relative = C.quat_mul(C.quat_conj(data["base_quat"][frame]), data["root_ref_quat"][target])
    matrix = C.quat_to_matrix(relative)
    rot6 = np.asarray([matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1], matrix[2, 0], matrix[2, 1]])
    return np.concatenate((delta, rot6)).astype(np.float32)


def _flat_environment(semi: np.ndarray, rows: int, corridor_frames: int = 36,
                      sdf_shape: tuple[int, int, int] = (10, 10, 8)) -> tuple[np.ndarray, np.ndarray]:
    corridor = np.zeros((rows, corridor_frames, 7), dtype=np.float32)
    corridor[..., 3:6] = (np.asarray(semi, dtype=np.float32) + 0.12)[None, None, :]
    one = corridor_sdf(corridor[0], CorridorConfig(sdf_shape=sdf_shape))
    sdf = np.repeat(one[None], rows, axis=0)
    return corridor, sdf.astype(np.float32)


def build(replay_root: Path, manifest: Path, out: Path) -> dict:
    with manifest.open(newline="", encoding="utf-8") as handle:
        manifest_rows = {row["motion_id"]: row for row in csv.DictReader(handle)}
    rows = []
    envelope_estimator = ExecutedEnvelopeEstimator()
    accepted = 0
    for summary_path in sorted((replay_root / "summaries").glob("*.json")):
        summary = json.loads(summary_path.read_text())
        if not summary.get("accepted"):
            continue
        motion_id = summary["motion_id"]
        row = manifest_rows.get(motion_id)
        record_path = replay_root / "records" / f"{motion_id}.npz"
        if row is None or not record_path.is_file():
            continue
        with np.load(record_path) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        state = _state_features({key: data[key] for key in (
            "q_exec", "dq_exec", "base_quat", "base_lin_vel", "foot_contact",
            "hand_contact", "nonfoot_floor_contact")}).astype(np.float32)
        # The route/self-manifold fields are fixed-shape and derived from the accepted execution.
        envelope_semi = envelope_estimator.sequence(data["q_exec"], data["base_pos"], data["base_quat"])
        semi = np.asarray(envelope_semi[0], dtype=np.float32)
        manifold = np.concatenate((envelope_semi.astype(np.float32),
                                   np.zeros((len(state), 3), dtype=np.float32)), axis=1)
        corridor, sdf = _flat_environment(semi, len(state))
        command = np.asarray([_command(data, frame) for frame in range(len(state))], dtype=np.float32)
        split_name = row.get("selection_role", "").rsplit("_", 1)[-1]
        split = {"train": 0, "validation": 1, "test": 2}.get(split_name, 0)
        rows.append({"state": state, "history": _history(state), "manifold": manifold,
                     "corridor": corridor, "sdf": sdf, "command": command,
                     "primitive": np.full(len(state), int(row["family_id"]), dtype=np.int64),
                     "split": np.full(len(state), split, dtype=np.uint8),
                     "base_action": data["action"].astype(np.float32),
                     "target_action": _target_action(data["q_ref"]),
                     "source_motion_id": np.repeat(motion_id, len(state))})
        accepted += 1
    if not rows:
        raise RuntimeError("no accepted supplemental replay records found")
    keys = rows[0].keys()
    output = {key: np.concatenate([row[key] for row in rows], axis=0) for key in keys}
    output["primitive_names"] = np.asarray([family.name for family in FAMILIES])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **output)
    report = {"schema": "manifold-motion.seed-adapter-dataset.v1", "replay_root": str(replay_root),
              "manifest": str(manifest), "accepted_clips": accepted, "action_rows": int(len(output["state"])),
              "split_counts": {str(i): int(np.sum(output["split"] == i)) for i in range(3)},
              "base_action_contract": "frozen SONIC action from SeedReplayRunner",
              "target_action_contract": "accepted SEED reference converted to SONIC IsaacLab units",
              "environment_note": "fixed-shape reverse-compatible corridor/SDF; only accepted rows admitted"}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.replay_root, args.manifest, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
