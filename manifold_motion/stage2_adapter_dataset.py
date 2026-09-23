"""Create an action-aligned frozen-SONIC/teacher archive for adapter warm-start."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C
from .seed_replay import ReplayConfig, SeedReplayRunner
from .seed_windows import _state_features
from .stage2_validate import _motion_from_trajectory


def _features(data: dict[str, np.ndarray]) -> np.ndarray:
    return _state_features({
        "q_exec": data["q_exec"], "dq_exec": data["dq_exec"],
        "base_quat": data["base_quat"], "base_lin_vel": data["base_lin_vel"],
        "foot_contact": data["foot_contact"], "hand_contact": data["hand_contact"],
        "nonfoot_floor_contact": data["nonfoot_floor_contact"],
    }).astype(np.float32)


def _target_action(q_ref_policy: np.ndarray) -> np.ndarray:
    q_ref_hw = np.asarray(q_ref_policy)[:, C.ISAACLAB_TO_MUJOCO]
    action_hw = (q_ref_hw - C.DEFAULT_ANGLES[None, :]) / C.ACTION_SCALE[None, :]
    return action_hw[:, C.MUJOCO_TO_ISAACLAB].astype(np.float32)


def _stratified_indices(transition: np.ndarray, split: np.ndarray, limit: int) -> np.ndarray:
    """Select clips across every available transition/split group deterministically."""
    count = len(transition)
    if limit <= 0 or limit >= count:
        return np.arange(count, dtype=np.int64)
    groups: list[np.ndarray] = []
    for name in sorted(np.unique(transition).tolist()):
        for split_id in sorted(np.unique(split[transition == name]).tolist()):
            index = np.flatnonzero((transition == name) & (split == split_id))
            # Start near the center, then alternate toward both ends. This avoids always
            # selecting the earliest temporally adjacent source window from each group.
            center = (len(index) - 1) / 2.0
            order = np.argsort(np.abs(np.arange(len(index)) - center), kind="stable")
            groups.append(index[order])
    selected: list[int] = []
    cursor = np.zeros(len(groups), dtype=np.int64)
    while len(selected) < limit:
        progressed = False
        for group_id, group in enumerate(groups):
            if cursor[group_id] >= len(group):
                continue
            selected.append(int(group[cursor[group_id]]))
            cursor[group_id] += 1
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


def build(transitions: Path, out: Path, limit: int = 0) -> dict:
    with np.load(transitions, allow_pickle=False) as archive:
        archive_data = {key: np.asarray(archive[key]) for key in archive.files}
    indices = _stratified_indices(
        archive_data["transition"], archive_data["split"], limit)
    runner = SeedReplayRunner(C.FLAT_SCENE)
    rows = []
    accepted = 0
    for index in indices:
        trajectory = archive_data["target_ref"][index]
        motion = _motion_from_trajectory(trajectory, transitions, 30.0, 50.0)
        data, summary = runner.replay(motion, ReplayConfig(),
                                      stratum=f"transition_{archive_data['transition'][index]}")
        if not summary["accepted"]:
            continue
        accepted += 1
        state = _features(data)
        history_rows = []
        for frame in range(len(state)):
            begin = max(0, frame - 11)
            value = state[begin:frame + 1]
            if len(value) < 12:
                value = np.concatenate([np.repeat(value[:1], 12 - len(value), axis=0), value], axis=0)
            history_rows.append(value[-12:])
        history = np.asarray(history_rows, dtype=np.float32)
        manifold = np.repeat(archive_data["manifold"][index][None, :], len(state), axis=0)
        corridor = np.repeat(archive_data["corridor"][index][None, :], len(state), axis=0)
        sdf = np.repeat(archive_data["sdf"][index][None, :], len(state), axis=0)
        command = np.repeat(archive_data["command"][index][None, :], len(state), axis=0)
        primitive = np.full(len(state), archive_data["primitive"][index], dtype=np.int64)
        split = np.full(len(state), archive_data["split"][index], dtype=np.uint8)
        rows.append({
            "state": state, "history": history, "manifold": manifold, "corridor": corridor,
            "sdf": sdf, "command": command, "primitive": primitive, "split": split,
            "base_action": data["action"].astype(np.float32),
            "target_action": _target_action(data["q_ref"]),
            "transition": np.repeat(archive_data["transition"][index], len(state)),
            "source_index": np.full(len(state), index, dtype=np.int32),
        })
    if not rows:
        raise RuntimeError("no physically accepted transition produced an adapter sample")
    keys = rows[0].keys()
    output = {key: np.concatenate([row[key] for row in rows], axis=0) for key in keys}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **output)
    report = {
        "accepted": True, "transitions": str(transitions), "output": str(out),
        "transition_samples_tested": int(len(indices)), "accepted_transitions": accepted,
        "action_rows": int(len(output["state"])),
        "selected_clip_splits": {
            str(int(value)): int(np.sum(archive_data["split"][indices] == value))
            for value in np.unique(archive_data["split"][indices])
        },
        "selected_transition_clips": {
            str(value): int(np.sum(archive_data["transition"][indices] == value))
            for value in np.unique(archive_data["transition"][indices])
        },
        "base_action_contract": "exact frozen SONIC decoder action from SeedReplayRunner",
        "target_action_contract": "policy-reference joint target converted to SONIC IsaacLab action units",
        "split_contract": "source transition split retained; no random window leakage",
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="number of transition clips to replay; 0 means all")
    args = parser.parse_args()
    report = build(args.transitions, args.out, args.limit)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
