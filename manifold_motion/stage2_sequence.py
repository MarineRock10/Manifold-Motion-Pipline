"""Assemble and physically validate a continuous Stage-2 primitive sequence.

The former demos replayed each primitive from its own recorded initial state, which is useful
for unit tests but is not a navigation task.  This module makes the hand-off explicit: it
rebases each next local-root reference at the preceding segment endpoint, inserts a smooth
joint/root bridge, and sends the single assembled reference to SONIC in one MuJoCo rollout.
There is no simulator reset at a segment boundary.  The pre-bridge mismatch is reported and can
be hard-gated, so interpolation cannot hide an incompatible primitive pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C
from .seed_replay import ReplayConfig, SeedReplayRunner
from .stage2_validate import _motion_from_trajectory, validate_trajectory


def _parse_segment(value: str) -> tuple[str, Path]:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as error:
        raise ValueError("--segment must be NAME=PATH_TO_SAMPLE_NPZ") from error
    if not name or not raw_path:
        raise ValueError("segment name and path must be non-empty")
    return name, Path(raw_path)


def _load_sample(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        trajectory = np.asarray(np.load(path), dtype=np.float32)
    else:
        with np.load(path) as archive:
            if "generated_ref" not in archive.files:
                raise ValueError(f"{path} has no generated_ref")
            trajectory = np.asarray(archive["generated_ref"], dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != 38 or len(trajectory) < 2:
        raise ValueError(f"{path} generated_ref must be [T>=2,38]")
    if not np.isfinite(trajectory).all():
        raise ValueError(f"{path} generated_ref contains non-finite values")
    return trajectory


def _bridge(start: np.ndarray, end: np.ndarray, frames: int) -> np.ndarray:
    """Cubic interpolation excluding both endpoints, which already occur in the sequence."""
    if frames == 0:
        return np.empty((0, 38), dtype=np.float32)
    alpha = np.arange(1, frames + 1, dtype=np.float32) / float(frames + 1)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return (start[None, :] + alpha[:, None] * (end[None, :] - start[None, :])).astype(np.float32)


def assemble(segments: list[tuple[str, np.ndarray]], bridge_frames: int,
             phase_match_frames: int = 0) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Join root-local references without changing their joint target samples."""
    if len(segments) < 2:
        raise ValueError("a continuous task requires at least two segments")
    names, first = segments[0]
    sequence = [first]
    boundaries: list[dict[str, object]] = []
    current_end = first[-1].copy()
    for name, next_segment in segments[1:]:
        match_index = 0
        if phase_match_frames > 0:
            search_stop = min(len(next_segment) - 1, phase_match_frames)
            errors = np.sqrt(np.mean(
                (next_segment[:search_stop, :29] - current_end[None, :29]) ** 2,
                axis=1,
            ))
            match_index = int(np.argmin(errors))
            next_segment = next_segment[match_index:].copy()
        raw_start = next_segment[0].copy()
        # Root xyz is local to each sampled segment.  Rebase it to the endpoint of the prior
        # segment; the 6-D orientation reference is left in the same heading convention used by
        # the current Stage-2 targets and is safely interpolated at the boundary.
        rebased = next_segment.copy()
        root_shift = current_end[29:32] - raw_start[29:32]
        rebased[:, 29:32] += root_shift[None, :]
        delta = current_end[:29] - rebased[0, :29]
        rms = float(np.sqrt(np.mean(delta ** 2)))
        maximum = float(np.max(np.abs(delta)))
        boundaries.append({"next_segment": name, "pre_bridge_joint_rms_rad": rms,
                           "pre_bridge_joint_abs_max_rad": maximum,
                           "root_translation_rebase_m": [float(value) for value in root_shift],
                           "phase_match_source_frame": match_index,
                           "bridge_frames": int(bridge_frames),
                           "first_frame_after_join": int(sum(len(part) for part in sequence) + bridge_frames)})
        sequence.append(_bridge(current_end, rebased[0], bridge_frames))
        sequence.append(rebased[1:])
        current_end = rebased[-1]
    return np.concatenate(sequence, axis=0).astype(np.float32), boundaries


def main() -> int:
    parser = argparse.ArgumentParser(description="execute a no-reset sequence of Stage-2 primitive references")
    parser.add_argument("--segment", action="append", required=True,
                        help="ordered primitive segment: NAME=selected_sample.npz (repeat at least twice)")
    parser.add_argument("--bridge-frames", type=int, default=18)
    parser.add_argument("--phase-match-frames", type=int, default=0,
                        help="search this many prefix frames for the safest learned handoff pose")
    parser.add_argument("--max-pre-bridge-rms", type=float, default=0.60,
                        help="reject an unacceptably discontinuous learned hand-off")
    parser.add_argument("--scene", type=Path, default=None,
                        help="optional physical MuJoCo scene; named obstacle contacts are hard failures")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-hz", type=float, default=30.0)
    parser.add_argument("--stratum", default="low_transition")
    parser.add_argument("--min-terminal-progress", type=float, default=ReplayConfig.min_progress_ratio,
                        help="minimum executed/reference planar progress for the final locomotion segment")
    args = parser.parse_args()
    if (args.bridge_frames < 0 or args.phase_match_frames < 0 or args.max_pre_bridge_rms <= 0
            or args.source_hz <= 0 or args.min_terminal_progress < 0):
        parser.error("bridge/phase-match frames must be non-negative; rms/source-hz positive; terminal progress non-negative")
    parsed = [_parse_segment(value) for value in args.segment]
    if len({name for name, _ in parsed}) != len(parsed):
        parser.error("segment names must be unique")
    trajectories = [(name, _load_sample(path)) for name, path in parsed]
    sequence, boundaries = assemble(trajectories, args.bridge_frames, args.phase_match_frames)
    join_ok = all(float(item["pre_bridge_joint_rms_rad"]) <= args.max_pre_bridge_rms for item in boundaries)
    args.out.mkdir(parents=True, exist_ok=True)
    sequence_path = args.out / "sequence.npz"
    np.savez_compressed(sequence_path, generated_ref=sequence,
                        segment_boundary_frames=np.asarray([item["first_frame_after_join"] for item in boundaries], dtype=np.int32),
                        segment_names=np.asarray([name for name, _ in parsed]),
                        bridge_frames=np.asarray(args.bridge_frames, dtype=np.int32))
    runner = SeedReplayRunner(C.REPO / args.scene if args.scene is not None else None)
    executed, summary = validate_trajectory(sequence, source=sequence_path, source_hz=args.source_hz,
                                             config=ReplayConfig(), stratum=args.stratum, runner=runner)
    # The global sequence can contain an in-place posture transition, whose learned local root
    # trace is not a navigation objective.  Score completion using the final primitive's own
    # 50 Hz reference, after the no-reset hand-off.  This preserves the ordinary 20% SONIC
    # progress requirement without incorrectly dividing by a transition's arbitrary local-root
    # coordinate variation.
    terminal_name, terminal_trajectory = trajectories[-1]
    terminal_motion = _motion_from_trajectory(terminal_trajectory, sequence_path, args.source_hz, 1.0 / C.CONTROL_DT)
    terminal_reference_path = float(np.linalg.norm(
        np.diff(terminal_motion.root_pos_seed_m[:, :2], axis=0), axis=1).sum())
    terminal_start_frame = int(boundaries[-1]["first_frame_after_join"])
    terminal_start_tick = int(np.searchsorted(executed["t"], terminal_start_frame / args.source_hz, side="left"))
    terminal_start_tick = min(max(terminal_start_tick, 0), len(executed["base_pos"]) - 1)
    terminal_executed_path = float(np.linalg.norm(
        np.diff(executed["base_pos"][terminal_start_tick:, :2], axis=0), axis=1).sum())
    terminal_required = terminal_reference_path >= ReplayConfig.min_reference_planar_path_m
    terminal_ratio = terminal_executed_path / max(terminal_reference_path, 1e-8)
    terminal_ok = (not terminal_required) or terminal_ratio >= args.min_terminal_progress
    summary["terminal_segment"] = terminal_name
    summary["terminal_reference_planar_path_m"] = terminal_reference_path
    summary["terminal_exec_path_m"] = terminal_executed_path
    summary["terminal_motion_progress_ratio"] = terminal_ratio
    summary["terminal_motion_progress_required"] = terminal_required
    if not join_ok:
        summary["failed_checks"] = list(summary["failed_checks"]) + ["handoff_discontinuity"]
        summary["accepted"] = False
    if not terminal_ok:
        summary["failed_checks"] = list(summary["failed_checks"]) + ["terminal_insufficient_motion_progress"]
        summary["accepted"] = False
    np.savez_compressed(args.out / "executed.npz", **executed)
    report = {"segments": [{"name": name, "sample": str(path), "frames": int(len(trajectory))}
                           for (name, path), (_, trajectory) in zip(parsed, trajectories)],
              "sequence_frames": int(len(sequence)), "bridge_frames": args.bridge_frames,
              "phase_match_frames": args.phase_match_frames,
              "boundaries": boundaries, "max_pre_bridge_rms_rad": args.max_pre_bridge_rms,
              "min_terminal_progress": args.min_terminal_progress,
              "no_reset_between_segments": True, "root_translation_rebased": True,
              "scene": str(C.REPO / args.scene) if args.scene is not None else str(C.FLAT_SCENE),
              "corridor_gate": "physical named-obstacle gate; no synthetic concatenated corridor supplied",
              "execution": summary, "accepted": bool(summary["accepted"]),
              "failed_checks": summary["failed_checks"]}
    (args.out / "sequence_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"accepted": report["accepted"], "frames": report["sequence_frames"],
                      "boundaries": boundaries, "failed_checks": report["failed_checks"]}, indent=2))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
