"""Execute a sampled Stage-2 reference through SONIC before it is considered usable.

``stage2_flow sample`` writes a 30 Hz future reference with 29 policy-order joints,
root translation in the local reference frame, and a 6-D root rotation.  This command
reconstructs a 50 Hz :class:`~manifold_motion.reference.ReferenceBuffer`, replays it in
MuJoCo through the frozen controller, and applies the same safety/tracking gate used for
BONES-SEED clips.

It is intentionally a validation command, not a clip repair command: joint-limit violations
are reported and make the sample fail.  Never clip an invalid generated reference and call
the clipped result a model success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from . import constants as C
from .corridor import execution_corridor_radius
from .seed_replay import ReplayConfig, SeedMotion, SeedReplayRunner


def _linear(values: np.ndarray, source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.stack([np.interp(target_t, source_t, values[:, i]) for i in range(values.shape[1])], axis=1)


def _rotation6_to_quat(rotation6: np.ndarray) -> np.ndarray:
    """Recover a proper rotation from the two-column 6-D representation."""
    result = np.empty((len(rotation6), 4), dtype=np.float64)
    for index, value in enumerate(np.asarray(rotation6, dtype=np.float64)):
        first = np.array([value[0], value[2], value[4]])
        second = np.array([value[1], value[3], value[5]])
        first /= max(np.linalg.norm(first), 1e-8)
        second -= first * np.dot(first, second)
        second /= max(np.linalg.norm(second), 1e-8)
        third = np.cross(first, second)
        matrix = np.column_stack([first, second, third])
        mujoco.mju_mat2Quat(result[index], matrix.reshape(-1))
    return result


def _motion_from_trajectory(trajectory: np.ndarray, source: Path, source_hz: float,
                            control_hz: float) -> SeedMotion:
    """Convert a stored 30 Hz model trajectory into SONIC's replay representation."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 38 or len(trajectory) < 2:
        raise ValueError("generated_ref must have shape [T, 38] with at least two frames")
    if not np.isfinite(trajectory).all():
        raise ValueError("generated_ref contains non-finite values")
    source_t = np.arange(len(trajectory), dtype=np.float64) / source_hz
    target_t = np.arange(0.0, source_t[-1] + 1e-9, 1.0 / control_hz)
    if target_t[-1] < source_t[-1] - 1e-7:
        target_t = np.append(target_t, source_t[-1])
    q_policy = _linear(trajectory[:, :29], source_t, target_t)
    root_pos = _linear(trajectory[:, 29:32], source_t, target_t)
    root_quat = _rotation6_to_quat(_linear(trajectory[:, 32:38], source_t, target_t))
    q_hw = q_policy[:, C.ISAACLAB_TO_MUJOCO]
    dq_hw = np.gradient(q_hw, target_t, axis=0, edge_order=1)
    return SeedMotion(source, target_t * source_hz, q_hw, dq_hw, root_pos, root_quat, control_hz)


def _generated_motion(path: Path, source_hz: float, control_hz: float,
                      candidate_index: int = 0) -> SeedMotion:
    with np.load(path) as sample:
        if "generated_ref_candidates" in sample.files:
            candidates = np.asarray(sample["generated_ref_candidates"], dtype=np.float64)
            if candidates.ndim != 3:
                raise ValueError("generated_ref_candidates must have shape [K, T, 38]")
            if not 0 <= candidate_index < len(candidates):
                raise ValueError(f"candidate index {candidate_index} is outside [0, {len(candidates) - 1}]")
            trajectory = candidates[candidate_index]
        elif candidate_index == 0 and "generated_ref" in sample.files:
            trajectory = np.asarray(sample["generated_ref"], dtype=np.float64)
        elif "generated_ref" not in sample.files:
            raise ValueError(f"{path} has no generated_ref array")
        else:
            raise ValueError("the sample contains one generated_ref only; candidate index must be 0")
    return _motion_from_trajectory(trajectory, Path(path), source_hz, control_hz)


def _joint_limit_report(runner: SeedReplayRunner, motion: SeedMotion) -> dict:
    model = runner.env.model
    joint_ids = model.actuator_trnid[runner.env.body_act, 0]
    lower = model.jnt_range[joint_ids, 0]
    upper = model.jnt_range[joint_ids, 1]
    below = motion.joint_pos_hw < lower[None, :]
    above = motion.joint_pos_hw > upper[None, :]
    violating = below | above
    return {
        "joint_limit_violations": int(violating.sum()),
        "joint_limit_frames": int(violating.any(axis=1).sum()),
        "joint_limit_names": [C.MOTOR_NAMES[index] for index in np.flatnonzero(violating.any(axis=0))],
    }


def validate_trajectory(trajectory: np.ndarray, *, source: Path, source_hz: float,
                        config: ReplayConfig, corridor: np.ndarray | None = None,
                        max_corridor_radius: float = 1.0,
                        stratum: str = "generated",
                        runner: SeedReplayRunner | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Replay one candidate and return its executed arrays and complete hard-gate report.

    This is shared by the command-line validator and multi-candidate selection, which prevents
    a candidate chooser from accidentally applying a looser safety definition than final
    validation.
    """
    runner = runner or SeedReplayRunner()
    motion = _motion_from_trajectory(trajectory, source, source_hz, 1.0 / C.CONTROL_DT)
    limits = _joint_limit_report(runner, motion)
    data, summary = runner.replay(motion, config, stratum=stratum)
    summary.update(limits)
    failed_checks = list(summary["failed_checks"])
    if corridor is not None:
        corridor_metrics = execution_corridor_radius(data["q_exec"], data["base_pos"], data["base_quat"], corridor)
        summary.update(corridor_metrics)
        summary["conditioned_corridor_shape"] = list(corridor.shape)
        if corridor_metrics["corridor_radius_max"] > max_corridor_radius:
            failed_checks.append("corridor_violation")
    if limits["joint_limit_violations"]:
        failed_checks.append("joint_limit_violation")
    summary["failed_checks"] = failed_checks
    summary["accepted"] = bool(not failed_checks)
    return data, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a generated Stage-2 R_ref through frozen SONIC")
    parser.add_argument("--sample", type=Path, required=True, help="sample.npz from manifold_motion.stage2_flow")
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/stage2_validation"))
    parser.add_argument("--source-hz", type=float, default=30.0)
    parser.add_argument("--warmup-seconds", type=float, default=ReplayConfig.warmup_seconds)
    parser.add_argument("--ignore-metrics-seconds", type=float, default=ReplayConfig.ignore_metrics_seconds)
    parser.add_argument("--fall-height", type=float, default=ReplayConfig.fall_height)
    parser.add_argument("--max-roll-deg", type=float, default=ReplayConfig.max_roll_deg)
    parser.add_argument("--max-track-err", type=float, default=ReplayConfig.max_track_err_rad)
    parser.add_argument("--max-leg-track-err", type=float, default=ReplayConfig.max_leg_track_err_rad)
    parser.add_argument("--min-reference-planar-path", type=float, default=ReplayConfig.min_reference_planar_path_m)
    parser.add_argument("--min-progress-ratio", type=float, default=ReplayConfig.min_progress_ratio)
    parser.add_argument("--max-corridor-radius", type=float, default=1.0,
                        help="maximum mesh-surface implicit radius in the conditioned corridor")
    parser.add_argument("--candidate-index", type=int, default=0,
                        help="candidate in generated_ref_candidates (0 for legacy one-sample files)")
    parser.add_argument("--trajectory-key", default="generated_ref",
                        help="NPZ trajectory key; use expected_ref only as an offline oracle diagnostic")
    args = parser.parse_args()
    if args.source_hz <= 0:
        parser.error("--source-hz must be positive")
    config = ReplayConfig(warmup_seconds=args.warmup_seconds,
                          ignore_metrics_seconds=args.ignore_metrics_seconds,
                          fall_height=args.fall_height,
                          max_roll_deg=args.max_roll_deg,
                          max_track_err_rad=args.max_track_err,
                          max_leg_track_err_rad=args.max_leg_track_err,
                          min_reference_planar_path_m=args.min_reference_planar_path,
                          min_progress_ratio=args.min_progress_ratio)
    with np.load(args.sample) as sample:
        corridor = np.asarray(sample["condition_corridor"]) if "condition_corridor" in sample.files else None
        source_trace = {key: int(np.asarray(sample[key])) for key in ("source_index", "clip_index", "source_origin")
                        if key in sample.files}
        stratum = str(np.asarray(sample["primitive_name"])) if "primitive_name" in sample.files else "generated"
        if args.trajectory_key == "generated_ref" and "generated_ref_candidates" in sample.files:
            trajectories = np.asarray(sample["generated_ref_candidates"])
            if not 0 <= args.candidate_index < len(trajectories):
                parser.error(f"candidate-index must be in [0, {len(trajectories) - 1}]")
            trajectory = trajectories[args.candidate_index]
        else:
            if args.candidate_index != 0:
                parser.error("candidate-index may only be nonzero for generated_ref_candidates")
            if args.trajectory_key not in sample.files:
                parser.error(f"sample has no trajectory key '{args.trajectory_key}'")
            trajectory = np.asarray(sample[args.trajectory_key])
    data, summary = validate_trajectory(trajectory, source=args.sample, source_hz=args.source_hz,
                                        config=config, corridor=corridor,
                                        max_corridor_radius=args.max_corridor_radius, stratum=stratum)
    summary.update({"sample": str(args.sample), "source_hz": args.source_hz,
                    "candidate_index": args.candidate_index,
                    "trajectory_key": args.trajectory_key,
                    **source_trace,
                    "record_layout": "q_ref/q_exec are SONIC policy order; generated root is local reference frame"})
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "executed.npz", **data)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
