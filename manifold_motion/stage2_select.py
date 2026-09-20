"""Select an executable Stage-2 Flow-Matching candidate with the real SONIC gate.

Sampling a conditional flow is intentionally stochastic.  This module evaluates every
candidate in MuJoCo with the same frozen GEAR-SONIC controller and exact mesh/corridor
check used by :mod:`stage2_validate`, then selects only among candidates that pass all hard
constraints.  It is the executable counterpart of the ``J(R)`` box in the system diagram.

The selector does not repair trajectories and it does not turn a rejected candidate into a
success by ranking it.  If no candidate is acceptable it emits the least-bad diagnostic
candidate for analysis and returns status 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .seed_replay import ReplayConfig, SeedReplayRunner
from .stage2_validate import validate_trajectory


def _smoothness(trajectory: np.ndarray) -> float:
    """Mean squared joint acceleration at the 30 Hz model output rate."""
    q = np.asarray(trajectory, dtype=np.float64)[:, :29]
    acceleration = np.diff(q, n=2, axis=0) * (30.0 ** 2)
    return float(np.mean(acceleration ** 2)) if len(acceleration) else 0.0


def _objective(summary: dict[str, Any], trajectory: np.ndarray) -> dict[str, float | int]:
    """Lexicographic cost: hard feasibility first, then tracking/corridor/smoothness.

    A feasible trajectory always scores below every failed candidate.  The continuous terms
    only rank candidates in the same feasibility class and are retained in the report so the
    user can audit why one viable candidate was chosen.
    """
    failures = len(set(summary.get("failed_checks", [])))
    tracking = float(summary.get("track_err_mean_rad", 10.0))
    leg_tracking = float(summary.get("track_err_legs_rad", 10.0))
    corridor_excess = max(0.0, float(summary.get("corridor_radius_max", 0.0)) - 1.0)
    progress = float(summary.get("motion_progress_ratio", 0.0))
    smoothness = _smoothness(trajectory)
    infeasible = 0 if bool(summary.get("accepted", False)) else 1
    # Hard feasibility dominates; within a class prefer lower tracking error, more corridor
    # margin, forward progress, and smoother references.  The smoothness term is deliberately
    # small because physical execution evidence matters more than imitation aesthetics.
    score = (1_000_000.0 * infeasible + 10_000.0 * failures + 100.0 * corridor_excess +
             10.0 * tracking + 10.0 * leg_tracking - progress + 1e-4 * smoothness)
    return {"score": float(score), "infeasible": infeasible, "hard_failures": failures,
            "tracking_term": tracking, "leg_tracking_term": leg_tracking,
            "corridor_excess": corridor_excess, "progress_term": progress,
            "joint_acceleration_mse": smoothness}


def _config(args: argparse.Namespace) -> ReplayConfig:
    return ReplayConfig(warmup_seconds=args.warmup_seconds,
                        ignore_metrics_seconds=args.ignore_metrics_seconds,
                        fall_height=args.fall_height,
                        max_roll_deg=args.max_roll_deg,
                        max_track_err_rad=args.max_track_err,
                        max_leg_track_err_rad=args.max_leg_track_err,
                        min_reference_planar_path_m=args.min_reference_planar_path,
                        min_progress_ratio=args.min_progress_ratio)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate and select Stage-2 flow candidates through frozen SONIC")
    parser.add_argument("--sample", type=Path, required=True,
                        help="sample.npz with generated_ref_candidates from stage2_flow sample")
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/stage2_selection"))
    parser.add_argument("--source-hz", type=float, default=30.0)
    parser.add_argument("--warmup-seconds", type=float, default=ReplayConfig.warmup_seconds)
    parser.add_argument("--ignore-metrics-seconds", type=float, default=ReplayConfig.ignore_metrics_seconds)
    parser.add_argument("--fall-height", type=float, default=ReplayConfig.fall_height)
    parser.add_argument("--max-roll-deg", type=float, default=ReplayConfig.max_roll_deg)
    parser.add_argument("--max-track-err", type=float, default=ReplayConfig.max_track_err_rad)
    parser.add_argument("--max-leg-track-err", type=float, default=ReplayConfig.max_leg_track_err_rad)
    parser.add_argument("--min-reference-planar-path", type=float, default=ReplayConfig.min_reference_planar_path_m)
    parser.add_argument("--min-progress-ratio", type=float, default=ReplayConfig.min_progress_ratio)
    parser.add_argument("--max-corridor-radius", type=float, default=1.0)
    args = parser.parse_args()
    if args.source_hz <= 0.0:
        parser.error("source-hz must be positive")

    with np.load(args.sample) as loaded:
        if "generated_ref_candidates" in loaded.files:
            candidates = np.asarray(loaded["generated_ref_candidates"], dtype=np.float32)
        elif "generated_ref" in loaded.files:
            candidates = np.asarray(loaded["generated_ref"], dtype=np.float32)[None, ...]
        else:
            parser.error("sample has neither generated_ref_candidates nor generated_ref")
        if candidates.ndim != 3 or candidates.shape[-1] != 38 or candidates.shape[1] < 2:
            parser.error("candidate trajectories must have shape [K, T>=2, 38]")
        corridor = np.asarray(loaded["condition_corridor"], dtype=np.float32) if "condition_corridor" in loaded.files else None
        original_arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
        source_trace = {key: int(np.asarray(loaded[key])) for key in ("source_index", "clip_index", "source_origin")
                        if key in loaded.files}
        stratum = str(np.asarray(loaded["primitive_name"])) if "primitive_name" in loaded.files else "generated"

    config = _config(args)
    runner = SeedReplayRunner()
    reports: list[dict[str, Any]] = []
    executions: list[dict[str, np.ndarray]] = []
    for index, trajectory in enumerate(candidates):
        executed, summary = validate_trajectory(trajectory, source=args.sample, source_hz=args.source_hz,
                                                 config=config, corridor=corridor,
                                                 max_corridor_radius=args.max_corridor_radius,
                                                 stratum=stratum,
                                                 runner=runner)
        report: dict[str, Any] = {"candidate_index": index, **_objective(summary, trajectory), **summary}
        reports.append(report)
        executions.append(executed)
        print(json.dumps({"candidate_index": index, "accepted": report["accepted"],
                          "score": report["score"], "failed_checks": report["failed_checks"]}), flush=True)

    viable = [report for report in reports if report["accepted"]]
    selected = min(viable or reports, key=lambda report: float(report["score"]))
    selected_index = int(selected["candidate_index"])
    # Make this a true one-trajectory hand-off.  Keeping the candidate tensor here would make
    # ``stage2_validate`` (correctly) use its candidate-0 default instead of the selected
    # trajectory.  All candidate evidence remains in selection.json and the original sample.
    original_arrays.pop("generated_ref_candidates", None)
    original_arrays.pop("candidate_indices", None)
    original_arrays["generated_ref"] = candidates[selected_index]
    original_arrays["selected_candidate_index"] = np.asarray(selected_index, dtype=np.int64)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "selected_sample.npz", **original_arrays)
    np.savez_compressed(args.out / "selected_executed.npz", **executions[selected_index])
    selection = {"sample": str(args.sample), "source_hz": args.source_hz,
                 "candidates_evaluated": int(len(candidates)), "viable_candidates": int(len(viable)),
                 "selected_candidate_index": selected_index, "accepted": bool(selected["accepted"]),
                 **source_trace,
                 "selection_rule": "hard feasibility, then tracking/corridor/progress/smoothness objective",
                 "selected": selected, "candidates": reports}
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps({key: selection[key] for key in ("candidates_evaluated", "viable_candidates",
                                                       "selected_candidate_index", "accepted")}, indent=2))
    return 0 if selected["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
