"""Summarize and visualize controlled Stage-2 environment-manifold experiments.

The rollouts must be produced by :mod:`stage2_manifold_adaptive` with the same
route, thresholds and Flow candidate count.  Relative to the wide control, the low
fixture adds a ceiling and the narrow fixture adds two side walls.  This processor
rejects the result unless all physical rollouts pass and both obstacles cause the
expected primitive change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _case(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = json.loads((path / "report.json").read_text())
    with np.load(path / "executed.npz") as archive:
        executed = {key: np.asarray(archive[key]) for key in archive.files}
    return report, executed


def _primitives(report: dict[str, Any]) -> list[str]:
    return [str(item["primitive"]) for item in report["decisions"]]


def _semis(report: dict[str, Any], axis: str) -> list[float]:
    return [float(item[f"{axis}_free_semi_min_m"]) for item in report["decisions"]]


def _route_geometry(report: dict[str, Any]) -> list[tuple[list[float], list[float]]]:
    return [(item["start_xy_m"], item["end_xy_m"]) for item in report["decisions"]]


def _candidate_counts(report: dict[str, Any]) -> list[int]:
    return [len(item["candidates"]) for item in report["candidate_evidence"]]


def _compact_panels(inputs: list[Path], out: Path, panel_width: int,
                    frame_duration_ms: int) -> int:
    images = [Image.open(path) for path in inputs]
    frames = []
    count = max(image.n_frames for image in images)
    source_width, source_height = images[0].size
    panel_height = round(source_height * panel_width / source_width)
    for index in range(count):
        canvas = Image.new("RGB", (len(images) * panel_width, panel_height), (10, 16, 24))
        for panel, image in enumerate(images):
            image.seek(round(index * (image.n_frames - 1) / max(1, count - 1)))
            resized = image.convert("RGB").resize((panel_width, panel_height), Image.BILINEAR)
            canvas.paste(resized, (panel * panel_width, 0))
            if panel:
                ImageDraw.Draw(canvas).line((panel * panel_width - 1, 0,
                                             panel * panel_width - 1, panel_height - 1),
                                            fill=(255, 255, 255), width=1)
        frames.append(canvas.quantize(colors=64, method=Image.MEDIANCUT))
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=frame_duration_ms, loop=0, optimize=True, disposal=2)
    return count


def build(args: argparse.Namespace) -> dict[str, Any]:
    wide_report, wide_data = _case(args.root / "wide")
    low_report, low_data = _case(args.root / "low")
    narrow_report, narrow_data = _case(args.root / "narrow")
    center_report, center_data = _case(args.root / "center")
    wide_route, low_route = _primitives(wide_report), _primitives(low_report)
    narrow_route = _primitives(narrow_report)
    wide_vertical, low_vertical = _semis(wide_report, "vertical"), _semis(low_report, "vertical")
    wide_lateral, narrow_lateral = _semis(wide_report, "lateral"), _semis(narrow_report, "lateral")
    checks = {
        "wide_physics_accepted": bool(wide_report["accepted"]),
        "low_physics_accepted": bool(low_report["accepted"]),
        "narrow_physics_accepted": bool(narrow_report["accepted"]),
        "wide_is_all_nominal": bool(wide_route and set(wide_route) == {"walk_nominal"}),
        "low_begins_nominal": bool(low_route and low_route[0] == "walk_nominal"),
        "low_switches_to_crouch": "crouch" in low_route[1:],
        "narrow_begins_nominal": bool(narrow_route and narrow_route[0] == "walk_nominal"),
        "narrow_switches_to_side_gait": "walk_lateral_reverse" in narrow_route[1:],
        "narrow_side_on_candidate": bool(narrow_report.get("side_gait_mode") == "side_on"
                                          and narrow_report["execution"].get("side_on_ticks", 0) > 0),
        "narrow_side_on_body_yaw": (narrow_report["execution"].get("side_on_body_yaw_error_p95_deg", 999.0) < 35.0),
        "same_segment_count": len(wide_route) == len(low_route) == len(narrow_route),
        "same_route_geometry": (_route_geometry(wide_report) == _route_geometry(low_report)
                                == _route_geometry(narrow_report)),
        "same_routing_thresholds": (wide_report["decisions"][0]["thresholds"]
                                    == low_report["decisions"][0]["thresholds"]
                                    == narrow_report["decisions"][0]["thresholds"]),
        "same_planner_config": (wide_report.get("planner") == low_report.get("planner")
                                == narrow_report.get("planner")),
        "three_flow_candidates_per_set": all(
            count == 3 for report in (wide_report, low_report, narrow_report)
            for count in _candidate_counts(report)
        ),
        "wide_zero_obstacle_contact": wide_report["execution"]["obstacle_contact_ticks"] == 0,
        "low_zero_obstacle_contact": low_report["execution"]["obstacle_contact_ticks"] == 0,
        "narrow_zero_obstacle_contact": narrow_report["execution"]["obstacle_contact_ticks"] == 0,
        "environment_changes_measured_vertical_aperture": any(
            low < wide - 1e-4 for low, wide in zip(low_vertical, wide_vertical)
        ),
        "environment_changes_measured_lateral_aperture": any(
            narrow < wide - 1e-4 for narrow, wide in zip(narrow_lateral, wide_lateral)
        ),
        "center_block_physics_accepted": bool(center_report["accepted"]),
        "center_block_reaches_all_keyframes": (
            center_report["execution"]["keyframes_reached"]
            == center_report["execution"]["keyframes_total_excluding_start"]
        ),
        "center_block_zero_obstacle_contact": center_report["execution"]["obstacle_contact_ticks"] == 0,
        "route_curvature_activates_turn": (
            any(item["requires_turn"] for item in center_report["decisions"])
            and center_report["execution"]["primitive_ticks"]["walk_turn"] > 0
        ),
    }
    passed = all(checks.values())
    frames = _compact_panels(
        [args.root / name / "manifold_adaptive.gif" for name in ("wide", "low", "narrow")],
        args.root / "manifold_driven_actions.gif",
        args.panel_width,
        args.frame_duration_ms,
    )
    summary = {
        "experiment": "controlled counterfactuals: environment M_e causes primitive changes",
        "passed": passed,
        "contract": {
            "routing": "deterministic from measured M_e aperture and route curvature; never segment index",
            "aperture_counterfactual_controls": "same route, code, thresholds, candidate count and seed",
            "changed_variables": {
                "low": "adds one physical overhead box",
                "narrow": "adds two physical side-wall boxes",
                "center": "adds a center block, causing A* route curvature",
            },
        },
        "checks": checks,
        "wide": {
            "route": wide_route,
            "vertical_free_semi_m": wide_vertical,
            "lateral_free_semi_m": wide_lateral,
            "pelvis_z_mean_m": float(wide_data["base_pos"][:, 2].mean()),
            "pelvis_z_min_m": float(wide_data["base_pos"][:, 2].min()),
            "keyframes": wide_report["execution"]["keyframes_reached"],
            "roll_abs_max_deg": wide_report["execution"]["roll_abs_max_deg"],
        },
        "low": {
            "route": low_route,
            "vertical_free_semi_m": low_vertical,
            "pelvis_z_mean_m": float(low_data["base_pos"][:, 2].mean()),
            "pelvis_z_min_m": float(low_data["base_pos"][:, 2].min()),
            "keyframes": low_report["execution"]["keyframes_reached"],
            "roll_abs_max_deg": low_report["execution"]["roll_abs_max_deg"],
        },
        "narrow": {
            "route": narrow_route,
            "lateral_free_semi_m": narrow_lateral,
            "pelvis_z_mean_m": float(narrow_data["base_pos"][:, 2].mean()),
            "keyframes": narrow_report["execution"]["keyframes_reached"],
            "roll_abs_max_deg": narrow_report["execution"]["roll_abs_max_deg"],
        },
        "center": {
            "route": _primitives(center_report),
            "turn_segments": [item["segment_index"] for item in center_report["decisions"]
                              if item["requires_turn"]],
            "primitive_ticks": center_report["execution"]["primitive_ticks"],
            "keyframes": center_report["execution"]["keyframes_reached"],
            "obstacle_contact_ticks": center_report["execution"]["obstacle_contact_ticks"],
            "route_deviation_p95_m": center_report["execution"]["route_deviation_p95_m"],
            "track_err_mean_rad": center_report["execution"]["track_err_mean_rad"],
            "roll_abs_max_deg": center_report["execution"]["roll_abs_max_deg"],
            "final_xy_m": center_data["base_pos"][-1, :2].astype(float).tolist(),
        },
        "effect": {
            "pelvis_mean_drop_m": float(
                wide_data["base_pos"][:, 2].mean() - low_data["base_pos"][:, 2].mean()
            ),
            "comparison_gif_frames": frames,
        },
    }
    (args.root / "comparison_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and pair manifold-adaptive rollouts")
    parser.add_argument("--root", type=Path, required=True,
                        help="directory containing wide/ and low/ rollout outputs")
    parser.add_argument("--panel-width", type=int, default=320)
    parser.add_argument("--frame-duration-ms", type=int, default=80)
    args = parser.parse_args()
    if min(args.panel_width, args.frame_duration_ms) <= 0:
        parser.error("render parameters must be positive")
    return 0 if build(args)["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
