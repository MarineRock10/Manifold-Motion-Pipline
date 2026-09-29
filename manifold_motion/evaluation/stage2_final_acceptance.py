"""Aggregate the reproducible Stage-2 acceptance artifacts.

The individual reports remain the source of truth.  This module only joins their hard-gate
results into one small manifest so a reviewer can audit the complete Stage-2 chain without
mistaking a qualitative GIF for a physical pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(root: Path, relative: str) -> tuple[Path, dict[str, Any] | None]:
    path = root / relative
    if not path.exists():
        return path, None
    return path, json.loads(path.read_text())


def _accepted(report: dict[str, Any]) -> bool:
    if "accepted" in report:
        return bool(report["accepted"])
    if "passed" in report:
        return bool(report["passed"])
    return False


def _execution_metrics(report: dict[str, Any]) -> dict[str, Any]:
    execution = report.get("execution", {})
    return {
        "keyframes_reached": execution.get("keyframes_reached"),
        "keyframes_total": execution.get("keyframes_total_excluding_start"),
        "obstacle_contact_ticks": execution.get("obstacle_contact_ticks"),
        "primitive_switch_count": execution.get("primitive_switch_count"),
        "physics_ticks": execution.get("physics_ticks"),
        "online_updates": report.get("online_composer", {}).get("updates",
                          report.get("online_perception", {}).get("updates")),
    }


def _row(root: Path, name: str, relative: str) -> dict[str, Any]:
    path, report = _load(root, relative)
    if report is None:
        return {"name": name, "report": relative, "available": False, "accepted": False}
    result = {
        "name": name,
        "report": relative,
        "available": True,
        "accepted": _accepted(report),
        "failed_checks": report.get("failed_checks", []),
    }
    result.update(_execution_metrics(report))
    if name == "incremental_dynamic_planning":
        result.update({
            "route_change_events": report.get("route_change_events"),
            "median_speedup": report.get("median_speedup"),
            "incremental_planning_ms_median_excluding_init": report.get(
                "incremental_planning_ms_median_excluding_init"),
        })
    if name == "environment_counterfactual":
        result["checks"] = report.get("checks", {})
    if name == "online_composer_gallery":
        result["scenarios"] = report.get("scenarios", [])
    return result


def build_summary(root: Path) -> dict[str, Any]:
    rows = [
        _row(root, "environment_counterfactual",
             "reports/manifold_motion/stage2_manifold_counterfactual_v1/comparison_report.json"),
        _row(root, "online_state_history_projection",
             "reports/manifold_motion/stage2_online_projection_v1/comparison_report.json"),
        _row(root, "online_composer_gallery",
             "reports/manifold_motion/online_composer_gallery_v1/comparison_report.json"),
        _row(root, "incremental_dynamic_planning",
             "reports/manifold_motion/incremental_dynamic_benchmark_final/report.json"),
        _row(root, "router_flow_long_horizon",
             "reports/manifold_motion/stage2_router_flow_long_horizon_v1/report.json"),
    ]

    autonomous_root = root / "reports/manifold_motion/autonomous_v3_motion_quality"
    autonomous = []
    if autonomous_root.exists():
        for report_path in sorted(autonomous_root.glob("*/report.json")):
            report = json.loads(report_path.read_text())
            autonomous.append({
                "name": report_path.parent.name,
                "report": str(report_path.relative_to(root)),
                "accepted": _accepted(report),
                "failed_checks": report.get("failed_checks", []),
                **_execution_metrics(report),
                "dynamic_obstacle_event": report.get("dynamic_obstacle_event"),
            })

    available_rows = [row for row in rows if row["available"]]
    all_rows = available_rows + autonomous
    return {
        "schema": "manifold-motion.stage2-completion.v1",
        "scope": "Stage 2 after Stage 1 temporal router, excluding real P1 point-cloud ingress",
        "contracts": [
            "time-varying M_e(t) and measured M_self(t)",
            "state/history-conditioned Flow candidates",
            "incremental ESDF + D* Lite route updates",
            "current-state shadow rollout and exact self-manifold gate",
            "continuous SONIC/MuJoCo execution without simulator reset",
        ],
        "overall_accepted": bool(all_rows) and all(row["accepted"] for row in all_rows),
        "required_reports_available": len(available_rows) == len(rows),
        "experiments": rows,
        "autonomous_scenarios": autonomous,
        "notes": [
            "A pass means the recorded hard gates passed; it is not a claim that every SEED family is executable.",
            "Crawl remains unsupported and jump remains partial under the frozen SONIC capability manifest.",
            "The router-Flow long-horizon route is accepted, while its wide static corridor selects nominal walk; narrow and dynamic tests provide the explicit semantic switches.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="aggregate Stage-2 hard-gate reports")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path,
                        default=Path("docs/experiments/stage2_completion_v1/acceptance.json"))
    args = parser.parse_args()
    summary = build_summary(args.root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["overall_accepted"] and summary["required_reports_available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
