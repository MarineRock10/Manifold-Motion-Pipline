"""Fast contract tests for the one-seed physical CVPR executor."""

from __future__ import annotations

import tempfile
from pathlib import Path

from manifold_motion.cvpr_benchmark import DEFAULT_CONFIG
from manifold_motion.cvpr_physical_smoke import (
    _report_to_result, _scenario_registry, _xml_for_generated_fixture, build_rows,
)
from manifold_motion.cvpr_benchmark import _read_json


def test_matrix_and_registry_are_complete() -> None:
    config = _read_json(DEFAULT_CONFIG)
    registry = _scenario_registry(config)
    assert len(registry) == len(config["scenario_variants"]) == 26
    rows = build_rows(DEFAULT_CONFIG)
    assert len(rows) == 104
    assert len({row["run_id"] for row in rows}) == 104
    assert {row["method"] for row in rows} == {"B2", "Ours-2", "Ours-3", "Ours-4"}
    assert sum(row["adapter"]["fidelity"] == "unsupported" for row in rows) == 0
    dynamic = [row for row in rows if row["scenario"].startswith("dynamic-")]
    assert len(dynamic) == 16
    assert all(row["adapter"]["kind"] == "generated_dynamic_mujoco_fixture"
               for row in dynamic)


def test_generated_fixtures_load_in_mujoco() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _xml_for_generated_fixture({
            "factory": "narrow_corridor", "width_m": 0.85, "side": "right",
        }, root / "narrow.xml")
        _xml_for_generated_fixture({
            "factory": "low_ceiling", "height_m": 1.0,
        }, root / "low.xml")
        _xml_for_generated_fixture({
            "factory": "dynamic_event_fixture", "event": "crossing",
        }, root / "dynamic.xml")


def test_report_conversion_preserves_physical_metrics() -> None:
    row = {"run_id": "abc", "method": "Ours-4", "scenario": "fixture", "seed": 31000}
    report = {
        "accepted": True, "failed_checks": [],
        "planner": {"initial_planning_ms": 4.0},
        "benchmark_method_profile": {"shadow_gate": True},
        "robot_self_manifold_safety": {"exact_surface_obstacle_clearance_min_m": 0.12},
        "online_perception": {"planning_ms": [6.0, 10.0], "online_updates": 2},
        "execution": {
            "obstacle_contact_ticks": 0, "terminal_error_m": 0.18,
            "route_deviation_p95_m": 0.09, "online_semantic_switch_count": 1,
            "online_semantic_updates": [{"committed": True}, {"committed": False}],
            "keyframes_reached": 8, "primitive_switch_count": 3,
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        result = _report_to_result(row, report, "exact_physics", 0.0, Path(directory))
    assert result["success"] is True
    assert result["min_clearance_m"] == 0.12
    assert result["planning_median_ms"] == 6.0
    assert result["planning_p95_ms"] > 9.0
    assert result["semantic_switches"] == 1
    assert result["false_switches"] == 1


if __name__ == "__main__":
    test_matrix_and_registry_are_complete()
    test_generated_fixtures_load_in_mujoco()
    test_report_conversion_preserves_physical_metrics()
    print("CVPR physical smoke tests passed")
