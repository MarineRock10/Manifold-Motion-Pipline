"""Regression tests for the frozen CVPR plan/audit contract."""

from __future__ import annotations

import json
from pathlib import Path

from manifold_motion.cvpr_benchmark import audit, build_plan


def _tiny_config(source: Path, destination: Path) -> None:
    config = json.loads(source.read_text(encoding="utf-8"))
    config["methods"] = {
        "B2": config["methods"]["B2"],
        "Ours-2": config["methods"]["Ours-2"],
        "ORCS-Grail": config["methods"]["ORCS-Grail"],
    }
    config["scenario_variants"] = [config["scenario_variants"][0]]
    config["robustness_scenario_ids"] = ["open-center"]
    config["evaluation_seeds_primary"] = 2
    config["evaluation_seeds_diagnostic"] = 1
    config["robustness_conditions"] = [config["robustness_conditions"][0]]
    destination.write_text(json.dumps(config), encoding="utf-8")


def test_plan_is_deterministic_and_audit_rejects_missing_rows(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "protocol.json"
    _tiny_config(root / "configs" / "cvpr_simulation_protocol.json", config_path)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    manifest1 = build_plan(config_path, first)
    manifest2 = build_plan(config_path, second)
    assert first.read_bytes() == second.read_bytes()
    assert manifest1["trial_count"] == 5
    assert manifest1["trial_count_by_tier"] == {"diagnostic": 1, "primary": 4}
    assert manifest1["config_sha256"] == manifest2["config_sha256"]

    rows = [json.loads(line) for line in first.read_text().splitlines()]
    required = json.loads(config_path.read_text())["required_result_fields"]
    result = {
        "run_id": rows[0]["run_id"],
        "method": rows[0]["method"],
        "scenario": rows[0]["scenario"],
        "seed": rows[0]["seed"],
    }
    result.update({
        "success": True,
        "collision": False,
        "min_clearance_m": 0.2,
        "terminal_error_m": 0.05,
        "route_deviation_p95_m": 0.04,
        "planning_median_ms": 20.0,
        "planning_p95_ms": 30.0,
        "semantic_switches": 1,
        "false_switches": 0,
        "failure_type": "none",
    })
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(result) + "\n", encoding="utf-8")
    report = audit(first, results, config_path)
    assert not report["accepted"]
    assert report["missing_count"] == 4


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_plan_is_deterministic_and_audit_rejects_missing_rows(Path(directory))
    print("CVPR benchmark tests passed")
