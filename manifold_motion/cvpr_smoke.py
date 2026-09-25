"""Run the 104-row structural preflight for the primary CVPR simulation matrix.

This is intentionally not a paper-result executor.  It freezes one nominal seed for every
primary method/scenario pair, resolves each scenario to a concrete fixture or parameterized
geometry adapter, and validates that every method has an explicit implementation profile.  The
output uses ``smoke_pass`` rather than the benchmark result schema so it cannot be mistaken for
MuJoCo success/collision statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cvpr_benchmark import DEFAULT_CONFIG, _read_json


ROOT = Path(__file__).resolve().parents[1]


METHOD_PROFILES: dict[str, dict[str, Any]] = {
    "B2": {
        "planner": "offline_geometry_astar_proxy",
        "primitive_router": "offline_geometry_snapshot",
        "projection": False,
        "shadow_gate": False,
        "history_condition": False,
    },
    "Ours-2": {
        "planner": "incremental_esdf_dstar_lite",
        "primitive_router": "online_environment_manifold",
        "projection": True,
        "shadow_gate": False,
        "history_condition": False,
    },
    "Ours-3": {
        "planner": "incremental_esdf_dstar_lite",
        "primitive_router": "online_environment_manifold",
        "projection": True,
        "shadow_gate": True,
        "switch_hysteresis": True,
        "history_condition": False,
    },
    "Ours-4": {
        "planner": "incremental_esdf_dstar_lite",
        "primitive_router": "online_environment_manifold",
        "projection": True,
        "shadow_gate": True,
        "switch_hysteresis": True,
        "history_condition": True,
    },
}


FIXTURE_SCENARIOS = {
    "open-center": ("data/g1_flat/scene_flat.xml", 3.6),
    "open-wide": ("data/g1_flat/scene_manifold_wide_long.xml", 5.0),
    "compound-side-low-turn": ("data/g1_flat/scene_manifold_low_side_turn.xml", 5.6),
    "compound-low-side-wide": ("data/g1_flat/scene_manifold_long_combo.xml", 5.2),
    "compound-repeated-low": ("data/g1_flat/scene_manifold_long_low_cycle.xml", 5.4),
    "chicane-four-turns": ("data/g1_flat/scene_manifold_chicane.xml", 6.0),
    "low-side-turn-long": ("data/g1_flat/scene_manifold_low_side_turn.xml", 5.6),
    "gate-cycle-low-side": ("data/g1_flat/scene_manifold_gate_cycle.xml", 6.4),
    "slalom-four-turns": ("data/g1_flat/scene_manifold_slalom.xml", 6.0),
    "heldout-box-layout": ("data/g1_flat/scene_manifold_block_right.xml", 3.6),
    "heldout-corridor-length": ("data/g1_flat/scene_manifold_narrow_long.xml", 5.0),
    "heldout-aperture-composition": ("data/g1_flat/scene_manifold_gate_cycle.xml", 6.4),
}


DYNAMIC_SCENARIOS = {
    "dynamic-crossing-block": "crossing",
    "dynamic-appear-disappear": "appear_disappear",
    "dynamic-moving-wall": "moving_wall",
    "dynamic-route-reopen": "route_reopen",
}


def _scenario_adapter(scenario: dict[str, Any]) -> dict[str, Any]:
    scenario_id = str(scenario["id"])
    if "width_m" in scenario:
        return {
            "kind": "parameterized_narrow_corridor",
            "width_m": float(scenario["width_m"]),
            "side": str(scenario["side"]),
            "factory": "cvpr_narrow_width_fixture",
            "compact_footprint_required": float(scenario["width_m"]) < 0.95,
        }
    if "height_m" in scenario:
        return {
            "kind": "parameterized_low_ceiling",
            "height_m": float(scenario["height_m"]),
            "factory": "cvpr_low_height_fixture",
        }
    if scenario_id in DYNAMIC_SCENARIOS:
        benchmark = ROOT / "manifold_motion" / "incremental_dynamic_benchmark.py"
        return {
            "kind": "incremental_dynamic_event",
            "event": DYNAMIC_SCENARIOS[scenario_id],
            "adapter": "manifold_motion.incremental_dynamic_benchmark",
            "adapter_exists": benchmark.is_file(),
        }
    if scenario_id in FIXTURE_SCENARIOS:
        relative, goal_x = FIXTURE_SCENARIOS[scenario_id]
        path = ROOT / relative
        return {
            "kind": "tracked_mujoco_fixture",
            "scene": relative,
            "goal_x_m": goal_x,
            "scene_exists": path.is_file(),
        }
    raise KeyError(f"CVPR scenario has no smoke adapter: {scenario_id}")


def run_smoke(config_path: Path, output: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    primary_methods = [
        method for method, value in config["methods"].items() if value["tier"] == "primary"
    ]
    if set(primary_methods) != set(METHOD_PROFILES):
        raise ValueError(
            f"primary method/profile mismatch: config={primary_methods}, profiles={sorted(METHOD_PROFILES)}"
        )
    seed = int(config["evaluation_seed_start"])
    rows = []
    for scenario in config["scenario_variants"]:
        adapter = _scenario_adapter(scenario)
        adapter_ready = bool(adapter.get("scene_exists", True) and adapter.get("adapter_exists", True))
        for method in primary_methods:
            rows.append({
                "smoke_id": f"{method}:{scenario['id']}:{seed}",
                "method": method,
                "scenario": scenario["id"],
                "scenario_family": scenario["family"],
                "seed": seed,
                "robustness": "nominal",
                "method_profile": METHOD_PROFILES[method],
                "scenario_adapter": adapter,
                "adapter_ready": adapter_ready,
                "profile_ready": True,
                "smoke_pass": adapter_ready,
                "evidence_type": "structural_preflight_only",
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "schema": "manifold-motion.cvpr-primary-smoke.v1",
        "accepted": bool(rows) and all(row["smoke_pass"] for row in rows),
        "rows": len(rows),
        "methods": primary_methods,
        "scenarios": len(config["scenario_variants"]),
        "seed": seed,
        "output": str(output),
        "failed_smoke_ids": [row["smoke_id"] for row in rows if not row["smoke_pass"]],
        "limitation": (
            "This validates 104 method/scenario adapters only. It performs no Flow generation or "
            "MuJoCo rollout and must not be reported as task success, collision, or paper statistics."
        ),
        "next_gate": "one-seed physical MuJoCo sweep, then paired eight-seed pilot",
    }
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_smoke(args.config, args.out)
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
