"""Execute and audit the one-seed primary CVPR MuJoCo matrix.

The committed CVPR plan is intentionally much larger than a smoke run (32 primary
seeds plus robustness perturbations).  This module is the reproducible bridge between
that plan and the simulator: it materializes one nominal row for every primary
method/scenario pair, runs the scenarios that have an honest MuJoCo adapter, and
records explicit ``unsupported`` rows for dynamic cases that still require a
time-varying scene adapter.  Nothing is silently dropped or relabelled as success.

Run from the repository root::

    python -m manifold_motion.cvpr_physical_smoke \
      --out reports/cvpr/physical_primary_seed31000 --resume --skip-render

This is a physical smoke/pilot, not a final CVPR table.  The report records exact
fixture, held-out proxy-fixture, and unsupported adapter fidelity separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .cvpr_benchmark import DEFAULT_CONFIG, _read_json
from .cvpr_smoke import DYNAMIC_SCENARIOS, FIXTURE_SCENARIOS
from .stage2_manifold_adaptive import BENCHMARK_METHOD_PROFILES


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_METHODS = tuple(BENCHMARK_METHOD_PROFILES)


def _run_id(method: str, scenario: str, seed: int) -> str:
    payload = json.dumps({"method": method, "scenario": scenario, "robustness": "nominal",
                          "seed": seed}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _scenario_registry(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a complete, explicit adapter record for every configured scenario."""
    registry: dict[str, dict[str, Any]] = {}
    for scenario in config["scenario_variants"]:
        scenario_id = str(scenario["id"])
        if "width_m" in scenario:
            registry[scenario_id] = {
                "kind": "generated_mujoco_fixture", "fidelity": "exact_physics",
                "factory": "narrow_corridor", "width_m": float(scenario["width_m"]),
                "side": str(scenario["side"]), "goal_x_m": 3.6,
            }
        elif "height_m" in scenario:
            registry[scenario_id] = {
                "kind": "generated_mujoco_fixture", "fidelity": "exact_physics",
                "factory": "low_ceiling", "height_m": float(scenario["height_m"]),
                "goal_x_m": 3.6,
            }
        elif scenario_id in DYNAMIC_SCENARIOS:
            registry[scenario_id] = {
                "kind": "unsupported_dynamic_stage2_adapter",
                "fidelity": "unsupported",
                "event": DYNAMIC_SCENARIOS[scenario_id],
                "reason": "requires a time-varying MuJoCo scene and synchronized radar frame adapter",
            }
        elif scenario_id in FIXTURE_SCENARIOS:
            relative, goal_x = FIXTURE_SCENARIOS[scenario_id]
            registry[scenario_id] = {
                "kind": "tracked_mujoco_fixture",
                "fidelity": "exact_physics" if not scenario_id.startswith("heldout-")
                             else "proxy_geometry",
                "scene": relative, "goal_x_m": float(goal_x),
            }
        else:
            registry[scenario_id] = {
                "kind": "unsupported_scenario_adapter", "fidelity": "unsupported",
                "reason": "no registered fixture factory",
            }
    return registry


def build_rows(config_path: Path, seed: int | None = None) -> list[dict[str, Any]]:
    config = _read_json(config_path)
    seed = int(config["evaluation_seed_start"] if seed is None else seed)
    registry = _scenario_registry(config)
    methods = [name for name, value in config["methods"].items() if value["tier"] == "primary"]
    if tuple(methods) != PRIMARY_METHODS:
        raise ValueError(f"primary method mismatch: config={methods}, implementation={PRIMARY_METHODS}")
    rows = []
    for scenario in config["scenario_variants"]:
        scenario_id = str(scenario["id"])
        for method in methods:
            rows.append({
                "run_id": _run_id(method, scenario_id, seed), "method": method,
                "scenario": scenario_id, "scenario_family": scenario["family"],
                "scenario_split": scenario["split"], "seed": seed,
                "robustness": "nominal", "method_profile": BENCHMARK_METHOD_PROFILES[method],
                "adapter": registry[scenario_id],
            })
    return rows


def _xml_for_generated_fixture(adapter: dict[str, Any], output: Path) -> Path:
    """Create a deterministic XML fixture without modifying the tracked data directory."""
    include = (ROOT / "data/g1_flat/g1_29dof_with_hand.xml").as_posix()
    # The included G1 file declares ``meshdir=meshes``. MuJoCo resolves that compiler path
    # relative to the top-level generated XML, so expose the tracked mesh directory beside it
    # instead of copying hundreds of MB into every run.
    mesh_link = output.parent / "meshes"
    if not mesh_link.exists():
        mesh_link.symlink_to(ROOT / "data/g1_flat/meshes", target_is_directory=True)
    if adapter["factory"] == "narrow_corridor":
        width = float(adapter["width_m"])
        # The gate stays centred so the benchmark goal remains outside its inflated wall end.
        # A mirrored approach marker keeps left/right geometry distinct without making the
        # nominal start state itself infeasible for the compact side footprint.
        blocker_y = -0.72 if adapter["side"] == "left" else 0.72
        title = f"cvpr narrow {width:.3f} {adapter['side']}"
        walls = f"""
    <geom name=\"obstacle_gate_left\" type=\"box\" pos=\"2.4 {width / 2 + 0.04:.4f} 0.55\" size=\"0.80 0.04 0.55\" rgba=\"0.95 0.30 0.18 0.58\"/>
    <geom name=\"obstacle_gate_right\" type=\"box\" pos=\"2.4 {-width / 2 - 0.04:.4f} 0.55\" size=\"0.80 0.04 0.55\" rgba=\"0.95 0.30 0.18 0.58\"/>
    <geom name=\"obstacle_approach_marker\" type=\"box\" pos=\"0.85 {blocker_y:.4f} 0.45\" size=\"0.22 0.10 0.45\" rgba=\"0.98 0.67 0.15 0.55\"/>
    <geom name=\"obstacle_boundary_left\" type=\"box\" pos=\"1.8 1.85 0.55\" size=\"2.35 0.04 0.55\" rgba=\"0.25 0.62 0.92 0.22\"/>
    <geom name=\"obstacle_boundary_right\" type=\"box\" pos=\"1.8 -1.85 0.55\" size=\"2.35 0.04 0.55\" rgba=\"0.25 0.62 0.92 0.22\"/>"""
    elif adapter["factory"] == "low_ceiling":
        height = float(adapter["height_m"])
        title = f"cvpr low ceiling {height:.3f}"
        walls = f"""
    <geom name=\"obstacle_low_ceiling\" type=\"box\" pos=\"1.8 0 {height + 0.04:.4f}\" size=\"1.0 0.90 0.04\" rgba=\"0.95 0.30 0.18 0.48\"/>
    <geom name=\"obstacle_boundary_left\" type=\"box\" pos=\"1.8 1.15 0.55\" size=\"1.8 0.03 0.55\" rgba=\"0.25 0.62 0.92 0.22\"/>
    <geom name=\"obstacle_boundary_right\" type=\"box\" pos=\"1.8 -1.15 0.55\" size=\"1.8 0.03 0.55\" rgba=\"0.25 0.62 0.92 0.22\"/>"""
    else:
        raise ValueError(f"unknown generated fixture: {adapter}")
    xml = f"""<mujoco model=\"{title}\">\n  <include file=\"{include}\"/>\n  <statistic center=\"1.8 0 0.65\" extent=\"4.5\"/>\n  <visual><headlight diffuse=\"0.7 0.7 0.7\" ambient=\"0.35 0.35 0.35\" specular=\"0 0 0\"/><global azimuth=\"-120\" elevation=\"-28\"/></visual>\n  <asset><texture type=\"skybox\" builtin=\"gradient\" rgb1=\"0.30 0.48 0.68\" rgb2=\"0 0 0\" width=\"512\" height=\"3072\"/><texture type=\"2d\" name=\"groundplane\" builtin=\"checker\" mark=\"edge\" rgb1=\"0.22 0.29 0.36\" rgb2=\"0.10 0.14 0.18\" markrgb=\"0.8 0.8 0.8\" width=\"300\" height=\"300\"/><material name=\"groundplane\" texture=\"groundplane\" texuniform=\"true\" texrepeat=\"8 5\" reflectance=\"0.12\"/></asset>\n  <worldbody><light pos=\"1.8 0 3.5\" dir=\"0 0 -1\" directional=\"true\"/><geom name=\"floor\" size=\"0 0 0.05\" type=\"plane\" material=\"groundplane\"/>{walls}\n  </worldbody>\n</mujoco>\n"""
    output.write_text(xml, encoding="utf-8")
    mujoco.MjModel.from_xml_path(str(output))
    return output


def _scene_for_row(row: dict[str, Any], run_dir: Path) -> tuple[Path | None, float, str]:
    adapter = row["adapter"]
    if adapter["kind"] == "tracked_mujoco_fixture":
        scene = ROOT / adapter["scene"]
        if not scene.is_file():
            return None, float(adapter["goal_x_m"]), "scene_missing"
        return scene, float(adapter["goal_x_m"]), str(adapter["fidelity"])
    if adapter["kind"] == "generated_mujoco_fixture":
        scene = _xml_for_generated_fixture(adapter, run_dir / "generated_scene.xml")
        return scene, float(adapter["goal_x_m"]), str(adapter["fidelity"])
    return None, 0.0, str(adapter["fidelity"])


def _empty_result(row: dict[str, Any], *, failure_type: str, fidelity: str,
                  started: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "run_id": row["run_id"], "method": row["method"], "scenario": row["scenario"],
        "seed": row["seed"], "robustness": "nominal", "success": False,
        "collision": False, "min_clearance_m": 0.0, "terminal_error_m": 999.0,
        "route_deviation_p95_m": 999.0, "planning_median_ms": 0.0,
        "planning_p95_ms": 0.0, "semantic_switches": 0, "false_switches": 0,
        "failure_type": failure_type, "implementation_fidelity": fidelity,
        "evidence_type": "physical_mujoco" if fidelity != "unsupported" else "explicit_adapter_gap",
        "wall_time_s": float(time.perf_counter() - started),
    }
    if extra:
        result.update(extra)
    return result


def _report_to_result(row: dict[str, Any], report: dict[str, Any], fidelity: str,
                      started: float, run_dir: Path) -> dict[str, Any]:
    execution = report.get("execution", {})
    safety = report.get("robot_self_manifold_safety", {})
    online = report.get("online_perception", {})
    times = [float(value) for value in online.get("planning_ms", []) if math.isfinite(float(value))]
    initial = report.get("planner", {}).get("initial_planning_ms")
    if initial is not None and math.isfinite(float(initial)):
        times.insert(0, float(initial))
    accepted = bool(report.get("accepted", False))
    collision = bool(execution.get("obstacle_contact_ticks", 0) > 0)
    failed = list(report.get("failed_checks", []))
    clearance = float(safety.get("exact_surface_obstacle_clearance_min_m", 0.0))
    clearance_censored = not math.isfinite(clearance)
    if clearance_censored:
        clearance = 10.0
    return {
        "run_id": row["run_id"], "method": row["method"], "scenario": row["scenario"],
        "seed": row["seed"], "robustness": "nominal", "success": accepted,
        "collision": collision,
        "min_clearance_m": clearance,
        "terminal_error_m": float(execution.get("terminal_error_m", 999.0)),
        "route_deviation_p95_m": float(execution.get("route_deviation_p95_m", 999.0)),
        "planning_median_ms": float(np.median(times)) if times else 0.0,
        "planning_p95_ms": float(np.quantile(times, 0.95)) if times else 0.0,
        "semantic_switches": int(execution.get("online_semantic_switch_count", 0)),
        "false_switches": int(sum(not bool(item.get("committed", False))
                                   for item in execution.get("online_semantic_updates", []))),
        "failure_type": "" if accepted else (";".join(map(str, failed)) or "rollout_rejected"),
        "implementation_fidelity": fidelity, "evidence_type": "physical_mujoco",
        "wall_time_s": 0.0,
        "source_report": str(run_dir / "rollout" / "report.json"),
        "keyframes_reached": int(execution.get("keyframes_reached", 0)),
        "online_updates": int(online.get("online_updates", 0)),
        "primitive_switches": int(execution.get("primitive_switch_count", 0)),
        "benchmark_method_profile": report.get("benchmark_method_profile"),
        "clearance_censored_at_m": 10.0 if clearance_censored else None,
    }


def execute_row(row: dict[str, Any], out: Path, *, skip_render: bool) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = out / "runs" / row["run_id"]
    result_path = run_dir / "result.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter = row["adapter"]
    scene, goal_x, fidelity = _scene_for_row(row, run_dir)
    if scene is None:
        return _empty_result(row, failure_type=("scenario_adapter_not_implemented"
                                               if fidelity == "unsupported" else "scene_missing"),
                             fidelity=fidelity, started=started,
                             extra={"adapter": adapter})
    rollout = run_dir / "rollout"
    command = [sys.executable, "-m", "manifold_motion.stage2_manifold_adaptive",
               "--benchmark-method", row["method"], "--scene", str(scene),
               "--title", f"CVPR {row['method']} {row['scenario']} seed={row['seed']}",
               "--out", str(rollout), "--goal-x", f"{goal_x:.5f}",
               "--segment-length-m", "0.45", "--side-gait-mode", "diagonal",
               "--side-semi-y-m", "0.42", "--num-candidates", "3",
               "--planner-body-radius-m", "0.40", "--planner-side-body-radius-m",
               f"{max(0.20, min(0.30, float(adapter.get('width_m', 1.20)) / 2.0 - 0.16)):.3f}",
               "--planner-clearance-m",
               ("0.06" if adapter.get("factory") == "narrow_corridor" else "0.10"),
               "--self-manifold-clearance-m", "0.02",
               "--max-ticks", "2600", "--seed", str(row["seed"])]
    if skip_render:
        command.append("--skip-render")
    started_process = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    report_path = rollout / "report.json"
    if not report_path.is_file():
        return _empty_result(row, failure_type=f"stage2_process_failed:{completed.returncode}",
                             fidelity=fidelity, started=started,
                             extra={"adapter": adapter, "stdout_tail": completed.stdout[-2000:],
                                    "stderr_tail": completed.stderr[-2000:]})
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = _report_to_result(row, report, fidelity, started, run_dir)
    result["wall_time_s"] = float(time.perf_counter() - started_process)
    result["adapter"] = adapter
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _read_existing(out: Path) -> dict[str, dict[str, Any]]:
    result_dir = out / "runs"
    existing = {}
    if not result_dir.is_dir():
        return existing
    for path in result_dir.glob("*/result.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("run_id"):
            existing[str(value["run_id"])] = value
    return existing


def run(config: Path, out: Path, *, seed: int | None, resume: bool,
        max_runs: int | None, skip_render: bool, rerun_failures: bool) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    rows = build_rows(config, seed)
    existing = _read_existing(out) if resume else {}
    executed = 0
    results: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["run_id"] in existing and not (
                rerun_failures and not bool(existing[row["run_id"]].get("success", False))):
            results[row["run_id"]] = existing[row["run_id"]]
            continue
        if max_runs is not None and executed >= max_runs:
            continue
        result = execute_row(row, out, skip_render=skip_render)
        results[row["run_id"]] = result
        result_path = out / "runs" / row["run_id"] / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        executed += 1
        print(json.dumps({"run_id": row["run_id"], "method": row["method"],
                          "scenario": row["scenario"], "success": result["success"],
                          "failure_type": result["failure_type"]}, ensure_ascii=False), flush=True)
    ordered = [results[key] for key in sorted(results)]
    results_path = out / "results.jsonl"
    results_path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in ordered),
                            encoding="utf-8")
    missing = sorted(set(row["run_id"] for row in rows) - set(results))
    fidelity_counts: dict[str, int] = {}
    for value in ordered:
        fidelity = str(value.get("implementation_fidelity", "unknown"))
        fidelity_counts[fidelity] = fidelity_counts.get(fidelity, 0) + 1
    report = {
        "schema": "manifold-motion.cvpr-physical-smoke.v1", "config": str(config),
        "seed": rows[0]["seed"] if rows else None, "planned": len(rows),
        "reported": len(ordered), "missing": len(missing), "missing_run_ids": missing[:100],
        "accepted": len(rows) == len(ordered), "executed_this_call": executed,
        "successes": int(sum(bool(value.get("success")) for value in ordered)),
        "fidelity_counts": fidelity_counts,
        "policy": "one nominal seed; explicit unsupported dynamic adapters; no seed exclusion",
        "limitation": "smoke/pilot evidence only; run the frozen multi-seed plan before paper claims",
        "results": str(results_path), "resume": bool(resume),
        "rerun_failures": bool(rerun_failures),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failures", action="store_true",
                        help="with --resume, rerun rows whose previous result was not successful")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="limit new rows for a bounded pilot; resume continues later")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    if args.max_runs is not None and args.max_runs <= 0:
        parser.error("--max-runs must be positive")
    report = run(args.config, args.out, seed=args.seed, resume=args.resume,
                 max_runs=args.max_runs, skip_render=args.skip_render,
                 rerun_failures=args.rerun_failures)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
