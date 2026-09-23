"""Pre-register, audit and summarize the paired CVPR simulation benchmark.

This module deliberately separates *planning* trials from executing them.  A
frozen JSONL plan makes missing or selectively omitted seeds visible before a
paper table is generated.  Result rows use the required schema declared in
``configs/cvpr_simulation_protocol.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cvpr_simulation_protocol.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_plan(config_path: Path, output: Path) -> dict:
    config = _read_json(config_path)
    if config.get("schema") != "manifold-motion.cvpr-simulation.v2":
        raise ValueError("unsupported benchmark config schema")
    conditions = {row["id"]: row for row in config["robustness_conditions"]}
    robustness_scenarios = set(config["robustness_scenario_ids"])
    start = int(config["evaluation_seed_start"])
    records = []
    for method, method_cfg in config["methods"].items():
        count = int(config[
            "evaluation_seeds_primary"
            if method_cfg["tier"] == "primary"
            else "evaluation_seeds_diagnostic"
        ])
        for scenario in config["scenario_variants"]:
            condition_ids = ["nominal"]
            if scenario["id"] in robustness_scenarios:
                condition_ids.extend(key for key in conditions if key != "nominal")
            for condition_id in condition_ids:
                condition = conditions[condition_id]
                for seed in range(start, start + count):
                    identity = {
                        "method": method,
                        "scenario": scenario["id"],
                        "robustness": condition_id,
                        "seed": seed,
                    }
                    records.append({
                        "run_id": _hash(identity)[:16],
                        **identity,
                        "method_tier": method_cfg["tier"],
                        "scenario_family": scenario["family"],
                        "scenario_split": scenario["split"],
                        "scenario_parameters": {
                            key: value for key, value in scenario.items()
                            if key not in {"id", "family", "split"}
                        },
                        "robustness_parameters": {
                            key: value for key, value in condition.items() if key != "id"
                        },
                    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    by_tier = defaultdict(int)
    for row in records:
        by_tier[row["method_tier"]] += 1
    manifest = {
        "schema": "manifold-motion.cvpr-plan.v1",
        "config": str(config_path),
        "config_sha256": _hash(config),
        "git_commit": _commit(ROOT),
        "trial_count": len(records),
        "trial_count_by_tier": dict(sorted(by_tier.items())),
        "plan": str(output),
        "policy": "all planned run_ids must be present; no post-execution seed exclusion",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _json_rows(path: Path) -> list[dict]:
    if path.is_dir():
        rows = []
        for candidate in sorted(path.rglob("*.json")):
            value = _read_json(candidate)
            if isinstance(value, dict) and "run_id" in value:
                rows.append(value)
        return rows
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def audit(plan_path: Path, result_path: Path, config_path: Path) -> dict:
    plan = _json_rows(plan_path)
    results = _json_rows(result_path)
    required = set(_read_json(config_path)["required_result_fields"])
    planned = {row["run_id"]: row for row in plan}
    seen: dict[str, dict] = {}
    duplicates = []
    malformed = []
    unexpected = []
    for row in results:
        run_id = row.get("run_id")
        if run_id in seen:
            duplicates.append(run_id)
        seen[run_id] = row
        missing = sorted(required - set(row))
        if missing:
            malformed.append({"run_id": run_id, "missing": missing})
        if run_id not in planned:
            unexpected.append(run_id)
    missing_runs = sorted(set(planned) - set(seen))
    report = {
        "accepted": not (duplicates or malformed or unexpected or missing_runs),
        "planned": len(plan), "reported": len(results),
        "missing_count": len(missing_runs), "duplicate_count": len(duplicates),
        "malformed_count": len(malformed), "unexpected_count": len(unexpected),
        "missing_run_ids": missing_runs[:100], "duplicate_run_ids": duplicates[:100],
        "malformed": malformed[:100], "unexpected_run_ids": unexpected[:100],
    }
    return report


def _bootstrap(values: np.ndarray, resamples: int, rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return [float("nan"), float("nan")]
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(results_path: Path, config_path: Path, output: Path) -> dict:
    config = _read_json(config_path)
    rows = _json_rows(results_path)
    audit_rows = [row for row in rows if all(key in row for key in config["required_result_fields"])]
    if not audit_rows:
        raise ValueError("no schema-complete result rows")
    rng = np.random.default_rng(20260923)
    resamples = int(config["bootstrap_resamples"])
    metrics = ("success", "collision", "min_clearance_m", "terminal_error_m",
               "route_deviation_p95_m", "planning_median_ms", "planning_p95_ms")
    by_method = defaultdict(list)
    for row in audit_rows:
        by_method[row["method"]].append(row)
    method_summary = {}
    for method, group in sorted(by_method.items()):
        item = {"trials": len(group)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            item[metric] = {"mean": float(values.mean()), "ci95": _bootstrap(values, resamples, rng)}
        method_summary[method] = item
    baseline_name = config["paired_baseline"]
    key = lambda row: (row["scenario"], row.get("robustness", "nominal"), int(row["seed"]))
    baseline = {key(row): row for row in by_method.get(baseline_name, [])}
    paired = {}
    for method, group in sorted(by_method.items()):
        if method == baseline_name:
            continue
        pairs = [(baseline[key(row)], row) for row in group if key(row) in baseline]
        item = {"pairs": len(pairs), "delta_is_method_minus_B2": {}}
        for metric in metrics:
            values = np.asarray([float(other[metric]) - float(base[metric])
                                 for base, other in pairs], dtype=np.float64)
            item["delta_is_method_minus_B2"][metric] = {
                "mean": float(values.mean()) if len(values) else float("nan"),
                "ci95": _bootstrap(values, resamples, rng),
            }
        paired[method] = item
    report = {
        "schema": "manifold-motion.cvpr-summary.v1",
        "result_rows": len(audit_rows), "baseline": baseline_name,
        "methods": method_summary, "paired_effects": paired,
        "warning": "descriptive pilot output until the frozen plan audit is complete",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--out", type=Path, required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--plan", type=Path, required=True)
    audit_parser.add_argument("--results", type=Path, required=True)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--results", type=Path, required=True)
    summary_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = build_plan(args.config, args.out)
    elif args.command == "audit":
        result = audit(args.plan, args.results, args.config)
    else:
        result = summarize(args.results, args.config, args.out)
    print(json.dumps(result, indent=2))
    return 0 if result.get("accepted", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
