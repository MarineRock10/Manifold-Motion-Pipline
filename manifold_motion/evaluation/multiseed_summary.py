"""Aggregate paired multi-seed physical rows and compute the proposed MCSA metric.

MCSA (Manifold-Conditioned Safe Adaptation) is a secondary metric for this project.  Ordinary
navigation success/collision scores do not distinguish a robot that reaches the goal with an
irrelevant fixed gait from one that changes posture for the measured corridor.  MCSA therefore
requires both safety and the expected environment-conditioned action, while retaining standard
success, clearance and planning-time columns next to it.

The metric is frozen before aggregation:

    MCSA = 0.45 * safe_success + 0.25 * condition_action +
           0.20 * clearance_score + 0.10 * realtime_score

``condition_action`` is scenario-family specific and is only a supplementary diagnostic; it is
not a replacement for success/collision or a claim of superiority over an external paper unless
the same simulator/robot/seed protocol is reproduced.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _rows(root: Path) -> list[dict[str, Any]]:
    values = []
    for path in sorted(root.glob("seed_*/results.jsonl")):
        values.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not values:
        raise ValueError(f"no results.jsonl under {root}")
    return values


def _condition_action(row: dict[str, Any]) -> float:
    """Score whether the executed route contains the geometry-required primitive family."""
    scenario = str(row.get("scenario", ""))
    result = Path(str(row.get("source_report", "")))
    if not result.is_file():
        # Rows can be copied independently of their rollout reports; a failed row is never
        # considered behaviourally aligned merely because its scenario name is informative.
        return 0.0
    try:
        report = json.loads(result.read_text())
    except (OSError, json.JSONDecodeError):
        return 0.0
    names = [str(item.get("primitive", "")) for item in report.get("decisions", [])]
    if not names:
        return 0.0
    if "low" in scenario:
        return float(any(name == "crouch" for name in names))
    if "narrow" in scenario or "side" in scenario:
        return float(any("lateral" in name for name in names))
    if scenario.startswith("dynamic-"):
        execution = report.get("execution", {})
        return float(int(execution.get("online_semantic_switch_count", 0)) > 0 or
                     any(name == "walk_turn" for name in names))
    if "open" in scenario:
        # Open-space false switches are the negative counterpart of adaptation correctness.
        return float(not any(name in {"crouch", "walk_lateral_reverse"} for name in names))
    return float(bool(row.get("success", False)))


def _mcsa(row: dict[str, Any]) -> dict[str, float]:
    success = float(bool(row.get("success", False)))
    collision = float(bool(row.get("collision", False)))
    safe_success = success * (1.0 - collision)
    clearance = float(row.get("min_clearance_m", 0.0))
    clearance_score = float(np.clip((clearance - 0.02) / 0.18, 0.0, 1.0))
    p95 = max(float(row.get("planning_p95_ms", 0.0)), 0.0)
    realtime_score = float(np.exp(-p95 / 200.0))
    action = _condition_action(row)
    score = 0.45 * safe_success + 0.25 * action + 0.20 * clearance_score + 0.10 * realtime_score
    return {"mcsa": float(score), "condition_action": action,
            "safe_success": safe_success, "clearance_score": clearance_score,
            "realtime_score": realtime_score}


def _seed_bootstrap(values: dict[int, float], *, draws: int = 4000,
                    seed: int = 20261002) -> dict[str, Any]:
    """Bootstrap a method mean over seeds, not individual correlated rows."""
    if not values:
        return {"mean": 0.0, "ci95": [0.0, 0.0], "seed_count": 0, "draws": 0}
    ordered = np.asarray([values[key] for key in sorted(values)], dtype=float)
    rng = np.random.default_rng(seed)
    if len(ordered) == 1:
        mean = float(ordered[0])
        return {"mean": mean, "ci95": [mean, mean], "seed_count": 1, "draws": 0}
    sample = rng.choice(ordered, size=(int(draws), len(ordered)), replace=True).mean(axis=1)
    return {"mean": float(ordered.mean()),
            "ci95": [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))],
            "seed_count": int(len(ordered)), "draws": int(draws)}


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    rows = _rows(args.root)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row = dict(row)
        row["mcsa_components"] = _mcsa(row)
        by_method[str(row.get("method", "unknown"))].append(row)
    methods = {}
    for method, group in sorted(by_method.items()):
        values = np.asarray([row["mcsa_components"]["mcsa"] for row in group], dtype=float)
        success = np.asarray([float(bool(row.get("success"))) for row in group])
        collision = np.asarray([float(bool(row.get("collision"))) for row in group])
        action = np.asarray([row["mcsa_components"]["condition_action"] for row in group])
        seed_values = defaultdict(list)
        for row in group:
            seed_values[int(row["seed"])].append(float(row["mcsa_components"]["mcsa"]))
        seed_means = {seed: float(np.mean(items)) for seed, items in seed_values.items()}
        methods[method] = {
            "rows": len(group), "seeds": sorted({int(row["seed"]) for row in group}),
            "success_rate": float(success.mean()), "collision_rate": float(collision.mean()),
            "condition_action_rate": float(action.mean()), "mcsa_mean": float(values.mean()),
            "mcsa_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "mcsa_seed_bootstrap": _seed_bootstrap(seed_means),
            "min_clearance_mean_m": float(np.mean([float(row.get("min_clearance_m", 0.0)) for row in group])),
            "planning_p95_mean_ms": float(np.mean([float(row.get("planning_p95_ms", 0.0)) for row in group])),
            "failure_types": dict(sorted({
                failure: sum(str(row.get("failure_type", "")) == failure for row in group)
                for failure in {str(row.get("failure_type", "")) for row in group}
                if failure
            }.items())),
        }
    baseline = methods.get(args.baseline)
    baseline_seed_means = {}
    if baseline is not None:
        baseline_seed_means = {
            int(seed): float(np.mean([row["mcsa_components"]["mcsa"] for row in group]))
            for seed, group in _group_by_seed(by_method[args.baseline]).items()
        }
    paired_delta = {}
    if baseline_seed_means:
        for method, group in sorted(by_method.items()):
            if method == args.baseline:
                continue
            method_seed_means = {
                int(seed): float(np.mean([row["mcsa_components"]["mcsa"] for row in seed_group]))
                for seed, seed_group in _group_by_seed(group).items()
            }
            common = sorted(set(baseline_seed_means) & set(method_seed_means))
            deltas = {seed: method_seed_means[seed] - baseline_seed_means[seed] for seed in common}
            paired_delta[method] = _seed_bootstrap(deltas)
    report = {
        "schema": "manifold-motion.multiseed-summary.v1",
        "input_root": str(args.root), "rows": len(rows), "methods": methods,
        "baseline": args.baseline,
        "mcsa_definition": "0.45 safe_success + 0.25 condition_action + 0.20 clearance_score + 0.10 realtime_score",
        "sota_status": {
            "claimable": False,
            "reason": "external methods are not yet evaluated under this identical G1/MuJoCo/SEED protocol",
            "internal_baseline_beaten": bool(
                baseline and any(value["mcsa_mean"] > baseline["mcsa_mean"] for key, value in methods.items()
                                 if key != args.baseline)
            ),
        },
        "paired_delta_vs_baseline_mcsa": paired_delta,
        "policy": "all available seed rows are aggregated; no post-hoc seed exclusion",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _group_by_seed(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["seed"])].append(row)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", default="B2")
    args = parser.parse_args()
    report = summarize(args)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
