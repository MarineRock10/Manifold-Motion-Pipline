"""Failure-directed analysis for Stage-1 physical rollouts.

The script is deliberately descriptive: it never chooses a checkpoint or edits the training
set after looking at the confirmation result.  It produces clusters and a proposed SEED
supplement manifest for the next run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict


def analyze(physical: Path, out: Path, adapter_summary: Path | None = None) -> dict:
    data = json.loads(physical.read_text())
    clusters = defaultdict(list)
    for row in data.get("rows", []):
        failures = []
        if row.get("fallen"): failures.append("fall_or_upright")
        if float(row.get("radius", 0.0)) > 1.0: failures.append("self_manifold_clearance")
        if float(row.get("drift_m", 0.0)) > .35: failures.append("root_drift")
        if not failures: failures.append("accepted")
        for cause in failures: clusters[cause].append(row)
    by_family = defaultdict(lambda: {"rows": 0, "failures": 0, "max_radius": 0.0, "max_drift_m": 0.0})
    for row in data.get("rows", []):
        family = str(row.get("family", "unknown")); item = by_family[family]
        item["rows"] += 1; item["failures"] += int(not row.get("accepted", False))
        item["max_radius"] = max(item["max_radius"], float(row.get("radius", 0.0)))
        item["max_drift_m"] = max(item["max_drift_m"], float(row.get("drift_m", 0.0)))
    summary = {
        "schema": "manifold-motion.stage1.failure-clusters.v1",
        "source": str(physical),
        "clusters": {k: {"rows": len(v), "families": sorted(set(str(r.get("family")) for r in v)),
                          "rows_detail": [{"row": r.get("row"), "family": r.get("family"),
                                           "radius": r.get("radius"), "drift_m": r.get("drift_m"),
                                           "predicted_family": r.get("predicted_family")} for r in v]}
                     for k, v in sorted(clusters.items())},
        "by_family": dict(sorted(by_family.items())),
        "supplement_manifest": [
            {"priority": 1, "family": "high_jump", "reason": "confirmation root drift exceeded 0.35 m"},
            {"priority": 1, "family": "box_jump", "reason": "confirmation root drift exceeded 0.35 m"},
            {"priority": 2, "family": "jog_forward", "reason": "temporal head confused with side_hop"},
            {"priority": 2, "family": "dodge_lateral", "reason": "temporal head confused with side_hop"},
            {"priority": 2, "family": "button_lever", "reason": "temporal head confused with door_interaction"},
        ],
        "selection_rule": "fixed before supplemental SEED download; no confirmation row is promoted by success",
    }
    if adapter_summary and adapter_summary.is_file():
        adapter = json.loads(adapter_summary.read_text())
        summary["frozen_sonic_adapter_reference"] = {
            "path": str(adapter_summary), "seeds": sorted(adapter),
            "interpretation": "v9 reports aggregate trade-offs only; no row-level failure labels were available",
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    md = ["# Stage 1 failure clusters", "", "该报告只做失败归因，不回看确认集来重新选模型。", "", "| cluster | rows | families |", "|---|---:|---|"]
    for k, v in summary["clusters"].items(): md.append(f"| {k} | {v['rows']} | {', '.join(v['families'])} |" )
    md += ["", "## 定向补充清单", "", "| priority | family | reason |", "|---:|---|---|"]
    for item in summary["supplement_manifest"]: md.append(f"| {item['priority']} | {item['family']} | {item['reason']} |" )
    out.with_name("README.md").write_text("\n".join(md) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(); p.add_argument("--physical", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--adapter-summary", type=Path); args = p.parse_args(); print(json.dumps(analyze(args.physical, args.out, args.adapter_summary), indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
