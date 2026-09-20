"""Aggregate Stage-2 hard-gate results into a compact reproducibility report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="summarize Stage-2 selector results")
    parser.add_argument("--root", type=Path, default=Path("reports/manifold_motion"))
    parser.add_argument("--pattern", default="stage2_residual_walk80_selection_test*/selection.json")
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/stage2_acceptance_report.json"))
    args = parser.parse_args()
    paths = sorted(args.root.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no selection reports matched {args.root / args.pattern}")
    rows = []
    for path in paths:
        report = json.loads(path.read_text())
        selected = report.get("selected", {})
        rows.append({"selection": str(path), "accepted": bool(report.get("accepted")),
                     "candidates_evaluated": int(report.get("candidates_evaluated", 0)),
                     "viable_candidates": int(report.get("viable_candidates", 0)),
                     "selected_candidate_index": int(report.get("selected_candidate_index", -1)),
                     "primitive": selected.get("stratum"),
                     "progress_ratio": selected.get("motion_progress_ratio"),
                     "track_err_mean_rad": selected.get("track_err_mean_rad"),
                     "corridor_radius_max": selected.get("corridor_radius_max"),
                     "failed_checks": selected.get("failed_checks", [])})
    aggregate = {"tests": len(rows), "accepted_tests": sum(row["accepted"] for row in rows),
                 "pass_rate": sum(row["accepted"] for row in rows) / len(rows),
                 "rows": rows,
                 "gate": "SONIC/MuJoCo hard gate; actor-disjoint walk80 test windows"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))
    return 0 if aggregate["accepted_tests"] == aggregate["tests"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

