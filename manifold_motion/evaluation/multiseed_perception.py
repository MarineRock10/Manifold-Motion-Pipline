"""Run the simulated radar/SLAM/P1 contract over a deterministic seed matrix.

This is intentionally separate from the expensive MuJoCo rollout matrix.  It stress-tests
the sensor noise/dropout, sliding 3-D probability map, 3-D A* and ellipsoid handoff for every
seed and records failures without selecting only successful seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from manifold_motion.perception.deploy import RadarConfig, build_simulated_deploy_condition


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for seed in args.seeds:
        out = args.out / f"seed_{int(seed)}"
        try:
            summary = build_simulated_deploy_condition(
                args.scene, np.asarray(args.root_pos, dtype=np.float64),
                np.asarray(args.root_quat, dtype=np.float64), np.asarray(args.goal, dtype=np.float64),
                out=out, radar_config=RadarConfig(seed=int(seed)),
                demo_pose_count=args.demo_pose_count,
                planner_body_radius_m=args.planner_body_radius_m,
                planner_clearance_m=args.planner_clearance_m,
            )
            checks = summary.get("checks", {})
            rows.append({"seed": int(seed), "accepted": bool(summary.get("accepted", False)),
                         "checks": checks, "total_returns": int(summary["radar"]["total_returns"]),
                         "occupied_voxels": int(summary["slam"]["occupied_voxels"]),
                         "route_length_m": float(summary["astar"]["route_length_m"]),
                         "max_lateral_detour_m": float(summary["astar"]["max_lateral_detour_m"]),
                         "output": str(out)})
        except Exception as error:  # preserve the seed as a failed, auditable trial
            rows.append({"seed": int(seed), "accepted": False, "failure_type": type(error).__name__,
                         "error": str(error), "output": str(out)})
    check_names = sorted({name for row in rows for name in row.get("checks", {})})
    report = {
        "schema": "manifold-motion.simulated-perception-multiseed.v1",
        "scene": str(args.scene), "seeds": [int(seed) for seed in args.seeds],
        "planned": len(args.seeds), "reported": len(rows),
        "accepted": len(rows) == len(args.seeds) and all(row["accepted"] for row in rows),
        "accepted_count": sum(bool(row["accepted"]) for row in rows),
        "check_pass_counts": {name: sum(bool(row.get("checks", {}).get(name, False)) for row in rows)
                              for name in check_names},
        "rows": rows,
        "policy": "all declared perception seeds are reported; no seed exclusion",
        "provenance": "simulated MuJoCo radar with ground-truth pose stream; not real SLAM accuracy",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("data/g1_flat/scene_long_avoidance.xml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31000, 31001, 31002, 31003, 31004, 31005, 31006, 31007])
    parser.add_argument("--root-pos", type=float, nargs=3, default=(0.0, 0.0, 0.78))
    parser.add_argument("--root-quat", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--goal", type=float, nargs=2, default=(3.60, 0.0))
    parser.add_argument("--demo-pose-count", type=int, default=5)
    parser.add_argument("--planner-body-radius-m", type=float, default=0.46)
    parser.add_argument("--planner-clearance-m", type=float, default=0.12)
    args = parser.parse_args()
    if not args.seeds or args.demo_pose_count < 2:
        parser.error("seeds must be non-empty and demo-pose-count must be at least 2")
    report = run(args)
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
