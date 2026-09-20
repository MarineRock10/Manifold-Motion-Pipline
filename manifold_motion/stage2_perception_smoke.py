"""Interface smoke test for the real-perception SDF/corridor adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .perception_corridor import PerceptionGridConfig, corridor_from_perception, validate_condition


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test point-cloud to Stage-2 condition conversion")
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/perception_smoke"))
    args = parser.parse_args()
    t = 48
    route = np.column_stack([np.linspace(0.0, 1.6, t), np.zeros(t), np.full(t, 0.78)])
    envelope = np.tile(np.array([0.42, 0.28, 0.78]), (t, 1))
    # A wall outside the route validates obstacle rasterization without making the smoke test
    # depend on any private sensor or ROS bag.
    # Keep the wall beyond both the route endpoint and the robot-envelope margin.  The adapter
    # now contracts aperture axes from observed surfaces, so the former 1.9 m placement was
    # correctly diagnosed as a near-route obstacle rather than a clear-space smoke case.
    wall = np.column_stack([np.full(160, 2.4), np.linspace(-1.0, 1.0, 160), np.full(160, 0.8)])
    corridor, sdf = corridor_from_perception(route, envelope, wall, config=PerceptionGridConfig())
    report = validate_condition(corridor, sdf)
    report.update({"wall_points": int(len(wall)), "corridor_shape": list(corridor.shape), "sdf_shape": list(sdf.shape),
                   "route_clear": bool(np.all(corridor[:, 3:6] > 0.30))})
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "condition.npz", corridor=corridor, sdf=sdf, route=route, points=wall)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] and report["route_clear"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
