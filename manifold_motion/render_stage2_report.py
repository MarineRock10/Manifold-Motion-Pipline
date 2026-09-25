"""Render a saved Stage-2 rollout without rerunning Flow, SONIC, or MuJoCo control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .stage2_manifold_adaptive import _ground_obstacles
from .stage2_long_horizon_avoidance import PlannerConfig, render


def render_report(run_dir: Path, output: Path, fps: float = 20.0,
                  scale: float = 1.0) -> None:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "executed.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(run_dir / "segment_conditions.npz", allow_pickle=False) as archive:
        keyframes = archive["keyframes"]
        route = archive["route"]
        corridor = archive["environment_corridor"]
    scene = Path(report["scene"])
    planner_row = report["planner"]
    planner = PlannerConfig(
        body_radius_m=float(planner_row["body_radius_m"]),
        clearance_m=float(planner_row["clearance_m"]),
        resolution_m=float(planner_row["resolution_m"]),
    )
    execution = dict(report["execution"])
    execution["render_title"] = str(report.get("scenario", "SAVED STAGE-2 ROLLOUT"))
    output.parent.mkdir(parents=True, exist_ok=True)
    render(
        scene, data, execution, keyframes, route, corridor,
        _ground_obstacles(scene), planner, output, fps,
        self_manifold=data.get("robot_manifold"),
        safe_manifold=data.get("robot_manifold_safe"),
        dynamic_obstacle_event=report.get("dynamic_obstacle_event"),
        output_scale=scale,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.fps <= 0 or not 0.0 < args.scale <= 1.0:
        parser.error("--fps must be positive and --scale must be in (0,1]")
    render_report(args.run_dir, args.out, args.fps, args.scale)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
