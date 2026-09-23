"""Deterministic moving-obstacle and map-change benchmark for the deploy planner."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .deploy_perception import ProbabilisticSlidingVoxelGrid, SlidingGridConfig, voxel_astar
from .incremental_planner import DStarLitePlanner, IncrementalPlannerConfig


def _set_box(grid: ProbabilisticSlidingVoxelGrid, center_xy: np.ndarray,
             half_xy: np.ndarray, occupied: bool) -> None:
    resolution = grid.config.resolution_m
    lo = np.floor(np.r_[center_xy - half_xy, 0.05] / resolution).astype(int)
    hi = np.ceil(np.r_[center_xy + half_xy, 1.45] / resolution).astype(int)
    lo_local = lo - grid.origin_cell_xyz
    hi_local = hi - grid.origin_cell_xyz
    x0, y0, z0 = np.maximum(lo_local, 0)
    x1, y1, z1 = np.minimum(hi_local + 1,
                            np.array([grid.shape_zyx[2], grid.shape_zyx[1], grid.shape_zyx[0]]))
    if x1 > x0 and y1 > y0 and z1 > z0:
        grid.log_odds[z0:z1, y0:y1, x0:x1] = 8.0 if occupied else -8.0


def _render_frame(grid: ProbabilisticSlidingVoxelGrid, route: np.ndarray,
                  old_route: np.ndarray, obstacle: np.ndarray | None,
                  label: str, scale: int = 5) -> Image.Image:
    occupied = grid.occupancy_xy_projection(0.05, 1.45)
    ny, nx = occupied.shape
    image = Image.new("RGB", (nx * scale, ny * scale + 34), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    for y, x in np.argwhere(occupied):
        draw.rectangle((x * scale, (ny - 1 - y) * scale,
                        (x + 1) * scale, (ny - y) * scale), fill=(42, 50, 64))

    def pixel(point: np.ndarray) -> tuple[int, int]:
        cell = np.floor(point[:2] / grid.config.resolution_m).astype(int) - grid.origin_cell_xyz[:2]
        return int((cell[0] + 0.5) * scale), int((ny - cell[1] - 0.5) * scale)

    for value, color, width in ((old_route, (170, 170, 180), 2), (route, (20, 132, 230), 4)):
        if len(value) > 1:
            draw.line([pixel(point) for point in value], fill=color, width=width)
    if obstacle is not None:
        px = pixel(np.r_[obstacle, 0.78])
        draw.ellipse((px[0] - 6, px[1] - 6, px[0] + 6, px[1] + 6),
                     fill=(235, 70, 55), outline=(120, 20, 15))
    draw.text((8, ny * scale + 9), label, fill=(10, 10, 10))
    return image


def run(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    config = SlidingGridConfig(resolution_m=0.08, size_xy=(96, 96), size_xyz=(96, 96, 24))
    grid = ProbabilisticSlidingVoxelGrid(config, np.array([1.8, 0.0, 0.78]))
    grid.log_odds[:] = -8.0
    # Static corridor boundaries make a crossing object cause a real route change.
    _set_box(grid, np.array([1.8, 1.45]), np.array([2.4, 0.06]), True)
    _set_box(grid, np.array([1.8, -1.45]), np.array([2.4, 0.06]), True)
    planner = DStarLitePlanner(config.resolution_m, IncrementalPlannerConfig(
        body_radius_m=0.20, clearance_m=0.08, body_half_height_m=0.65,
        esdf_truncation_m=0.90, route_preference_weight=0.15,
    ))
    start = np.array([0.0, 0.0, 0.78])
    goal = np.array([3.6, 0.0, 0.78])
    centres: list[np.ndarray | None] = [
        None, np.array([1.8, -0.95]), np.array([1.8, -0.60]),
        np.array([1.8, -0.25]), np.array([1.8, 0.10]),
        np.array([1.8, 0.45]), np.array([1.8, 0.80]), None,
    ]
    half = np.array([0.20, 0.24])
    previous_obstacle: np.ndarray | None = None
    previous_route = np.linspace(start, goal, 2)
    frames, rows = [], []
    changed_routes = 0
    for step, center in enumerate(centres):
        if previous_obstacle is not None:
            _set_box(grid, previous_obstacle, half, False)
        if center is not None:
            _set_box(grid, center, half, True)
        grid.update_count += 1
        route, report = planner.plan(grid, start, goal, preferred_world_xy=previous_route[:, :2])
        baseline_started = time.perf_counter()
        baseline = voxel_astar(
            grid, start, goal, body_radius_m=0.20, body_half_height_m=0.65,
            clearance_m=0.08, allow_unknown=True, preferred_world_xy=previous_route[:, :2],
            preference_weight=0.15,
        )
        baseline_ms = (time.perf_counter() - baseline_started) * 1000.0
        if center is not None:
            clearance = float(np.min(np.linalg.norm(
                np.maximum(np.abs(route[:, :2] - center[None, :]) - half[None, :], 0.0),
                axis=1)))
            # The ESDF inflation applies around the box surface, so route centres must not
            # enter the raw box. Exact robot-envelope clearance is covered by Stage-2 gates.
            if np.any(np.all(np.abs(route[:, :2] - center[None, :]) <= half[None, :], axis=1)):
                raise RuntimeError(f"incremental route intersects moving obstacle at step {step}")
        else:
            clearance = None
        route_change = float(np.max(np.abs(np.interp(
            np.linspace(0, 1, 50), np.linspace(0, 1, len(route)), route[:, 1])
            - np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, len(previous_route)),
                        previous_route[:, 1]))))
        changed_routes += int(route_change > 0.04)
        row = {
            "step": step, "event": ("obstacle_absent" if center is None else "obstacle_moved"),
            "obstacle_center_xy_m": (None if center is None else center.astype(float).tolist()),
            "incremental_ms": report["planning_ms"], "full_voxel_astar_ms": baseline_ms,
            "speedup": baseline_ms / max(report["planning_ms"], 1.0e-9),
            "changed_cost_cells": report["changed_cost_cells"],
            "expanded_vertices": report["expanded_vertices"],
            "route_length_m": report["route_length_m"],
            "route_change_max_y_m": route_change,
            "raw_box_clearance_proxy_m": (None if clearance is None else float(clearance)),
            "baseline_route_points": int(len(baseline)),
        }
        rows.append(row)
        frames.append(_render_frame(
            grid, route, previous_route, center,
            f"step {step}: {row['event']} | D* {row['incremental_ms']:.1f} ms | "
            f"changed={row['changed_cost_cells']}",
        ))
        previous_route = route.copy()
        previous_obstacle = None if center is None else center.copy()
    frames[0].save(out / "moving_obstacle_replanning.gif", save_all=True,
                   append_images=frames[1:], duration=450, loop=0)
    incremental = np.asarray([row["incremental_ms"] for row in rows[1:]], dtype=np.float64)
    baseline = np.asarray([row["full_voxel_astar_ms"] for row in rows[1:]], dtype=np.float64)
    report = {
        "accepted": bool(changed_routes >= 3 and np.all(np.isfinite(incremental))),
        "scenario": "appearing + crossing + disappearing obstacle",
        "steps": rows, "route_change_events": int(changed_routes),
        "incremental_planning_ms_median_excluding_init": float(np.median(incremental)),
        "full_voxel_astar_ms_median": float(np.median(baseline)),
        "median_speedup": float(np.median(baseline / np.maximum(incremental, 1.0e-9))),
        "artifact": str(out / "moving_obstacle_replanning.gif"),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.out)
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
