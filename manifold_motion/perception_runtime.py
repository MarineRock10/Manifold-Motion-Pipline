"""Online point-cloud -> Stage-2 condition handoff with explicit coordinate provenance.

The training corpus's reverse constructor is intentionally not used here.  This runtime accepts
an obstacle point cloud in the world frame (RGB-D/LiDAR fusion may write ``.npy [N,3]``), a robot
root pose, and a navigation route.  It transforms both into the root-local coordinate contract
used by the dynamic model and emits the exact ``condition.npz`` consumed by
``stage2_flow sample --condition-npz``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C
from .perception_corridor import (PerceptionGridConfig, corridor_from_perception,
                                  pointcloud_sdf, validate_condition)


def world_to_root(points_world: np.ndarray, root_pos_world: np.ndarray, root_quat_wxyz: np.ndarray) -> np.ndarray:
    """Transform world points to root-local metres, x-forward/y-left/z-up."""
    points = np.asarray(points_world, dtype=np.float64)
    root_pos = np.asarray(root_pos_world, dtype=np.float64)
    quat = np.asarray(root_quat_wxyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or root_pos.shape != (3,) or quat.shape != (4,):
        raise ValueError("points must be [N,3], root position [3], and root quaternion [4]")
    if not (np.isfinite(points).all() and np.isfinite(root_pos).all() and np.isfinite(quat).all()):
        raise ValueError("perception inputs must be finite")
    quat = quat / max(float(np.linalg.norm(quat)), 1e-8)
    return (points - root_pos[None, :]) @ C.quat_to_matrix(quat)


def route_world_to_root(route_world: np.ndarray, root_pos_world: np.ndarray,
                        root_quat_wxyz: np.ndarray) -> np.ndarray:
    return world_to_root(np.asarray(route_world, dtype=np.float64), root_pos_world, root_quat_wxyz)


def build_condition(points_world: np.ndarray, route_world: np.ndarray, root_pos_world: np.ndarray,
                    root_quat_wxyz: np.ndarray, envelope_semi_m: np.ndarray,
                    *, route_yaw_local_rad: np.ndarray | None = None,
                    config: PerceptionGridConfig = PerceptionGridConfig(), clearance_m: float = 0.08) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    points_local = world_to_root(points_world, root_pos_world, root_quat_wxyz)
    route_local = route_world_to_root(route_world, root_pos_world, root_quat_wxyz)
    corridor, sdf = corridor_from_perception(route_local, envelope_semi_m, points_local,
                                             route_yaw_local_rad=route_yaw_local_rad,
                                             config=config, clearance_m=clearance_m)
    obstacle_sdf = pointcloud_sdf(points_local, config)
    report = validate_condition(corridor, sdf, config)
    report.update({"points_world": int(len(points_world)), "coordinate_frame": "root_local_m",
                   "provenance": "online_world_pointcloud", "route_frames": int(len(route_world)),
                   "condition_sdf_convention": "positive_inside_safe_corridor_union",
                   "obstacle_sdf_convention": "positive_obstacle_clearance",
                   "obstacle_sdf_min_m": float(obstacle_sdf.min()),
                   "obstacle_sdf_max_m": float(obstacle_sdf.max()),
                   "corridor_semi_min_m": [float(value) for value in corridor[:, 3:6].min(axis=0)],
                   "corridor_semi_mean_m": [float(value) for value in corridor[:, 3:6].mean(axis=0)]})
    return corridor, sdf, obstacle_sdf, report


def main() -> int:
    parser = argparse.ArgumentParser(description="create a Stage-2 condition from online world-frame point-cloud data")
    parser.add_argument("--points-world", type=Path, required=True, help=".npy finite [N,3] obstacle cloud in metres")
    parser.add_argument("--route-world", type=Path, required=True, help=".npy [T,3] global planner route in metres")
    parser.add_argument("--root-pos", type=float, nargs=3, required=True)
    parser.add_argument("--root-quat", type=float, nargs=4, required=True, metavar=("W", "X", "Y", "Z"))
    envelope_group = parser.add_mutually_exclusive_group(required=True)
    envelope_group.add_argument("--envelope-semi", type=float, nargs=3,
                                help="constant G1 body-envelope half-extents in root metres")
    envelope_group.add_argument("--envelope-semi-npy", type=Path,
                                help="planner-predicted root-local body envelopes, finite [T,3] metres")
    parser.add_argument("--route-yaw-local-npy", type=Path, default=None,
                        help="optional planner-predicted root-local headings, finite [T] radians; geometric route headings are used otherwise")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clearance", type=float, default=0.08)
    args = parser.parse_args()
    points, route = np.load(args.points_world), np.load(args.route_world)
    if route.ndim != 2 or route.shape[1] != 3 or len(route) < 2:
        parser.error("route-world must have shape [T>=2,3]")
    if args.envelope_semi_npy is None:
        envelope = np.tile(np.asarray(args.envelope_semi, dtype=np.float64), (len(route), 1))
    else:
        envelope = np.asarray(np.load(args.envelope_semi_npy), dtype=np.float64)
        if envelope.shape != (len(route), 3) or not np.isfinite(envelope).all() or np.any(envelope <= 0):
            parser.error("envelope-semi-npy must be a finite [len(route-world),3] array with positive values")
    route_yaw = None if args.route_yaw_local_npy is None else np.asarray(np.load(args.route_yaw_local_npy), dtype=np.float64)
    if route_yaw is not None and (route_yaw.shape != (len(route),) or not np.isfinite(route_yaw).all()):
        parser.error("route-yaw-local-npy must be a finite [len(route-world)] array")
    corridor, sdf, obstacle_sdf, report = build_condition(
        points, route, np.asarray(args.root_pos), np.asarray(args.root_quat), envelope,
        route_yaw_local_rad=route_yaw, clearance_m=args.clearance)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "condition.npz", corridor=corridor, sdf=sdf,
                        obstacle_sdf=obstacle_sdf,
                        points_world=points, route_world=route, root_pos_world=np.asarray(args.root_pos),
                        root_quat_wxyz=np.asarray(args.root_quat),
                        route_yaw_local_rad=(route_yaw if route_yaw is not None else np.array([], dtype=np.float32)))
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
