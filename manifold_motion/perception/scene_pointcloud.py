"""Deterministically sample named MuJoCo obstacles as a perception integration fixture.

The live runtime accepts a real RGB-D/LiDAR cloud.  For an end-to-end test without a sensor bag,
this module samples only geometries named ``obstacle_*`` from the actual MuJoCo obstacle scene.
It is not used for training and makes no reverse use of executed trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def _box_surface(size: np.ndarray, spacing: float) -> np.ndarray:
    """Regular surface samples for a MuJoCo half-size box, in its local frame."""
    half = np.asarray(size, dtype=np.float64)
    axes = [np.arange(-value, value + spacing * 0.5, spacing) for value in half]
    points = []
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        grid = np.meshgrid(axes[other[0]], axes[other[1]], indexing="ij")
        for sign in (-1.0, 1.0):
            value = np.zeros((grid[0].size, 3), dtype=np.float64)
            value[:, axis] = sign * half[axis]
            value[:, other[0]] = grid[0].ravel()
            value[:, other[1]] = grid[1].ravel()
            points.append(value)
    return np.unique(np.round(np.concatenate(points), decimals=8), axis=0)


def obstacle_pointcloud(scene: Path, spacing_m: float = 0.06) -> tuple[np.ndarray, list[str]]:
    if spacing_m <= 0:
        raise ValueError("spacing must be positive")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pieces, names = [], []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name.startswith("obstacle_"):
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"fixture supports box obstacle only, got {name}")
        local = _box_surface(model.geom_size[geom_id, :3], spacing_m)
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        pieces.append(local @ rotation.T + data.geom_xpos[geom_id])
        names.append(name)
    if not pieces:
        raise ValueError(f"{scene} has no obstacle_* geoms")
    return np.concatenate(pieces).astype(np.float32), names


def main() -> int:
    parser = argparse.ArgumentParser(description="sample a MuJoCo obstacle fixture as a world-frame point cloud")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=0.06)
    args = parser.parse_args()
    points, names = obstacle_pointcloud(args.scene, args.spacing)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, points)
    report = {"scene": str(args.scene), "out": str(args.out), "points": int(len(points)),
              "obstacles": names, "coordinate_frame": "MuJoCo world metres", "provenance": "scene_fixture"}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
