"""Perception-side adapter for Stage-2 local SDF and safe-corridor conditions.

The training corpus currently stores reverse-synthesized corridors.  This module defines the
drop-in runtime representation for a real RGB-D/LiDAR producer: a local point cloud in the
current root frame is rasterized into a signed free-space grid and a route-aligned sequence of
ellipsoids.  It intentionally has no MuJoCo dependency, so a navigation stack can call it at
50 Hz and pass the returned arrays to the dynamic model.

Coordinates are root-local metres (x forward, y left, z up).  The raw obstacle-distance field
is positive in free space and negative inside inflated obstacle voxels.  It is retained as a
diagnostic, but is deliberately *not* fed into Stage-2: the training corpus calls ``sdf`` the
signed field of the **safe corridor union** (positive inside the traversable tube, negative
outside).  The returned ``sdf`` therefore follows that learned contract exactly.  The adapter
is conservative: a corridor element whose centreline is occupied is collapsed to a small
positive semi-axis rather than silently declared safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerceptionGridConfig:
    shape: tuple[int, int, int] = (10, 10, 8)
    lower_m: tuple[float, float, float] = (-0.50, -1.20, -0.15)
    upper_m: tuple[float, float, float] = (2.50, 1.20, 1.65)
    obstacle_inflation_m: float = 0.10
    max_sdf_m: float = 1.0
    min_corridor_semi_m: float = 0.06

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.shape) != 3 or min(self.shape) < 2:
            raise ValueError("SDF shape must contain three dimensions >= 2")
        lower, upper = np.asarray(self.lower_m, dtype=np.float64), np.asarray(self.upper_m, dtype=np.float64)
        if np.any(upper <= lower):
            raise ValueError("SDF upper bounds must be greater than lower bounds")
        if self.obstacle_inflation_m < 0 or self.max_sdf_m <= 0 or self.min_corridor_semi_m <= 0:
            raise ValueError("perception margins must be positive")
        return tuple(np.linspace(lower[i], upper[i], int(self.shape[i])) for i in range(3))


def _grid_points(config: PerceptionGridConfig) -> np.ndarray:
    axes = config.axes()
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def pointcloud_sdf(points_local_m: np.ndarray, config: PerceptionGridConfig = PerceptionGridConfig()) -> np.ndarray:
    """Rasterize a local obstacle cloud into the fixed positive-free SDF contract."""
    points = np.asarray(points_local_m, dtype=np.float64)
    if points.size == 0:
        return np.full(config.shape, config.max_sdf_m, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_local_m must be a finite [N,3] array")
    grid = _grid_points(config)
    # Nearest-obstacle distance is sufficient for the low-resolution policy condition.  The
    # sign changes inside the configured inflation radius; distance is clipped for robust model
    # normalization and kept in metres.
    nearest = np.min(np.linalg.norm(grid[:, None, :] - points[None, :, :], axis=2), axis=1)
    signed = nearest - float(config.obstacle_inflation_m)
    return np.clip(signed, -config.max_sdf_m, config.max_sdf_m).reshape(config.shape).astype(np.float32)


def _yaw_from_route(route_centres: np.ndarray) -> np.ndarray:
    route = np.asarray(route_centres, dtype=np.float64)
    if len(route) == 1:
        return np.zeros(1, dtype=np.float64)
    delta = np.gradient(route[:, :2], axis=0)
    return np.arctan2(delta[:, 1], delta[:, 0])


def corridor_condition_sdf(corridor: np.ndarray,
                           config: PerceptionGridConfig = PerceptionGridConfig()) -> np.ndarray:
    """Rasterise the safe-corridor union using the Stage-2 training SDF convention.

    This mirrors :func:`corridor.corridor_sdf` without importing MuJoCo.  A point is positive
    if it is inside any admissible local corridor ellipse and negative otherwise.  Keeping this
    representation distinct from obstacle clearance prevents a distribution shift where a wide
    but obstacle-free room would misleadingly look unlike every reverse-constructed training
    window.
    """
    corridor = np.asarray(corridor, dtype=np.float64)
    if corridor.ndim != 2 or corridor.shape[1] != 7 or len(corridor) == 0:
        raise ValueError("corridor must be a non-empty [T,7] array")
    if not np.isfinite(corridor).all() or np.any(corridor[:, 3:6] <= 0):
        raise ValueError("corridor must have finite centres/yaws and positive semis")
    points = _grid_points(config)
    centres, semis, yaw = corridor[:, :3], corridor[:, 3:6], corridor[:, 6]
    delta = points[:, None, :] - centres[None, :, :]
    cos, sin = np.cos(yaw)[None, :], np.sin(yaw)[None, :]
    aligned = np.stack([delta[:, :, 0] * cos + delta[:, :, 1] * sin,
                        -delta[:, :, 0] * sin + delta[:, :, 1] * cos,
                        delta[:, :, 2]], axis=2)
    normalized_radius = np.linalg.norm(aligned / semis[None, :, :], axis=2)
    signed = (1.0 - normalized_radius) * semis.min(axis=1)[None, :]
    return np.clip(signed.max(axis=1), -config.max_sdf_m, config.max_sdf_m).reshape(config.shape).astype(np.float32)


def corridor_from_perception(route_centres_local_m: np.ndarray, envelope_semi_m: np.ndarray,
                             points_local_m: np.ndarray, *, route_yaw_local_rad: np.ndarray | None = None,
                             config: PerceptionGridConfig = PerceptionGridConfig(),
                             clearance_m: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(corridor, sdf)`` from a planned local route and perceived obstacles.

    ``route_centres_local_m`` and ``envelope_semi_m`` have one row per future policy frame.
    The output corridor layout is ``[centre xyz, semi xyz, yaw]`` and matches
    :func:`manifold_motion.corridor.corridor_sdf`.
    """
    route = np.asarray(route_centres_local_m, dtype=np.float64)
    semi = np.asarray(envelope_semi_m, dtype=np.float64)
    if route.ndim != 2 or route.shape[1] != 3 or semi.shape != route.shape:
        raise ValueError("route centres and envelope semis must both be [T,3]")
    if not np.isfinite(route).all() or not np.isfinite(semi).all() or np.any(semi <= 0):
        raise ValueError("route and envelope values must be finite with positive semis")
    if clearance_m < 0:
        raise ValueError("clearance_m must be non-negative")
    # Rasterize first even though the raw field is not returned: it validates the point-cloud
    # input and documents the physical clearance computation that informed the contracted tube.
    pointcloud_sdf(points_local_m, config)
    if route_yaw_local_rad is None:
        yaw = _yaw_from_route(route)
    else:
        yaw = np.asarray(route_yaw_local_rad, dtype=np.float64)
        if yaw.shape != (len(route),) or not np.isfinite(yaw).all():
            raise ValueError("route_yaw_local_rad must be finite [T] when supplied")
    # Constrain each ellipse axis by nearby observed surfaces.  This is deliberately a
    # conservative point-cloud construction, rather than calling a planned route free merely
    # because its *centre* happens to be empty.  For every axis and sign we consider points in
    # the robot's transverse envelope and use the closest signed surface distance as the free
    # half-extent.  A low ceiling therefore contracts z, while a side wall contracts y.
    #
    # ``semi`` is already the body envelope plus caller-selected margin.  The point cloud may
    # only reduce that proposal; a reduction below the body envelope is retained explicitly so
    # Stage-2's hard gate can reject an infeasible route instead of silently enlarging space.
    points = np.asarray(points_local_m, dtype=np.float64)
    safe = semi + float(clearance_m)
    if points.size:
        for frame, (centre, proposal) in enumerate(zip(route, safe)):
            delta = points - centre
            for axis in range(3):
                other = [index for index in range(3) if index != axis]
                transverse = np.all(np.abs(delta[:, other]) <= proposal[other][None, :], axis=1)
                for sign in (-1.0, 1.0):
                    ahead = transverse & (sign * delta[:, axis] > 0.0)
                    if np.any(ahead):
                        nearest = float(np.min(sign * delta[ahead, axis]))
                        proposal[axis] = min(proposal[axis], max(config.min_corridor_semi_m,
                                                                  nearest - config.obstacle_inflation_m - clearance_m))
            # If the centre itself lies inside the inflated cloud, every axis advertises the
            # impossibility through the minimum semi-axis.  The downstream hard gate remains
            # responsible for rejecting any generated body that does not fit.
            if np.min(np.linalg.norm(delta, axis=1)) < config.obstacle_inflation_m:
                proposal[:] = config.min_corridor_semi_m
    corridor = np.concatenate([route, safe, yaw[:, None]], axis=1).astype(np.float32)
    return corridor, corridor_condition_sdf(corridor, config)


def validate_condition(corridor: np.ndarray, sdf: np.ndarray, config: PerceptionGridConfig = PerceptionGridConfig()) -> dict[str, float | bool]:
    """Small interface-level health check used before handing conditions to Stage 2."""
    corridor = np.asarray(corridor)
    sdf = np.asarray(sdf)
    valid = bool(corridor.ndim == 2 and corridor.shape[1] == 7 and sdf.shape == config.shape and
                 np.isfinite(corridor).all() and np.isfinite(sdf).all() and
                 np.all(corridor[:, 3:6] >= config.min_corridor_semi_m))
    return {"valid": valid, "corridor_frames": int(corridor.shape[0]) if corridor.ndim == 2 else 0,
            "sdf_min_m": float(np.min(sdf)) if sdf.size else float("nan"),
            "sdf_max_m": float(np.max(sdf)) if sdf.size else float("nan")}
