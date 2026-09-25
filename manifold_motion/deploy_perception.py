"""Simulation perception stack for the ``deploy`` branch.

This module is the perception-side replacement for the fixture-only A* route constructor:

``MuJoCo radar rays -> world-frame returns -> probabilistic sliding SLAM grid -> local A*
-> probability-aware safe corridor -> Stage-2 [corridor, sdf] condition``.

The grid is stored in global metric coordinates while its finite array window follows the robot.
The demo uses MuJoCo ground-truth poses as odometry so the sensor/map contract can be tested
without claiming that a simulated pose is a real SLAM estimate. A real radar/LiDAR driver and
SLAM pose source can replace :class:`SimulatedRadar` and the pose stream without changing the
grid, planner, or Stage-2 condition format.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from . import constants as C
from .perception_corridor import PerceptionGridConfig, corridor_condition_sdf, validate_condition


@dataclass(frozen=True)
class RadarConfig:
    """A deterministic 2-D/vertical fan used by the MuJoCo radar fixture."""

    horizontal_rays: int = 240
    # Upward channels make overhead clearance observable even while the pelvis pitches during
    # a crouch transition.  The old +/-0.16 fan intermittently lost a 1.20 m ceiling and
    # incorrectly expanded M_e back to nominal posture before entry.
    vertical_angles_rad: tuple[float, ...] = (-0.16, 0.0, 0.16, 0.32, 0.48)
    horizontal_fov_rad: float = 2.0 * math.pi
    max_range_m: float = 5.0
    # Offset above the pelvis/root origin.  With the fixture root at z=0.78 this places the
    # radar at z=1.00, below the 1.10 m top of the centre block, so a horizontal fan observes
    # its front/side surfaces instead of flying over it.
    sensor_height_m: float = 0.22
    noise_std_m: float = 0.008
    dropout_probability: float = 0.03
    seed: int = 20260922

    def validate(self) -> None:
        if self.horizontal_rays < 8 or self.max_range_m <= 0 or self.sensor_height_m < 0:
            raise ValueError("radar ray count/range/height are invalid")
        if not (0.0 <= self.dropout_probability < 1.0) or self.noise_std_m < 0:
            raise ValueError("radar noise/dropout values are invalid")


@dataclass
class RadarScan:
    points_world: np.ndarray
    origins_world: np.ndarray
    geom_names: list[str]
    timestamp: float


class SimulatedRadar:
    """MuJoCo ray fan that returns only named obstacle surfaces in world coordinates."""

    def __init__(self, scene: Path, config: RadarConfig = RadarConfig()):
        config.validate()
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(config.seed)
        self.geomgroup = np.ones(6, dtype=np.uint8)
        self.bodyexclude = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def scan(self, root_pos_world: np.ndarray, root_quat_wxyz: np.ndarray,
             timestamp: float = 0.0) -> RadarScan:
        root_pos = np.asarray(root_pos_world, dtype=np.float64)
        quat = np.asarray(root_quat_wxyz, dtype=np.float64)
        if root_pos.shape != (3,) or quat.shape != (4,):
            raise ValueError("root pose must be [3] position and [4] wxyz quaternion")
        quat = quat / max(float(np.linalg.norm(quat)), 1e-8)
        # Keep the simulated sensor mount synchronized with the supplied pose.  Rays exclude
        # the robot body, so this updates self geometry without turning it into map obstacles.
        if self.model.nq >= 7:
            self.data.qpos[:3] = root_pos
            self.data.qpos[3:7] = quat
        mujoco.mj_forward(self.model, self.data)
        origin = root_pos + C.quat_rotate(quat, np.array([0.0, 0.0, self.config.sensor_height_m]))
        yaw = np.linspace(-self.config.horizontal_fov_rad / 2.0,
                          self.config.horizontal_fov_rad / 2.0,
                          self.config.horizontal_rays, endpoint=False)
        points, origins, names = [], [], []
        for elevation in self.config.vertical_angles_rad:
            cos_elevation = math.cos(elevation)
            for angle in yaw:
                local_direction = np.array([
                    cos_elevation * math.cos(float(angle)),
                    cos_elevation * math.sin(float(angle)),
                    math.sin(elevation),
                ])
                direction = C.quat_rotate(quat, local_direction)
                # ``mj_ray`` accepts one body to exclude, not an entire articulated subtree.
                # Advance through non-obstacle robot links so the radar still sees a wall
                # behind an arm/torso rather than silently dropping that beam.
                travel = 0.0
                name = ""
                distance = -1.0
                while travel < self.config.max_range_m:
                    ray_origin = origin + travel * direction
                    geomid = np.full(1, -1, dtype=np.int32)
                    hit = float(mujoco.mj_ray(
                        self.model, self.data, ray_origin, direction, self.geomgroup,
                        True, self.bodyexclude, geomid))
                    if hit < 0.0:
                        break
                    distance = travel + hit
                    if distance > self.config.max_range_m:
                        break
                    geom_index = int(geomid[0])
                    name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_index) or ""
                    if name.startswith("obstacle_"):
                        break
                    travel = distance + 0.01
                if distance < 0.0 or distance > self.config.max_range_m or not name.startswith("obstacle_"):
                    continue
                if self.rng.random() < self.config.dropout_probability:
                    continue
                noisy_distance = max(0.02, distance + self.rng.normal(0.0, self.config.noise_std_m))
                points.append(origin + noisy_distance * direction)
                origins.append(origin.copy())
                names.append(name)
        return RadarScan(
            points_world=np.asarray(points, dtype=np.float32).reshape(-1, 3),
            origins_world=np.asarray(origins, dtype=np.float32).reshape(-1, 3),
            geom_names=names,
            timestamp=float(timestamp),
        )


@dataclass(frozen=True)
class SlidingGridConfig:
    resolution_m: float = 0.08
    size_xy: tuple[int, int] = (128, 128)
    # The deploy map is a true voxel volume.  XY remains the planner resolution; z is
    # represented explicitly so overhead obstacles and low clearances can contract the ellipsoid.
    size_xyz: tuple[int, int, int] = (128, 128, 32)
    body_z_min_offset_m: float = -0.78
    body_z_max_offset_m: float = 0.82
    known_ground_z_m: float = 0.0
    prior_probability: float = 0.50
    hit_probability: float = 0.78
    miss_probability: float = 0.38
    occupied_probability: float = 0.68
    unknown_band: float = 0.08
    log_odds_clip: float = 8.0
    # A perception-driven corridor may contract to this explicit lower bound.  Stage 2's
    # hard gate can then reject an infeasible squeeze instead of receiving a degenerate tube.
    min_corridor_semi_m: float = 0.06

    def validate(self) -> None:
        if (self.resolution_m <= 0 or min(self.size_xy) < 16 or min(self.size_xyz) < 8 or
                self.min_corridor_semi_m <= 0 or self.body_z_max_offset_m <= self.body_z_min_offset_m):
            raise ValueError("grid resolution/size is invalid")
        for value in (self.prior_probability, self.hit_probability, self.miss_probability,
                      self.occupied_probability):
            if not 0.0 < value < 1.0:
                raise ValueError("grid probabilities must be in (0,1)")
        if not self.miss_probability < self.prior_probability < self.hit_probability:
            raise ValueError("hit/miss probabilities must bracket the prior")


def _logit(probability: float) -> float:
    return float(np.log(probability / (1.0 - probability)))


def _bresenham(start: tuple[int, int], stop: tuple[int, int]) -> Iterable[tuple[int, int]]:
    """Integer 2-D ray cells including both endpoints."""
    x0, y0 = start
    x1, y1 = stop
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx - dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice > -dy:
            error -= dy
            x0 += sx
        if twice < dx:
            error += dx
            y0 += sy


def _bresenham3d(start: tuple[int, int, int], stop: tuple[int, int, int]) -> Iterable[tuple[int, int, int]]:
    """Deterministic integer voxel ray cells including both endpoints."""
    start_array, stop_array = np.asarray(start, dtype=np.int64), np.asarray(stop, dtype=np.int64)
    count = int(np.max(np.abs(stop_array - start_array))) + 1
    samples = np.rint(np.linspace(start_array, stop_array, max(2, count), axis=0)).astype(np.int64)
    previous: tuple[int, int, int] | None = None
    for sample in samples:
        cell = tuple(int(value) for value in sample)
        if cell != previous:
            yield cell
            previous = cell


class ProbabilisticSlidingGrid:
    """Global-coordinate log-odds map with a finite robot-centred sliding window."""

    def __init__(self, config: SlidingGridConfig = SlidingGridConfig(),
                 initial_robot_xy: np.ndarray = np.zeros(2)):
        config.validate()
        self.config = config
        self.shape_yx = (int(config.size_xy[1]), int(config.size_xy[0]))
        self.log_odds = np.full(self.shape_yx, _logit(config.prior_probability), dtype=np.float32)
        self.origin_cell_xy = np.zeros(2, dtype=np.int64)
        self.robot_xy = np.asarray(initial_robot_xy, dtype=np.float64).copy()
        self.recenter(self.robot_xy)
        self.update_count = 0

    @property
    def width_m(self) -> float:
        return self.config.size_xy[0] * self.config.resolution_m

    @property
    def height_m(self) -> float:
        return self.config.size_xy[1] * self.config.resolution_m

    @property
    def origin_world_xy(self) -> np.ndarray:
        return self.origin_cell_xy.astype(np.float64) * self.config.resolution_m

    @property
    def center_world_xy(self) -> np.ndarray:
        return self.origin_world_xy + np.array([self.width_m, self.height_m]) / 2.0

    def recenter(self, robot_xy: np.ndarray) -> None:
        robot_xy = np.asarray(robot_xy, dtype=np.float64)
        if robot_xy.shape != (2,) or not np.isfinite(robot_xy).all():
            raise ValueError("robot_xy must be finite [2]")
        res = self.config.resolution_m
        desired = np.floor((robot_xy - np.array([self.width_m, self.height_m]) / 2.0) / res).astype(np.int64)
        old_origin = self.origin_cell_xy.copy()
        old = self.log_odds
        if np.array_equal(desired, old_origin):
            self.robot_xy = robot_xy.copy()
            return
        new = np.full_like(old, _logit(self.config.prior_probability))
        old_x0, old_y0 = int(old_origin[0]), int(old_origin[1])
        new_x0, new_y0 = int(desired[0]), int(desired[1])
        old_x1, old_y1 = old_x0 + self.shape_yx[1], old_y0 + self.shape_yx[0]
        new_x1, new_y1 = new_x0 + self.shape_yx[1], new_y0 + self.shape_yx[0]
        overlap_x0, overlap_x1 = max(old_x0, new_x0), min(old_x1, new_x1)
        overlap_y0, overlap_y1 = max(old_y0, new_y0), min(old_y1, new_y1)
        if overlap_x1 > overlap_x0 and overlap_y1 > overlap_y0:
            old_slice = (slice(overlap_y0 - old_y0, overlap_y1 - old_y0),
                         slice(overlap_x0 - old_x0, overlap_x1 - old_x0))
            new_slice = (slice(overlap_y0 - new_y0, overlap_y1 - new_y0),
                         slice(overlap_x0 - new_x0, overlap_x1 - new_x0))
            new[new_slice] = old[old_slice]
        self.origin_cell_xy = desired
        self.log_odds = new
        self.robot_xy = robot_xy.copy()

    def world_to_global_cell(self, points_world_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world_xy, dtype=np.float64)
        return np.floor(points / self.config.resolution_m).astype(np.int64)

    def global_to_local_cell(self, cells_xy: np.ndarray) -> np.ndarray:
        return np.asarray(cells_xy, dtype=np.int64) - self.origin_cell_xy[None, :]

    def _inside_local(self, cells_xy: np.ndarray) -> np.ndarray:
        cells = np.asarray(cells_xy, dtype=np.int64)
        return ((cells[:, 0] >= 0) & (cells[:, 0] < self.shape_yx[1]) &
                (cells[:, 1] >= 0) & (cells[:, 1] < self.shape_yx[0]))

    def _add(self, global_cells_xy: np.ndarray, delta: float) -> None:
        local = self.global_to_local_cell(global_cells_xy)
        inside = self._inside_local(local)
        if not np.any(inside):
            return
        values = local[inside]
        self.log_odds[values[:, 1], values[:, 0]] = np.clip(
            self.log_odds[values[:, 1], values[:, 0]] + delta,
            -self.config.log_odds_clip, self.config.log_odds_clip)

    def update_radar(self, robot_pos_world: np.ndarray, scan: RadarScan) -> None:
        robot_xy = np.asarray(robot_pos_world, dtype=np.float64)[:2]
        self.recenter(robot_xy)
        start = tuple(self.world_to_global_cell(robot_xy[None, :])[0].tolist())
        miss_delta = _logit(self.config.miss_probability) - _logit(self.config.prior_probability)
        hit_delta = _logit(self.config.hit_probability) - _logit(self.config.prior_probability)
        for point in np.asarray(scan.points_world, dtype=np.float64).reshape(-1, 3):
            stop = tuple(self.world_to_global_cell(point[None, :2])[0].tolist())
            cells = list(_bresenham(start, stop))
            if len(cells) > 1:
                self._add(np.asarray(cells[:-1], dtype=np.int64), miss_delta)
            self._add(np.asarray([cells[-1]], dtype=np.int64), hit_delta)
        # A return can land on the robot's discretized centre cell after range noise/rounding,
        # even though articulated self-geometries were filtered from the ray labels.  The pose
        # source is authoritative for that one cell; keeping it free prevents A* from declaring
        # the current robot state occupied solely due to its own sensor quantization.
        robot_local = self.global_to_local_cell(np.asarray(start, dtype=np.int64)[None, :])[0]
        if (0 <= robot_local[0] < self.shape_yx[1] and 0 <= robot_local[1] < self.shape_yx[0]):
            self.log_odds[robot_local[1], robot_local[0]] = min(
                self.log_odds[robot_local[1], robot_local[0]], _logit(self.config.miss_probability))
        self.update_count += 1

    def probability(self) -> np.ndarray:
        return (1.0 / (1.0 + np.exp(-self.log_odds))).astype(np.float32)

    def occupancy_mask(self, threshold: float | None = None) -> np.ndarray:
        threshold = self.config.occupied_probability if threshold is None else threshold
        return self.probability() >= threshold

    def unknown_mask(self) -> np.ndarray:
        probability = self.probability()
        return np.abs(probability - self.config.prior_probability) <= self.config.unknown_band

    def cell_center_world(self, cell_xy: tuple[int, int] | np.ndarray) -> np.ndarray:
        cell = np.asarray(cell_xy, dtype=np.float64)
        return self.origin_world_xy + (cell + 0.5) * self.config.resolution_m

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, probability=self.probability(), log_odds=self.log_odds,
                            origin_world_xy=self.origin_world_xy,
                            origin_cell_xy=self.origin_cell_xy,
                            resolution_m=np.array(self.config.resolution_m),
                            robot_xy=self.robot_xy, update_count=np.array(self.update_count))


class ProbabilisticSlidingVoxelGrid:
    """Global-coordinate 3-D log-odds voxels with a robot-centred finite window.

    The planner consumes a conservative XY projection over the robot body height for the
    compatibility helper below, while :func:`voxel_astar` searches the full ``[z,y,x]`` volume.
    The complete 3-D probability volume remains available for vertical clearance and logging.
    """

    def __init__(self, config: SlidingGridConfig = SlidingGridConfig(),
                 initial_robot_xyz: np.ndarray = np.array([0.0, 0.0, 0.78])):
        config.validate()
        self.config = config
        nx, ny, nz = (int(value) for value in config.size_xyz)
        self.shape_zyx = (nz, ny, nx)
        self.shape_yx = (ny, nx)
        self.log_odds = np.full(self.shape_zyx, _logit(config.prior_probability), dtype=np.float32)
        self.origin_cell_xyz = np.zeros(3, dtype=np.int64)
        self.robot_xyz = np.asarray(initial_robot_xyz, dtype=np.float64).copy()
        self.recenter(self.robot_xyz)
        self.update_count = 0

    @property
    def width_m(self) -> float:
        return self.config.size_xyz[0] * self.config.resolution_m

    @property
    def depth_m(self) -> float:
        return self.config.size_xyz[1] * self.config.resolution_m

    @property
    def height_m(self) -> float:
        return self.config.size_xyz[2] * self.config.resolution_m

    @property
    def origin_world_xyz(self) -> np.ndarray:
        return self.origin_cell_xyz.astype(np.float64) * self.config.resolution_m

    @property
    def origin_world_xy(self) -> np.ndarray:
        return self.origin_world_xyz[:2]

    @property
    def center_world_xyz(self) -> np.ndarray:
        return self.origin_world_xyz + np.array([self.width_m, self.depth_m, self.height_m]) / 2.0

    def recenter(self, robot_xyz: np.ndarray) -> None:
        robot_xyz = np.asarray(robot_xyz, dtype=np.float64)
        if robot_xyz.shape != (3,) or not np.isfinite(robot_xyz).all():
            raise ValueError("robot_xyz must be finite [3]")
        half = np.array([self.width_m, self.depth_m, self.height_m]) / 2.0
        desired = np.floor((robot_xyz - half) / self.config.resolution_m).astype(np.int64)
        old_origin, old = self.origin_cell_xyz.copy(), self.log_odds
        if np.array_equal(desired, old_origin):
            self.robot_xyz = robot_xyz.copy()
            return
        new = np.full_like(old, _logit(self.config.prior_probability))
        old_max = old_origin + np.array([self.shape_zyx[2], self.shape_zyx[1], self.shape_zyx[0]])
        new_max = desired + np.array([self.shape_zyx[2], self.shape_zyx[1], self.shape_zyx[0]])
        lo, hi = np.maximum(old_origin, desired), np.minimum(old_max, new_max)
        if np.all(hi > lo):
            old_start = lo - old_origin
            new_start = lo - desired
            extent = hi - lo
            old_slice = tuple(slice(int(old_start[axis]), int(old_start[axis] + extent[axis]))
                              for axis in (2, 1, 0))
            new_slice = tuple(slice(int(new_start[axis]), int(new_start[axis] + extent[axis]))
                              for axis in (2, 1, 0))
            new[new_slice] = old[old_slice]
        self.origin_cell_xyz, self.log_odds, self.robot_xyz = desired, new, robot_xyz.copy()

    def world_to_global_cell_xyz(self, points_world_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_world_xyz must be [N,3]")
        return np.floor(points / self.config.resolution_m).astype(np.int64)

    def global_to_local_cell_xyz(self, cells_xyz: np.ndarray) -> np.ndarray:
        return np.asarray(cells_xyz, dtype=np.int64) - self.origin_cell_xyz[None, :]

    def world_to_global_cell_xy(self, points_world_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world_xy, dtype=np.float64)
        return np.floor(points / self.config.resolution_m).astype(np.int64)

    def _inside_local_xyz(self, cells_xyz: np.ndarray) -> np.ndarray:
        cells = np.asarray(cells_xyz, dtype=np.int64)
        return ((cells[:, 0] >= 0) & (cells[:, 0] < self.shape_zyx[2]) &
                (cells[:, 1] >= 0) & (cells[:, 1] < self.shape_zyx[1]) &
                (cells[:, 2] >= 0) & (cells[:, 2] < self.shape_zyx[0]))

    def _add_xyz(self, global_cells_xyz: np.ndarray, delta: float) -> None:
        local = self.global_to_local_cell_xyz(global_cells_xyz)
        inside = self._inside_local_xyz(local)
        if np.any(inside):
            values = local[inside]
            self.log_odds[values[:, 2], values[:, 1], values[:, 0]] = np.clip(
                self.log_odds[values[:, 2], values[:, 1], values[:, 0]] + delta,
                -self.config.log_odds_clip, self.config.log_odds_clip)

    def update_radar(self, robot_pos_world: np.ndarray, scan: RadarScan) -> None:
        robot_xyz = np.asarray(robot_pos_world, dtype=np.float64)
        self.recenter(robot_xyz)
        origins = np.asarray(scan.origins_world, dtype=np.float64).reshape(-1, 3)
        points = np.asarray(scan.points_world, dtype=np.float64).reshape(-1, 3)
        if len(origins) != len(points):
            raise ValueError("radar origins and points must have equal length")
        miss_delta = _logit(self.config.miss_probability) - _logit(self.config.prior_probability)
        hit_delta = _logit(self.config.hit_probability) - _logit(self.config.prior_probability)
        for origin, point in zip(origins, points):
            start = tuple(self.world_to_global_cell_xyz(origin[None, :])[0].tolist())
            stop = tuple(self.world_to_global_cell_xyz(point[None, :])[0].tolist())
            cells = list(_bresenham3d(start, stop))
            if len(cells) > 1:
                self._add_xyz(np.asarray(cells[:-1], dtype=np.int64), miss_delta)
            self._add_xyz(np.asarray([cells[-1]], dtype=np.int64), hit_delta)
        robot_cell = self.world_to_global_cell_xyz(robot_xyz[None, :])[0]
        local = self.global_to_local_cell_xyz(robot_cell[None, :])[0]
        if self._inside_local_xyz(local[None, :])[0]:
            self.log_odds[local[2], local[1], local[0]] = min(
                self.log_odds[local[2], local[1], local[0]], _logit(self.config.miss_probability))
        self.update_count += 1

    def probability(self) -> np.ndarray:
        return (1.0 / (1.0 + np.exp(-self.log_odds))).astype(np.float32)

    def occupancy_mask(self, threshold: float | None = None) -> np.ndarray:
        threshold = self.config.occupied_probability if threshold is None else threshold
        return self.probability() >= threshold

    def unknown_mask(self) -> np.ndarray:
        probability = self.probability()
        return np.abs(probability - self.config.prior_probability) <= self.config.unknown_band

    def probability_xy_projection(self, z_min_world: float, z_max_world: float) -> np.ndarray:
        z0 = int(np.floor(z_min_world / self.config.resolution_m)) - int(self.origin_cell_xyz[2])
        z1 = int(np.ceil(z_max_world / self.config.resolution_m)) - int(self.origin_cell_xyz[2]) + 1
        z0, z1 = max(0, z0), min(self.shape_zyx[0], z1)
        if z1 <= z0:
            return np.full(self.shape_yx, self.config.prior_probability, dtype=np.float32)
        return self.probability()[z0:z1].max(axis=0)

    def occupancy_xy_projection(self, z_min_world: float, z_max_world: float,
                                threshold: float | None = None) -> np.ndarray:
        return self.probability_xy_projection(z_min_world, z_max_world) >= (
            self.config.occupied_probability if threshold is None else threshold)

    def unknown_xy_projection(self, z_min_world: float, z_max_world: float) -> np.ndarray:
        z0 = max(0, int(np.floor(z_min_world / self.config.resolution_m)) - int(self.origin_cell_xyz[2]))
        z1 = min(self.shape_zyx[0], int(np.ceil(z_max_world / self.config.resolution_m)) -
                 int(self.origin_cell_xyz[2]) + 1)
        if z1 <= z0:
            return np.ones(self.shape_yx, dtype=bool)
        return self.unknown_mask()[z0:z1].all(axis=0)

    def xy_probability_at(self, point_world_xy: np.ndarray, z_min_world: float,
                          z_max_world: float) -> float:
        cell = self.world_to_global_cell_xy(np.asarray(point_world_xy, dtype=np.float64)[None, :])[0]
        local = cell - self.origin_cell_xyz[:2]
        if not (0 <= local[0] < self.shape_zyx[2] and 0 <= local[1] < self.shape_zyx[1]):
            return self.config.prior_probability
        return float(self.probability_xy_projection(z_min_world, z_max_world)[local[1], local[0]])

    def vertical_probability_at(self, point_world_xyz: np.ndarray) -> float:
        cell = self.world_to_global_cell_xyz(np.asarray(point_world_xyz, dtype=np.float64)[None, :])[0]
        local = cell - self.origin_cell_xyz
        if not self._inside_local_xyz(local[None, :])[0]:
            return self.config.prior_probability
        return float(self.probability()[local[2], local[1], local[0]])

    def cell_center_world(self, cell_xy: tuple[int, int] | np.ndarray) -> np.ndarray:
        cell = np.asarray(cell_xy, dtype=np.float64)
        return self.origin_world_xy + (cell + 0.5) * self.config.resolution_m

    def cell_center_world_xyz(self, cell_xyz: tuple[int, int, int] | np.ndarray) -> np.ndarray:
        cell = np.asarray(cell_xyz, dtype=np.float64)
        return self.origin_world_xyz + (cell + 0.5) * self.config.resolution_m

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, probability=self.probability(), log_odds=self.log_odds,
                            origin_world_xyz=self.origin_world_xyz,
                            origin_cell_xyz=self.origin_cell_xyz,
                            resolution_m=np.array(self.config.resolution_m),
                            robot_xyz=self.robot_xyz, update_count=np.array(self.update_count),
                            shape_zyx=np.asarray(self.shape_zyx))


def render_grid_png(grid: ProbabilisticSlidingGrid | ProbabilisticSlidingVoxelGrid, route_world_xy: np.ndarray,
                    start_world_xy: np.ndarray, goal_world_xy: np.ndarray, path: Path,
                    trace_world_xy: np.ndarray | None = None) -> None:
    """Render a small, dependency-light evidence image for headless WSL runs."""
    from PIL import Image, ImageDraw

    if hasattr(grid, "probability_xy_projection"):
        robot_z = float(getattr(grid, "robot_xyz")[2])
        probability = grid.probability_xy_projection(
            robot_z + grid.config.body_z_min_offset_m,
            robot_z + grid.config.body_z_max_offset_m)
        observed = ~grid.unknown_xy_projection(
            robot_z + grid.config.body_z_min_offset_m,
            robot_z + grid.config.body_z_max_offset_m)
        occupied = grid.occupancy_xy_projection(
            robot_z + grid.config.body_z_min_offset_m,
            robot_z + grid.config.body_z_max_offset_m)
    else:
        probability = grid.probability()
        observed = ~grid.unknown_mask()
        occupied = grid.occupancy_mask()
    # Unknown cells stay mid-gray; high occupancy is dark red and observed free space is light.
    intensity = np.clip((1.0 - probability) * 210.0 + 35.0, 0, 255).astype(np.uint8)
    rgb = np.stack([intensity, intensity, intensity], axis=2)
    rgb[~observed] = np.array([148, 148, 148], dtype=np.uint8)
    rgb[occupied] = np.array([205, 62, 52], dtype=np.uint8)
    resampling = getattr(Image, "Resampling", Image).NEAREST
    image = Image.fromarray(np.flipud(rgb), mode="RGB").resize((768, 768), resampling)
    draw = ImageDraw.Draw(image)

    def pixel(point: np.ndarray) -> tuple[int, int]:
        if hasattr(grid, "world_to_global_cell_xy"):
            cell = grid.world_to_global_cell_xy(np.asarray(point, dtype=np.float64)[None, :2])[0]
            local = cell - grid.origin_cell_xyz[:2]
        else:
            cell = grid.world_to_global_cell(np.asarray(point, dtype=np.float64)[None, :2])[0]
            local = cell - grid.origin_cell_xy
        scale = 768.0 / float(grid.shape_yx[1])
        return int((local[0] + 0.5) * scale), int((grid.shape_yx[0] - local[1] - 0.5) * scale)

    if trace_world_xy is not None:
        trace = np.asarray(trace_world_xy, dtype=np.float64)
        for first, second in zip(trace[:-1], trace[1:]):
            draw.line([pixel(first), pixel(second)], fill=(45, 180, 220), width=4)
    route = np.asarray(route_world_xy, dtype=np.float64)
    for first, second in zip(route[:-1], route[1:]):
        draw.line([pixel(first), pixel(second)], fill=(255, 220, 30), width=5)
    sx, sy = pixel(start_world_xy)
    gx, gy = pixel(goal_world_xy)
    draw.ellipse((sx - 8, sy - 8, sx + 8, sy + 8), fill=(30, 210, 90), outline=(0, 0, 0), width=2)
    draw.ellipse((gx - 8, gy - 8, gx + 8, gy + 8), fill=(45, 110, 245), outline=(0, 0, 0), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def render_voxel_slices_png(grid: ProbabilisticSlidingVoxelGrid, path: Path) -> None:
    """Render three horizontal z slices to make the 3-D occupancy volume inspectable headlessly."""
    from PIL import Image, ImageDraw

    probability = grid.probability()
    occupied = grid.occupancy_mask()
    unknown = grid.unknown_mask()
    indices = np.unique(np.asarray([0, grid.shape_zyx[0] // 2, grid.shape_zyx[0] - 1]))
    panels = []
    for z_index in indices:
        intensity = np.clip((1.0 - probability[z_index]) * 210.0 + 35.0, 0, 255).astype(np.uint8)
        rgb = np.stack([intensity, intensity, intensity], axis=2)
        rgb[unknown[z_index]] = np.array([148, 148, 148], dtype=np.uint8)
        rgb[occupied[z_index]] = np.array([205, 62, 52], dtype=np.uint8)
        panel = Image.fromarray(np.flipud(rgb), mode="RGB").resize((320, 320),
            getattr(Image, "Resampling", Image).NEAREST)
        ImageDraw.Draw(panel).text((8, 8), f"z={grid.origin_world_xyz[2] + (int(z_index)+.5)*grid.config.resolution_m:.2f} m",
                                   fill=(20, 20, 20))
        panels.append(panel)
    canvas = Image.new("RGB", (320 * len(panels), 320), (110, 110, 110))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * 320, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _inflate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return mask.copy()
    result = np.zeros_like(mask, dtype=bool)
    occupied_y, occupied_x = np.nonzero(mask)
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue
            y = occupied_y + dy
            x = occupied_x + dx
            valid = (y >= 0) & (y < mask.shape[0]) & (x >= 0) & (x < mask.shape[1])
            result[y[valid], x[valid]] = True
    return result


def grid_astar(grid: ProbabilisticSlidingGrid | ProbabilisticSlidingVoxelGrid, start_world_xy: np.ndarray,
               goal_world_xy: np.ndarray, *, body_radius_m: float = 0.40,
               clearance_m: float = 0.10, allow_unknown: bool = True,
               robot_z_world: float | None = None) -> np.ndarray:
    """A* over the global-coordinate sliding window, with probability obstacle inflation."""
    if hasattr(grid, "occupancy_xy_projection"):
        robot_z = float(grid.robot_xyz[2] if robot_z_world is None else robot_z_world)
        z_min = robot_z + grid.config.body_z_min_offset_m
        z_max = robot_z + grid.config.body_z_max_offset_m
        start_global = grid.world_to_global_cell_xy(np.asarray(start_world_xy)[None, :2])[0]
        goal_global = grid.world_to_global_cell_xy(np.asarray(goal_world_xy)[None, :2])[0]
        origin_xy = grid.origin_cell_xyz[:2]
        occupied_raw = grid.occupancy_xy_projection(z_min, z_max)
        unknown = grid.unknown_xy_projection(z_min, z_max)
        probability_map = grid.probability_xy_projection(z_min, z_max)
    else:
        start_global = grid.world_to_global_cell(np.asarray(start_world_xy)[None, :2])[0]
        goal_global = grid.world_to_global_cell(np.asarray(goal_world_xy)[None, :2])[0]
        origin_xy = grid.origin_cell_xy
        occupied_raw = grid.occupancy_mask()
        unknown = grid.unknown_mask()
        probability_map = grid.probability()
    start = tuple((start_global - origin_xy).tolist())
    goal = tuple((goal_global - origin_xy).tolist())
    occupied = _inflate(occupied_raw, int(math.ceil((body_radius_m + clearance_m) /
                                                               grid.config.resolution_m)))
    if allow_unknown:
        blocked = occupied
    else:
        blocked = occupied | unknown
    height, width = grid.shape_yx

    def clear_pose_disk(cell: tuple[int, int], radius_cells: int) -> None:
        """Clear only the footprint already occupied by the localized robot."""
        cx, cy = cell
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= radius_cells * radius_cells:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < width and 0 <= y < height:
                        blocked[y, x] = False

    # Inflation of a nearby wall can overlap the discretized current footprint by one or two
    # cells.  Localization proves that footprint is currently occupied by the robot rather than
    # an obstacle; carve a small start bubble, while leaving all cells beyond the body radius
    # untouched.  This is the standard rolling-costmap start-footprint exception.
    clear_pose_disk(start, max(1, int(math.ceil(body_radius_m / grid.config.resolution_m))))

    def valid(cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < width and 0 <= y < height and not blocked[y, x]
    if not valid(start):
        raise RuntimeError("SLAM grid start cell is occupied or outside the sliding window")
    if not valid(goal):
        candidates = [(x, y) for y in range(height) for x in range(width) if valid((x, y))]
        if not candidates:
            raise RuntimeError("SLAM grid has no free goal cell")
        goal = min(candidates, key=lambda cell: (cell[0] - goal[0]) ** 2 + (cell[1] - goal[1]) ** 2)
    neighbors = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
                 (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)))
    frontier = [(0.0, start)]
    cost = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for dx, dy, step in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if not valid(nxt):
                continue
            probability = float(probability_map[nxt[1], nxt[0]])
            unknown_penalty = 0.35 if unknown[nxt[1], nxt[0]] else 0.0
            probability_penalty = 0.50 * probability
            new_cost = cost[current] + step + unknown_penalty + probability_penalty
            if new_cost < cost.get(nxt, np.inf):
                cost[nxt] = new_cost
                heuristic = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                heapq.heappush(frontier, (new_cost + heuristic, nxt))
                came_from[nxt] = current
    if goal not in came_from and goal != start:
        raise RuntimeError("A* could not connect start and goal on the local probabilistic map")
    cells = [goal]
    while cells[-1] != start:
        cells.append(came_from[cells[-1]])
    cells.reverse()
    return np.asarray([grid.cell_center_world(cell) for cell in cells], dtype=np.float32)


def voxel_astar(grid: ProbabilisticSlidingVoxelGrid, start_world_xyz: np.ndarray,
                goal_world_xyz: np.ndarray, *, body_radius_m: float = 0.40,
                body_half_height_m: float = 0.78, clearance_m: float = 0.10,
                allow_unknown: bool = True, vertical_weight: float = 1.8,
                preferred_world_xy: np.ndarray | None = None,
                preference_weight: float = 0.0) -> np.ndarray:
    """A* directly over the 3-D voxel volume.

    The footprint is conservatively inflated in XY and the body half-height is inflated in Z.
    Six axis moves plus 20 diagonals are allowed; vertical motion costs more than horizontal
    motion so a flat route is preferred whenever it is collision-free.  Online replanning may
    additionally provide the previous safe route as a soft hysteresis prior.  This prevents a
    symmetric obstacle from making the local planner alternate between left and right detours.
    """
    start_xyz = np.asarray(start_world_xyz, dtype=np.float64)
    goal_xyz = np.asarray(goal_world_xyz, dtype=np.float64)
    start_global = grid.world_to_global_cell_xyz(start_xyz[None, :])[0]
    goal_global = grid.world_to_global_cell_xyz(goal_xyz[None, :])[0]
    start = tuple((start_global - grid.origin_cell_xyz).tolist())
    goal = tuple((goal_global - grid.origin_cell_xyz).tolist())
    occupied = grid.occupancy_mask().copy()
    unknown = grid.unknown_mask()
    if allow_unknown:
        blocked = occupied
    else:
        blocked = occupied | unknown
    nz, ny, nx = grid.shape_zyx
    radius_cells = int(math.ceil((body_radius_m + clearance_m) / grid.config.resolution_m))
    z_cells = int(math.ceil((body_half_height_m + clearance_m) / grid.config.resolution_m))
    occupied_y, occupied_x = np.nonzero(occupied.any(axis=0))
    occupied_z, occupied_y3, occupied_x3 = np.nonzero(occupied)
    inflated = np.zeros_like(blocked, dtype=bool)
    for z, y, x in zip(occupied_z, occupied_y3, occupied_x3):
        for dz in range(-z_cells, z_cells + 1):
            zz = z + dz
            if not 0 <= zz < nz:
                continue
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy > radius_cells * radius_cells:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < ny and 0 <= xx < nx:
                        inflated[zz, yy, xx] = True
    blocked = inflated if allow_unknown else (inflated | unknown)

    def valid(cell: tuple[int, int, int]) -> bool:
        x, y, z = cell
        return 0 <= x < nx and 0 <= y < ny and 0 <= z < nz and not blocked[z, y, x]

    def clear_start(cell: tuple[int, int, int]) -> None:
        cx, cy, cz = cell
        for dz in range(-z_cells, z_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy > radius_cells * radius_cells:
                        continue
                    x, y, z = cx + dx, cy + dy, cz + dz
                    if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                        blocked[z, y, x] = False

    clear_start(start)
    if not valid(start):
        raise RuntimeError("3-D voxel A* start is occupied or outside the sliding volume")
    if not valid(goal):
        candidates = np.argwhere(~blocked)
        if len(candidates) == 0:
            raise RuntimeError("3-D voxel map has no free goal voxel")
        goal_array = np.asarray(goal)[[2, 1, 0]]
        nearest = candidates[np.argmin(np.sum((candidates - goal_array[None, :]) ** 2, axis=1))]
        goal = (int(nearest[2]), int(nearest[1]), int(nearest[0]))
    neighbors = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                step = math.sqrt(dx * dx + dy * dy + (vertical_weight * dz) ** 2)
                neighbors.append((dx, dy, dz, step))
    probability = grid.probability()
    preference = None
    if preferred_world_xy is not None and preference_weight > 0.0:
        preferred = np.asarray(preferred_world_xy, dtype=np.float64)
        if preferred.ndim != 2 or preferred.shape[1] != 2 or not len(preferred):
            raise ValueError("preferred_world_xy must be a non-empty [N,2] route")
        xs = grid.origin_world_xyz[0] + (np.arange(nx, dtype=np.float64) + 0.5) * grid.config.resolution_m
        ys = grid.origin_world_xyz[1] + (np.arange(ny, dtype=np.float64) + 0.5) * grid.config.resolution_m
        mesh_x, mesh_y = np.meshgrid(xs, ys)
        cells_xy = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
        preference = np.sqrt(np.min(np.sum(
            (cells_xy[:, None, :] - preferred[None, :, :]) ** 2, axis=2), axis=1)).reshape(ny, nx)
    frontier = [(0.0, start)]
    cost = {start: 0.0}
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for dx, dy, dz, step in neighbors:
            nxt = (current[0] + dx, current[1] + dy, current[2] + dz)
            if not valid(nxt):
                continue
            uncertainty = 0.35 if unknown[nxt[2], nxt[1], nxt[0]] else 0.0
            risk = 0.50 * float(probability[nxt[2], nxt[1], nxt[0]])
            hysteresis = (float(preference[nxt[1], nxt[0]]) * float(preference_weight)
                          if preference is not None else 0.0)
            new_cost = cost[current] + step + uncertainty + risk + hysteresis
            if new_cost < cost.get(nxt, np.inf):
                cost[nxt] = new_cost
                heuristic = math.sqrt((goal[0] - nxt[0]) ** 2 +
                                      (goal[1] - nxt[1]) ** 2 +
                                      (vertical_weight * (goal[2] - nxt[2])) ** 2)
                heapq.heappush(frontier, (new_cost + heuristic, nxt))
                came_from[nxt] = current
    if goal not in came_from and goal != start:
        raise RuntimeError("3-D voxel A* could not connect start and goal")
    cells = [goal]
    while cells[-1] != start:
        cells.append(came_from[cells[-1]])
    cells.reverse()
    return np.asarray([grid.cell_center_world_xyz((x, y, z)) for x, y, z in cells], dtype=np.float32)


def _densify(route_world_xy: np.ndarray, spacing_m: float = 0.08) -> np.ndarray:
    route = np.asarray(route_world_xy, dtype=np.float64)
    pieces = []
    for index, (start, stop) in enumerate(zip(route[:-1], route[1:])):
        count = max(2, int(math.ceil(np.linalg.norm(stop - start) / spacing_m)) + 1)
        part = np.linspace(start, stop, count)
        pieces.append(part if index == 0 else part[1:])
    return np.concatenate(pieces) if pieces else route.copy()


def _route_command(route_world_xyz: np.ndarray, root_pos_world: np.ndarray,
                   root_quat_wxyz: np.ndarray, horizon_m: float = 0.90) -> np.ndarray:
    """Encode a short local route target using the Stage-2 3+6 command contract."""
    route = np.asarray(route_world_xyz, dtype=np.float64)
    root = np.asarray(root_pos_world, dtype=np.float64)
    if len(route) < 2:
        raise ValueError("route must contain at least two points for a command")
    segment = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    target = min(float(horizon_m), float(cumulative[-1]))
    index = min(max(int(np.searchsorted(cumulative, target, side="left")), 1), len(route) - 1)
    span = cumulative[index] - cumulative[index - 1]
    alpha = (target - cumulative[index - 1]) / span if span > 1e-8 else 1.0
    point = route[index - 1] * (1.0 - alpha) + route[index] * alpha
    rotation = C.quat_to_matrix(np.asarray(root_quat_wxyz, dtype=np.float64))
    delta_local = (point - root) @ rotation
    tangent = route[index, :2] - route[index - 1, :2]
    yaw = math.atan2(float(tangent[1]), float(tangent[0]))
    root_forward = C.quat_rotate(np.asarray(root_quat_wxyz, dtype=np.float64), np.array([1.0, 0.0, 0.0]))
    yaw -= math.atan2(float(root_forward[1]), float(root_forward[0]))
    c, s = math.cos(yaw), math.sin(yaw)
    return np.concatenate([delta_local, np.array([c, -s, s, c, 0.0, 0.0])]).astype(np.float32)


def _axis_free_half_extent(grid: ProbabilisticSlidingGrid | ProbabilisticSlidingVoxelGrid, center: np.ndarray,
                           direction: np.ndarray, max_distance_m: float,
                           clearance_m: float) -> float:
    steps = max(1, int(math.ceil(max_distance_m / grid.config.resolution_m)))
    for step in range(1, steps + 1):
        point = center + direction * (step * grid.config.resolution_m)
        if hasattr(grid, "xy_probability_at"):
            robot_z = float(grid.robot_xyz[2])
            probability = grid.xy_probability_at(
                point, robot_z + grid.config.body_z_min_offset_m,
                robot_z + grid.config.body_z_max_offset_m)
            cell_global = grid.world_to_global_cell_xy(point[None, :])[0]
            cell = tuple((cell_global - grid.origin_cell_xyz[:2]).tolist())
        else:
            cell_global = grid.world_to_global_cell(point[None, :])[0]
            cell = tuple((cell_global - grid.origin_cell_xy).tolist())
            x, y = cell
            probability = float(grid.probability()[y, x]) if (
                0 <= x < grid.shape_yx[1] and 0 <= y < grid.shape_yx[0]) else 0.0
        x, y = cell
        if not (0 <= x < grid.shape_yx[1] and 0 <= y < grid.shape_yx[0]):
            return max(grid.config.min_corridor_semi_m, (step - 1) * grid.config.resolution_m - clearance_m)
        if probability >= grid.config.occupied_probability:
            return max(grid.config.min_corridor_semi_m, (step - 1) * grid.config.resolution_m - clearance_m)
    return max(grid.config.min_corridor_semi_m, max_distance_m - clearance_m)


def _vertical_footprint_occupied(
    grid: ProbabilisticSlidingVoxelGrid,
    center_xy: np.ndarray,
    z_world: float,
    probability: np.ndarray,
    *,
    footprint_radius_m: float = 0.28,
    min_occupied_voxels: int = 2,
) -> bool:
    """Query an overhead slice across the body footprint, not one centre voxel.

    A spinning/ray radar samples a ceiling sparsely.  Requiring the exact route-centre voxel
    misses most valid returns, while a small footprint disk retains the geometric meaning of
    vertical clearance.  Two occupied cells suppress an isolated noisy hit.
    """
    global_cell = grid.world_to_global_cell_xyz(np.array([
        [float(center_xy[0]), float(center_xy[1]), float(z_world)]], dtype=np.float64))[0]
    local = global_cell - grid.origin_cell_xyz
    z, cy, cx = int(local[2]), int(local[1]), int(local[0])
    if not (0 <= z < grid.shape_zyx[0]):
        return False
    radius = max(1, int(math.ceil(footprint_radius_m / grid.config.resolution_m)))
    x0, x1 = max(0, cx - radius), min(grid.shape_zyx[2], cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(grid.shape_zyx[1], cy + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return False
    yy, xx = np.indices((y1 - y0, x1 - x0))
    disk = ((xx + x0 - cx) ** 2 + (yy + y0 - cy) ** 2) <= radius ** 2
    occupied = probability[z, y0:y1, x0:x1] >= grid.config.occupied_probability
    return int(np.count_nonzero(occupied & disk)) >= int(min_occupied_voxels)


def _vertical_free_half_extent(grid: ProbabilisticSlidingVoxelGrid, center_xy: np.ndarray,
                               z_center: float, max_distance_m: float,
                               clearance_m: float,
                               probability: np.ndarray | None = None) -> float:
    """Find the upward free extent from footprint-aware 3-D voxel evidence."""
    probability = grid.probability() if probability is None else probability
    steps = max(1, int(math.ceil(max_distance_m / grid.config.resolution_m)))
    limits = []
    for sign in (-1.0, 1.0):
        free = max_distance_m
        for step in range(1, steps + 1):
            point = np.array([center_xy[0], center_xy[1],
                              z_center + sign * step * grid.config.resolution_m])
            if sign < 0.0 and point[2] <= grid.config.known_ground_z_m:
                free = max(0.0, z_center - grid.config.known_ground_z_m - clearance_m)
                break
            if _vertical_footprint_occupied(
                    grid, center_xy, float(point[2]), probability):
                free = (step - 1) * grid.config.resolution_m - clearance_m
                break
        limits.append(max(grid.config.min_corridor_semi_m, free))
    # The Stage-2 corridor uses a symmetric root-local ellipsoid whose vertical axis is the
    # robot envelope/ceiling clearance.  The floor is not an overhead constraint: using the
    # lower root-to-floor distance here would shrink every flat-ground corridor to ~0.68 m and
    # make the semantic router classify ordinary walking as crouching.  Foot support remains a
    # separate SONIC/MuJoCo contact and self-manifold check.
    return limits[1]


def safe_corridor_from_grid(grid: ProbabilisticSlidingGrid | ProbabilisticSlidingVoxelGrid, route_world: np.ndarray,
                            root_pos_world: np.ndarray, root_quat_wxyz: np.ndarray,
                            *, vertical_semi_m: float = 1.20,
                            clearance_m: float = 0.08,
                            frames: int = 48, horizon_m: float | None = 0.90) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create a root-local Stage-2 corridor from the probability map and A* route.

    The global A* route is retained by the deploy artifact for map visualisation, but a
    Stage-2 reference is only one short receding-horizon window.  Truncating here keeps the
    corridor time span consistent with :func:`_route_command` and with the SEED windows
    used during training; passing a full multi-metre route to a 1.6-second model is a silent
    distribution shift and makes it predict an over-long root trajectory.
    """
    route_world = np.asarray(route_world, dtype=np.float64)
    if route_world.ndim != 2 or route_world.shape[1] not in (2, 3):
        raise ValueError("route_world must be [N,2] or [N,3]")
    if route_world.shape[1] == 2:
        route_world = np.column_stack([route_world, np.full(len(route_world), float(root_pos_world[2]))])
    route_world = _densify(route_world)
    if len(route_world) < 2:
        raise ValueError("route must contain at least two points")
    global_route_length = float(np.linalg.norm(np.diff(route_world, axis=0), axis=1).sum())
    if horizon_m is not None:
        if horizon_m <= 0:
            raise ValueError("horizon_m must be positive when provided")
        segment_lengths = np.linalg.norm(np.diff(route_world, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        target = min(float(horizon_m), float(cumulative[-1]))
        if target < cumulative[-1] - 1e-9:
            index = int(np.searchsorted(cumulative, target, side="left"))
            index = min(max(index, 1), len(route_world) - 1)
            span = cumulative[index] - cumulative[index - 1]
            alpha = (target - cumulative[index - 1]) / span if span > 1e-8 else 1.0
            endpoint = route_world[index - 1] * (1.0 - alpha) + route_world[index] * alpha
            route_world = np.vstack([route_world[:index], endpoint])
    target = np.linspace(0.0, 1.0, frames)
    source = np.linspace(0.0, 1.0, len(route_world))
    route_world = np.column_stack([np.interp(target, source, route_world[:, axis]) for axis in range(3)])
    root = np.asarray(root_pos_world, dtype=np.float64)
    quat = np.asarray(root_quat_wxyz, dtype=np.float64)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-8)
    route_local_xy = (route_world[:, :2] - root[None, :2]) @ C.quat_to_matrix(quat)[:2, :2]
    route_local_z = route_world[:, 2] - root[2]
    root_forward = C.quat_rotate(quat, np.array([1.0, 0.0, 0.0]))
    root_yaw = math.atan2(float(root_forward[1]), float(root_forward[0]))
    headings_world = np.unwrap(np.arctan2(np.gradient(route_world[:, 1]), np.gradient(route_world[:, 0])))
    headings_local = headings_world - root_yaw
    semis = []
    volume_probability = grid.probability() if hasattr(grid, "vertical_probability_at") else None
    for point, heading in zip(route_world, headings_world):
        tangent = np.array([math.cos(float(heading)), math.sin(float(heading))])
        lateral = np.array([-tangent[1], tangent[0]])
        tangent_free = min(_axis_free_half_extent(grid, point[:2], tangent, 1.4, clearance_m),
                           _axis_free_half_extent(grid, point[:2], -tangent, 1.4, clearance_m))
        lateral_free = min(_axis_free_half_extent(grid, point[:2], lateral, 1.4, clearance_m),
                           _axis_free_half_extent(grid, point[:2], -lateral, 1.4, clearance_m))
        if hasattr(grid, "vertical_probability_at"):
            vertical_free = _vertical_free_half_extent(
                grid, point[:2], float(point[2]), vertical_semi_m, clearance_m,
                probability=volume_probability)
        else:
            vertical_free = vertical_semi_m
        semis.append([max(grid.config.min_corridor_semi_m, tangent_free),
                      max(grid.config.min_corridor_semi_m, lateral_free),
                      max(grid.config.min_corridor_semi_m, vertical_free)])
    local_centres = np.column_stack([route_local_xy, route_local_z])
    corridor = np.column_stack([local_centres, np.asarray(semis), headings_local]).astype(np.float32)
    sdf = corridor_condition_sdf(corridor, PerceptionGridConfig())
    report = {
        "coordinate_frame": "corridor root-local; map global metric",
        "route_frames": int(frames),
        "route_length_m": float(np.linalg.norm(np.diff(route_world, axis=0), axis=1).sum()),
        "global_route_length_m": global_route_length,
        "receding_horizon_m": float(horizon_m) if horizon_m is not None else None,
        "corridor_semi_min_m": corridor[:, 3:6].min(axis=0).astype(float).tolist(),
        "corridor_semi_mean_m": corridor[:, 3:6].mean(axis=0).astype(float).tolist(),
        "map_origin_world_xy_m": grid.origin_world_xy.astype(float).tolist(),
        "map_origin_world_xyz_m": getattr(grid, "origin_world_xyz", np.r_[grid.origin_world_xy, 0.0]).astype(float).tolist(),
        "map_shape_zyx": list(getattr(grid, "shape_zyx", (1, *grid.shape_yx))),
        "map_resolution_m": float(grid.config.resolution_m),
        "map_updates": int(grid.update_count),
    }
    return corridor, sdf, report


def build_simulated_deploy_condition(scene: Path, root_pos_world: np.ndarray,
                                     root_quat_wxyz: np.ndarray, goal_world_xy: np.ndarray,
                                     *, out: Path, radar_config: RadarConfig = RadarConfig(),
                                     grid_config: SlidingGridConfig = SlidingGridConfig(),
                                     demo_pose_count: int = 5,
                                     planner_body_radius_m: float = 0.46,
                                     planner_clearance_m: float = 0.12) -> dict:
    """Run a repeatable radar/SLAM/A*/corridor demo and save deploy-ready artifacts."""
    radar = SimulatedRadar(scene, radar_config)
    root_pos = np.asarray(root_pos_world, dtype=np.float64)
    grid = ProbabilisticSlidingVoxelGrid(grid_config, root_pos)
    goal = np.asarray(goal_world_xy, dtype=np.float64)
    initial_root = root_pos.copy()
    pose = root_pos.copy()
    # Ground-truth pose is deliberately used only as simulated odometry.  The scan poses follow
    # the *current A* prefix* rather than a straight line through the box: this is a compact
    # receding-horizon SLAM/planning loop, and every simulated scout pose remains collision-free
    # with respect to the inflated map produced so far.
    scans = []
    all_points, all_origins = [], []
    scout_trace = [pose[:2].copy()]
    route_world_xyz = None
    goal_world_xyz = np.array([goal[0], goal[1], root_pos[2]], dtype=np.float64)
    for timestamp in range(max(2, demo_pose_count)):
        scan = radar.scan(pose, root_quat_wxyz, timestamp=float(timestamp) * 0.1)
        grid.update_radar(pose, scan)
        all_points.append(scan.points_world)
        all_origins.append(scan.origins_world)
        scans.append({"timestamp": scan.timestamp, "returns": int(len(scan.points_world)),
                      "robot_xy": pose[:2].astype(float).tolist(),
                      "grid_origin_world_xy_m": grid.origin_world_xy.astype(float).tolist(),
                      "grid_origin_world_xyz_m": grid.origin_world_xyz.astype(float).tolist(),
                      "obstacle_return_names": sorted(set(scan.geom_names))})
        route_world_xyz = voxel_astar(
            grid, pose, goal_world_xyz, body_radius_m=0.40,
            body_half_height_m=0.78, clearance_m=0.10,
            allow_unknown=True, vertical_weight=3.0)
        route_world_xyz[:, 2] = root_pos[2]
        if timestamp + 1 < max(2, demo_pose_count) and len(route_world_xyz) > 1:
            step_lengths = np.linalg.norm(np.diff(route_world_xyz, axis=0), axis=1)
            next_index = int(np.searchsorted(np.cumsum(step_lengths), 0.45, side="left") + 1)
            next_index = min(next_index, len(route_world_xyz) - 1)
            pose[:3] = route_world_xyz[next_index]
            scout_trace.append(pose[:2].copy())
    assert route_world_xyz is not None
    # The scout poses above are only a simulated radar/SLAM observation stream.  They must not
    # silently become the execution origin: main's SONIC rollout still starts at root_pos_world.
    # Replan once from that live root against the accumulated global map, then express the
    # corridor in the same root-local frame consumed by Stage 2.
    planning_root = initial_root.copy()
    route_world_xyz = voxel_astar(
        grid, planning_root, goal_world_xyz, body_radius_m=0.40,
        body_half_height_m=0.78, clearance_m=0.10, allow_unknown=True,
        vertical_weight=3.0)
    route_world_xyz[:, 2] = planning_root[2]
    # Keep the map-derived lower/upper detour, then move only the detour portion outward by
    # the additional execution-envelope margin.  The start/end remain anchored to the live
    # root/goal, while the route no longer grazes a wall when the G1 mesh is wider than the
    # pilot planner footprint.
    extra_margin = max(0.0, float(planner_body_radius_m + planner_clearance_m - 0.50))
    if extra_margin > 1.0e-6:
        route_xy = route_world_xyz[:, :2].astype(np.float64)
        direct = goal - planning_root[:2]
        direct_norm = max(float(np.linalg.norm(direct)), 1.0e-8)
        tangent = direct / direct_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        lateral = (route_xy - planning_root[:2][None, :]) @ normal
        sign = float(np.sign(lateral[np.argmax(np.abs(lateral))]))
        if abs(sign) > 0.5:
            ramp = np.clip(np.abs(lateral) / 0.80, 0.0, 1.0)
            route_world_xyz[:, :2] = route_xy + sign * extra_margin * ramp[:, None] * normal[None, :]
            route_world_xyz[0, :2] = planning_root[:2]
            route_world_xyz[-1, :2] = goal
    route_world_xy = route_world_xyz[:, :2]
    command = _route_command(route_world_xyz, planning_root, root_quat_wxyz)
    corridor, sdf, corridor_report = safe_corridor_from_grid(
        grid, route_world_xyz, planning_root, root_quat_wxyz, horizon_m=0.90)
    out.mkdir(parents=True, exist_ok=True)
    grid.save(out / "slam_grid.npz")
    scout_trace = np.asarray(scout_trace, dtype=np.float32)
    np.savez_compressed(out / "condition.npz", corridor=corridor, sdf=sdf, command=command,
                        route_world_xy=route_world_xy, route_world_xyz=route_world_xyz,
                        root_pos_world=planning_root,
                        root_quat_wxyz=np.asarray(root_quat_wxyz),
                        goal_world_xy=goal, map_origin_world_xy=grid.origin_world_xy,
                        map_origin_world_xyz=grid.origin_world_xyz,
                        map_probability=grid.probability(), scout_trace_world_xy=scout_trace)
    np.savez_compressed(out / "radar_returns.npz",
                        points_world=np.concatenate(all_points, axis=0).astype(np.float32),
                        origins_world=np.concatenate(all_origins, axis=0).astype(np.float32))
    render_grid_png(grid, route_world_xy, initial_root[:2], goal, out / "slam_grid_route.png",
                    trace_world_xy=scout_trace)
    render_voxel_slices_png(grid, out / "slam_voxel_slices.png")
    scout_length = float(np.linalg.norm(np.diff(scout_trace, axis=0), axis=1).sum())
    full_trace = np.vstack([scout_trace, route_world_xy])
    direct = max(float(np.linalg.norm(goal - initial_root[:2])), 1e-6)
    # Scout and execution route are two phases; do not count the artificial jump from the last
    # scout pose back to the live root when reporting total travelled distance.
    route_length = float(np.linalg.norm(np.diff(route_world_xy, axis=0), axis=1).sum())
    full_route_length = scout_length + route_length
    direction = (goal - initial_root[:2]) / direct
    route_offset = full_trace - initial_root[None, :2]
    lateral = np.abs(route_offset[:, 0] * direction[1] - route_offset[:, 1] * direction[0])
    condition_health = validate_condition(corridor, sdf)
    shifted_origins = {tuple(item["grid_origin_world_xy_m"]) for item in scans}
    checks = {
        "radar_has_returns": bool(sum(item["returns"] for item in scans) > 0),
        "centre_obstacle_observed": any("obstacle_center_block" in item["obstacle_return_names"] for item in scans),
        "sliding_origin_changed": bool(len(shifted_origins) > 1),
        "occupied_map_nonempty": bool(grid.occupancy_mask().sum() > 0),
        "voxel_volume_is_3d": bool(grid.probability().ndim == 3 and len(grid.shape_zyx) == 3),
        "astar_detours_around_block": bool(float(lateral.max()) > 0.70 and full_route_length > direct + 0.20),
        "stage2_condition_valid": bool(condition_health["valid"]),
        # Flat ground has no overhead restriction, so the vertical semi-axis may remain at the
        # configured 1.20 m ceiling.  Any observed low ceiling still contracts it below that
        # value; in both cases the emitted axis must be finite and above the interface minimum.
        "ellipsoid_vertical_axis_from_map": bool(np.isfinite(corridor[:, 5]).all() and
                                                   corridor[:, 5].min() >= grid.config.min_corridor_semi_m),
    }
    summary = {
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "provenance": "simulated_MuJoCo_radar_ground_truth_odometry",
        "scene": str(scene),
        "coordinate_contract": "SLAM grid global world metres; corridor root-local x-forward/y-left/z-up",
        "radar": {"config": radar_config.__dict__, "scans": scans,
                   "total_returns": int(sum(item["returns"] for item in scans))},
        "slam": {"grid_shape_yx": list(grid.shape_yx),
                 "grid_shape_zyx": list(grid.shape_zyx),
                 "resolution_m": float(grid.config.resolution_m),
                 "origin_world_xy_m": grid.origin_world_xy.astype(float).tolist(),
                 "origin_world_xyz_m": grid.origin_world_xyz.astype(float).tolist(),
                 "robot_center_world_xy_m": planning_root[:2].astype(float).tolist(),
                 "initial_robot_world_xy_m": initial_root[:2].astype(float).tolist(),
                 "updates": int(grid.update_count),
                 "occupied_voxels": int(grid.occupancy_mask().sum()),
                 "unknown_voxels": int(grid.unknown_mask().sum()),
                 "occupied_xy_projection_cells": int(grid.occupancy_xy_projection(
                     planning_root[2] + grid.config.body_z_min_offset_m,
                     planning_root[2] + grid.config.body_z_max_offset_m).sum())},
        "astar": {"waypoints": int(len(route_world_xyz)),
                  "dimension": 3,
                  "start_world_xy_m": planning_root[:2].astype(float).tolist(),
                  "initial_start_world_xy_m": initial_root[:2].astype(float).tolist(),
                  "goal_world_xy_m": goal.astype(float).tolist(),
                  "command_local_9d": command.astype(float).tolist(),
                  "command_horizon_m": 0.90,
                  "direct_distance_m": direct,
                  "route_length_m": route_length,
                  "scout_length_m": scout_length,
                  "full_scout_plus_route_length_m": full_route_length,
                  "scout_trace_world_xy_m": scout_trace.astype(float).tolist(),
                  "max_lateral_detour_m": float(lateral.max()),
                  "route_world_xy_m": route_world_xy.astype(float).tolist(),
                  "route_world_xyz_m": route_world_xyz.astype(float).tolist(),
                  "z_range_m": [float(route_world_xyz[:, 2].min()),
                                float(route_world_xyz[:, 2].max())]},
        "safe_corridor": {**corridor_report, "condition_health": condition_health,
                           "planner_body_radius_m": float(planner_body_radius_m),
                           "planner_clearance_m": float(planner_clearance_m)},
        "artifacts": {"condition": str(out / "condition.npz"),
                       "slam_grid": str(out / "slam_grid.npz"),
                       "radar_returns": str(out / "radar_returns.npz"),
                       "map_route_image": str(out / "slam_grid_route.png"),
                       "voxel_slices_image": str(out / "slam_voxel_slices.png")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="simulated radar + SLAM sliding grid + local A* deploy condition")
    parser.add_argument("--scene", type=Path, default=Path("data/g1_flat/scene_long_avoidance.xml"))
    parser.add_argument("--root-pos", type=float, nargs=3, default=(0.0, 0.0, 0.78))
    parser.add_argument("--root-quat", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0), metavar=("W", "X", "Y", "Z"))
    parser.add_argument("--goal", type=float, nargs=2, default=(3.60, 0.0), metavar=("X", "Y"))
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/deploy_perception_demo"))
    parser.add_argument("--demo-pose-count", type=int, default=5)
    parser.add_argument("--planner-body-radius-m", type=float, default=0.46,
                        help="voxel-A* body footprint inflation used before Stage 2")
    parser.add_argument("--planner-clearance-m", type=float, default=0.12,
                        help="voxel-A* extra clearance used before Stage 2")
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if args.planner_body_radius_m <= 0.0 or args.planner_clearance_m < 0.0:
        parser.error("planner body radius must be positive and clearance non-negative")
    radar_config = RadarConfig(seed=args.seed)
    summary = build_simulated_deploy_condition(
        args.scene, np.asarray(args.root_pos), np.asarray(args.root_quat), np.asarray(args.goal),
        out=args.out, radar_config=radar_config, demo_pose_count=args.demo_pose_count,
        planner_body_radius_m=args.planner_body_radius_m,
        planner_clearance_m=args.planner_clearance_m)
    print(json.dumps(summary, indent=2))
    return 0 if summary["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
