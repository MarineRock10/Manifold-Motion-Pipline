"""Incremental ESDF and D* Lite planning over a sliding 3-D SLAM map.

The map stays three dimensional: overhead clearance is consumed by ``M_e`` while the
ground-constrained G1 route is planned on an XY projection through the current body-height
band.  The ESDF updates only cells inside the truncated influence region of changed occupancy
evidence.  D* Lite keeps ``g/rhs`` values in *global grid coordinates*, so a moving local map
origin does not invalidate the search state.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from .deploy_perception import ProbabilisticSlidingVoxelGrid


Cell = tuple[int, int]
_INF = float("inf")


def _edt_1d(values: np.ndarray) -> np.ndarray:
    """Felzenszwalb-Huttenlocher squared Euclidean distance transform."""
    f = np.asarray(values, dtype=np.float64)
    n = int(len(f))
    finite = np.flatnonzero(np.isfinite(f))
    if not len(finite):
        return np.full(n, _INF, dtype=np.float64)
    vertices = np.empty(n, dtype=np.int64)
    boundaries = np.empty(n + 1, dtype=np.float64)
    k = 0
    vertices[0] = int(finite[0])
    boundaries[0], boundaries[1] = -_INF, _INF
    for q_value in finite[1:]:
        q = int(q_value)
        while True:
            p = int(vertices[k])
            s = ((f[q] + q * q) - (f[p] + p * p)) / (2.0 * (q - p))
            if s > boundaries[k] or k == 0:
                break
            k -= 1
        if s <= boundaries[k] and k == 0:
            # The new parabola dominates the first one everywhere to its right.
            vertices[0] = q
            boundaries[1] = _INF
            continue
        k += 1
        vertices[k] = q
        boundaries[k] = s
        boundaries[k + 1] = _INF
    result = np.empty(n, dtype=np.float64)
    k_eval = 0
    for q in range(n):
        while boundaries[k_eval + 1] < q:
            k_eval += 1
        p = int(vertices[k_eval])
        result[q] = (q - p) ** 2 + f[p]
    return result


def _euclidean_distance_cells(occupied: np.ndarray) -> np.ndarray:
    """Exact dependency-free 2-D EDT in cell units."""
    occupied = np.asarray(occupied, dtype=bool)
    if occupied.ndim != 2:
        raise ValueError("occupied map must be 2-D")
    if not np.any(occupied):
        return np.full(occupied.shape, _INF, dtype=np.float32)
    first = np.empty(occupied.shape, dtype=np.float64)
    base = np.where(occupied, 0.0, _INF)
    for x in range(occupied.shape[1]):
        first[:, x] = _edt_1d(base[:, x])
    result = np.empty_like(first)
    for y in range(occupied.shape[0]):
        result[y] = _edt_1d(first[y])
    return np.sqrt(result).astype(np.float32)


def _shift_into_new_window(array: np.ndarray, shift_xy: np.ndarray, fill: Any) -> np.ndarray:
    """Translate a local [y,x] array after the global sliding-window origin changes."""
    result = np.full_like(array, fill)
    dx, dy = int(shift_xy[0]), int(shift_xy[1])
    ny, nx = array.shape
    source_x0, source_x1 = max(0, dx), min(nx, nx + dx)
    source_y0, source_y1 = max(0, dy), min(ny, ny + dy)
    target_x0, target_x1 = max(0, -dx), min(nx, nx - dx)
    target_y0, target_y1 = max(0, -dy), min(ny, ny - dy)
    if source_x1 > source_x0 and source_y1 > source_y0:
        result[target_y0:target_y1, target_x0:target_x1] = array[
            source_y0:source_y1, source_x0:source_x1]
    return result


@dataclass(frozen=True)
class IncrementalPlannerConfig:
    body_radius_m: float = 0.40
    clearance_m: float = 0.10
    body_half_height_m: float = 0.78
    ground_projection_height_m: float = 0.45
    esdf_truncation_m: float = 1.20
    risk_weight: float = 0.50
    unknown_penalty: float = 0.35
    route_preference_weight: float = 2.0
    allow_unknown: bool = True
    diagonal_motion: bool = True
    max_compute_steps: int = 250_000

    def validate(self) -> None:
        if min(self.body_radius_m, self.clearance_m, self.body_half_height_m,
               self.ground_projection_height_m,
               self.esdf_truncation_m, self.max_compute_steps) <= 0:
            raise ValueError("incremental planner geometry and budget must be positive")
        if min(self.risk_weight, self.unknown_penalty, self.route_preference_weight) < 0:
            raise ValueError("incremental planner weights must be non-negative")


class IncrementalTruncatedESDF:
    """Exact truncated ESDF with dirty-region updates and sliding-window alignment."""

    def __init__(self, resolution_m: float, truncation_m: float):
        if resolution_m <= 0 or truncation_m <= 0:
            raise ValueError("ESDF resolution/truncation must be positive")
        self.resolution_m = float(resolution_m)
        self.truncation_m = float(truncation_m)
        self.radius_cells = int(math.ceil(truncation_m / resolution_m))
        self.occupied: np.ndarray | None = None
        self.distance_m: np.ndarray | None = None
        self.origin_cell_xy: np.ndarray | None = None

    def update(self, occupied: np.ndarray, origin_cell_xy: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter()
        current = np.asarray(occupied, dtype=bool)
        origin = np.asarray(origin_cell_xy, dtype=np.int64).reshape(2)
        full = self.occupied is None or self.occupied.shape != current.shape
        shifted = False
        if full:
            previous = np.zeros_like(current)
            changed = np.ones_like(current, dtype=bool)
            distance = _euclidean_distance_cells(current)
        else:
            assert self.distance_m is not None and self.origin_cell_xy is not None
            shift = origin - self.origin_cell_xy
            shifted = bool(np.any(shift))
            previous = _shift_into_new_window(self.occupied, shift, False)
            distance = _shift_into_new_window(
                self.distance_m, shift, np.float32(self.truncation_m))
            changed = previous != current
            changed_count = int(changed.sum())
            if changed_count:
                ys, xs = np.nonzero(changed)
                r = self.radius_cells
                core_y0, core_y1 = max(0, int(ys.min()) - r), min(current.shape[0], int(ys.max()) + r + 1)
                core_x0, core_x1 = max(0, int(xs.min()) - r), min(current.shape[1], int(xs.max()) + r + 1)
                # An obstacle outside the affected core can still be the nearest source.  The
                # second halo makes the recomputation exact up to the truncation radius.
                patch_y0, patch_y1 = max(0, core_y0 - r), min(current.shape[0], core_y1 + r)
                patch_x0, patch_x1 = max(0, core_x0 - r), min(current.shape[1], core_x1 + r)
                patch = current[patch_y0:patch_y1, patch_x0:patch_x1]
                patch_distance = _euclidean_distance_cells(patch) * self.resolution_m
                sy0, sy1 = core_y0 - patch_y0, core_y1 - patch_y0
                sx0, sx1 = core_x0 - patch_x0, core_x1 - patch_x0
                distance[core_y0:core_y1, core_x0:core_x1] = patch_distance[sy0:sy1, sx0:sx1]
        distance = np.minimum(distance * (self.resolution_m if full else 1.0),
                              self.truncation_m).astype(np.float32)
        self.occupied = current.copy()
        self.distance_m = distance
        self.origin_cell_xy = origin.copy()
        return {
            "changed_cells": int(changed.sum()),
            "window_shifted": shifted,
            "full_rebuild": bool(full),
            "update_ms": float((time.perf_counter() - started) * 1000.0),
            "truncation_m": self.truncation_m,
        }


class DStarLitePlanner:
    """D* Lite over global XY cells backed by an incremental truncated ESDF."""

    _MOVES_8 = ((1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1))
    _MOVES_4 = _MOVES_8[:4]

    def __init__(self, resolution_m: float, config: IncrementalPlannerConfig):
        config.validate()
        self.resolution_m = float(resolution_m)
        self.config = config
        self.esdf = IncrementalTruncatedESDF(resolution_m, config.esdf_truncation_m)
        self.g: dict[Cell, float] = {}
        self.rhs: dict[Cell, float] = {}
        self.queue: list[tuple[float, float, int, Cell]] = []
        self.queue_version: dict[Cell, int] = {}
        self.version = 0
        self.km = 0.0
        self.start: Cell | None = None
        self.last_start: Cell | None = None
        self.goal: Cell | None = None
        self.origin_cell_xy = np.zeros(2, dtype=np.int64)
        self.blocked: np.ndarray | None = None
        self.probability: np.ndarray | None = None
        self.unknown: np.ndarray | None = None
        self.preference: np.ndarray | None = None
        self.last_report: dict[str, Any] = {}

    def _value(self, table: dict[Cell, float], cell: Cell) -> float:
        return float(table.get(cell, _INF))

    @staticmethod
    def _heuristic(a: Cell, b: Cell) -> float:
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        diagonal = min(dx, dy)
        return float(diagonal * math.sqrt(2.0) + max(dx, dy) - diagonal)

    def _key(self, cell: Cell) -> tuple[float, float]:
        assert self.start is not None
        minimum = min(self._value(self.g, cell), self._value(self.rhs, cell))
        return minimum + self._heuristic(self.start, cell) + self.km, minimum

    def _push(self, cell: Cell) -> None:
        self.version += 1
        self.queue_version[cell] = self.version
        key = self._key(cell)
        heapq.heappush(self.queue, (key[0], key[1], self.version, cell))

    def _remove(self, cell: Cell) -> None:
        self.queue_version.pop(cell, None)

    def _peek(self) -> tuple[tuple[float, float], Cell | None]:
        while self.queue:
            k1, k2, version, cell = self.queue[0]
            if self.queue_version.get(cell) != version:
                heapq.heappop(self.queue)
                continue
            return (float(k1), float(k2)), cell
        return (_INF, _INF), None

    def _pop(self) -> tuple[tuple[float, float], Cell | None]:
        key, cell = self._peek()
        if cell is None:
            return key, None
        heapq.heappop(self.queue)
        self.queue_version.pop(cell, None)
        return key, cell

    def _inside(self, cell: Cell) -> bool:
        if self.blocked is None:
            return False
        x, y = cell[0] - int(self.origin_cell_xy[0]), cell[1] - int(self.origin_cell_xy[1])
        return 0 <= x < self.blocked.shape[1] and 0 <= y < self.blocked.shape[0]

    def _local(self, cell: Cell) -> tuple[int, int]:
        return cell[0] - int(self.origin_cell_xy[0]), cell[1] - int(self.origin_cell_xy[1])

    def _is_blocked(self, cell: Cell) -> bool:
        if not self._inside(cell):
            return True
        x, y = self._local(cell)
        return bool(self.blocked[y, x])

    def _neighbors(self, cell: Cell) -> Iterable[Cell]:
        moves = self._MOVES_8 if self.config.diagonal_motion else self._MOVES_4
        for dx, dy in moves:
            nxt = (cell[0] + dx, cell[1] + dy)
            if self._inside(nxt):
                yield nxt

    def _cost(self, source: Cell, target: Cell) -> float:
        if self._is_blocked(source) or self._is_blocked(target):
            return _INF
        dx, dy = target[0] - source[0], target[1] - source[1]
        step = math.sqrt(float(dx * dx + dy * dy))
        x, y = self._local(target)
        assert self.probability is not None and self.unknown is not None
        risk = self.config.risk_weight * float(self.probability[y, x])
        uncertainty = self.config.unknown_penalty if bool(self.unknown[y, x]) else 0.0
        preference = (self.config.route_preference_weight * float(self.preference[y, x])
                      if self.preference is not None else 0.0)
        return step + risk + uncertainty + preference

    def _update_vertex(self, cell: Cell) -> None:
        assert self.goal is not None
        if cell != self.goal:
            values = [self._cost(cell, successor) + self._value(self.g, successor)
                      for successor in self._neighbors(cell)]
            self.rhs[cell] = min(values, default=_INF)
        self._remove(cell)
        if not math.isclose(self._value(self.g, cell), self._value(self.rhs, cell),
                            rel_tol=1e-10, abs_tol=1e-10):
            self._push(cell)

    def _initialize(self, start: Cell, goal: Cell) -> None:
        self.g.clear(); self.rhs.clear(); self.queue.clear(); self.queue_version.clear()
        self.version = 0; self.km = 0.0
        self.start = self.last_start = start
        self.goal = goal
        self.rhs[goal] = 0.0
        self._push(goal)

    def _compute_shortest_path(self) -> int:
        assert self.start is not None
        expanded = 0
        while True:
            top_key, _ = self._peek()
            start_key = self._key(self.start)
            consistent = math.isclose(self._value(self.rhs, self.start), self._value(self.g, self.start),
                                      rel_tol=1e-10, abs_tol=1e-10)
            if not (top_key < start_key or not consistent):
                break
            old_key, cell = self._pop()
            if cell is None:
                break
            expanded += 1
            if expanded > self.config.max_compute_steps:
                raise RuntimeError("D* Lite compute budget exceeded")
            new_key = self._key(cell)
            if old_key < new_key:
                self._push(cell)
            elif self._value(self.g, cell) > self._value(self.rhs, cell):
                self.g[cell] = self._value(self.rhs, cell)
                for predecessor in self._neighbors(cell):
                    self._update_vertex(predecessor)
            else:
                self.g[cell] = _INF
                self._update_vertex(cell)
                for predecessor in self._neighbors(cell):
                    self._update_vertex(predecessor)
        return expanded

    def _nearest_free(self, cell: Cell) -> Cell:
        if self._inside(cell) and not self._is_blocked(cell):
            return cell
        if self.blocked is None:
            raise RuntimeError("planner map is uninitialized")
        free_yx = np.argwhere(~self.blocked)
        if not len(free_yx):
            raise RuntimeError("incremental map contains no free cell")
        globals_xy = np.column_stack([
            free_yx[:, 1] + int(self.origin_cell_xy[0]),
            free_yx[:, 0] + int(self.origin_cell_xy[1]),
        ])
        index = int(np.argmin(np.sum((globals_xy - np.asarray(cell)[None, :]) ** 2, axis=1)))
        return int(globals_xy[index, 0]), int(globals_xy[index, 1])

    def _extract_path(self) -> list[Cell]:
        assert self.start is not None and self.goal is not None
        if not np.isfinite(self._value(self.g, self.start)):
            raise RuntimeError("D* Lite could not connect start and goal")
        path = [self.start]
        seen = {self.start}
        limit = int(np.prod(self.blocked.shape)) if self.blocked is not None else 0
        while path[-1] != self.goal and len(path) <= limit:
            current = path[-1]
            candidates = [(self._cost(current, successor) + self._value(self.g, successor), successor)
                          for successor in self._neighbors(current)]
            candidates = [(cost, cell) for cost, cell in candidates if np.isfinite(cost)]
            if not candidates:
                raise RuntimeError("D* Lite path extraction reached a dead end")
            _, successor = min(candidates, key=lambda item: (item[0], self._heuristic(item[1], self.goal)))
            if successor in seen:
                raise RuntimeError("D* Lite path extraction found a cycle")
            path.append(successor); seen.add(successor)
        if path[-1] != self.goal:
            raise RuntimeError("D* Lite path extraction exceeded map size")
        return path

    @staticmethod
    def _preference_field(shape: tuple[int, int], origin: np.ndarray, resolution: float,
                          preferred_world_xy: np.ndarray | None) -> np.ndarray | None:
        if preferred_world_xy is None:
            return None
        preferred = np.asarray(preferred_world_xy, dtype=np.float64)
        if preferred.ndim != 2 or preferred.shape[1] != 2 or not len(preferred):
            raise ValueError("preferred_world_xy must be a non-empty [N,2] route")
        y, x = np.indices(shape)
        cells = np.column_stack([x.ravel() + origin[0], y.ravel() + origin[1]])
        world = (cells.astype(np.float64) + 0.5) * resolution
        distance = np.sqrt(np.min(np.sum((world[:, None, :] - preferred[None, :, :]) ** 2,
                                         axis=2), axis=1))
        return distance.reshape(shape).astype(np.float32)

    def plan(self, grid: ProbabilisticSlidingVoxelGrid, start_world_xyz: np.ndarray,
             goal_world_xyz: np.ndarray, *, preferred_world_xy: np.ndarray | None = None
             ) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        start_xyz = np.asarray(start_world_xyz, dtype=np.float64).reshape(3)
        goal_xyz = np.asarray(goal_world_xyz, dtype=np.float64).reshape(3)
        # D* Lite searches a ground-bound XY route.  Projecting the complete robot-height band
        # here makes a low ceiling look like a vertical wall and causes the route to detour
        # around a passage that is actually traversable by crouching.  Keep the 3-D map intact
        # for M_e/vertical clearance; only the near-ground obstacle band feeds XY occupancy.
        ground_z = start_xyz[2] - self.config.body_half_height_m
        z_min = ground_z - self.config.clearance_m
        z_max = ground_z + self.config.ground_projection_height_m
        occupied = grid.occupancy_xy_projection(z_min, z_max)
        unknown = grid.unknown_xy_projection(z_min, z_max)
        probability = grid.probability_xy_projection(z_min, z_max)
        origin = grid.origin_cell_xyz[:2].astype(np.int64)
        esdf_report = self.esdf.update(occupied, origin)
        assert self.esdf.distance_m is not None
        blocked = self.esdf.distance_m <= (self.config.body_radius_m + self.config.clearance_m)
        if not self.config.allow_unknown:
            blocked = blocked | unknown
        old_origin = self.origin_cell_xy.copy()
        old_blocked = self.blocked
        old_probability = self.probability
        old_unknown = self.unknown
        self.origin_cell_xy = origin.copy()
        self.blocked = blocked
        self.probability = probability
        self.unknown = unknown
        self.preference = self._preference_field(blocked.shape, origin, self.resolution_m,
                                                 preferred_world_xy)
        start_cell = tuple(np.floor(start_xyz[:2] / self.resolution_m).astype(np.int64).tolist())
        goal_cell = tuple(np.floor(goal_xyz[:2] / self.resolution_m).astype(np.int64).tolist())
        # The robot's measured collision-free state is authoritative. If the robot is already
        # inside an obstacle's conservative inflation band, clearing only its centre cell traps
        # D* inside a ring of blocked neighbors. Clear the current footprint disk (the same
        # egress rule used by the previous voxel A*) while the exact 50 Hz body-surface gate
        # remains responsible for the real obstacle boundary.
        local_start = (start_cell[0] - int(origin[0]), start_cell[1] - int(origin[1]))
        if 0 <= local_start[0] < blocked.shape[1] and 0 <= local_start[1] < blocked.shape[0]:
            clear_radius = int(math.ceil(
                (self.config.body_radius_m + self.config.clearance_m) / self.resolution_m))
            for dy in range(-clear_radius, clear_radius + 1):
                for dx in range(-clear_radius, clear_radius + 1):
                    if dx * dx + dy * dy > clear_radius * clear_radius:
                        continue
                    x, y = local_start[0] + dx, local_start[1] + dy
                    if 0 <= x < blocked.shape[1] and 0 <= y < blocked.shape[0]:
                        blocked[y, x] = False
        start_cell = self._nearest_free(start_cell)
        goal_cell = self._nearest_free(goal_cell)
        reinitialized = self.goal is None or self.goal != goal_cell
        changed_global: set[Cell] = set()
        if reinitialized:
            self._initialize(start_cell, goal_cell)
        else:
            assert self.last_start is not None and self.start is not None
            self.km += self._heuristic(self.last_start, start_cell)
            self.start = start_cell
            self.last_start = start_cell
            # Align old planner costs into the new map window, then update only changed cells.
            if old_blocked is None:
                changed = np.ones_like(blocked, dtype=bool)
            else:
                shift = origin - old_origin
                aligned_blocked = _shift_into_new_window(old_blocked, shift, True)
                aligned_probability = _shift_into_new_window(old_probability, shift, np.float32(0.5))
                aligned_unknown = _shift_into_new_window(old_unknown, shift, True)
                changed = (aligned_blocked != blocked)
                # D* Lite assumes every edge-cost change is reported.  Even a small log-odds
                # update changes the risk term; suppressing sub-0.04 deltas accumulates stale
                # rhs values and can eventually produce a false disconnected map.
                changed |= np.abs(aligned_probability - probability) > 1.0e-6
                changed |= aligned_unknown != unknown
            for y, x in np.argwhere(changed):
                changed_global.add((int(x + origin[0]), int(y + origin[1])))
            affected = set(changed_global)
            for cell in list(changed_global):
                affected.update(self._neighbors(cell))
            for cell in affected:
                self._update_vertex(cell)
        expanded = self._compute_shortest_path()
        repaired = False
        try:
            path_cells = self._extract_path()
        except RuntimeError:
            # Defensive exact repair. Incremental updates are the normal path, but a finite
            # sliding window can invalidate a large fringe at once; rebuilding D* state from
            # the same ESDF is preferable to falsely reporting that the physical map is closed.
            self._initialize(start_cell, goal_cell)
            expanded += self._compute_shortest_path()
            path_cells = self._extract_path()
            repaired = True
        route_xy = (np.asarray(path_cells, dtype=np.float64) + 0.5) * self.resolution_m
        route = np.column_stack([route_xy, np.full(len(route_xy), start_xyz[2])]).astype(np.float32)
        report = {
            "planner": "D* Lite",
            "esdf": esdf_report,
            "reinitialized": bool(reinitialized),
            "repair_reinitialized": bool(repaired),
            "changed_cost_cells": int(len(changed_global)),
            "expanded_vertices": int(expanded),
            "route_cells": int(len(path_cells)),
            "route_length_m": float(np.linalg.norm(np.diff(route_xy, axis=0), axis=1).sum()),
            "planning_ms": float((time.perf_counter() - started) * 1000.0),
            "start_global_cell": list(start_cell),
            "goal_global_cell": list(goal_cell),
            "config": asdict(self.config),
            "map_origin_global_cell_xy": origin.astype(int).tolist(),
        }
        self.last_report = report
        return route, report


def incremental_route_from_grid(grid: ProbabilisticSlidingVoxelGrid,
                                start_world_xyz: np.ndarray, goal_world_xyz: np.ndarray,
                                *, planner: DStarLitePlanner | None = None,
                                config: IncrementalPlannerConfig | None = None,
                                preferred_world_xy: np.ndarray | None = None,
                                ) -> tuple[np.ndarray, DStarLitePlanner, dict[str, Any]]:
    """Convenience entry point retaining the planner object for the next sensor frame."""
    config = IncrementalPlannerConfig() if config is None else config
    planner = (DStarLitePlanner(grid.config.resolution_m, config)
               if planner is None else planner)
    route, report = planner.plan(grid, start_world_xyz, goal_world_xyz,
                                 preferred_world_xy=preferred_world_xy)
    return route, planner, report
