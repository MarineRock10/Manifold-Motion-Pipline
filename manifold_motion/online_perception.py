"""Online radar, sliding-voxel SLAM, incremental planning and ``M_e`` routing.

The map remains a 3-D probability volume.  A body-height projection feeds incremental ESDF
and D* Lite, while the full volume constructs the route-local ellipsoid corridor consumed by
Stage 2. Primitive decisions are debounced across sensor updates before reaching the executor.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .deploy_perception import (
    ProbabilisticSlidingVoxelGrid,
    RadarConfig,
    SimulatedRadar,
    SlidingGridConfig,
    safe_corridor_from_grid,
)
from .incremental_planner import DStarLitePlanner, IncrementalPlannerConfig
from .dynamic_scene import obstacle_state


def _lookahead_yaw(route: np.ndarray, position_xy: np.ndarray, distance_m: float) -> float:
    points = np.asarray(route, dtype=np.float64)[:, :2]
    position = np.asarray(position_xy, dtype=np.float64)
    if len(points) < 2:
        delta = points[-1] - position
        return float(math.atan2(delta[1], delta[0]))
    nearest = int(np.argmin(np.linalg.norm(points - position[None, :], axis=1)))
    remaining = points[nearest:]
    if len(remaining) < 2:
        target = points[-1]
    else:
        lengths = np.linalg.norm(np.diff(remaining, axis=0), axis=1)
        target_index = min(int(np.searchsorted(np.cumsum(lengths), distance_m, side="left") + 1),
                           len(remaining) - 1)
        target = remaining[target_index]
    delta = target - position
    if float(np.linalg.norm(delta)) < 1e-6:
        delta = points[-1] - position
    return float(math.atan2(delta[1], delta[0]))


def _bilateral_lateral_free_semi_from_radar(
    route_world_xyz: np.ndarray,
    radar_points_world: np.ndarray,
    root_z: float,
    *,
    cap_m: float = 0.70,
    surface_margin_m: float = 0.18,
    longitudinal_window_m: float = 0.32,
    min_points_per_side: int = 2,
    min_bilateral_stations: int = 4,
) -> float:
    """Estimate bilateral lateral aperture from the *local* radar surfaces.

    ``safe_corridor_from_grid`` reports a symmetric lateral semi-axis.  Feeding that value
    directly to the semantic router makes a one-sided pole look like a doorway, especially when
    the sliding grid has accumulated hits from several route positions.  This query retains the
    signed route-frame evidence: a side gait is eligible only when surfaces are observed on both
    sides of the same short route interval.  A vertical band rejects overhead returns, while the
    exact mesh/self-manifold gate remains authoritative after the decision.
    """
    route = np.asarray(route_world_xyz, dtype=np.float64).reshape(-1, 3)
    points = np.asarray(radar_points_world, dtype=np.float64).reshape(-1, 3)
    if len(route) < 2 or len(points) == 0:
        return float(cap_m)
    # Use body-level returns only.  The lower band retains side-wall feet/torso returns; the
    # upper cutoff prevents a ceiling underside from becoming a bilateral lateral wall.
    points = points[
        (points[:, 2] >= float(root_z) - 0.55)
        & (points[:, 2] <= float(root_z) + 0.08)
    ]
    if len(points) < 2 * min_points_per_side:
        return float(cap_m)
    sample_indices = np.unique(np.linspace(0, len(route) - 1,
                                           min(32, len(route)), dtype=int))
    candidates: list[float] = []
    for index in sample_indices:
        if index == 0:
            delta = route[1, :2] - route[0, :2]
        elif index == len(route) - 1:
            delta = route[-1, :2] - route[-2, :2]
        else:
            delta = route[index + 1, :2] - route[index - 1, :2]
        norm = float(np.linalg.norm(delta))
        if norm <= 1.0e-8:
            continue
        tangent = delta / norm
        lateral_axis = np.array([-tangent[1], tangent[0]])
        offset = points[:, :2] - route[index, :2]
        longitudinal = offset @ tangent
        lateral = offset @ lateral_axis
        nearby = ((np.abs(longitudinal) <= longitudinal_window_m)
                  & (np.abs(lateral) <= cap_m + surface_margin_m))
        signed = lateral[nearby]
        left = signed[signed > 0.0]
        right = -signed[signed < 0.0]
        if len(left) < min_points_per_side or len(right) < min_points_per_side:
            continue
        left_clearance = max(0.06, float(left.min()) - surface_margin_m)
        right_clearance = max(0.06, float(right.min()) - surface_margin_m)
        if left_clearance < cap_m and right_clearance < cap_m:
            candidates.append(min(left_clearance, right_clearance))
    # A real passage persists over several adjacent route samples.  Requiring four stations
    # rejects the momentary left/right pairing produced by two longitudinally staggered poles.
    return (float(min(candidates))
            if len(candidates) >= min_bilateral_stations else float(cap_m))


class OnlinePerceptionNavigator:
    """Update P1 and provide a D* Lite route, ``M_e`` and primitive decision."""

    def __init__(self, scene: Path, *, seed_grid: Path | None = None,
                 scan_ticks: int = 20, lookahead_m: float = 0.45,
                 body_radius_m: float = 0.40, clearance_m: float = 0.10,
                 side_body_radius_m: float | None = None,
                 route_preference_weight: float = 2.0,
                 crouch_semi_z_m: float = 1.12, side_semi_y_m: float = 0.40,
                 decision_confirm_updates: int = 2,
                 decision_release_confirm_updates: int = 4,
                 initial_primitive_id: int = 5,
                 vertical_lookahead_m: float = 1.20,
                 dynamic_event: str | None = None, seed: int = 20260923):
        if scan_ticks <= 0 or lookahead_m <= 0 or body_radius_m <= 0 or clearance_m <= 0:
            raise ValueError("online perception cadence and geometry must be positive")
        if decision_confirm_updates <= 0:
            raise ValueError("decision_confirm_updates must be positive")
        if decision_release_confirm_updates < decision_confirm_updates:
            raise ValueError(
                "decision_release_confirm_updates must be no smaller than decision_confirm_updates")
        if initial_primitive_id not in (2, 4, 5):
            raise ValueError("initial_primitive_id must be crouch(2), side(4), or nominal(5)")
        if vertical_lookahead_m < 0.90:
            raise ValueError("vertical_lookahead_m must cover the nominal online horizon")
        self.scene = Path(scene)
        self.seed_grid = Path(seed_grid) if seed_grid is not None else None
        self.scan_ticks = int(scan_ticks)
        self.lookahead_m = float(lookahead_m)
        self.body_radius_m = float(body_radius_m)
        self.side_body_radius_m = float(
            min(body_radius_m, 0.30) if side_body_radius_m is None else side_body_radius_m)
        if not 0.0 < self.side_body_radius_m <= self.body_radius_m:
            raise ValueError("side body radius must be positive and no larger than nominal radius")
        self.active_body_radius_m = self.body_radius_m
        self.clearance_m = float(clearance_m)
        self.route_preference_weight = float(route_preference_weight)
        self.crouch_semi_z_m = float(crouch_semi_z_m)
        self.side_semi_y_m = float(side_semi_y_m)
        self.decision_confirm_updates = int(decision_confirm_updates)
        self.decision_release_confirm_updates = int(decision_release_confirm_updates)
        self.vertical_lookahead_m = float(vertical_lookahead_m)
        self.dynamic_event = dynamic_event
        self.config = SlidingGridConfig()
        self.radar = SimulatedRadar(
            self.scene, RadarConfig(seed=int(seed)), dynamic_event=dynamic_event)
        self.grid: ProbabilisticSlidingVoxelGrid | None = None
        self.planner: DStarLitePlanner | None = None
        self.initial_update_count = 0
        self.last_route: np.ndarray | None = None
        self.last_corridor: np.ndarray | None = None
        self.last_sdf: np.ndarray | None = None
        self.last_decision: dict[str, Any] | None = None
        # Start from the posture selected by the initial P1 manifold instead of silently
        # expanding to nominal walk while the first radar scans are still incomplete.
        self.stable_primitive_id = int(initial_primitive_id)
        self.pending_primitive_id: int | None = None
        self.pending_primitive_updates = 0
        self.failure: str | None = None
        self.temporary_failures: list[dict[str, Any]] = []
        self.max_temporary_no_route_updates = 8
        self.override_ticks = 0
        self.update_ticks: list[int] = []
        self.pose_history: list[np.ndarray] = []
        self.origin_history: list[np.ndarray] = []
        self.map_history: list[np.ndarray] = []
        self.route_history: list[np.ndarray] = []
        self.yaw_history: list[float] = []
        self.route_length_history: list[float] = []
        self.occupied_history: list[int] = []
        self.scan_points: list[np.ndarray] = []
        self.scan_origins: list[np.ndarray] = []
        self.corridor_history: list[np.ndarray] = []
        self.sdf_history: list[np.ndarray] = []
        self.primitive_history: list[int] = []
        self.raw_primitive_history: list[int] = []
        self.planner_history: list[dict[str, Any]] = []
        self.body_radius_history: list[float] = []
        self.planner_reinitializations: list[dict[str, Any]] = []

    def _initialize_grid(self, position: np.ndarray) -> None:
        self.grid = ProbabilisticSlidingVoxelGrid(self.config, position)
        if self.seed_grid is None:
            return
        with np.load(self.seed_grid) as archive:
            required = {"log_odds", "origin_cell_xyz", "robot_xyz", "update_count"}
            if not required.issubset(archive.files):
                raise ValueError(f"seed SLAM grid is missing {sorted(required - set(archive.files))}")
            log_odds = np.asarray(archive["log_odds"], dtype=np.float32)
            if log_odds.shape != self.grid.shape_zyx:
                raise ValueError(f"seed SLAM shape {log_odds.shape} != {self.grid.shape_zyx}")
            self.grid.log_odds = log_odds.copy()
            self.grid.origin_cell_xyz = np.asarray(archive["origin_cell_xyz"], dtype=np.int64).copy()
            self.grid.robot_xyz = np.asarray(archive["robot_xyz"], dtype=np.float64).copy()
            self.grid.update_count = int(np.asarray(archive["update_count"]).item())
        self.initial_update_count = int(self.grid.update_count)
        self.grid.recenter(position)

    def _primitive_decision(self, corridor: np.ndarray,
                            bilateral_lateral_semi_m: float | None = None,
                            dynamic_lateral_blocker: bool = False) -> dict[str, Any]:
        vertical = float(np.min(corridor[:, 5]))
        lateral = float(np.min(corridor[:, 4]))
        bilateral_lateral = lateral if bilateral_lateral_semi_m is None else float(bilateral_lateral_semi_m)
        # A moving wall is a lateral blocker, not a ceiling.  Prefer the compact side gait
        # when the synchronized dynamic event is close; otherwise a tall radar return can make
        # the vertical test win and commit a crouch clip that has no forward progress.
        if dynamic_lateral_blocker and (
                vertical < self.crouch_semi_z_m or bilateral_lateral < self.side_semi_y_m):
            raw, reason = 4, "dynamic_lateral_blocker_prefers_side"
        elif (getattr(self, "dynamic_event", None) == "moving_wall" and self.stable_primitive_id == 4
              and vertical < self.crouch_semi_z_m):
            # The probability map deliberately decays slower than one radar period.  Once the
            # moving-wall response has contracted laterally, keep that safe mode while stale
            # vertical returns clear instead of reinterpreting the same wall as a ceiling.
            raw, reason = 4, "dynamic_wall_side_hysteresis_over_stale_vertical_returns"
        elif vertical < self.crouch_semi_z_m:
            raw, reason = 2, "vertical_free_semi_below_crouch_threshold"
        elif bilateral_lateral < self.side_semi_y_m:
            raw, reason = 4, "bilateral_lateral_free_semi_below_side_threshold"
        elif (self.stable_primitive_id == 4
              and lateral < self.side_semi_y_m + 0.12):
            # Entry needs fresh signed evidence from both sides.  Once inside a gate, however,
            # a forward-facing radar can temporarily lose the walls behind its field of view.
            # Retain the compact gait until the accumulated probability-map corridor itself
            # widens; otherwise the robot expands to nominal posture while still in the gate.
            raw, reason = 4, "side_gait_release_hysteresis_until_corridor_widens"
        else:
            raw, reason = 5, "wide_and_tall_enough_for_nominal_walk"
        previous = int(self.stable_primitive_id)
        if raw == previous:
            self.pending_primitive_id = None
            self.pending_primitive_updates = 0
        elif self.pending_primitive_id == raw:
            self.pending_primitive_updates += 1
        else:
            self.pending_primitive_id = raw
            self.pending_primitive_updates = 1
        # Contracting the body is safety-positive and may be committed quickly.  Expanding
        # from crouch/side to nominal needs more consecutive observations because an unseen
        # ceiling/wall initially looks like free space to a forward-looking radar.
        required_confirm_updates = int(
            self.decision_release_confirm_updates
            if raw == 5 and previous in (2, 4)
            else self.decision_confirm_updates
        )
        committed = False
        if (self.pending_primitive_id is not None
                and self.pending_primitive_updates >= required_confirm_updates):
            self.stable_primitive_id = int(self.pending_primitive_id)
            self.pending_primitive_id = None
            self.pending_primitive_updates = 0
            committed = True
        names = {2: "crouch", 4: "walk_lateral_reverse", 5: "walk_nominal"}
        return {
            "raw_primitive_id": int(raw), "primitive_id": int(self.stable_primitive_id),
            "primitive": names[int(self.stable_primitive_id)], "reason": reason,
            "primitive_changed": bool(committed), "previous_primitive_id": previous,
            "pending_primitive_id": self.pending_primitive_id,
            "pending_updates": int(self.pending_primitive_updates),
            "confirm_updates": required_confirm_updates,
            "contraction_confirm_updates": int(self.decision_confirm_updates),
            "release_confirm_updates": int(self.decision_release_confirm_updates),
            "vertical_free_semi_min_m": vertical,
            "lateral_free_semi_min_m": lateral,
            "bilateral_lateral_free_semi_min_m": bilateral_lateral,
            "thresholds": {"crouch_semi_z_m": self.crouch_semi_z_m,
                           "side_semi_y_m": self.side_semi_y_m,
                           "side_release_semi_y_m": self.side_semi_y_m + 0.12},
        }

    def __call__(self, tick: int, state: dict[str, np.ndarray], goal_world_xy: np.ndarray,
                 preferred_world_route: np.ndarray) -> dict[str, Any]:
        position = np.asarray(state["base_pos"], dtype=np.float64)
        quat = np.asarray(state["base_quat"], dtype=np.float64)
        if self.grid is None:
            self._initialize_grid(position)
        assert self.grid is not None
        updated = tick == 0 or tick % self.scan_ticks == 0
        if updated:
            try:
                scan = self.radar.scan(position, quat, timestamp=float(tick) * 0.02)
                self.grid.update_radar(position, scan)
                goal = np.array([goal_world_xy[0], goal_world_xy[1], position[2]], dtype=np.float64)
                preferred_xyz = np.column_stack([
                    np.asarray(preferred_world_route, dtype=np.float64)[:, :2],
                    np.full(len(preferred_world_route), position[2]),
                ])
                preplan_bilateral = _bilateral_lateral_free_semi_from_radar(
                    preferred_xyz, scan.points_world, float(position[2]),
                    cap_m=max(0.70, self.side_semi_y_m + 0.30),
                )
                compact_requested = bool(
                    preplan_bilateral < self.side_semi_y_m or self.stable_primitive_id == 4)
                requested_radius = (self.side_body_radius_m
                                    if compact_requested else self.body_radius_m)
                if self.planner is None or abs(requested_radius - self.active_body_radius_m) > 1e-9:
                    if self.planner is not None:
                        self.planner_reinitializations.append({
                            "tick": int(tick),
                            "from_body_radius_m": float(self.active_body_radius_m),
                            "to_body_radius_m": float(requested_radius),
                            "reason": ("bilateral_side_affordance"
                                       if compact_requested else "corridor_widened"),
                        })
                    self.active_body_radius_m = float(requested_radius)
                    self.planner = DStarLitePlanner(
                        self.grid.config.resolution_m,
                        IncrementalPlannerConfig(
                            body_radius_m=self.active_body_radius_m,
                            clearance_m=self.clearance_m,
                            body_half_height_m=0.78,
                            route_preference_weight=self.route_preference_weight,
                        ),
                    )
                route, planner_report = self.planner.plan(
                    self.grid, position, goal,
                    preferred_world_xy=np.asarray(preferred_world_route)[:, :2],
                )
                route[:, 2] = position[2]
                corridor, sdf, corridor_report = safe_corridor_from_grid(
                    self.grid, route, position, quat, vertical_semi_m=1.20,
                    clearance_m=self.clearance_m, frames=48,
                    # The vertical aperture must look far enough ahead to see a ceiling
                    # before the robot's head enters it; the route planner still uses the
                    # shorter local horizon for latency, while M_e is anticipatory.
                    horizon_m=self.vertical_lookahead_m,
                )
                bilateral_lateral = _bilateral_lateral_free_semi_from_radar(
                    route, scan.points_world, float(position[2]),
                    cap_m=max(0.70, self.side_semi_y_m + 0.30),
                )
                dynamic_state = obstacle_state(self.dynamic_event, tick * 0.02)
                dynamic_lateral_blocker = bool(
                    dynamic_state is not None and dynamic_state.active
                    and abs(float(dynamic_state.center_xy[0] - position[0])) <= 1.25
                    and abs(float(dynamic_state.center_xy[1] - position[1])) <= 0.95
                )
                decision = self._primitive_decision(
                    corridor, bilateral_lateral,
                    dynamic_lateral_blocker=dynamic_lateral_blocker,
                )
                decision["dynamic_lateral_blocker"] = dynamic_lateral_blocker
                yaw = _lookahead_yaw(route, position[:2], self.lookahead_m)
                length = float(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1).sum())
                self.last_route = route.astype(np.float32)
                self.last_corridor = corridor.astype(np.float32)
                self.last_sdf = sdf.astype(np.float32)
                self.last_decision = decision
                self.update_ticks.append(int(tick))
                self.pose_history.append(position.astype(np.float32))
                self.origin_history.append(self.grid.origin_world_xyz.astype(np.float32))
                self.map_history.append(self.grid.probability().astype(np.float16))
                self.route_history.append(self.last_route.copy())
                self.yaw_history.append(float(yaw))
                self.route_length_history.append(length)
                self.occupied_history.append(int(self.grid.occupancy_mask().sum()))
                self.scan_points.append(scan.points_world.astype(np.float32))
                self.scan_origins.append(scan.origins_world.astype(np.float32))
                self.corridor_history.append(self.last_corridor.copy())
                self.sdf_history.append(self.last_sdf.copy())
                self.primitive_history.append(int(decision["primitive_id"]))
                self.raw_primitive_history.append(int(decision["raw_primitive_id"]))
                self.body_radius_history.append(float(self.active_body_radius_m))
                self.planner_history.append({**planner_report, "corridor": corridor_report})
            except RuntimeError as error:
                # A finite sliding map can be transiently over-occupied after several noisy
                # hit updates. Keep the last *validated* route for a bounded number of scans;
                # the executor's exact self-manifold/contact gate remains active meanwhile.
                message = str(error)
                if self.last_route is not None and "could not connect" in message:
                    item = {"tick": int(tick), "failure": message}
                    self.temporary_failures.append(item)
                    if len(self.temporary_failures) <= self.max_temporary_no_route_updates:
                        return {
                            "updated": True, "hard_stop": False,
                            "temporary_no_route": True, "planner_degraded": True,
                            "planner_failure": item,
                            "desired_yaw_rad": float(_lookahead_yaw(
                                self.last_route, position[:2], self.lookahead_m)),
                            "map_update_count": int(self.grid.update_count),
                            "online_update_count": int(len(self.update_ticks)),
                            "occupied_voxels": int(self.occupied_history[-1]),
                            "route_length_m": float(self.route_length_history[-1]),
                            "route_world_xyz": self.last_route,
                            "corridor": self.last_corridor, "sdf": self.last_sdf,
                            "primitive_id": int(self.last_decision["primitive_id"]),
                            "raw_primitive_id": int(self.last_decision["raw_primitive_id"]),
                            "primitive": str(self.last_decision["primitive"]),
                            "primitive_changed": False,
                            "primitive_decision": dict(self.last_decision),
                        }
                self.failure = f"{type(error).__name__}: {error}"
                return {"updated": True, "hard_stop": True, "failure": self.failure}
            except Exception as error:
                self.failure = f"{type(error).__name__}: {error}"
                return {"updated": True, "hard_stop": True, "failure": self.failure}
        if self.last_route is None or self.last_decision is None:
            self.failure = "online perception produced no route"
            return {"updated": updated, "hard_stop": True, "failure": self.failure}
        desired_yaw = _lookahead_yaw(self.last_route, position[:2], self.lookahead_m)
        self.override_ticks += 1
        return {
            "updated": bool(updated), "hard_stop": False,
            "desired_yaw_rad": float(desired_yaw),
            "map_update_count": int(self.grid.update_count),
            "online_update_count": int(len(self.update_ticks)),
            "occupied_voxels": int(self.occupied_history[-1]),
            "route_length_m": float(self.route_length_history[-1]),
            "route_world_xyz": self.last_route, "corridor": self.last_corridor,
            "sdf": self.last_sdf, "primitive_id": int(self.last_decision["primitive_id"]),
            "raw_primitive_id": int(self.last_decision["raw_primitive_id"]),
            "primitive": str(self.last_decision["primitive"]),
            "primitive_changed": bool(self.last_decision["primitive_changed"] if updated else False),
            "primitive_decision": dict(self.last_decision),
            "planner_report": (dict(self.planner_history[-1]) if updated else None),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True, "sensor": "MuJoCo ray radar with ground-truth odometry",
            "planner": "incremental truncated ESDF + D* Lite",
            "scan_period_ticks": self.scan_ticks, "scan_rate_hz": 50.0 / self.scan_ticks,
            "initial_map_updates": self.initial_update_count,
            "online_updates": len(self.update_ticks),
            "total_map_updates": (int(self.grid.update_count) if self.grid is not None else 0),
            "planning_override_ticks": int(self.override_ticks),
            "route_preference_weight": self.route_preference_weight, "failure": self.failure,
            "vertical_lookahead_m": self.vertical_lookahead_m,
            "dynamic_event": self.dynamic_event,
            "nominal_body_radius_m": self.body_radius_m,
            "side_body_radius_m": self.side_body_radius_m,
            "active_body_radius_m": self.body_radius_history,
            "planner_reinitializations": self.planner_reinitializations,
            "temporary_no_route_updates": self.temporary_failures,
            "update_ticks": self.update_ticks, "route_length_m": self.route_length_history,
            "occupied_voxels": self.occupied_history,
            "primitive_ids": self.primitive_history,
            "raw_primitive_ids": self.raw_primitive_history,
            "primitive_switches": (int(np.sum(np.diff(self.primitive_history) != 0))
                                   if len(self.primitive_history) > 1 else 0),
            "planning_ms": [float(item["planning_ms"]) for item in self.planner_history],
            "expanded_vertices": [int(item["expanded_vertices"]) for item in self.planner_history],
            "changed_cost_cells": [int(item["changed_cost_cells"]) for item in self.planner_history],
            "execution_contract": (
                "live radar -> 3-D sliding map -> incremental ESDF/D* Lite -> M_e -> "
                "debounced primitive decision and route-yaw command"
            ),
        }

    def save(self, path: Path) -> None:
        if self.grid is None or not self.map_history:
            raise RuntimeError("online perception has no updates to save")
        route_offsets = [0]
        routes = []
        for route in self.route_history:
            routes.append(route)
            route_offsets.append(route_offsets[-1] + len(route))
        scan_offsets = [0]
        points, origins = [], []
        for scan_points, scan_origins in zip(self.scan_points, self.scan_origins):
            points.append(scan_points); origins.append(scan_origins)
            scan_offsets.append(scan_offsets[-1] + len(scan_points))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            probability=np.stack(self.map_history, axis=0),
            map_origin_world_xyz=np.stack(self.origin_history, axis=0),
            robot_pos_world=np.stack(self.pose_history, axis=0),
            update_ticks=np.asarray(self.update_ticks, dtype=np.int32),
            desired_yaw_rad=np.asarray(self.yaw_history, dtype=np.float32),
            route_length_m=np.asarray(self.route_length_history, dtype=np.float32),
            occupied_voxels=np.asarray(self.occupied_history, dtype=np.int32),
            corridor=np.stack(self.corridor_history, axis=0),
            sdf=np.stack(self.sdf_history, axis=0),
            primitive_id=np.asarray(self.primitive_history, dtype=np.int16),
            raw_primitive_id=np.asarray(self.raw_primitive_history, dtype=np.int16),
            active_body_radius_m=np.asarray(self.body_radius_history, dtype=np.float32),
            planning_ms=np.asarray([item["planning_ms"] for item in self.planner_history], dtype=np.float32),
            expanded_vertices=np.asarray([item["expanded_vertices"] for item in self.planner_history], dtype=np.int32),
            changed_cost_cells=np.asarray([item["changed_cost_cells"] for item in self.planner_history], dtype=np.int32),
            route_world_xyz=np.concatenate(routes, axis=0),
            route_offsets=np.asarray(route_offsets, dtype=np.int32),
            radar_points_world=(np.concatenate(points, axis=0) if points else np.empty((0, 3), np.float32)),
            radar_origins_world=(np.concatenate(origins, axis=0) if origins else np.empty((0, 3), np.float32)),
            radar_offsets=np.asarray(scan_offsets, dtype=np.int32),
            resolution_m=np.asarray(self.config.resolution_m),
            initial_map_updates=np.asarray(self.initial_update_count, dtype=np.int32),
            final_log_odds=self.grid.log_odds.astype(np.float32),
            final_origin_cell_xyz=self.grid.origin_cell_xyz.astype(np.int64),
        )
