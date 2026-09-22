"""Environment-caused primitive routing for long-horizon Stage-2 execution.

Unlike the earlier demonstration, this runner never assigns an action from the route-segment
index.  Every segment is classified from measurable environment-manifold quantities:

* low vertical free semi-axis -> ``crouch``;
* genuinely narrow lateral free semi-axis -> ``walk_lateral_reverse``;
* otherwise -> ``walk_nominal``;
* a route heading change caused by obstacle avoidance temporarily activates ``walk_turn``.

The selected token is then passed with that segment's actual ``M_e`` corridor/SDF to latent
Flow Matching.  Multiple generated references are physically screened, and the selected
reference is executed in one continuous, keyframe-gated MuJoCo rollout.  Running the same code
on the paired wide/low scenes is the counterfactual test: only the obstacle manifold changes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from . import constants as C
from .corridor import ExecutedEnvelopeEstimator
from .perception_corridor import PerceptionGridConfig, corridor_condition_sdf
from .scene_pointcloud import obstacle_pointcloud
from .seed_replay import SeedReplayRunner
from .stage2_flow_route_candidates import (
    CandidatePlan, PRIMITIVE_NAMES, RouteFlowSampler, _calibrate_envelope,
    _candidate_screen, _screen_cost, _segment_condition, execute_plan,
)
from .stage2_long_horizon_avoidance import (
    Box2D, PlannerConfig, _astar, _boxes, _densify, _route_yaw, _simplify, render,
)
from .stage2_validate import _motion_from_trajectory
from .stage2_projection import ProjectionConfig, project_reference
from .seed_windows import _state_features
from .stage2_capability import CAPABILITIES, supported_ids


def _ground_obstacles(scene: Path) -> list[Box2D]:
    """Return only floor-connected boxes for 2-D A*; overhead boxes remain in M_e."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result: list[Box2D] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name.startswith("obstacle_"):
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError(f"adaptive fixture supports box obstacles only, got {name}")
        bottom = float(data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 2])
        # A ceiling constrains body pose but does not occupy the ground-projected route.
        if bottom > 0.35:
            continue
        result.append(Box2D(name, data.geom_xpos[geom_id, :2].copy(),
                            model.geom_size[geom_id, :2].copy()))
    return result


def _physical_boxes(scene: Path) -> list[dict[str, Any]]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith("obstacle_"):
            result.append({"name": name, "center": data.geom_xpos[geom_id].copy(),
                           "half": model.geom_size[geom_id, :3].copy()})
    return result


def _adaptive_segment_condition(route_segment: np.ndarray, boxes: list[dict[str, Any]],
                                envelope_semi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build axis-aware M_e from box geometry without mixing ceiling and side-wall normals.

    A raw unoriented point cloud can make the underside of a ceiling look like a lateral wall.
    These fixtures retain the MuJoCo box axes, so overhead surfaces contract z only and
    floor-connected walls contract x/y only.  This is the geometry that should drive routing.
    """
    start = route_segment[0]
    heading = float(np.arctan2(route_segment[-1, 1] - start[1],
                               route_segment[-1, 0] - start[0]))
    c, s = np.cos(heading), np.sin(heading)
    rotation = np.array([[c, s], [-s, c]])
    source_t = np.linspace(0.0, 1.0, len(route_segment))
    target_t = np.linspace(0.0, 1.0, 48)
    route_world = np.stack([np.interp(target_t, source_t, route_segment[:, axis]) for axis in range(2)], axis=1)
    local_xy = (route_world - start[None, :]) @ rotation.T
    route = np.column_stack([local_xy, np.zeros(48)])
    # These caps match the reverse-corridor training scale.  Physical obstacles can only
    # contract them; the exact robot mesh/contact gate remains authoritative at execution.
    semis = np.repeat(np.array([[0.95, 0.70, float(envelope_semi[2])]], dtype=np.float64), 48, axis=0)
    safety = 0.18  # 0.10 m obstacle inflation + 0.08 m clearance used by perception adapter.
    for frame, point in enumerate(route_world):
        for box in boxes:
            center, half = box["center"], box["half"]
            bottom = float(center[2] - half[2])
            if bottom > 0.35:
                # Expand the ceiling footprint by the robot's calibrated horizontal body
                # envelope.  Testing the pelvis point alone releases crouch too early while
                # the head/arms are still physically below the ceiling edge.
                if (abs(point[0] - center[0]) <= half[0] + envelope_semi[0] + 0.08 and
                        abs(point[1] - center[1]) <= half[1] + envelope_semi[1] + 0.08):
                    semis[frame, 2] = min(semis[frame, 2], max(0.06, bottom - safety))
                continue
            # For ground obstacles, contract only the horizontal direction in which the box
            # surface is actually visible from this route point.
            if abs(point[0] - center[0]) <= half[0] + semis[frame, 0]:
                lateral = abs(point[1] - center[1]) - half[1] - safety
                if lateral > 0.0:
                    semis[frame, 1] = min(semis[frame, 1], lateral)
            if abs(point[1] - center[1]) <= half[1] + semis[frame, 1]:
                longitudinal = abs(point[0] - center[0]) - half[0] - safety
                if longitudinal > 0.0:
                    semis[frame, 0] = min(semis[frame, 0], longitudinal)
    semis = np.maximum(semis, 0.06)
    corridor = np.column_stack([route, semis, np.zeros(48)]).astype(np.float32)
    sdf = corridor_condition_sdf(corridor, PerceptionGridConfig())
    command = RouteFlowSampler._yaw_command(route[-1], 0.0)
    return corridor, sdf, command


def _subdivide(polyline: np.ndarray, max_length_m: float) -> np.ndarray:
    pieces = []
    for index, (start, stop) in enumerate(zip(polyline[:-1], polyline[1:])):
        count = max(2, int(math.ceil(float(np.linalg.norm(stop - start)) / max_length_m)) + 1)
        segment = np.linspace(start, stop, count)
        pieces.append(segment if index == 0 else segment[1:])
    return np.concatenate(pieces)


def _segments(keyframes: np.ndarray, spacing_m: float = 0.06) -> tuple[list[np.ndarray], np.ndarray]:
    pieces = []
    for start, stop in zip(keyframes[:-1], keyframes[1:]):
        count = max(3, int(math.ceil(float(np.linalg.norm(stop - start)) / spacing_m)) + 1)
        pieces.append(np.linspace(start, stop, count))
    dense = np.concatenate([piece if i == 0 else piece[1:] for i, piece in enumerate(pieces)])
    return pieces, dense


def _angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _environment_corridor_world(route_segments: list[np.ndarray],
                                segment_inputs: list[dict[str, Any]],
                                dense_route: np.ndarray,
                                root_z: float) -> np.ndarray:
    """Expand each local segment condition into a world-frame ``M_e(t)`` sequence.

    The adaptive sampler stores each corridor in the segment's route-local frame.  The
    renderer used to throw those apertures away and draw one fixed calibration ellipsoid.
    Keep the measured aperture and interpolate it onto the executed route instead, so a low
    ceiling or a narrow side passage is visible as a change in the environment manifold.
    """
    pieces: list[np.ndarray] = []
    for segment, item in zip(route_segments, segment_inputs):
        condition = np.asarray(item["corridor"], dtype=np.float64)
        if len(segment) < 2 or condition.ndim != 2 or condition.shape[1] < 6:
            raise ValueError("route segment/corridor shapes are inconsistent")
        source_t = np.linspace(0.0, 1.0, len(condition))
        target_t = np.linspace(0.0, 1.0, len(segment))
        semis = np.column_stack([
            np.interp(target_t, source_t, condition[:, axis]) for axis in (3, 4, 5)
        ])
        heading = float(np.arctan2(segment[-1, 1] - segment[0, 1],
                                   segment[-1, 0] - segment[0, 0]))
        piece = np.column_stack([
            segment[:, 0], segment[:, 1], np.full(len(segment), root_z), semis,
            np.full(len(segment), heading),
        ])
        pieces.append(piece if not pieces else piece[1:])
    result = np.concatenate(pieces).astype(np.float32)
    if len(result) != len(dense_route) or not np.allclose(result[:, :2], dense_route, atol=1e-6):
        raise RuntimeError("world environment corridor is not aligned with the dense route")
    return result


def _robot_self_manifold(data: dict[str, np.ndarray],
                         environment_corridor: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build the executed, primitive-conditioned robot manifold ``M_r(t)``.

    ``measured`` is the exact mesh-fitted envelope in the instantaneous body frame.  The
    planning manifold is then expressed in the route frame and contracts according to the
    selected primitive: crouch is capped vertically and side gait is explicitly narrowed
    laterally.  The mesh/contact gate remains authoritative; this layer is the interpretable
    geometry passed to the planner and shown in the diagnostic GIF.
    """
    q_exec = np.asarray(data["q_exec"])
    base_pos = np.asarray(data["base_pos"])
    base_quat = np.asarray(data["base_quat"])
    if len(q_exec) != len(environment_corridor):
        # M_e is sampled on route points while execution is sampled at 50 Hz.  Associate each
        # executed state with its nearest route element; this preserves abrupt aperture changes
        # at a ceiling/wall boundary without pretending that the robot followed a teleported
        # time parameterisation.
        nearest = np.argmin(
            np.linalg.norm(base_pos[:, None, :2] - environment_corridor[None, :, :2], axis=2),
            axis=1)
        environment_at_execution = environment_corridor[nearest]
    else:
        environment_at_execution = environment_corridor
    measured_body = ExecutedEnvelopeEstimator().sequence(q_exec, base_pos, base_quat)
    body_yaw = np.unwrap(np.asarray([
        np.arctan2(C.quat_rotate(quat, np.array([1.0, 0.0, 0.0]))[1],
                                C.quat_rotate(quat, np.array([1.0, 0.0, 0.0]))[0])
        for quat in base_quat
    ], dtype=np.float64))
    route_yaw = np.unwrap(np.asarray(environment_at_execution[:, 6], dtype=np.float64))
    delta = body_yaw - route_yaw
    cosine, sine = np.abs(np.cos(delta)), np.abs(np.sin(delta))
    measured_route = np.column_stack([
        cosine * measured_body[:, 0] + sine * measured_body[:, 1],
        sine * measured_body[:, 0] + cosine * measured_body[:, 1],
        measured_body[:, 2],
    ]).astype(np.float32)
    # ``safe`` is the physical, mesh-fitted envelope.  ``task`` is a smaller target shape used
    # to express the posture requested by the environment; it must never replace ``safe`` in a
    # deployment collision check.
    task_semi = measured_route.astype(np.float32).copy()
    names = np.asarray(data.get("active_primitive_name", [
        PRIMITIVE_NAMES[int(value)] for value in data["active_primitive"]
    ]), dtype=str)
    crouch = names == "crouch"
    side = names == "walk_lateral_reverse"
    # Explicit task-conditioned contraction.  The measured mesh envelope is never enlarged;
    # only the planning shape is contracted to encode the posture required by M_e.
    task_semi[crouch, 2] = np.minimum(task_semi[crouch, 2],
                                      0.90 * environment_at_execution[crouch, 5])
    task_semi[side, 0] *= 1.05
    task_semi[side, 1] = np.minimum(
        0.72 * task_semi[side, 1], 0.90 * environment_at_execution[side, 4])
    task_semi = np.maximum(task_semi, 0.06)
    task_manifold = np.column_stack([base_pos, task_semi, route_yaw]).astype(np.float32)
    safe_manifold = np.column_stack([base_pos, measured_route, route_yaw]).astype(np.float32)
    def _mean(mask: np.ndarray, values: np.ndarray) -> list[float] | None:
        return values[mask].mean(axis=0).astype(float).tolist() if np.any(mask) else None
    stats: dict[str, Any] = {
        "contract": "M_e(t) -> primitive selection -> measured body envelope -> primitive-conditioned M_r(t)",
        "environment_axes": "route-frame [tangent, lateral, vertical]",
        "measured_body_frame_semi_mean_m": measured_body.mean(axis=0).astype(float).tolist(),
        "measured_body_frame_semi_min_m": measured_body.min(axis=0).astype(float).tolist(),
        "measured_body_frame_semi_max_m": measured_body.max(axis=0).astype(float).tolist(),
        "robot_self_manifold_semi_mean_m": task_semi.mean(axis=0).astype(float).tolist(),
        "robot_self_manifold_semi_min_m": task_semi.min(axis=0).astype(float).tolist(),
        "robot_self_manifold_semi_max_m": task_semi.max(axis=0).astype(float).tolist(),
        "measured_safe_manifold_semi_mean_m": measured_route.mean(axis=0).astype(float).tolist(),
        "measured_safe_manifold_semi_min_m": measured_route.min(axis=0).astype(float).tolist(),
        "measured_safe_manifold_semi_max_m": measured_route.max(axis=0).astype(float).tolist(),
        "primitive_means_m": {
            "crouch": _mean(crouch, task_semi),
            "walk_lateral_reverse": _mean(side, task_semi),
            "walk_nominal": _mean(names == "walk_nominal", task_semi),
            "walk_turn": _mean(names == "walk_turn", task_semi),
        },
        "measured_safe_primitive_means_m": {
            "crouch": _mean(crouch, measured_route),
            "walk_lateral_reverse": _mean(side, measured_route),
            "walk_nominal": _mean(names == "walk_nominal", measured_route),
            "walk_turn": _mean(names == "walk_turn", measured_route),
        },
        "contraction_policy": {
            "crouch_vertical_cap_fraction_of_Me": 0.90,
            "side_lateral_cap_fraction_of_measured": 0.72,
            "side_lateral_cap_fraction_of_Me": 0.90,
        },
        "measured_mesh_is_safety_authority": True,
    }
    return task_manifold, safe_manifold, measured_route, stats


def _box_signed_distance(points: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Signed distance of points to an axis-aligned obstacle box (positive is clear)."""
    delta = np.abs(points - center[None, :]) - half[None, :]
    outside = np.maximum(delta, 0.0)
    distance = np.linalg.norm(outside, axis=1)
    inside = np.all(delta <= 0.0, axis=1)
    distance[inside] = np.max(delta[inside], axis=1)
    return distance


def _ellipsoid_surface(element: np.ndarray, theta_count: int = 18,
                       phi_count: int = 36) -> np.ndarray:
    """Deterministic surface samples for the conservative M_r broad phase."""
    theta = np.linspace(0.0, np.pi, theta_count)
    phi = np.linspace(0.0, 2.0 * np.pi, phi_count, endpoint=False)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    local = np.stack([
        element[3] * np.sin(th) * np.cos(ph),
        element[4] * np.sin(th) * np.sin(ph),
        element[5] * np.cos(th),
    ], axis=-1).reshape(-1, 3)
    yaw = float(element[6])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                         [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    return local @ rotation.T + element[:3]


def _self_manifold_safety(data: dict[str, np.ndarray], environment_corridor: np.ndarray,
                          scene: Path, required_clearance_m: float
                          ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Run the deployment safety gate using M_r, then an exact mesh narrow phase.

    The colored task manifold is intentionally contracted for posture generation and is not a
    collision proxy.  The hard gate uses the mesh-fitted safe M_r to generate a conservative
    ellipsoid broad phase, then checks every sampled G1 surface point against every physical
    obstacle box.  This is the same ordering used on a real robot: fast self-manifold query,
    followed by the known robot geometry against the local obstacle/SDF representation.
    """
    estimator = ExecutedEnvelopeEstimator()
    boxes = _physical_boxes(scene)
    base_pos = np.asarray(data["base_pos"])
    q_exec, base_quat = data["q_exec"], data["base_quat"]
    safe = np.asarray(data["robot_manifold_safe"])
    if len(base_pos) != len(environment_corridor) and len(base_pos):
        nearest = np.argmin(np.linalg.norm(
            base_pos[:, None, :2] - environment_corridor[None, :, :2], axis=2), axis=1)
        env_at = environment_corridor[nearest]
    else:
        env_at = environment_corridor
    exact_clearance = np.full(len(base_pos), np.inf, dtype=np.float32)
    broad_clearance = np.full(len(base_pos), np.inf, dtype=np.float32)
    corridor_radius = np.zeros(len(base_pos), dtype=np.float32)
    task = np.asarray(data["robot_manifold"])
    names = np.asarray(data.get("active_primitive_name", [
        PRIMITIVE_NAMES[int(value)] for value in data["active_primitive"]
    ]), dtype=str)
    task_aperture_ratio = np.ones(len(base_pos), dtype=np.float32)
    for tick, (q, pos, quat) in enumerate(zip(q_exec, base_pos, base_quat)):
        points = estimator.surface_points(q, pos, quat)
        for box in boxes:
            exact_clearance[tick] = min(
                float(exact_clearance[tick]),
                float(_box_signed_distance(points, box["center"], box["half"]).min()))
        ellipsoid_points = _ellipsoid_surface(safe[tick])
        for box in boxes:
            broad_clearance[tick] = min(
                float(broad_clearance[tick]),
                float(_box_signed_distance(ellipsoid_points, box["center"], box["half"]).min()))
        if len(env_at):
            element = env_at[tick]
            delta = points - element[:3]
            c, s = np.cos(element[6]), np.sin(element[6])
            aligned = np.column_stack([delta[:, 0] * c + delta[:, 1] * s,
                                        -delta[:, 0] * s + delta[:, 1] * c,
                                        delta[:, 2]])
            corridor_radius[tick] = float(np.linalg.norm(aligned / element[3:6], axis=1).max())
            if names[tick] == "crouch":
                task_aperture_ratio[tick] = float(task[tick, 5] / max(element[5], 1e-6))
            elif names[tick] == "walk_lateral_reverse":
                task_aperture_ratio[tick] = float(task[tick, 4] / max(element[4], 1e-6))
    task_aperture_bad = task_aperture_ratio > 1.0 + 1e-3
    failed = bool(np.any(exact_clearance < required_clearance_m) or np.any(task_aperture_bad))
    stats: dict[str, Any] = {
        "contract": "measured safe M_r broad phase -> exact G1 surface/obstacle narrow phase",
        "accepted": not failed,
        "required_clearance_m": float(required_clearance_m),
        "exact_surface_obstacle_clearance_min_m": float(exact_clearance.min()) if len(exact_clearance) else None,
        "exact_surface_obstacle_clearance_p05_m": float(np.quantile(exact_clearance, 0.05)) if len(exact_clearance) else None,
        "ellipsoid_broadphase_clearance_min_m": float(broad_clearance.min()) if len(broad_clearance) else None,
        "ellipsoid_broadphase_overlap_ticks": int(np.sum(broad_clearance < 0.0)),
        "environment_corridor_radius_max": float(corridor_radius.max()) if len(corridor_radius) else None,
        "environment_corridor_radius_p95": float(np.quantile(corridor_radius, 0.95)) if len(corridor_radius) else None,
        "environment_corridor_over_1_ticks": int(np.sum(corridor_radius > 1.0)),
        "task_manifold_aperture_ratio_max": float(task_aperture_ratio.max()) if len(task_aperture_ratio) else None,
        "task_manifold_aperture_ratio_p95": float(np.quantile(task_aperture_ratio, 0.95)) if len(task_aperture_ratio) else None,
        "task_manifold_aperture_violation_ticks": int(np.sum(task_aperture_bad)),
        "task_manifold_aperture_contract": "crouch checks vertical M_r/M_e; side gait checks route-lateral M_r/M_e",
        "physical_obstacle_count": len(boxes),
        "mesh_narrowphase_is_hard_gate": True,
        "ellipsoid_broadphase_is_conservative_query": True,
    }
    return stats, exact_clearance, corridor_radius


def _online_condition_overrides(data: dict[str, np.ndarray], execution: dict[str, Any],
                                segment_count: int) -> dict[int, dict[str, Any]]:
    """Extract 69-D executed states and 12-frame histories at measured segment boundaries.

    The replay is sampled at SONIC's 50 Hz control rate.  History indices are spread over the
    preceding 0.4 seconds, matching the 12-frame/30 Hz Stage-2 training contract.  This keeps
    the dynamic model conditioned on what the controller actually did, including contacts and
    base velocity, rather than on the SEED anchor state used for the first proposal.
    """
    count = len(data["q_exec"])
    if count == 0:
        return {}
    features = _state_features({
        "q_exec": data["q_exec"], "dq_exec": data["dq_exec"],
        "base_quat": data["base_quat"], "base_lin_vel": data["base_lin_vel"],
        "foot_contact": data["foot_contact"], "hand_contact": data["hand_contact"],
        "nonfoot_floor_contact": data["nonfoot_floor_contact"],
    }).astype(np.float32)
    ticks: dict[int, int] = {}
    for event in execution.get("primitive_switches", []):
        target = event.get("to")
        if not isinstance(target, (list, tuple)) or not target:
            continue
        segment = int(target[0])
        if 0 <= segment < segment_count and segment not in ticks:
            ticks[segment] = int(np.clip(event.get("tick", 0), 0, count - 1))
    for segment in range(segment_count):
        ticks.setdefault(segment, int(round((segment + 0.5) * count / max(segment_count, 1))))
        ticks[segment] = int(np.clip(ticks[segment], 0, count - 1))
    overrides: dict[int, dict[str, Any]] = {}
    for segment, tick in ticks.items():
        history_idx = np.rint(np.linspace(max(0, tick - 20), tick, 12)).astype(int)
        overrides[segment] = {
            "state": features[tick].copy(), "history": features[history_idx].copy(),
            "tick": int(tick), "history_ticks": history_idx.tolist(),
        }
    return overrides


def _route_decision(corridor: np.ndarray, heading: float, previous_heading: float,
                    *, crouch_semi_z_m: float, side_semi_y_m: float,
                    turn_threshold_rad: float) -> dict[str, Any]:
    vertical = float(np.min(corridor[:, 5]))
    lateral = float(np.min(corridor[:, 4]))
    heading_change = _angle(heading - previous_heading)
    if vertical < crouch_semi_z_m:
        primitive_id, reason = 2, "vertical_free_semi_below_crouch_threshold"
    elif lateral < side_semi_y_m:
        primitive_id, reason = 4, "lateral_free_semi_below_side_threshold"
    else:
        primitive_id, reason = 5, "wide_and_tall_enough_for_nominal_walk"
    return {
        "primitive_id": primitive_id,
        "primitive": PRIMITIVE_NAMES[primitive_id],
        "reason": reason,
        "vertical_free_semi_min_m": vertical,
        "lateral_free_semi_min_m": lateral,
        "heading_rad": float(heading),
        "heading_change_rad": heading_change,
        "requires_turn": bool(abs(heading_change) > turn_threshold_rad),
        "thresholds": {"crouch_semi_z_m": crouch_semi_z_m,
                       "side_semi_y_m": side_semi_y_m,
                       "turn_heading_change_rad": turn_threshold_rad},
    }


def _choose_candidate(primitive_id: int, generated: np.ndarray, source_index: int,
                      segment_index: int, out: Path, runner: SeedReplayRunner, *,
                      min_forward_progress_m: float,
                      max_heading_error_rad: float,
                      side_on: bool = False,
                      side_heading_tolerance_rad: float = 0.55,
                      ) -> tuple[CandidatePlan, list[dict[str, Any]]]:
    rows = []
    for candidate_index, trajectory in enumerate(generated):
        executed, summary = _candidate_screen(trajectory, primitive_id,
                                              out / f"segment_{segment_index}_candidates.npz", runner)
        displacement = executed["base_pos"][-1, :2] - executed["base_pos"][0, :2]
        heading_offset = float(np.arctan2(displacement[1], displacement[0]))
        # The executor rotates a body-frame candidate onto the route.  A candidate whose
        # isolated motion points backward would therefore require turning the *body* around
        # while still following the route.  That was the source of the visually incorrect
        # 180-degree crouched walk.  Direction is a semantic constraint, not a soft score.
        side_heading_error = abs(abs(heading_offset) - np.pi / 2.0)
        if primitive_id == 4 and side_on:
            directional = (float(abs(displacement[1])) >= min_forward_progress_m
                           and side_heading_error <= side_heading_tolerance_rad)
        else:
            directional = (primitive_id == 6 or (
                float(displacement[0]) >= min_forward_progress_m
                and abs(heading_offset) <= max_heading_error_rad
            ))
        lateral_drift = abs(float(displacement[1]))
        row = {"candidate_index": candidate_index, "accepted": bool(summary["accepted"]),
               "directional_semantics_passed": bool(directional),
               "side_on_semantics_passed": bool(side_on and primitive_id == 4 and directional),
               "score": _screen_cost(summary, trajectory), "failed_checks": summary["failed_checks"],
               "track_err_mean_rad": summary.get("track_err_mean_rad"),
               "base_z_min_m": summary.get("base_z_min_m"),
               "base_z_mean_m": summary.get("base_z_mean_m"),
               "exec_path_m": summary.get("exec_path_m"),
               "exec_displacement_xy_m": displacement.astype(float).tolist(),
               "forward_progress_m": float(displacement[0]),
               "lateral_drift_m": lateral_drift,
               "side_heading_error_rad": float(side_heading_error),
               "exec_heading_offset_rad": heading_offset,
               "source_index": source_index}
        rows.append(row)
    viable = [row for row in rows if row["accepted"] and row["directional_semantics_passed"]]
    if primitive_id == 2:
        # A useful crouch must be both visibly low and able to advance through a long ceiling.
        # Use minimum executed height as the posture qualification because a locomoting crouch
        # necessarily contains a short stand->crouch transition; after qualification, maximize
        # measured path so a stationary deep squat cannot beat a traversing crouched gait.
        low = [row for row in viable if float(row.get("base_z_min_m") or 10.0) < 0.62]
        # Prefer real forward progress, then penalize side drift and heading error.  The
        # height qualification remains a hard semantic requirement when such a candidate
        # exists, so an upright walk cannot win merely by moving faster.
        winner = max(low or viable, key=lambda row: (
            float(row["forward_progress_m"])
            - 0.40 * float(row["lateral_drift_m"])
            - 0.10 * abs(float(row["exec_heading_offset_rad"]))
        )) if viable else None
    elif primitive_id == 4 and side_on:
        winner = max(viable, key=lambda row: (
            abs(float(row["exec_displacement_xy_m"][1]))
            - 0.35 * abs(float(row["exec_displacement_xy_m"][0]))
            - 0.15 * float(row["side_heading_error_rad"])
        )) if viable else None
    elif primitive_id in (4, 5):
        winner = max(viable, key=lambda row: (
            float(row["forward_progress_m"])
            - 0.45 * float(row["lateral_drift_m"])
            - 0.10 * abs(float(row["exec_heading_offset_rad"]))
        )) if viable else None
    else:
        winner = min(viable, key=lambda row: float(row["score"])) if viable else None
    if winner is None:
        physical = sum(bool(row["accepted"]) for row in rows)
        directional = sum(bool(row["accepted"] and row["directional_semantics_passed"])
                          for row in rows)
        raise RuntimeError(
            f"segment {segment_index}: primitive {PRIMITIVE_NAMES[primitive_id]} has no viable "
            f"candidate (physical={physical}, directionally valid={directional}); "
            f"screen_rows={rows}; refusing a backward-facing locomotion workaround"
        )
    index = int(winner["candidate_index"])
    trajectory = generated[index]
    return CandidatePlan(
        segment_index, primitive_id, PRIMITIVE_NAMES[primitive_id], index, trajectory,
        _motion_from_trajectory(trajectory, out / "adaptive_candidate.npz", 30.0, 50.0), winner,
    ), rows


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    args.out.mkdir(parents=True, exist_ok=True)
    planner = PlannerConfig(body_radius_m=args.planner_body_radius_m,
                            clearance_m=args.planner_clearance_m)
    ground = _ground_obstacles(args.scene)
    raw = _astar(np.array([0.0, 0.0]), np.array([args.goal_x, 0.0]), ground, planner)
    corners = _simplify(raw, ground, planner)
    keyframes = _subdivide(corners, args.segment_length_m)
    route_segments, dense_route = _segments(keyframes)
    _, obstacle_names = obstacle_pointcloud(args.scene, spacing_m=0.06)
    physical_boxes = _physical_boxes(args.scene)
    envelope = _calibrate_envelope()
    sampler = RouteFlowSampler(args.windows, args.autoencoder, args.flow, args.device,
                               mean_model_root=args.mean_model_root)
    # Window 590 is a physically verified *forward* crouched locomotion phase.  Under the real
    # low-ceiling M_e/SDF it advances about 0.60 m with ~9 degree heading error, while the old
    # window 669 advances backward (~149 degree offset) and forced the executor to face away
    # from the route.  The directional gate below independently checks every generated result.
    sampler.validated_exemplars[2] = args.crouch_locomotion_source_index
    # The original p4 anchor (1348) is a stable but near-stationary phase.  Anchor 1375 was
    # selected by an exhaustive stride-12 scan of all p4 windows: its conditional-mean Flow
    # proposal passed the same SONIC gate and advanced 0.76 m without obstacle contact.
    sampler.validated_exemplars[4] = (args.side_on_source_index
                                      if args.side_gait_mode == "side_on"
                                      else args.side_locomotion_source_index)
    runner = SeedReplayRunner(C.FLAT_SCENE)

    decisions: list[dict[str, Any]] = []
    condition_arrays: dict[str, np.ndarray] = {}
    segment_inputs: list[dict[str, Any]] = []
    previous_heading = 0.0
    for segment_index, segment in enumerate(route_segments):
        corridor, sdf, command = _adaptive_segment_condition(segment, physical_boxes, envelope)
        condition_arrays[f"segment_{segment_index}_corridor"] = corridor
        condition_arrays[f"segment_{segment_index}_sdf"] = sdf
        heading = float(np.arctan2(segment[-1, 1] - segment[0, 1],
                                   segment[-1, 0] - segment[0, 0]))
        decision = _route_decision(
            corridor, heading, previous_heading,
            crouch_semi_z_m=args.crouch_semi_z_m,
            side_semi_y_m=args.side_semi_y_m,
            turn_threshold_rad=args.turn_threshold_rad,
        )
        decision.update({"segment_index": segment_index,
                         "start_xy_m": segment[0].tolist(), "end_xy_m": segment[-1].tolist()})
        decisions.append(decision)
        previous_heading = heading

        base_id = int(decision["primitive_id"])
        # A turn helper is caused by route curvature only.  The generated base gait's measured
        # displacement offset is handled by reference-yaw calibration in execute_plan, so a
        # straight corridor never receives an arbitrary extra action.
        primitive_ids = ([6, base_id] if decision["requires_turn"] and base_id != 6 else [base_id])
        segment_inputs.append({"corridor": corridor, "sdf": sdf, "command": command,
                               "primitive_ids": primitive_ids})

    projection_config = ProjectionConfig(
        iterations=args.projection_iterations, step_size=args.projection_step_size,
        smooth_weight=args.projection_smooth_weight,
        velocity_weight=args.projection_velocity_weight,
        acceleration_weight=args.projection_acceleration_weight,
        jerk_weight=args.projection_jerk_weight,
        handoff_weight=args.projection_handoff_weight,
        corridor_weight=args.projection_corridor_weight,
        root_blend=args.projection_root_blend,
        max_joint_step=args.projection_max_joint_step,
        max_root_step_m=args.projection_max_root_step_m,
    )

    def build_plan_options(overrides: dict[int, dict[str, Any]] | None = None
                           ) -> tuple[list[list[CandidatePlan]], list[dict[str, Any]]]:
        """Generate, project and physically screen one condition pass."""
        plan_options_local: list[list[CandidatePlan]] = []
        evidence_local: list[dict[str, Any]] = []
        overrides = {} if overrides is None else overrides
        for segment_index, item in enumerate(segment_inputs):
            corridor, sdf, command = item["corridor"], item["sdf"], item["command"]
            options: list[CandidatePlan] = []
            condition = overrides.get(segment_index, {})
            for primitive_id in item["primitive_ids"]:
                generated, source_index = sampler.sample(
                    primitive_id, corridor, sdf, command, args.num_candidates,
                    seed=args.seed + segment_index * 101 + primitive_id,
                    state=condition.get("state"), history=condition.get("history"),
                    # Sparse semantic clips are still weakly covered by the pilot Flow corpus.
                    # Candidate zero remains a verified SEED reference while the other K-1
                    # candidates use the live state/history.  The same projection and SONIC
                    # gate evaluate both, so this is an explicit safety set, not a hidden
                    # replacement after a model failure.
                    raw_exemplar=(primitive_id == 4 and not args.disable_anchor),
                    include_mean_anchor=not args.pure_stochastic_flow,
                )
                safety_anchor = False
                if (condition and primitive_id in (2, 5)
                        and not args.disable_learned_anchor and not args.pure_stochastic_flow):
                    # The current pilot corpus is sparse at walk->crouch boundary states.
                    # Keep one model proposal from the verified training-support anchor and
                    # retain K-1 genuinely online-conditioned proposals.  A raw clip is not
                    # used here: the anchored candidate is still decoded by the same Flow/AE.
                    anchor_generated, _ = sampler.sample(
                        primitive_id, corridor, sdf, command, 1,
                        seed=args.seed + segment_index * 101 + primitive_id,
                        include_mean_anchor=True,
                    )
                    generated[0] = anchor_generated[0]
                    safety_anchor = True
                raw_generated = generated.copy()
                projected = []
                projection_reports = []
                for trajectory in generated:
                    projected_trajectory, projection_report = project_reference(
                        trajectory, corridor, projection_config)
                    projected.append(projected_trajectory)
                    projection_reports.append(projection_report)
                generated = np.asarray(projected, dtype=np.float32)
                selected, rows = _choose_candidate(
                    primitive_id, generated, source_index, segment_index, args.out, runner,
                    min_forward_progress_m=args.min_forward_progress_m,
                    max_heading_error_rad=args.max_candidate_heading_error_rad,
                    side_on=(args.side_gait_mode == "side_on"),
                    side_heading_tolerance_rad=args.side_heading_tolerance_rad)
                for row in rows:
                    row["projection"] = projection_reports[int(row["candidate_index"])]
                selected.screen_summary["projection"] = projection_reports[selected.candidate_index]
                selected.screen_summary["online_condition"] = {
                    "used": bool(condition), "tick": condition.get("tick"),
                    "history_ticks": condition.get("history_ticks"),
                    "candidate0_training_support_anchor": safety_anchor,
                }
                options.append(selected)
                evidence_local.append({
                    "segment_index": segment_index, "primitive_id": primitive_id,
                    "primitive": PRIMITIVE_NAMES[primitive_id],
                    "selected_candidate_index": selected.candidate_index,
                    "online_condition": selected.screen_summary["online_condition"],
                    "candidates": rows,
                })
                np.save(args.out / f"segment_{segment_index}_{PRIMITIVE_NAMES[primitive_id]}_raw_flow.npy",
                        raw_generated)
                np.save(args.out / f"segment_{segment_index}_{selected.primitive}_candidate_{selected.candidate_index}.npy",
                        selected.trajectory)
            plan_options_local.append(options)
        return plan_options_local, evidence_local

    # execute_plan switches [turn, manifold-selected] by measured body-yaw error when needed.
    args.manifold_adaptive = True
    plan_options, evidence = build_plan_options()

    def receding_replan(segment: int, primitive_id: int, state: np.ndarray,
                        history: np.ndarray, current_q: np.ndarray, tick: int):
        """Refresh one future reference from the actual online state/history.

        This callback intentionally performs only decode + projection and a continuity ranking;
        it does not pretend that a reset replay is an online physical gate.  The enclosing
        continuous MuJoCo rollout remains the authority and can reject the complete run.
        """
        if segment < 0 or segment >= len(segment_inputs):
            return None, {"reason": "segment_out_of_range"}
        item = segment_inputs[segment]
        generated, source_index = sampler.sample(
            primitive_id, item["corridor"], item["sdf"], item["command"],
            max(2, args.num_candidates), seed=args.seed + 900001 + tick + segment * 17,
            state=state, history=history, raw_exemplar=False,
        include_mean_anchor=not args.pure_stochastic_flow,
        )
        projected = []
        reports = []
        for trajectory in generated:
            value, projection = project_reference(trajectory, item["corridor"], projection_config,
                                                  handoff_q=current_q)
            projected.append(value)
            reports.append(projection)
        projected = np.asarray(projected, dtype=np.float32)
        # Continuity is a ranking term, but a refresh that only holds the current pose is not a
        # useful locomotion proposal.  Include the model-predicted route-local displacement as a
        # small progress prior; the final continuous MuJoCo gate remains authoritative.
        distances = np.sqrt(np.mean((projected[:, 0, :29] - current_q[None, :]) ** 2, axis=1))
        displacement = projected[:, -1, 29:31] - projected[:, 0, 29:31]
        expected_progress = (np.abs(displacement[:, 1]) if primitive_id == 4
                             else displacement[:, 0])
        scores = distances + 0.35 * np.maximum(0.0, 0.08 - expected_progress)
        winner = int(np.argmin(scores))
        trajectory = projected[winner]
        plan = CandidatePlan(
            segment, primitive_id, PRIMITIVE_NAMES[primitive_id], winner, trajectory,
            _motion_from_trajectory(trajectory, args.out / "receding_candidate.npz", 30.0, 50.0),
            {"accepted": True, "online_refresh": True,
             "candidate_index": winner, "candidate_count": int(len(projected)),
             "continuity_rms_rad": float(distances[winner]),
             "predicted_route_progress_m": float(expected_progress[winner]),
             "refresh_score": float(scores[winner]), "source_index": int(source_index),
             "projection": reports[winner]},
        )
        return plan, {"candidate_index": winner, "candidate_count": int(len(projected)),
                      "continuity_rms_rad": float(distances[winner]),
                      "predicted_route_progress_m": float(expected_progress[winner]),
                      "refresh_score": float(scores[winner]),
                      "anchor_disabled": bool(args.disable_anchor),
                      "commit": not args.receding_horizon_shadow}

    probe_execution: dict[str, Any] | None = None
    online_overrides: dict[int, dict[str, Any]] = {}
    for online_iteration in range(args.online_condition_iterations):
        probe_data, probe_execution = execute_plan(
            args.scene, keyframes, dense_route, plan_options, args,
            replan_callback=(receding_replan if args.receding_horizon_ticks > 0 else None),
        )
        online_overrides = _online_condition_overrides(probe_data, probe_execution, len(segment_inputs))
        plan_options, evidence = build_plan_options(online_overrides)
        if online_iteration + 1 < args.online_condition_iterations:
            # The next loop iteration probes the newly reconditioned candidates.
            continue
    data, execution = execute_plan(
        args.scene, keyframes, dense_route, plan_options, args,
        replan_callback=(receding_replan if args.receding_horizon_ticks > 0 else None),
    )
    execution["render_title"] = args.title
    origin = np.asarray(execution["start_world_xy_m"])
    world_keyframes, world_route = keyframes + origin, dense_route + origin
    render_corridor = _environment_corridor_world(
        route_segments, segment_inputs, dense_route, float(data["base_pos"][0, 2]))
    render_corridor[:, :2] += origin[None, :]
    robot_manifold, robot_manifold_safe, measured_route_semi, self_manifold_stats = _robot_self_manifold(
        data, render_corridor)
    data["robot_manifold"] = robot_manifold
    data["robot_manifold_safe"] = robot_manifold_safe
    data["measured_body_manifold_semi"] = measured_route_semi
    safety_stats, exact_clearance, corridor_radius = _self_manifold_safety(
        data, render_corridor, args.scene, args.self_manifold_clearance_m)
    data["self_manifold_obstacle_clearance_m"] = exact_clearance
    data["self_manifold_environment_radius"] = corridor_radius
    if not safety_stats["accepted"]:
        execution["failed_checks"].append("self_manifold_obstacle_clearance")
        execution["accepted"] = False
    if not args.skip_render:
        render(args.scene, data, execution, world_keyframes, world_route, render_corridor,
               _boxes(args.scene), planner, args.out / "manifold_adaptive.gif", args.fps,
               self_manifold=robot_manifold, safe_manifold=robot_manifold_safe)
    report = {
        "experiment": "environment-manifold-caused primitive routing",
        "scenario": args.title, "scene": str(args.scene), "obstacles": obstacle_names,
        "planner": {"body_radius_m": planner.body_radius_m,
                    "clearance_m": planner.clearance_m,
                    "inflation_m": planner.inflation_m,
                    "resolution_m": planner.resolution_m},
        "routing_contract": "primitive is a deterministic function of measured M_e aperture and route heading change; never segment index",
        "capability_manifest": {
            "strict_supported_primitive_ids": list(supported_ids(False)),
            "partial_primitive_ids": list(supported_ids(True)),
            "unsupported_not_routed": [name for name, row in CAPABILITIES.items()
                                        if row["status"] == "unsupported"],
        },
        "side_gait_mode": args.side_gait_mode,
        "anchor_policy": {
            "handcrafted_disabled": bool(args.disable_anchor),
            "learned_support_anchor_disabled": bool(args.disable_learned_anchor),
            "pure_stochastic_flow": bool(args.pure_stochastic_flow),
            "description": ("conditional mean is disabled; all candidates are stochastic Flow samples"
                             if args.pure_stochastic_flow else
                             "raw SEED anchors are disabled; learned conditional mean support remains available"
                             if args.disable_anchor else
                             "candidate zero may use a validated conditional-mean/SEED anchor"),
        },
        "online_conditioning": {
            "iterations": args.online_condition_iterations,
            "contract": "probe rollout -> measured executed 69-D state and 12-frame history -> per-segment Flow reconditioning",
            "segments": {str(k): {"tick": v.get("tick"), "history_ticks": v.get("history_ticks")}
                         for k, v in online_overrides.items()},
            "probe_execution": probe_execution,
        },
        "optimization_embedded_projection": {
            "contract": "projected-gradient feasibility layer between Flow decode and SONIC gate",
            "config": projection_config.__dict__,
            "root_position_note": "root position is projected for target consistency; current SONIC action head consumes joint pose/velocity and base orientation, so MuJoCo execution remains authoritative",
        },
        "receding_horizon": {
            "ticks": int(args.receding_horizon_ticks),
            "shadow": bool(args.receding_horizon_shadow),
            "state_dim": 69,
            "history_shape": [12, 69],
            "contract": "actual state/history -> Flow decode -> projection -> continuous rollout gate",
        },
        "decisions": decisions,
        "environment_manifold": {
            "contract": "world-aligned M_e(t) interpolated from each route segment condition",
            "semi_mean_m": render_corridor[:, 3:6].mean(axis=0).astype(float).tolist(),
            "semi_min_m": render_corridor[:, 3:6].min(axis=0).astype(float).tolist(),
            "semi_max_m": render_corridor[:, 3:6].max(axis=0).astype(float).tolist(),
        },
        "robot_self_manifold": self_manifold_stats,
        "robot_self_manifold_safety": safety_stats,
        "selected_plans": [[{"primitive": plan.primitive, "candidate_index": plan.candidate_index,
                              "screen_summary": plan.screen_summary} for plan in options]
                            for options in plan_options],
        "candidate_evidence": evidence, "execution": execution,
        "accepted": execution["accepted"], "failed_checks": execution["failed_checks"],
    }
    np.savez_compressed(args.out / "segment_conditions.npz", **condition_arrays,
                        keyframes=world_keyframes, route=world_route,
                        environment_corridor=render_corridor)
    np.savez_compressed(args.out / "executed.npz", **data)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"scenario": args.title, "accepted": report["accepted"],
                      "failed_checks": report["failed_checks"],
                      "routed_primitives": [item["primitive"] for item in decisions],
                      "turn_segments": [item["segment_index"] for item in decisions if item["requires_turn"]],
                      "keyframes_reached": execution["keyframes_reached"],
                      "obstacle_contact_ticks": execution["obstacle_contact_ticks"]}, indent=2))
    return report, data


def main() -> int:
    parser = argparse.ArgumentParser(description="environment-manifold adaptive Stage-2 route")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--title", default="MANIFOLD-ADAPTIVE ROUTE")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows", type=Path, default=C.REPO / "reports/manifold_motion/seed_windows_corridor_stage2_v2/seed_stage2_windows.npz")
    parser.add_argument("--autoencoder", type=Path, default=C.REPO / "reports/manifold_motion/stage2_flow_corridor_v1/autoencoder.pt")
    parser.add_argument("--flow", type=Path, default=C.REPO / "reports/manifold_motion/stage2_flow_corridor_v1/flow.pt")
    parser.add_argument("--mean-model-root", type=Path, default=None,
                        help="optional root containing primitive<ID>/conditional_mean.pt")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--goal-x", type=float, default=3.60)
    parser.add_argument("--segment-length-m", type=float, default=0.60)
    parser.add_argument("--crouch-semi-z-m", type=float, default=1.12)
    parser.add_argument("--side-semi-y-m", type=float, default=0.40)
    parser.add_argument("--planner-body-radius-m", type=float, default=0.46,
                        help="A* footprint radius; reduce only with a physically validated compact primitive")
    parser.add_argument("--planner-clearance-m", type=float, default=0.12)
    parser.add_argument("--self-manifold-clearance-m", type=float, default=0.02,
                        help="minimum exact G1 surface-to-obstacle clearance for deployment gate")
    parser.add_argument("--turn-threshold-rad", type=float, default=0.28)
    parser.add_argument("--keyframe-tolerance-m", type=float, default=0.22)
    parser.add_argument("--warmup-ticks", type=int, default=20)
    parser.add_argument("--max-ticks", type=int, default=1800)
    parser.add_argument("--fall-height-m", type=float, default=0.30)
    parser.add_argument("--max-roll-deg", type=float, default=45.0)
    parser.add_argument("--max-tracking-error-rad", type=float, default=0.35)
    parser.add_argument("--max-route-deviation-m", type=float, default=0.45)
    parser.add_argument("--max-switch-rms-rad", type=float, default=0.75)
    parser.add_argument("--handoff-blend-ticks", type=int, default=12,
                        help="preview-space crossfade length for semantic primitive changes")
    parser.add_argument("--side-body-yaw-offset-rad", type=float, default=0.0,
                        help="side-gait body yaw offset relative to route tangent; zero keeps legacy alignment")
    parser.add_argument("--max-side-body-yaw-error-p95-deg", type=float, default=35.0)
    parser.add_argument("--max-body-route-yaw-p95-deg", type=float, default=55.0,
                        help="reject sustained body/route misalignment outside turn helper ticks")
    parser.add_argument("--action-hold-ticks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--crouch-locomotion-source-index", type=int, default=590)
    parser.add_argument("--side-locomotion-source-index", type=int, default=1375)
    parser.add_argument("--side-on-source-index", type=int, default=1450,
                        help="SEED window with measured lateral displacement for strict side-on gait")
    parser.add_argument("--side-gait-mode", choices=("side_on", "diagonal"), default="side_on",
                        help="side_on requires p4 displacement heading near +/-90 degrees")
    parser.add_argument("--side-heading-tolerance-rad", type=float, default=0.55,
                        help="allowed deviation from +/-90 degrees for side-on candidates")
    parser.add_argument("--online-condition-iterations", type=int, default=1,
                        help="probe/recondition passes; 0 uses the SEED anchor state only")
    parser.add_argument("--receding-horizon-ticks", type=int, default=0,
                        help="refresh the future Flow reference every N control ticks (0 disables)")
    parser.add_argument("--receding-horizon-shadow", action="store_true",
                        help="generate/score online candidates but keep the verified reference")
    parser.add_argument("--progress-horizon-s", type=float, default=1.6,
                        help="time constant for measured route-progress velocity command")
    parser.add_argument("--disable-anchor", action="store_true",
                        help="disable hand-selected raw SEED anchors; learned conditional mean remains")
    parser.add_argument("--disable-learned-anchor", action="store_true",
                        help="strictly remove the training-state learned support anchor as well")
    parser.add_argument("--pure-stochastic-flow", action="store_true",
                        help="strict stochastic ablation: also disable the learned conditional-mean candidate")
    parser.add_argument("--projection-iterations", type=int, default=1)
    parser.add_argument("--projection-step-size", type=float, default=0.08)
    parser.add_argument("--projection-smooth-weight", type=float, default=0.01)
    parser.add_argument("--projection-velocity-weight", type=float, default=0.002)
    parser.add_argument("--projection-acceleration-weight", type=float, default=0.01)
    parser.add_argument("--projection-jerk-weight", type=float, default=0.004)
    parser.add_argument("--projection-handoff-weight", type=float, default=0.35)
    parser.add_argument("--projection-corridor-weight", type=float, default=0.85)
    parser.add_argument("--projection-root-blend", type=float, default=0.85)
    parser.add_argument("--projection-max-joint-step", type=float, default=0.005)
    parser.add_argument("--projection-max-root-step-m", type=float, default=0.05)
    parser.add_argument("--min-forward-progress-m", type=float, default=0.15,
                        help="hard route-tangent progress gate for locomotion candidates")
    parser.add_argument("--max-candidate-heading-error-rad", type=float, default=0.96,
                        help="hard body/displacement heading gate (0.96 rad ~= 55 degrees)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    if (args.num_candidates < 2 or args.online_condition_iterations < 0
            or args.receding_horizon_ticks < 0 or args.progress_horizon_s <= 0
            or args.projection_iterations < 0
            or min(args.goal_x, args.segment_length_m, args.fps,
                   args.min_forward_progress_m, args.max_candidate_heading_error_rad,
                   args.planner_body_radius_m, args.planner_clearance_m,
                   args.projection_step_size, args.projection_smooth_weight,
                   args.projection_velocity_weight, args.projection_acceleration_weight,
                   args.projection_jerk_weight, args.projection_handoff_weight,
                   args.projection_corridor_weight, args.projection_root_blend,
                   args.projection_max_joint_step, args.projection_max_root_step_m) <= 0):
        parser.error("candidate count and geometric parameters must be positive")
    if args.self_manifold_clearance_m < 0:
        parser.error("self-manifold clearance must be non-negative")
    report, _ = run(args)
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
