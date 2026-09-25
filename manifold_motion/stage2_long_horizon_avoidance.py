"""Closed-loop long-horizon obstacle avoidance with keyframe-safe primitive switching.

This experiment closes the gap between the short, independently replayed Stage-2 clips and
an actual navigation task.  It plans around the physical ``obstacle_*`` boxes in MuJoCo,
turns the A* polyline into a continuous route/corridor condition, then executes the whole
task in one simulator rollout.  The waypoint controller changes only the reference supplied
to frozen SONIC:

* ``walk_turn`` while the measured pelvis heading is far from the next keyframe;
* ``walk_nominal`` once it is aligned, so the gait advances along the segment.

A keyframe is released only after the *executed* pelvis reaches it.  There is no simulator
reset or state teleport at a primitive boundary.  Completion, collision, tracking, stability,
route deviation, and hand-off continuity are all reported from the physics rollout.
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
from PIL import Image, ImageDraw, ImageFont

from . import constants as C
from .corridor import ExecutedEnvelopeEstimator
from .env import G1FlatEnv
from .perception_corridor import PerceptionGridConfig, corridor_from_perception
from .reference import ReferenceBuffer
from .scene_pointcloud import obstacle_pointcloud
from .seed_replay import _ContactMonitor, _roll_degrees
from .sonic import SonicController
from .stage2_validate import _generated_motion
from .dynamic_scene import apply_dynamic_obstacle, obstacle_state


@dataclass(frozen=True)
class Box2D:
    name: str
    centre: np.ndarray
    half: np.ndarray

    def contains(self, point: np.ndarray, inflation: float = 0.0) -> bool:
        return bool(np.all(np.abs(np.asarray(point) - self.centre) <= self.half + inflation))


@dataclass(frozen=True)
class PlannerConfig:
    x_bounds: tuple[float, float] = (-0.05, 3.65)
    y_bounds: tuple[float, float] = (-1.35, 1.35)
    resolution_m: float = 0.05
    body_radius_m: float = 0.46
    clearance_m: float = 0.12
    route_spacing_m: float = 0.07

    @property
    def inflation_m(self) -> float:
        return self.body_radius_m + self.clearance_m


def _boxes(scene: Path) -> list[Box2D]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    boxes: list[Box2D] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name.startswith("obstacle_"):
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError(f"A* fixture supports box obstacles only, got {name}")
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        if not np.allclose(rotation, np.eye(3), atol=1e-6):
            raise ValueError(f"A* fixture requires axis-aligned boxes, got rotated {name}")
        boxes.append(Box2D(name, data.geom_xpos[geom_id, :2].copy(),
                           model.geom_size[geom_id, :2].copy()))
    if not boxes:
        raise ValueError(f"{scene} contains no obstacle_* boxes")
    return boxes


def _grid_axes(config: PlannerConfig) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(config.x_bounds[0], config.x_bounds[1] + config.resolution_m * 0.5,
                   config.resolution_m)
    ys = np.arange(config.y_bounds[0], config.y_bounds[1] + config.resolution_m * 0.5,
                   config.resolution_m)
    return xs, ys


def _occupied(point: np.ndarray, boxes: Iterable[Box2D], inflation: float) -> bool:
    return any(box.contains(point, inflation) for box in boxes)


def _astar(start: np.ndarray, goal: np.ndarray, boxes: list[Box2D],
           config: PlannerConfig) -> np.ndarray:
    xs, ys = _grid_axes(config)

    def cell(point: np.ndarray) -> tuple[int, int]:
        return (int(np.argmin(np.abs(xs - point[0]))), int(np.argmin(np.abs(ys - point[1]))))

    def point(node: tuple[int, int]) -> np.ndarray:
        return np.array([xs[node[0]], ys[node[1]]], dtype=np.float64)

    start_node, goal_node = cell(start), cell(goal)
    for label, node in (("start", start_node), ("goal", goal_node)):
        if _occupied(point(node), boxes, config.inflation_m):
            raise RuntimeError(f"{label} lies inside an inflated obstacle")

    neighbours = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0),
                  (-1, -1), (0, -1), (1, -1)]
    frontier: list[tuple[float, int, tuple[int, int]]] = [(0.0, 0, start_node)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_node: None}
    cost = {start_node: 0.0}
    serial = 0
    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal_node:
            break
        for dx, dy in neighbours:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < len(xs) and 0 <= nxt[1] < len(ys)):
                continue
            nxt_point = point(nxt)
            if _occupied(nxt_point, boxes, config.inflation_m):
                continue
            if dx and dy:
                # Do not cut through the corner of an inflated box.
                if (_occupied(point((current[0] + dx, current[1])), boxes, config.inflation_m)
                        or _occupied(point((current[0], current[1] + dy)), boxes,
                                     config.inflation_m)):
                    continue
            step = math.hypot(dx, dy) * config.resolution_m
            # The scene is symmetric.  A tiny deterministic cost selects the +y route and
            # makes repeated experiment renders directly comparable.
            side_preference = max(0.0, -float(nxt_point[1])) * 1e-4
            candidate = cost[current] + step + side_preference
            if candidate >= cost.get(nxt, float("inf")):
                continue
            cost[nxt] = candidate
            came_from[nxt] = current
            heuristic = float(np.linalg.norm(nxt_point - point(goal_node)))
            serial += 1
            heapq.heappush(frontier, (candidate + heuristic, serial, nxt))
    if goal_node not in came_from:
        raise RuntimeError("A* found no collision-free route")
    nodes: list[tuple[int, int]] = []
    node: tuple[int, int] | None = goal_node
    while node is not None:
        nodes.append(node)
        node = came_from[node]
    nodes.reverse()
    route = np.asarray([point(node) for node in nodes])
    route[0], route[-1] = start, goal
    return route


def _segment_free(a: np.ndarray, b: np.ndarray, boxes: list[Box2D],
                  config: PlannerConfig) -> bool:
    length = float(np.linalg.norm(b - a))
    samples = max(2, int(math.ceil(length / (config.resolution_m * 0.35))) + 1)
    for weight in np.linspace(0.0, 1.0, samples):
        if _occupied((1.0 - weight) * a + weight * b, boxes, config.inflation_m):
            return False
    return True


def _simplify(route: np.ndarray, boxes: list[Box2D], config: PlannerConfig) -> np.ndarray:
    result = [route[0]]
    current = 0
    while current < len(route) - 1:
        nxt = len(route) - 1
        while nxt > current + 1 and not _segment_free(route[current], route[nxt], boxes, config):
            nxt -= 1
        result.append(route[nxt])
        current = nxt
    return np.asarray(result)


def _densify(keyframes: np.ndarray, spacing: float) -> np.ndarray:
    pieces = []
    for index, (start, stop) in enumerate(zip(keyframes[:-1], keyframes[1:])):
        count = max(2, int(math.ceil(float(np.linalg.norm(stop - start)) / spacing)) + 1)
        segment = np.linspace(start, stop, count)
        pieces.append(segment if index == 0 else segment[1:])
    return np.concatenate(pieces)


def _route_yaw(route: np.ndarray) -> np.ndarray:
    delta = np.gradient(route, axis=0)
    return np.arctan2(delta[:, 1], delta[:, 0])


def _angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _yaw(quat: np.ndarray) -> float:
    forward = C.quat_rotate(quat, np.array([1.0, 0.0, 0.0]))
    return float(np.arctan2(forward[1], forward[0]))


def _distance_to_polyline(points: np.ndarray, route: np.ndarray) -> np.ndarray:
    starts, delta = route[:-1], np.diff(route, axis=0)
    denom = np.maximum(np.sum(delta * delta, axis=1), 1e-10)
    values = []
    for point in points:
        weight = np.clip(np.sum((point - starts) * delta, axis=1) / denom, 0.0, 1.0)
        closest = starts + weight[:, None] * delta
        values.append(float(np.min(np.linalg.norm(closest - point, axis=1))))
    return np.asarray(values)


def _load_motion(sample: Path):
    return _generated_motion(sample, source_hz=30.0, control_hz=1.0 / C.CONTROL_DT)


def _rolling_reference(motion, phase: int, desired_yaw: float,
                       horizon: int = 50) -> tuple[ReferenceBuffer, int]:
    # SONIC encodes ten future frames up to 45 ticks ahead.  Clamping that preview to the
    # terminal pose near the end of every short generated clip, then jumping back to frame 6,
    # makes a physically valid one-window gait reverse or drift when repeated.  Present a
    # genuinely cyclic preview over the interior frames instead.  Frame 6 skips the generated
    # boundary transient and is also the cycle entry used by the executor.
    cycle_start = min(6, max(0, motion.T - 2))
    cycle_stop = max(cycle_start + 1, motion.T - 1)  # exclusive; omit terminal duplicate
    cycle_length = cycle_stop - cycle_start
    phase = cycle_start + (int(phase) - cycle_start) % cycle_length
    indices = cycle_start + (np.arange(horizon) + phase - cycle_start) % cycle_length
    joint_pos = motion.joint_pos_policy[indices]
    joint_vel = motion.joint_vel_policy[indices]
    root_pos = np.zeros((horizon, 3), dtype=np.float64)
    # Preserve the generated time-varying root orientation: it is one of the G1 SONIC
    # encoder's required observations.  Root *position* is not consumed by this controller,
    # but flattening root orientation to a constant changes the policy token and made a
    # forward crouch candidate drift in long execution despite passing isolated replay.
    # Apply one yaw alignment to the entire orientation sequence so its cycle remains intact
    # while its anchor faces the route.
    anchor_yaw = _yaw(motion.root_quat[cycle_start])
    align = C.quat_from_yaw(_angle(desired_yaw - anchor_yaw))
    root_quat = np.asarray([C.quat_mul(align, motion.root_quat[index]) for index in indices])
    reference = ReferenceBuffer(joint_pos, joint_vel, root_pos, root_quat, play=True)
    next_phase = cycle_start + (phase + 1 - cycle_start) % cycle_length
    return reference, next_phase


def _calibrated_envelope(walk, turn) -> dict[str, list[float]]:
    estimator = ExecutedEnvelopeEstimator()
    values = []
    for motion in (walk, turn):
        indices = np.unique(np.linspace(0, motion.T - 1, min(24, motion.T), dtype=int))
        values.append(estimator.sequence(
            motion.joint_pos_policy[indices], np.zeros((len(indices), 3)),
            np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]]), len(indices), axis=0)
        ))
    all_values = np.concatenate(values)
    return {
        "semi_min_m": all_values.min(axis=0).astype(float).tolist(),
        "semi_mean_m": all_values.mean(axis=0).astype(float).tolist(),
        "semi_max_m": all_values.max(axis=0).astype(float).tolist(),
    }


def _condition(route: np.ndarray, pointcloud: np.ndarray,
               envelope: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    # The full-body fitted ellipsoid is retained as calibration metadata, but it is overly
    # conservative in x/y because one ellipsoid must cover head, arms and feet simultaneously.
    # The navigation tube therefore uses the same calibrated circular footprint as A*; exact
    # G1 meshes and contacts remain the final physical safety gate.
    maximum = np.asarray(envelope["semi_max_m"], dtype=np.float64)
    proposed_semi = np.array([PlannerConfig.body_radius_m, PlannerConfig.body_radius_m,
                              maximum[2] + 0.08])
    proposed = np.repeat(proposed_semi[None, :], len(route), axis=0)
    route3 = np.column_stack([route, np.zeros(len(route))])
    # The default SDF volume covers only a short local horizon.  This experiment's saved
    # condition spans the complete long task, so expand the diagnostic raster accordingly.
    grid = PerceptionGridConfig(
        shape=(18, 16, 8), lower_m=(-0.3, -1.6, -0.2), upper_m=(4.0, 1.6, 1.6),
        obstacle_inflation_m=0.10, max_sdf_m=1.0, min_corridor_semi_m=0.06,
    )
    return corridor_from_perception(
        route3, proposed, pointcloud, route_yaw_local_rad=_route_yaw(route),
        config=grid, clearance_m=0.08,
    )


def execute(scene: Path, walk_sample: Path, turn_sample: Path, keyframes: np.ndarray,
            dense_route: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict]:
    walk, turn = _load_motion(walk_sample), _load_motion(turn_sample)
    motions = {"walk_nominal": walk, "walk_turn": turn}
    env = G1FlatEnv(scene)
    controller = SonicController()
    contacts = _ContactMonitor(env.model)
    env.reset(joint_offset=walk.joint_pos_hw[0] - C.DEFAULT_ANGLES)
    controller.reset()

    # Fill the policy history without advancing the task or resetting it later.
    hold_q = np.repeat(walk.joint_pos_policy[0:1], 50, axis=0)
    hold = ReferenceBuffer(hold_q, np.zeros_like(hold_q), np.zeros((50, 3)),
                           np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]]), 50, axis=0))
    for _ in range(args.warmup_ticks):
        state = env.state()
        controller.append_state(state["q_hw"], state["dq_hw"], state["base_quat"],
                                state["base_ang_vel"])
        _, target, _ = controller.act(hold, state["base_quat"])
        env.set_target(target)
        env.step()

    start_world = env.state()["base_pos"][:2].copy()
    world_keyframes = keyframes + start_world
    world_route = dense_route + start_world
    target_index = 1
    phases = {name: min(6, motion.T - 1) for name, motion in motions.items()}
    active_previous: str | None = None
    previous_q_ref: np.ndarray | None = None
    switch_events: list[dict] = []
    reached: list[dict] = []
    log_keys = (
        "t", "q_ref", "q_exec", "dq_exec", "base_pos", "base_quat", "action",
        "foot_contact", "hand_contact", "nonfoot_floor_contact", "obstacle_contact",
        "active_primitive", "target_keyframe", "desired_yaw", "yaw_error",
    )
    log: dict[str, list] = {key: [] for key in log_keys}

    for tick in range(args.max_ticks):
        state = env.state()
        position = state["base_pos"][:2]
        while target_index < len(world_keyframes):
            error = float(np.linalg.norm(world_keyframes[target_index] - position))
            if error > args.keyframe_tolerance_m:
                break
            reached.append({
                "keyframe_index": target_index,
                "tick": tick,
                "planned_xy_m": world_keyframes[target_index].astype(float).tolist(),
                "executed_xy_m": position.astype(float).tolist(),
                "position_error_m": error,
                "q_tracking_rms_rad": (float(np.sqrt(np.mean((previous_q_ref -
                    state["q_hw"][C.MUJOCO_TO_ISAACLAB]) ** 2)))
                    if previous_q_ref is not None else 0.0),
            })
            target_index += 1
        if target_index >= len(world_keyframes):
            break

        delta = world_keyframes[target_index] - position
        desired_yaw = float(np.arctan2(delta[1], delta[0]))
        yaw_error = _angle(desired_yaw - _yaw(state["base_quat"]))
        active = "walk_turn" if abs(yaw_error) > args.turn_threshold_rad else "walk_nominal"
        motion = motions[active]
        reference, phases[active] = _rolling_reference(
            motion, phases[active], desired_yaw, horizon=50
        )
        q_ref = reference.joint_pos[0].copy()
        if active != active_previous:
            switch_events.append({
                "tick": tick,
                "from": active_previous or "warmup_hold",
                "to": active,
                "target_keyframe": target_index,
                "yaw_error_rad": yaw_error,
                "pre_switch_joint_rms_rad": (float(np.sqrt(np.mean((previous_q_ref - q_ref) ** 2)))
                    if previous_q_ref is not None else 0.0),
            })
            active_previous = active

        controller.append_state(state["q_hw"], state["dq_hw"], state["base_quat"],
                                state["base_ang_vel"])
        action, target, _ = controller.act(reference, state["base_quat"])
        env.set_target(target)
        env.step()
        executed = env.state()
        feet, hands, nonfoot, obstacle = contacts.flags(env.data)
        log["t"].append(env.time)
        log["q_ref"].append(q_ref)
        log["q_exec"].append(executed["q_hw"][C.MUJOCO_TO_ISAACLAB])
        log["dq_exec"].append(executed["dq_hw"][C.MUJOCO_TO_ISAACLAB])
        log["base_pos"].append(executed["base_pos"])
        log["base_quat"].append(executed["base_quat"])
        log["action"].append(action)
        log["foot_contact"].append(feet)
        log["hand_contact"].append(hands)
        log["nonfoot_floor_contact"].append(nonfoot)
        log["obstacle_contact"].append(obstacle)
        log["active_primitive"].append(0 if active == "walk_nominal" else 1)
        log["target_keyframe"].append(target_index)
        log["desired_yaw"].append(desired_yaw)
        log["yaw_error"].append(yaw_error)
        previous_q_ref = q_ref

    data = {key: np.asarray(value) for key, value in log.items()}
    if not len(data["t"]):
        raise RuntimeError("long-horizon runner produced no execution ticks")
    positions = data["base_pos"][:, :2]
    route_deviation = _distance_to_polyline(positions, world_route)
    tracking = np.abs(data["q_ref"] - data["q_exec"])
    roll = np.abs(_roll_degrees(data["base_quat"]))
    final_error = float(np.linalg.norm(positions[-1] - world_keyframes[-1]))
    failed = []
    if target_index < len(world_keyframes):
        failed.append("goal_or_keyframe_not_reached")
    if bool(data["obstacle_contact"].any()):
        failed.append("scene_obstacle_contact")
    if float(data["base_pos"][:, 2].min()) < args.fall_height_m or float(roll.max()) > 60.0:
        failed.append("fall_or_extreme_roll")
    if float(roll.max()) > args.max_roll_deg:
        failed.append("roll_limit")
    if float(tracking.mean()) > args.max_tracking_error_rad:
        failed.append("tracking_error")
    if float(np.quantile(route_deviation, 0.95)) > args.max_route_deviation_m:
        failed.append("route_deviation")
    max_switch = max((float(item["pre_switch_joint_rms_rad"]) for item in switch_events), default=0.0)
    if max_switch > args.max_switch_rms_rad:
        failed.append("handoff_discontinuity")
    if final_error > args.keyframe_tolerance_m:
        failed.append("terminal_keyframe_error")
    summary = {
        "accepted": not failed,
        "failed_checks": failed,
        "no_reset_between_primitives": True,
        "physics_ticks": int(len(data["t"])),
        "duration_s": float(len(data["t"]) * C.CONTROL_DT),
        "keyframes_total_excluding_start": int(len(keyframes) - 1),
        "keyframes_reached": int(len(reached)),
        "keyframe_events": reached,
        "terminal_error_m": final_error,
        "route_deviation_mean_m": float(route_deviation.mean()),
        "route_deviation_p95_m": float(np.quantile(route_deviation, 0.95)),
        "route_deviation_max_m": float(route_deviation.max()),
        "track_err_mean_rad": float(tracking.mean()),
        "track_err_legs_rad": float(tracking[:, :12].mean()),
        "base_z_min_m": float(data["base_pos"][:, 2].min()),
        "roll_abs_max_deg": float(roll.max()),
        "obstacle_contact_ticks": int(data["obstacle_contact"].sum()),
        "nonfoot_floor_ratio": float(data["nonfoot_floor_contact"].mean()),
        "primitive_switches": switch_events,
        "primitive_switch_count": int(max(0, len(switch_events) - 1)),
        "max_pre_switch_joint_rms_rad": max_switch,
        "walk_nominal_ticks": int(np.sum(data["active_primitive"] == 0)),
        "walk_turn_ticks": int(np.sum(data["active_primitive"] == 1)),
        "start_world_xy_m": start_world.astype(float).tolist(),
        "final_world_xy_m": positions[-1].astype(float).tolist(),
    }
    return data, summary


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _add_sphere(scene: mujoco.MjvScene, position: np.ndarray, radius: float,
                color: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0.0, 0.0]), position, np.eye(3).ravel(),
                        np.asarray(color, dtype=np.float32))
    scene.ngeom += 1


def _add_ellipsoid(scene: mujoco.MjvScene, element: np.ndarray,
                   color: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    yaw = float(element[6])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                         [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        element[3:6], element[:3], rotation.ravel(),
                        np.asarray(color, dtype=np.float32))
    scene.ngeom += 1


def _plan_pixels(points: np.ndarray, rect: tuple[int, int, int, int],
                 bounds: tuple[tuple[float, float], tuple[float, float]]) -> list[tuple[int, int]]:
    left, top, right, bottom = rect
    (xmin, xmax), (ymin, ymax) = bounds
    return [(int(left + (point[0] - xmin) / (xmax - xmin) * (right - left)),
             int(bottom - (point[1] - ymin) / (ymax - ymin) * (bottom - top)))
            for point in points]


def render(scene_path: Path, data: dict[str, np.ndarray], summary: dict,
           keyframes: np.ndarray, route: np.ndarray, corridor: np.ndarray,
           boxes: list[Box2D], planner: PlannerConfig, out: Path, fps: float,
           self_manifold: np.ndarray | None = None,
           safe_manifold: np.ndarray | None = None,
           dynamic_obstacle_event: str | None = None,
           output_scale: float = 1.0) -> None:
    # The bundled G1 XML has a 640-pixel offscreen framebuffer.
    width, scene_height, plan_height, header = 640, 430, 245, 105
    env = G1FlatEnv(scene_path)
    renderer = mujoco.Renderer(env.model, height=scene_height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (1.8, 0.0, 0.55)
    camera.distance, camera.azimuth, camera.elevation = 5.2, 90.0, -42.0
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    title_font, body_font, small_font = _font(21), _font(16), _font(14)
    frames: list[Image.Image] = []
    stride = max(1, int(round((1.0 / C.CONTROL_DT) / fps)))
    ticks = list(range(0, len(data["t"]), stride))
    if ticks[-1] != len(data["t"]) - 1:
        ticks.append(len(data["t"]) - 1)
    try:
        for tick in ticks:
            env.data.qpos[:3] = data["base_pos"][tick]
            env.data.qpos[3:7] = data["base_quat"][tick]
            env.data.qpos[env.body_qadr] = data["q_exec"][tick][C.ISAACLAB_TO_MUJOCO]
            env.data.qpos[env.hand_qadr] = 0.0
            env.data.qvel[:] = 0.0
            if dynamic_obstacle_event is None:
                mujoco.mj_forward(env.model, env.data)
            else:
                apply_dynamic_obstacle(env.model, env.data, dynamic_obstacle_event,
                                       tick * C.CONTROL_DT)
            renderer.update_scene(env.data, camera=camera, scene_option=option)
            mjscene = renderer.scene
            for point in route[::max(1, len(route) // 45)]:
                _add_sphere(mjscene, np.array([point[0], point[1], 0.035]), 0.022,
                            (0.1, 0.9, 1.0, 0.75))
            for index, point in enumerate(keyframes):
                reached = index <= int(data["target_keyframe"][tick]) - 1
                color = (0.2, 1.0, 0.35, 0.95) if reached else (1.0, 0.85, 0.1, 0.95)
                _add_sphere(mjscene, np.array([point[0], point[1], 0.09]), 0.055, color)
            for point in data["base_pos"][max(0, tick - 250):tick + 1:4]:
                _add_sphere(mjscene, point, 0.015, (1.0, 0.45, 0.08, 0.88))
            route_index = int(np.argmin(np.linalg.norm(route - data["base_pos"][tick, :2], axis=1)))
            _add_ellipsoid(mjscene, corridor[route_index], (0.18, 0.58, 1.0, 0.065))
            if safe_manifold is not None:
                _add_ellipsoid(mjscene, safe_manifold[tick], (0.85, 0.95, 1.0, 0.12))
            if self_manifold is not None:
                primitive_id = int(data["active_primitive"][tick])
                self_color = {
                    2: (1.0, 0.42, 0.12, 0.22),   # crouch: orange, lower z
                    4: (0.15, 1.0, 0.55, 0.24),   # side: green, narrower y
                    5: (1.0, 0.82, 0.10, 0.18),   # nominal: yellow
                    6: (0.78, 0.35, 1.0, 0.20),   # turn: violet
                }.get(primitive_id, (1.0, 0.82, 0.10, 0.18))
                _add_ellipsoid(mjscene, self_manifold[tick], self_color)
            rendered = Image.fromarray(np.asarray(renderer.render()), mode="RGB")

            canvas = Image.new("RGB", (width, header + scene_height + plan_height), (11, 15, 22))
            canvas.paste(rendered, (0, header))
            draw = ImageDraw.Draw(canvas)
            if "active_primitive_name" in data:
                active = str(data["active_primitive_name"][tick])
            else:
                active = "walk_nominal" if int(data["active_primitive"][tick]) == 0 else "walk_turn"
            target = int(data["target_keyframe"][tick])
            draw.rectangle((0, 0, width, header), fill=(17, 23, 33))
            render_title = str(summary.get("render_title", "LONG-HORIZON A* OBSTACLE AVOIDANCE"))
            draw.text((16, 10), render_title, font=title_font,
                      fill=(245, 247, 250))
            draw.text((16, 43), f"active primitive: {active}    target keyframe: {target}/{len(keyframes)-1}",
                      font=body_font, fill=(105, 222, 255))
            if self_manifold is not None:
                self_semi = self_manifold[tick, 3:6]
                draw.text((16, 67),
                          "blue=M_e   white=M_r^safe   colored=M_r^task",
                          font=small_font, fill=(255, 220, 130))
                draw.text((16, 86),
                          f"task semi=({self_semi[0]:.2f},{self_semi[1]:.2f},{self_semi[2]:.2f}) m | NO RESET",
                          font=small_font, fill=(105, 238, 135))
            else:
                draw.text((16, 70), "single MuJoCo rollout | keyframe-gated switching | NO RESET",
                          font=small_font, fill=(105, 238, 135))

            plan_top = header + scene_height
            draw.rectangle((0, plan_top, width, plan_top + plan_height), fill=(15, 21, 30))
            rect = (48, plan_top + 30, width - 35, plan_top + plan_height - 25)
            bounds = (planner.x_bounds, planner.y_bounds)
            draw.rounded_rectangle(rect, radius=6, fill=(20, 28, 39), outline=(71, 83, 101), width=2)
            dynamic_state = (obstacle_state(dynamic_obstacle_event, tick * C.CONTROL_DT)
                             if dynamic_obstacle_event is not None else None)
            for box in boxes:
                if box.name == "obstacle_dynamic_block" and dynamic_state is not None:
                    if not dynamic_state.active:
                        continue
                    box = Box2D(box.name,
                                np.array(dynamic_state.center_xy, dtype=np.float64),
                                box.half)
                corners = np.array([box.centre - box.half, box.centre + box.half])
                pixels = _plan_pixels(corners, rect, bounds)
                draw.rectangle((pixels[0][0], pixels[1][1], pixels[1][0], pixels[0][1]),
                               fill=(176, 45, 42), outline=(255, 105, 90), width=2)
                inflated = np.array([box.centre - box.half - planner.inflation_m,
                                     box.centre + box.half + planner.inflation_m])
                ip = _plan_pixels(inflated, rect, bounds)
                draw.rectangle((ip[0][0], ip[1][1], ip[1][0], ip[0][1]),
                               outline=(255, 130, 115), width=1)
            route_px = _plan_pixels(route, rect, bounds)
            draw.line(route_px, fill=(35, 220, 248), width=3, joint="curve")
            actual_px = _plan_pixels(data["base_pos"][:tick + 1, :2], rect, bounds)
            if len(actual_px) > 1:
                draw.line(actual_px, fill=(255, 142, 33), width=4, joint="curve")
            for point in _plan_pixels(keyframes, rect, bounds):
                draw.ellipse((point[0]-5, point[1]-5, point[0]+5, point[1]+5),
                             fill=(255, 226, 54))
            draw.text((16, plan_top + 5),
                      f"cyan=A* safe centerline  orange=executed  red=physical / outline=inflated   "
                      f"contact ticks={summary['obstacle_contact_ticks']}",
                      font=small_font, fill=(224, 230, 238))
            frames.append(canvas)
    finally:
        renderer.close()
    if output_scale <= 0.0 or output_scale > 1.0:
        raise ValueError("output_scale must be in (0, 1]")
    if output_scale < 1.0:
        target_size = (max(1, int(round(width * output_scale))),
                       max(1, int(round((header + scene_height + plan_height) * output_scale))))
        resampling = getattr(Image, "Resampling", Image)
        frames = [frame.resize(target_size, resampling.LANCZOS) for frame in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(round(1000.0 / fps)), loop=0, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="closed-loop long-horizon Stage-2 obstacle avoidance")
    parser.add_argument("--scene", type=Path,
                        default=C.REPO / "data/g1_flat/scene_long_avoidance.xml")
    parser.add_argument("--walk-sample", type=Path,
                        default=C.REPO / "reports/manifold_motion/stage2_routed_walk_test_pass/sample.npz")
    parser.add_argument("--turn-sample", type=Path,
                        default=C.REPO / "reports/manifold_motion/stage2_mean_primitive6_test/sample.npz")
    parser.add_argument("--out", type=Path,
                        default=C.REPO / "reports/manifold_motion/stage2_long_horizon_avoidance_v1")
    parser.add_argument("--goal-x", type=float, default=3.60)
    parser.add_argument("--keyframe-tolerance-m", type=float, default=0.30)
    parser.add_argument("--turn-threshold-rad", type=float, default=0.28)
    parser.add_argument("--warmup-ticks", type=int, default=20)
    parser.add_argument("--max-ticks", type=int, default=900)
    parser.add_argument("--fall-height-m", type=float, default=0.30)
    parser.add_argument("--max-roll-deg", type=float, default=45.0)
    parser.add_argument("--max-tracking-error-rad", type=float, default=0.35)
    parser.add_argument("--max-route-deviation-m", type=float, default=0.45)
    parser.add_argument("--max-switch-rms-rad", type=float, default=0.65)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    if args.max_ticks <= 0 or args.warmup_ticks < 0 or min(args.fps, args.goal_x,
            args.keyframe_tolerance_m, args.turn_threshold_rad) <= 0:
        parser.error("tick counts and positive thresholds must be valid")
    args.out.mkdir(parents=True, exist_ok=True)

    planner = PlannerConfig()
    boxes = _boxes(args.scene)
    raw_route = _astar(np.array([0.0, 0.0]), np.array([args.goal_x, 0.0]), boxes, planner)
    keyframes = _simplify(raw_route, boxes, planner)
    dense_route = _densify(keyframes, planner.route_spacing_m)
    walk, turn = _load_motion(args.walk_sample), _load_motion(args.turn_sample)
    envelope = _calibrated_envelope(walk, turn)
    points, obstacle_names = obstacle_pointcloud(args.scene, spacing_m=0.06)
    corridor, sdf = _condition(dense_route, points, envelope)

    data, execution = execute(args.scene, args.walk_sample, args.turn_sample,
                              keyframes, dense_route, args)
    # Warmup drift defines the execution frame.  Rebase planned geometry for saved/rendered
    # world coordinates using exactly the origin reported by execute().
    origin = np.asarray(execution["start_world_xy_m"], dtype=np.float64)
    world_keyframes, world_route = keyframes + origin, dense_route + origin
    world_corridor = corridor.copy()
    world_corridor[:, :2] += origin
    world_corridor[:, 2] += float(data["base_pos"][0, 2])

    condition_report = {
        "source": "A* over inflated physical obstacle_* boxes; point cloud sampled from the same MuJoCo scene",
        "obstacles": obstacle_names,
        "planner_resolution_m": planner.resolution_m,
        "body_radius_m": planner.body_radius_m,
        "clearance_m": planner.clearance_m,
        "inflation_m": planner.inflation_m,
        "raw_astar_points": int(len(raw_route)),
        "keyframes": world_keyframes.astype(float).tolist(),
        "dense_route_points": int(len(world_route)),
        "corridor_frames": int(len(corridor)),
        "corridor_semi_min_m": corridor[:, 3:6].min(axis=0).astype(float).tolist(),
        "sdf_shape": list(sdf.shape),
        "sdf_min_m": float(sdf.min()),
        "sdf_max_m": float(sdf.max()),
        "reference_envelope_calibration": envelope,
    }
    report = {
        "experiment": "Stage-2 long-horizon keyframe-preserving primitive switching",
        "scene": str(args.scene),
        "walk_sample": str(args.walk_sample),
        "turn_sample": str(args.turn_sample),
        "condition": condition_report,
        "execution": execution,
        "accepted": execution["accepted"],
        "failed_checks": execution["failed_checks"],
    }
    np.savez_compressed(args.out / "planned_condition.npz", raw_astar_route=raw_route,
                        keyframes=world_keyframes, route=world_route, corridor=world_corridor,
                        sdf=sdf, obstacle_pointcloud=points)
    np.savez_compressed(args.out / "executed.npz", **data)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    render(args.scene, data, execution, world_keyframes, world_route, world_corridor,
           boxes, planner, args.out / "long_horizon_avoidance.gif", args.fps)
    print(json.dumps({
        "accepted": execution["accepted"],
        "failed_checks": execution["failed_checks"],
        "keyframes_reached": execution["keyframes_reached"],
        "keyframes_total": execution["keyframes_total_excluding_start"],
        "primitive_switch_count": execution["primitive_switch_count"],
        "obstacle_contact_ticks": execution["obstacle_contact_ticks"],
        "terminal_error_m": execution["terminal_error_m"],
        "route_deviation_p95_m": execution["route_deviation_p95_m"],
        "gif": str(args.out / "long_horizon_avoidance.gif"),
    }, indent=2))
    return 0 if execution["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
