"""Stage-2 Flow Matching candidates on A* route segments.

The long-horizon executor previously used two hand-selected SONIC clips.  This module keeps
the same hard physical gate, but replaces those clips with conditional model proposals:

``M_e(segment) + state/history + command + z_p -> {R_ref^k}``

For each route segment, the route-local safe corridor and SDF are packed in the exact 48-frame
Stage-2 condition layout.  A latent Flow Matching model generates several candidates, and each
candidate is replayed through SONIC/MuJoCo before any candidate can enter the continuous task.
The preferred primitive differs by segment (turn, side-step, crouch, nominal walk); it falls
back to another *accepted* candidate only when the preferred one fails the physics gate.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from . import constants as C
from .body_envelope import body_points
from .corridor import ExecutedEnvelopeEstimator
from .perception_corridor import PerceptionGridConfig, corridor_from_perception
from .scene_pointcloud import obstacle_pointcloud
from .seed_replay import ReplayConfig, SeedReplayRunner
from .stage2_flow import (Normalizer, WindowData, _load_ae, _load_flow,
                          _load_mean, _target_from_model)
from .stage2_long_horizon_avoidance import (
    PlannerConfig, _astar, _boxes, _densify, _distance_to_polyline, _roll_degrees,
    _route_yaw, _simplify, _angle, _yaw, _rolling_reference, render,
)
from .stage2_validate import _motion_from_trajectory, validate_trajectory
from .env import G1FlatEnv
from .reference import ReferenceBuffer
from .sonic import SonicController
from .seed_replay import _ContactMonitor
from .seed_windows import _state_features


PRIMITIVE_NAMES = {
    2: "crouch",
    4: "walk_lateral_reverse",
    5: "walk_nominal",
    6: "walk_turn",
}


def _runtime_obstacle_boxes(model: mujoco.MjModel, data: mujoco.MjData
                            ) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Extract the axis-aligned obstacle boxes used by the online self-manifold stop."""
    boxes = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name.startswith("obstacle_"):
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError(f"runtime self-manifold gate requires box obstacle, got {name}")
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        if not np.allclose(rotation, np.eye(3), atol=1e-6):
            raise ValueError(f"runtime self-manifold gate requires axis-aligned obstacle, got {name}")
        boxes.append((name, data.geom_xpos[geom_id].copy(), model.geom_size[geom_id, :3].copy()))
    return boxes


def _runtime_surface_clearance(points: np.ndarray,
                               boxes: list[tuple[str, np.ndarray, np.ndarray]]) -> float:
    """Minimum signed G1-surface distance to the current obstacle representation."""
    minimum = np.inf
    for _, center, half in boxes:
        delta = np.abs(points - center[None, :]) - half[None, :]
        outside = np.maximum(delta, 0.0)
        distance = np.linalg.norm(outside, axis=1)
        inside = np.all(delta <= 0.0, axis=1)
        distance[inside] = np.max(delta[inside], axis=1)
        minimum = min(minimum, float(distance.min()))
    return float(minimum)


@dataclass
class CandidatePlan:
    segment_index: int
    primitive_id: int
    primitive: str
    candidate_index: int
    trajectory: np.ndarray
    motion: Any
    screen_summary: dict[str, Any]


class RouteFlowSampler:
    """Build exact model-condition rows and decode stochastic latent-flow trajectories."""

    def __init__(self, windows: Path, autoencoder: Path, flow: Path, device: str = "cpu",
                 mean_model_root: Path | None = None):
        self.device = torch.device(device)
        self.ae, self.ae_checkpoint = _load_ae(autoencoder, self.device)
        self.flow, self.flow_checkpoint = _load_flow(flow, self.device)
        normalizer = Normalizer.from_state_dict(self.ae_checkpoint["normalizer"])
        self.data = WindowData.load(windows, normalizer=normalizer)
        self.normalizer = normalizer
        self.validated_exemplars = {2: 554, 4: 1348, 5: 2072, 6: 2113}
        if self.flow.condition_dim != self.data.condition.shape[1]:
            raise ValueError("Flow checkpoint and route window condition dimensions differ")
        if tuple(self.data.target_shape) != tuple(self.ae_checkpoint["target_shape"]):
            raise ValueError("route windows and autoencoder target shape differ")
        self.mean_models: dict[int, Any] = {}
        self.mean_normalizers: dict[int, Normalizer] = {}
        for primitive_id in PRIMITIVE_NAMES:
            paths: list[Path] = []
            if mean_model_root is not None:
                suffix = {2: "crouch", 4: "side", 5: "walk"}.get(primitive_id)
                paths.append(mean_model_root / f"primitive{primitive_id}" / "conditional_mean.pt")
                if suffix is not None:
                    paths.append(mean_model_root / f"primitive{primitive_id}_{suffix}" /
                                 "conditional_mean.pt")
            # A deploy run may cover only locomotion primitives. Keep the verified model for
            # turn/crawl as an explicit fallback instead of silently disabling the candidate.
            paths.append(C.REPO / "reports" / "manifold_motion" /
                         f"stage2_mean_primitive{primitive_id}_v1" / "conditional_mean.pt")
            path = next((candidate for candidate in paths if candidate.is_file()), paths[-1])
            if path.is_file():
                model, checkpoint = _load_mean(path, self.device)
                if model.condition_dim == self.data.condition.shape[1]:
                    self.mean_models[primitive_id] = model
                    if "normalizer" in checkpoint:
                        self.mean_normalizers[primitive_id] = Normalizer.from_state_dict(
                            checkpoint["normalizer"]
                        )

    def _window_index(self, primitive_id: int) -> int:
        values = np.flatnonzero(self.data.raw["primitive"] == primitive_id)
        if not len(values):
            raise ValueError(f"route windows contain no primitive {primitive_id}")
        # Use the same physically validated state/history anchors as the Stage-2 primitive
        # demonstrations.  An arbitrary first training row can be a near-terminal/slow phase
        # and make every otherwise valid locomotion candidate stationary.  Environment and
        # command are still replaced by the live route segment below.
        exemplar = self.validated_exemplars.get(primitive_id)
        if exemplar is not None and exemplar < len(self.data.raw["primitive"]):
            if int(self.data.raw["primitive"][exemplar]) == primitive_id:
                return exemplar
        # Fallback to a real training state/history rather than a zero placeholder.
        train = values[self.data.split[values] == 0]
        return int(train[0] if len(train) else values[0])

    @staticmethod
    def _yaw_command(delta_local: np.ndarray, yaw_local: float) -> np.ndarray:
        c, s = np.cos(yaw_local), np.sin(yaw_local)
        # target_ref's rotation6 is [R00,R01,R10,R11,R20,R21].
        return np.asarray([delta_local[0], delta_local[1], 0.0, c, -s, s, c, 0.0, 0.0], dtype=np.float32)

    def condition(self, primitive_id: int, corridor: np.ndarray, sdf: np.ndarray,
                  command: np.ndarray, *, state: np.ndarray | None = None,
                  history: np.ndarray | None = None,
                  normalizer: Normalizer | None = None) -> tuple[np.ndarray, int]:
        index = self._window_index(primitive_id)
        normalizer = self.normalizer if normalizer is None else normalizer
        if corridor.shape != (48, 7) or sdf.shape != (10, 10, 8):
            raise ValueError(f"route condition must be (48,7)/(10,10,8), got {corridor.shape}/{sdf.shape}")
        environment = np.concatenate([
            self.data.raw["manifold"][index].reshape(1, -1),
            corridor.reshape(1, -1), sdf.reshape(1, -1),
        ], axis=1).astype(np.float32)
        one_hot = np.eye(self.data.primitive_count, dtype=np.float32)[[primitive_id]]
        state_value = self.data.raw["state"][index:index + 1] if state is None else np.asarray(state, dtype=np.float32).reshape(1, -1)
        history_value = self.data.raw["history"][index:index + 1] if history is None else np.asarray(history, dtype=np.float32).reshape(1, 12, -1)
        if state_value.shape != (1, self.data.raw["state"].shape[1]):
            raise ValueError(f"online state must have shape ({self.data.raw['state'].shape[1]},), got {state_value.shape}")
        if history_value.shape != (1, 12, self.data.raw["state"].shape[1]):
            raise ValueError(f"online history must have shape (12,{self.data.raw['state'].shape[1]}), got {history_value.shape}")
        row = np.concatenate([
            normalizer.state(state_value).astype(np.float32),
            normalizer.state(history_value).reshape(1, -1).astype(np.float32),
            one_hot,
            normalizer.manifold(environment).astype(np.float32),
            normalizer.command(command[None]).astype(np.float32),
        ], axis=1)
        return row, index

    def sample(self, primitive_id: int, corridor: np.ndarray, sdf: np.ndarray,
               command: np.ndarray, count: int, seed: int, *,
               state: np.ndarray | None = None,
               history: np.ndarray | None = None,
               raw_exemplar: bool = False,
               include_mean_anchor: bool = True) -> tuple[np.ndarray, int]:
        condition_row, source_index = self.condition(
            primitive_id, corridor, sdf, command, state=state, history=history
        )
        condition = torch.as_tensor(condition_row, device=self.device).expand(count, -1)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.no_grad():
            latent = torch.randn((count, self.flow.latent_dim), device=self.device, generator=generator)
            dt = 1.0 / 32.0
            for step in range(32):
                time = torch.full((count,), step * dt, device=self.device)
                latent = latent + dt * self.flow(latent, time, condition)
            normalized = self.ae.decode(latent, condition).cpu().numpy().reshape(
                (count,) + self.data.target_shape
            )
        generated = _target_from_model(
            self.normalizer.inverse_target(normalized), self.data.joint_lower, self.data.joint_upper
        ).astype(np.float32)
        # Add a deterministic conditional-mean anchor as candidate 0 when available.  This is
        # still a model candidate and prevents a stochastic draw from removing a known stable
        # primitive from the candidate set.
        if include_mean_anchor and primitive_id in self.mean_models:
            mean_normalizer = self.mean_normalizers.get(primitive_id, self.normalizer)
            mean_condition_row, _ = self.condition(
                primitive_id, corridor, sdf, command, state=state, history=history,
                normalizer=mean_normalizer,
            )
            with torch.no_grad():
                mean_condition = torch.as_tensor(mean_condition_row, device=self.device)
                mean = self.mean_models[primitive_id](mean_condition).cpu().numpy().reshape(
                    (1,) + self.data.target_shape
                )
            generated[0] = _target_from_model(
                mean_normalizer.inverse_target(mean), self.data.joint_lower, self.data.joint_upper
            )[0]
        if raw_exemplar:
            # Some semantic motions are sparse in the pilot Flow corpus. Keep a physically
            # replayed SEED target as candidate 0 while stochastic Flow samples remain available
            # for diversity. The exact same SONIC/MuJoCo gate still decides whether it is usable.
            generated[0] = self.data.raw["target_ref"][source_index].astype(np.float32)
        return generated, source_index


def _segment_condition(route_segment: np.ndarray, start: np.ndarray, points_world: np.ndarray,
                       envelope_semi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make a route-local 48-frame M_e and the exact fixed Stage-2 SDF shape."""
    delta = route_segment - start[None, :]
    heading = float(np.arctan2(delta[-1, 1], delta[-1, 0]))
    c, s = np.cos(heading), np.sin(heading)
    rotation = np.array([[c, s], [-s, c]])
    local_xy = delta @ rotation.T
    local_route = np.column_stack([local_xy, np.zeros(len(local_xy))])
    local_points = np.column_stack([(points_world[:, :2] - start[None, :]) @ rotation.T,
                                    points_world[:, 2]])
    source_t = np.linspace(0.0, 1.0, len(local_route))
    target_t = np.linspace(0.0, 1.0, 48)
    route = np.stack([np.interp(target_t, source_t, local_route[:, axis]) for axis in range(3)], axis=1)
    semi = np.repeat(np.asarray(envelope_semi, dtype=np.float64)[None, :], 48, axis=0)
    yaw = np.zeros(48, dtype=np.float64)
    config = PerceptionGridConfig()
    corridor, sdf = corridor_from_perception(
        route, semi, local_points, route_yaw_local_rad=yaw, config=config, clearance_m=0.08
    )
    command = RouteFlowSampler._yaw_command(route[-1], 0.0)
    return corridor, sdf, command


def _candidate_screen(trajectory: np.ndarray, primitive_id: int, source: Path,
                      runner: SeedReplayRunner) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    # The stored generated root displacement is only a training target.  In this route task the
    # actual navigation progress is measured by the continuous executor, so candidate screening
    # checks stability/tracking/contact but does not reject a side/crouch/turn candidate merely
    # because its isolated source clip has little planar displacement.
    stratum = "candidate_physical_gate"
    return validate_trajectory(
        trajectory, source=source, source_hz=30.0, config=ReplayConfig(),
        stratum=stratum, runner=runner,
    )


def _screen_cost(summary: dict[str, Any], trajectory: np.ndarray) -> float:
    failed = len(summary.get("failed_checks", []))
    return (1_000_000.0 * (not summary.get("accepted", False)) + 10_000.0 * failed
            + 10.0 * float(summary.get("track_err_mean_rad", 10.0))
            + 1e-4 * float(np.mean(np.diff(trajectory[:, :29], n=2, axis=0) ** 2)))


def _route_segments(dense: np.ndarray, keyframes: np.ndarray) -> list[np.ndarray]:
    pieces = []
    start = 0
    for stop_point in keyframes[1:]:
        stop = int(np.argmin(np.linalg.norm(dense - stop_point[None, :], axis=1)))
        pieces.append(dense[start:stop + 1])
        start = stop
    return pieces


def _route_progress(point: np.ndarray, route: np.ndarray) -> tuple[float, float]:
    """Project one xy point onto a polyline and return (arc length, distance)."""
    starts, delta = route[:-1], np.diff(route, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    denom = np.maximum(lengths * lengths, 1e-10)
    weights = np.clip(np.sum((point[None, :] - starts) * delta, axis=1) / denom, 0.0, 1.0)
    closest = starts + weights[:, None] * delta
    distance = np.linalg.norm(closest - point[None, :], axis=1)
    index = int(np.argmin(distance))
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    return float(cumulative[index] + weights[index] * lengths[index]), float(distance[index])


def _handoff_phase(motion: Any, previous_q: np.ndarray | None, default: int = 6,
                   search_stop: int | None = None) -> int:
    """Find the nearest pose in a new clip instead of resetting every route segment.

    Consecutive segments often select the same semantic primitive but receive slightly
    different Flow outputs because M_e changes.  Restarting each clip at frame six repeated
    its transition before every obstacle interval.  Nearest-pose phase alignment preserves
    the gait/crouch phase and minimizes the reference discontinuity at the handoff.
    """
    stop = max(1, motion.T - 1)
    if search_stop is not None:
        stop = max(1, min(stop, int(search_stop)))
    if previous_q is None:
        return min(default, stop - 1)
    error = np.sqrt(np.mean((motion.joint_pos_policy[:stop] - previous_q[None, :]) ** 2, axis=1))
    return int(np.argmin(error))


def _clone_sonic_controller(controller: SonicController) -> SonicController:
    """Clone mutable SONIC history while sharing immutable ONNX Runtime sessions."""
    clone = SonicController.__new__(SonicController)
    for name in ("encoder", "decoder", "enc_layout", "dec_layout", "mode_id", "required",
                 "enc_dim", "dec_dim"):
        setattr(clone, name, getattr(controller, name))
    clone.enc_in = controller.enc_in.copy()
    clone.dec_in = controller.dec_in.copy()
    clone.history = deque(({
        key: value.copy() for key, value in row.items()
    } for row in controller.history), maxlen=10)
    clone.last_action = controller.last_action.copy()
    clone.delta_heading = (None if controller.delta_heading is None
                           else controller.delta_heading.copy())
    return clone


def _shadow_screen_current_state(scene: Path, env: G1FlatEnv, controller: SonicController,
                                 plan: CandidatePlan, previous_q: np.ndarray | None,
                                 args: argparse.Namespace) -> dict[str, Any]:
    """Short current-state MuJoCo rollout used before an online semantic switch.

    This is deliberately different from the offline reset screen: qpos/qvel, mocap state and
    the ten-frame SONIC decoder history are copied from the live executor.  The shadow never
    mutates the live world.
    """
    shadow = G1FlatEnv(scene)
    if shadow.model.nq != env.model.nq or shadow.model.nv != env.model.nv:
        return {"accepted": False, "failed_checks": ["shadow_model_shape_mismatch"]}
    shadow.data.qpos[:] = env.data.qpos
    shadow.data.qvel[:] = env.data.qvel
    if shadow.data.act.size == env.data.act.size:
        shadow.data.act[:] = env.data.act
    if shadow.data.mocap_pos.shape == env.data.mocap_pos.shape:
        shadow.data.mocap_pos[:] = env.data.mocap_pos
        shadow.data.mocap_quat[:] = env.data.mocap_quat
    shadow.data.time = env.data.time
    shadow.time = env.time
    shadow.q_des = env.q_des.copy()
    mujoco.mj_forward(shadow.model, shadow.data)
    shadow_controller = _clone_sonic_controller(controller)
    contacts = _ContactMonitor(shadow.model)
    phase = min(6, plan.motion.T - 1)
    q_previous = (shadow.state()["q_hw"][C.MUJOCO_TO_ISAACLAB]
                  if previous_q is None else np.asarray(previous_q, dtype=np.float64))
    ticks = int(getattr(args, "online_shadow_ticks", 24))
    blend_ticks = max(1, int(getattr(args, "handoff_blend_ticks", 12)))
    root_z, roll_deg, tracking, clearances = [], [], [], []
    obstacle_contact = False
    nonfoot_contact = False
    for step in range(ticks):
        state = shadow.state()
        reference, phase = _rolling_reference(
            plan.motion, phase, _yaw(state["base_quat"]), horizon=50)
        weights = np.clip((step + np.arange(1, reference.T + 1)) / blend_ticks,
                          0.0, 1.0)[:, None]
        reference.joint_pos = (1.0 - weights) * q_previous[None, :] + weights * reference.joint_pos
        reference.joint_vel = weights * reference.joint_vel
        shadow_controller.append_state(
            state["q_hw"], state["dq_hw"], state["base_quat"], state["base_ang_vel"])
        _, target, _ = shadow_controller.act(reference, state["base_quat"])
        shadow.set_target(target); shadow.step()
        executed = shadow.state()
        _, _, nonfoot, obstacle = contacts.flags(shadow.data)
        boxes = _runtime_obstacle_boxes(shadow.model, shadow.data)
        clearance = _runtime_surface_clearance(body_points(shadow.model, shadow.data), boxes)
        root_z.append(float(executed["base_pos"][2]))
        roll_deg.append(float(np.abs(_roll_degrees(executed["base_quat"][None, :]))[0]))
        tracking.append(float(np.sqrt(np.mean((reference.joint_pos[0]
                                                - executed["q_hw"][C.MUJOCO_TO_ISAACLAB]) ** 2))))
        clearances.append(float(clearance))
        obstacle_contact |= bool(obstacle)
        nonfoot_contact |= bool(nonfoot)
    threshold = float(getattr(args, "self_manifold_clearance_m", 0.02))
    failures = []
    if min(root_z) < float(getattr(args, "fall_height_m", 0.30)):
        failures.append("shadow_fall_height")
    if max(roll_deg) > float(getattr(args, "max_roll_deg", 45.0)):
        failures.append("shadow_roll_limit")
    if float(np.mean(tracking)) > float(getattr(args, "max_tracking_error_rad", 0.35)):
        failures.append("shadow_tracking_error")
    if obstacle_contact:
        failures.append("shadow_obstacle_contact")
    if nonfoot_contact:
        failures.append("shadow_nonfoot_floor_contact")
    if min(clearances) < threshold:
        failures.append("shadow_self_manifold_clearance")
    return {
        "accepted": not failures, "failed_checks": failures, "ticks": ticks,
        "base_z_min_m": min(root_z), "roll_abs_max_deg": max(roll_deg),
        "track_err_mean_rad": float(np.mean(tracking)),
        "self_manifold_clearance_min_m": min(clearances),
        "obstacle_contact": obstacle_contact, "nonfoot_floor_contact": nonfoot_contact,
        "contract": "cloned current qpos/qvel/mocap + cloned 10-frame SONIC history",
    }


def _calibrate_envelope() -> np.ndarray:
    estimator = ExecutedEnvelopeEstimator()
    paths = [
        C.REPO / "reports/manifold_motion/stage2_routed_walk_test_pass/sample.npz",
        C.REPO / "reports/manifold_motion/stage2_mean_primitive6_test/sample.npz",
        C.REPO / "reports/manifold_motion/stage2_routed_side_test/sample.npz",
        C.REPO / "reports/manifold_motion/stage2_routed_crouch_train0/sample.npz",
    ]
    values = []
    for path in paths:
        with np.load(path) as archive:
            trajectory = archive["generated_ref"]
        motion = _motion_from_trajectory(trajectory, path, 30.0, 50.0)
        indices = np.unique(np.linspace(0, motion.T - 1, min(24, motion.T), dtype=int))
        values.append(estimator.sequence(
            motion.joint_pos_policy[indices], np.zeros((len(indices), 3)),
            np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]]), len(indices), axis=0)
        ))
    # A route tube needs a footprint radius, while z retains the conservative full-body extent.
    return np.array([0.46, 0.46, float(np.concatenate(values).max(axis=0)[2] + 0.08)])


def execute_plan(scene: Path, keyframes: np.ndarray, dense_route: np.ndarray,
                 plan_options: list[list[CandidatePlan]], args: argparse.Namespace,
                 replan_callback: Any | None = None,
                 perception_callback: Any | None = None,
                 semantic_replan_callback: Any | None = None
                 ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    env = G1FlatEnv(scene)
    controller = SonicController()
    contacts = _ContactMonitor(env.model)
    mujoco.mj_forward(env.model, env.data)
    runtime_boxes = _runtime_obstacle_boxes(env.model, env.data)
    runtime_clearance_threshold = float(getattr(args, "self_manifold_clearance_m", -1.0))
    # In manifold-adaptive mode option zero can be a transient turn helper.  Initialize from
    # the environment-selected locomotion posture (the last option), not from that helper.
    first = (plan_options[0][-1].motion if getattr(args, "manifold_adaptive", False)
             else plan_options[0][0].motion)
    env.reset(joint_offset=first.joint_pos_hw[0] - C.DEFAULT_ANGLES)
    controller.reset()
    hold_q = np.repeat(first.joint_pos_policy[0:1], 50, axis=0)
    hold = ReferenceBuffer(hold_q, np.zeros_like(hold_q), np.zeros((50, 3)),
                           np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]]), 50, axis=0))
    for _ in range(args.warmup_ticks):
        state = env.state()
        controller.append_state(state["q_hw"], state["dq_hw"], state["base_quat"], state["base_ang_vel"])
        _, target, _ = controller.act(hold, state["base_quat"])
        env.set_target(target); env.step()
    origin = env.state()["base_pos"][:2].copy()
    world_keyframes = keyframes + origin
    world_route = dense_route + origin
    phases = [[min(6, plan.motion.T - 1) for plan in options] for options in plan_options]
    active: tuple[int, int] | None = None
    segment_ticks = [0 for _ in plan_options]
    previous_q: np.ndarray | None = None
    blend_from_q: np.ndarray | None = None
    blend_step = 0
    switches: list[dict[str, Any]] = []
    receding_updates: list[dict[str, Any]] = []
    reached: list[dict[str, Any]] = []
    log: dict[str, list[Any]] = {key: [] for key in (
        "t", "q_ref", "q_exec", "dq_exec", "base_pos", "base_quat", "base_lin_vel", "action",
        "foot_contact", "hand_contact", "nonfoot_floor_contact", "obstacle_contact",
        "active_primitive", "active_primitive_name", "target_keyframe", "desired_yaw", "yaw_error",
        "route_progress_m", "route_progress_error_m", "route_velocity_cmd_mps", "yaw_command_rad",
        "reference_phase", "body_route_yaw_error", "side_on_active",
        "runtime_self_manifold_clearance_m",
    )}
    runtime_safety_stop = False
    runtime_safety_stop_tick: int | None = None
    runtime_safety_stop_clearance: float | None = None
    online_perception_failure: dict[str, Any] | None = None
    online_perception_updates: list[dict[str, Any]] = []
    online_semantic_updates: list[dict[str, Any]] = []
    online_semantic_choice: dict[int, int] = {}
    last_semantic_request: tuple[int, int, int] | None = None
    target_index = 1
    recent_features: list[np.ndarray] = []
    for tick in range(args.max_ticks):
        state = env.state(); position = state["base_pos"][:2]
        feet_now, hands_now, nonfoot_now, _ = contacts.flags(env.data)
        if runtime_clearance_threshold >= 0.0:
            pre_step_clearance = _runtime_surface_clearance(
                body_points(env.model, env.data), runtime_boxes)
            if pre_step_clearance < runtime_clearance_threshold:
                runtime_safety_stop = True
                runtime_safety_stop_tick = int(tick)
                runtime_safety_stop_clearance = float(pre_step_clearance)
                break
        current_feature = _state_features({
            "q_exec": state["q_hw"][None, :], "dq_exec": state["dq_hw"][None, :],
            "base_quat": state["base_quat"][None, :], "base_lin_vel": state["base_lin_vel"][None, :],
            "foot_contact": np.atleast_1d(np.asarray(feet_now, dtype=np.float32))[None, :],
            "hand_contact": np.atleast_1d(np.asarray(hands_now, dtype=np.float32))[None, :],
            "nonfoot_floor_contact": np.atleast_1d(np.asarray(nonfoot_now, dtype=np.float32)),
        })[0].astype(np.float32)
        if not recent_features:
            recent_features = [current_feature.copy() for _ in range(12)]
        else:
            recent_features.append(current_feature.copy())
            recent_features = recent_features[-12:]
        online_navigation: dict[str, Any] = {}
        if perception_callback is not None:
            online_navigation = dict(perception_callback(
                int(tick), state, np.asarray(world_route[-1], dtype=np.float64),
                np.asarray(world_route, dtype=np.float64)) or {})
            if bool(online_navigation.get("updated", False)):
                online_perception_updates.append({
                    key: value for key, value in online_navigation.items()
                    if key not in {"route_world_xyz", "radar_points_world", "corridor", "sdf"}
                })
            if bool(online_navigation.get("hard_stop", False)):
                online_perception_failure = {
                    "tick": int(tick),
                    "failure": str(online_navigation.get("failure", "unknown online perception failure")),
                }
                break
        route_progress_m, _ = _route_progress(position, world_route)
        while target_index < len(world_keyframes):
            error = float(np.linalg.norm(world_keyframes[target_index] - position))
            target_progress_gate, _ = _route_progress(
                world_keyframes[target_index], world_route)
            # A live local A* route can move an intermediate waypoint laterally while retaining
            # monotonic progress toward the same goal.  Do not wait forever for an obsolete
            # offline waypoint after the robot has already passed its route station.  The final
            # goal remains a strict Euclidean position gate.
            progress_passed = bool(
                perception_callback is not None
                and target_index < len(world_keyframes) - 1
                and route_progress_m >= target_progress_gate
            )
            if error > args.keyframe_tolerance_m and not progress_passed: break
            reached.append({"keyframe_index": target_index, "tick": tick,
                            "planned_xy_m": world_keyframes[target_index].tolist(),
                            "executed_xy_m": position.tolist(), "position_error_m": error,
                            "reached_by": ("online_route_progress" if progress_passed else "position"),
                            "q_tracking_rms_rad": (float(np.sqrt(np.mean((previous_q - state["q_hw"][C.MUJOCO_TO_ISAACLAB]) ** 2)))
                                                    if previous_q is not None else 0.0)})
            target_index += 1
        if target_index >= len(world_keyframes): break
        segment = min(target_index - 1, len(plan_options) - 1)
        # A new online M_e may contradict the primitive chosen from the initial map. Generate
        # candidates from the live corridor/state/history, then gate each one in a cloned
        # current-state MuJoCo world before changing the reference used by the live controller.
        if (semantic_replan_callback is not None
                and bool(online_navigation.get("updated", False))
                and "primitive_id" in online_navigation
                and "corridor" in online_navigation and "sdf" in online_navigation):
            requested_id = int(online_navigation["primitive_id"])
            current_option_index = online_semantic_choice.get(segment, len(plan_options[segment]) - 1)
            current_id = int(plan_options[segment][current_option_index].primitive_id)
            request_key = (int(online_navigation.get("online_update_count", 0)),
                           int(segment), requested_id)
            if requested_id != current_id and request_key != last_semantic_request:
                last_semantic_request = request_key
                proposals, semantic_report = semantic_replan_callback(
                    segment, requested_id,
                    np.asarray(online_navigation["corridor"], dtype=np.float32),
                    np.asarray(online_navigation["sdf"], dtype=np.float32),
                    current_feature.copy(), np.asarray(recent_features, dtype=np.float32),
                    state["q_hw"][C.MUJOCO_TO_ISAACLAB].copy(), tick,
                )
                if proposals is None:
                    proposals = []
                if isinstance(proposals, CandidatePlan):
                    proposals = [proposals]
                shadow_rows = []
                accepted_plan = None
                for proposal in proposals:
                    shadow = _shadow_screen_current_state(
                        scene, env, controller, proposal, previous_q, args)
                    shadow_rows.append({"candidate_index": proposal.candidate_index, **shadow})
                    if shadow["accepted"] and accepted_plan is None:
                        accepted_plan = proposal
                committed = accepted_plan is not None
                previous_id = current_id
                if committed:
                    # Keep the nominal fallback. The preferred semantic slot is index 0 unless
                    # index 0 is the geometric turn helper, in which case it is index 1.
                    preferred_index = next((index for index, option in enumerate(plan_options[segment])
                                            if option.primitive_id == requested_id), None)
                    if preferred_index is None:
                        preferred_index = 1 if plan_options[segment][0].primitive_id == 6 else 0
                        plan_options[segment].insert(preferred_index, accepted_plan)
                        phases[segment].insert(preferred_index, min(6, accepted_plan.motion.T - 1))
                    else:
                        plan_options[segment][preferred_index] = accepted_plan
                        phases[segment][preferred_index] = min(6, accepted_plan.motion.T - 1)
                    online_semantic_choice[segment] = int(preferred_index)
                    # Force the normal semantic handoff branch below to apply preview-space
                    # crossfade even though the list index itself did not change.
                    active = None
                online_semantic_updates.append({
                    "tick": int(tick), "segment": int(segment),
                    "from_primitive_id": previous_id, "requested_primitive_id": requested_id,
                    "committed": bool(committed), "preferred_option_index": (
                        int(preferred_index) if committed else None),
                    "shadow_candidates": shadow_rows,
                    **(semantic_report or {}),
                })
        delta = world_keyframes[target_index] - position
        desired_yaw = float(np.arctan2(delta[1], delta[0]))
        if "desired_yaw_rad" in online_navigation:
            desired_yaw = float(online_navigation["desired_yaw_rad"])
        target_progress, _ = _route_progress(world_keyframes[target_index], world_route)
        progress_error = float(target_progress - route_progress_m)
        progress_horizon_s = max(float(getattr(args, "progress_horizon_s", 1.6)), 0.1)
        route_velocity_cmd = float(np.clip(progress_error / progress_horizon_s, -0.8, 0.8))
        route_yaw_error = _angle(desired_yaw - _yaw(state["base_quat"]))
        # A generated gait need not move along its pelvis forward axis.  Infer the body yaw
        # that makes the selected locomotion displacement align with the route, then use the
        # turn helper only until that measured body-yaw target is reached.
        semantic_option_index = online_semantic_choice.get(segment, len(plan_options[segment]) - 1)
        base_plan = plan_options[segment][semantic_option_index]
        base_heading_offset = (0.0 if base_plan.primitive_id == 6 else
                               float(base_plan.screen_summary.get("exec_heading_offset_rad", 0.0)))
        side_yaw_offset = float(getattr(args, "side_body_yaw_offset_rad", 0.0))
        desired_body_yaw = (desired_yaw + side_yaw_offset
                            if base_plan.primitive_id == 4 and abs(side_yaw_offset) > 1e-6
                            else desired_yaw - base_heading_offset)
        alignment_yaw_error = _angle(desired_body_yaw - _yaw(state["base_quat"]))
        option_index = 0
        if len(plan_options[segment]) > 1:
            if getattr(args, "manifold_adaptive", False):
                # A real turn helper is primitive 6 at option zero. Online M_e choices may add
                # another preferred slot while retaining nominal fallback; never mistake that
                # semantic slot for a turn helper merely because its index is zero.
                has_turn_helper = plan_options[segment][0].primitive_id == 6
                option_index = (0 if has_turn_helper
                                and abs(alignment_yaw_error) > args.turn_threshold_rad
                                else semantic_option_index)
            elif segment_ticks[segment] >= args.action_hold_ticks:
                option_index = 1
        plan = plan_options[segment][option_index]
        # Optional receding-horizon refresh.  The callback receives the *actual* current
        # state/history and may replace the live future reference for the same semantic option.
        # The final continuous MuJoCo gate remains authoritative; a refresh is never accepted
        # merely because its decoded joints are in range.
        refresh_ticks = int(getattr(args, "receding_horizon_ticks", 0))
        if (replan_callback is not None and refresh_ticks > 0 and tick > 0
                and tick % refresh_ticks == 0 and active is not None):
            refreshed, refresh_report = replan_callback(
                segment, plan.primitive_id, current_feature.copy(),
                np.asarray(recent_features, dtype=np.float32),
                state["q_hw"][C.MUJOCO_TO_ISAACLAB].copy(), tick,
            )
            if refreshed is not None:
                commit = bool((refresh_report or {}).get("commit", True))
                if commit:
                    plan_options[segment][option_index] = refreshed
                    plan = refreshed
                    # A rolling refresh of the same semantic primitive must preserve gait phase;
                    # restarting at the transition prefix every N ticks turns a walk into a
                    # series of starts and explains the zero-progress failure mode of the first
                    # prototype.
                    phases[segment][option_index] = (
                        _handoff_phase(plan.motion, previous_q)
                        if previous_q is not None else min(6, plan.motion.T - 1)
                    )
                if commit and previous_q is not None:
                    blend_from_q = previous_q.copy()
                    blend_step = 0
                row = {"tick": int(tick), "segment": int(segment),
                       "primitive": plan.primitive, "committed": commit,
                       **(refresh_report or {})}
                receding_updates.append(row)
        switch_event: dict[str, Any] | None = None
        if active != (segment, option_index):
            previous_plan = (plan_options[active[0]][active[1]] if active is not None else None)
            # A semantic change (e.g. walk -> crouch) must enter through the generated clip's
            # transition prefix.  Searching the whole clip for the nearest pose can jump into
            # its terminal phase; looping from there changes the gait dynamics and can reverse
            # an otherwise forward candidate.  Nearest-pose alignment is used only when the
            # semantic primitive stays the same across an M_e segment boundary.
            if previous_plan is not None and previous_plan.primitive_id == plan.primitive_id:
                phases[segment][option_index] = _handoff_phase(plan.motion, previous_q)
                phase_handoff = "same_primitive_nearest_pose"
                blend_from_q = None
            elif previous_q is not None:
                # Enter a new semantic motion at its validated prefix, then crossfade the
                # complete SONIC preview below.  Jumping to a later nearest pose changed the
                # learned gait phase and increased route drift; preview-space crossfade keeps
                # both the validated phase and a bounded first-step handoff.
                phases[segment][option_index] = min(6, plan.motion.T - 1)
                phase_handoff = "semantic_transition_prefix_crossfade"
                blend_from_q = previous_q.copy()
                blend_step = 0
            else:
                phases[segment][option_index] = min(6, plan.motion.T - 1)
                phase_handoff = "semantic_transition_prefix"
                blend_from_q = None
            switch_event = {"tick": tick, "from": (list(active) if active is not None else None),
                            "to": [segment, option_index],
                            "primitive": plan.primitive, "candidate_index": plan.candidate_index,
                            "target_keyframe": target_index,
                            "route_progress_m": route_progress_m,
                            "selected_start_phase": phases[segment][option_index],
                            "phase_handoff": phase_handoff,
                            "route_yaw_error_rad": route_yaw_error,
                            "alignment_yaw_error_rad": alignment_yaw_error}
            active = (segment, option_index)
        reference_yaw = desired_body_yaw
        reference_phase = phases[segment][option_index]
        reference, phases[segment][option_index] = _rolling_reference(
            plan.motion, reference_phase, reference_yaw, horizon=50
        )
        if blend_from_q is not None:
            blend_ticks = max(1, int(getattr(args, "handoff_blend_ticks", 12)))
            weights = np.clip(
                (blend_step + np.arange(1, reference.T + 1, dtype=np.float64)) / blend_ticks,
                0.0, 1.0,
            )[:, None]
            reference.joint_pos = (1.0 - weights) * blend_from_q[None, :] + weights * reference.joint_pos
            reference.joint_vel = weights * reference.joint_vel
            blend_step += 1
            if blend_step >= blend_ticks:
                blend_from_q = None
        q_ref = reference.joint_pos[0].copy()
        if switch_event is not None:
            switch_event["pre_switch_joint_rms_rad"] = (
                float(np.sqrt(np.mean((previous_q - q_ref) ** 2))) if previous_q is not None else 0.0
            )
            switch_event["handoff_blend_ticks"] = (
                int(getattr(args, "handoff_blend_ticks", 12))
                if switch_event["phase_handoff"].endswith("crossfade") else 0
            )
            switches.append(switch_event)
        controller.append_state(state["q_hw"], state["dq_hw"], state["base_quat"], state["base_ang_vel"])
        action, target, _ = controller.act(reference, state["base_quat"])
        env.set_target(target); env.step()
        executed = env.state()
        feet, hands, nonfoot, obstacle = contacts.flags(env.data)
        runtime_clearance = _runtime_surface_clearance(
            body_points(env.model, env.data), runtime_boxes)
        log["t"].append(env.time); log["q_ref"].append(q_ref); log["q_exec"].append(executed["q_hw"][C.MUJOCO_TO_ISAACLAB])
        log["dq_exec"].append(executed["dq_hw"][C.MUJOCO_TO_ISAACLAB]); log["base_pos"].append(executed["base_pos"])
        log["base_quat"].append(executed["base_quat"]); log["base_lin_vel"].append(executed["base_lin_vel"]); log["action"].append(action)
        log["foot_contact"].append(feet); log["hand_contact"].append(hands); log["nonfoot_floor_contact"].append(nonfoot)
        log["obstacle_contact"].append(obstacle); log["active_primitive"].append(plan.primitive_id)
        log["active_primitive_name"].append(plan.primitive); log["target_keyframe"].append(target_index)
        log["desired_yaw"].append(desired_yaw); log["yaw_error"].append(alignment_yaw_error)
        log["route_progress_m"].append(route_progress_m)
        log["route_progress_error_m"].append(progress_error)
        log["route_velocity_cmd_mps"].append(route_velocity_cmd)
        log["yaw_command_rad"].append(desired_yaw)
        log["reference_phase"].append(reference_phase)
        log["body_route_yaw_error"].append(_angle(_yaw(executed["base_quat"]) - desired_yaw))
        log["side_on_active"].append(bool(plan.screen_summary.get("side_on_semantics_passed", False)))
        log["runtime_self_manifold_clearance_m"].append(runtime_clearance)
        previous_q = q_ref
        segment_ticks[segment] += 1
        executed_feature = _state_features({
            "q_exec": executed["q_hw"][None, :], "dq_exec": executed["dq_hw"][None, :],
            "base_quat": executed["base_quat"][None, :], "base_lin_vel": executed["base_lin_vel"][None, :],
            "foot_contact": np.atleast_1d(np.asarray(feet, dtype=np.float32))[None, :],
            "hand_contact": np.atleast_1d(np.asarray(hands, dtype=np.float32))[None, :],
            "nonfoot_floor_contact": np.atleast_1d(np.asarray(nonfoot, dtype=np.float32)),
        })[0].astype(np.float32)
        recent_features.append(executed_feature)
        recent_features = recent_features[-12:]
        if (runtime_clearance_threshold >= 0.0
                and runtime_clearance < runtime_clearance_threshold):
            runtime_safety_stop = True
            runtime_safety_stop_tick = int(tick)
            runtime_safety_stop_clearance = float(runtime_clearance)
            break
    data = {key: np.asarray(value) for key, value in log.items()}
    positions = data["base_pos"][:, :2]
    deviation = _distance_to_polyline(positions, world_route)
    tracking = np.abs(data["q_ref"] - data["q_exec"])
    roll = np.abs(_roll_degrees(data["base_quat"]))
    locomotion = data["active_primitive"] != 6
    yaw_abs = np.abs(data["yaw_error"][locomotion]) if np.any(locomotion) else np.zeros(1)
    side_mask = np.asarray(data["side_on_active"], dtype=bool)
    side_yaw_error = np.abs(np.degrees(np.abs(data["body_route_yaw_error"][side_mask]) - np.pi / 2.0)) if np.any(side_mask) else np.zeros(1)
    terminal_error = float(np.linalg.norm(positions[-1] - world_keyframes[-1]))
    failures = []
    if online_perception_failure is not None: failures.append("online_perception_failure")
    if target_index < len(world_keyframes): failures.append("goal_or_keyframe_not_reached")
    if runtime_safety_stop: failures.append("runtime_self_manifold_clearance_stop")
    if bool(data["obstacle_contact"].any()): failures.append("scene_obstacle_contact")
    if float(data["base_pos"][:, 2].min()) < args.fall_height_m: failures.append("fall_or_extreme_roll")
    if float(roll.max()) > args.max_roll_deg: failures.append("roll_limit")
    if float(tracking.mean()) > args.max_tracking_error_rad: failures.append("tracking_error")
    if float(np.quantile(deviation, 0.95)) > args.max_route_deviation_m: failures.append("route_deviation")
    switch_rms = max((float(x["pre_switch_joint_rms_rad"]) for x in switches), default=0.0)
    if switch_rms > args.max_switch_rms_rad: failures.append("handoff_discontinuity")
    if terminal_error > args.keyframe_tolerance_m: failures.append("terminal_keyframe_error")
    yaw_p95_deg = float(np.degrees(np.quantile(yaw_abs, 0.95)))
    if yaw_p95_deg > getattr(args, "max_body_route_yaw_p95_deg", 55.0):
        failures.append("body_route_heading_mismatch")
    side_yaw_p95_deg = float(np.quantile(side_yaw_error, 0.95)) if np.any(side_mask) else 0.0
    if np.any(side_mask) and side_yaw_p95_deg > getattr(args, "max_side_body_yaw_error_p95_deg", 35.0):
        failures.append("side_on_body_heading_mismatch")
    summary = {
        "accepted": not failures, "failed_checks": failures, "no_reset_between_primitives": True,
        "physics_ticks": int(len(data["t"])), "duration_s": float(len(data["t"]) * C.CONTROL_DT),
        "keyframes_reached": int(len(reached)), "keyframes_total_excluding_start": int(len(keyframes) - 1),
        "keyframe_events": reached, "terminal_error_m": terminal_error,
        "route_deviation_mean_m": float(deviation.mean()), "route_deviation_p95_m": float(np.quantile(deviation, .95)),
        "route_deviation_max_m": float(deviation.max()), "track_err_mean_rad": float(tracking.mean()),
        "base_z_min_m": float(data["base_pos"][:, 2].min()), "roll_abs_max_deg": float(roll.max()),
        "body_route_yaw_abs_p95_deg": yaw_p95_deg,
        "body_route_yaw_abs_max_deg": float(np.degrees(yaw_abs.max())),
        "route_progress_error_abs_p95_m": float(np.quantile(np.abs(data["route_progress_error_m"]), 0.95)),
        "route_velocity_cmd_abs_max_mps": float(np.max(np.abs(data["route_velocity_cmd_mps"]))),
        "side_on_ticks": int(side_mask.sum()),
        "side_on_body_yaw_error_p95_deg": side_yaw_p95_deg,
        "side_on_body_yaw_abs_median_deg": (float(np.median(np.abs(np.degrees(data["body_route_yaw_error"][side_mask]))))
                                             if np.any(side_mask) else 0.0),
        "obstacle_contact_ticks": int(data["obstacle_contact"].sum()), "primitive_switches": switches,
        "runtime_self_manifold_clearance_min_m": (
            float(np.min(data["runtime_self_manifold_clearance_m"]))
            if len(data["runtime_self_manifold_clearance_m"]) else None),
        "runtime_self_manifold_clearance_threshold_m": (
            runtime_clearance_threshold if runtime_clearance_threshold >= 0.0 else None),
        "runtime_self_manifold_safety_stop": bool(runtime_safety_stop),
        "runtime_self_manifold_safety_stop_tick": runtime_safety_stop_tick,
        "runtime_self_manifold_safety_stop_clearance_m": runtime_safety_stop_clearance,
        "online_perception_enabled": bool(perception_callback is not None),
        "online_perception_updates": online_perception_updates,
        "online_semantic_updates": online_semantic_updates,
        "online_semantic_switch_count": int(sum(bool(row.get("committed"))
                                                  for row in online_semantic_updates)),
        "online_perception_failure": online_perception_failure,
        "online_perception_contract": (
            "live radar -> 3-D sliding map -> incremental ESDF/D* Lite -> M_e -> "
            "current-state shadow-gated primitive switch and lookahead yaw"
            if perception_callback is not None else "disabled"
        ),
        "primitive_switch_count": int(max(0, len(switches) - 1)), "max_pre_switch_joint_rms_rad": switch_rms,
        "primitive_ticks": {name: int(np.sum(data["active_primitive_name"] == name)) for name in PRIMITIVE_NAMES.values()},
        "start_world_xy_m": origin.tolist(), "final_world_xy_m": positions[-1].tolist(),
        "phase_contract": "nearest-pose handoff plus continuous 50 Hz phase; no per-segment phase reset",
        "receding_horizon_ticks": int(getattr(args, "receding_horizon_ticks", 0)),
        "receding_horizon_updates": receding_updates,
        "receding_horizon_contract": (
            "actual 69-D state/history refreshes future Flow reference at the configured tick interval; "
            "continuous MuJoCo execution and final physical gates are retained"
            if receding_updates else "disabled"),
        "root_progress_contract": (
            "route progress/yaw command is computed from measured pelvis state; frozen SONIC does not "
            "consume root position directly, so these commands are logged for the closed-loop "
            "scheduler and future controller fine-tuning"
        ),
    }
    return data, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Flow Matching candidates per A* route segment")
    parser.add_argument("--scene", type=Path, default=C.REPO / "data/g1_flat/scene_long_avoidance.xml")
    parser.add_argument("--windows", type=Path, default=C.REPO / "reports/manifold_motion/seed_windows_corridor_stage2_v2/seed_stage2_windows.npz")
    parser.add_argument("--autoencoder", type=Path, default=C.REPO / "reports/manifold_motion/stage2_flow_corridor_v1/autoencoder.pt")
    parser.add_argument("--flow", type=Path, default=C.REPO / "reports/manifold_motion/stage2_flow_corridor_v1/flow.pt")
    parser.add_argument("--out", type=Path, default=C.REPO / "reports/manifold_motion/stage2_flow_route_avoidance_v1")
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--goal-x", type=float, default=3.60)
    parser.add_argument("--keyframe-tolerance-m", type=float, default=.22)
    parser.add_argument("--warmup-ticks", type=int, default=20)
    parser.add_argument("--max-ticks", type=int, default=900)
    parser.add_argument("--turn-threshold-rad", type=float, default=.28)
    parser.add_argument("--fall-height-m", type=float, default=.30)
    parser.add_argument("--max-roll-deg", type=float, default=45.)
    parser.add_argument("--max-tracking-error-rad", type=float, default=.35)
    parser.add_argument("--max-route-deviation-m", type=float, default=.45)
    parser.add_argument("--max-switch-rms-rad", type=float, default=.75)
    parser.add_argument("--handoff-blend-ticks", type=int, default=12)
    parser.add_argument("--side-body-yaw-offset-rad", type=float, default=0.0,
                        help="optional side-gait body yaw offset relative to route tangent")
    parser.add_argument("--max-side-body-yaw-error-p95-deg", type=float, default=35.0)
    parser.add_argument("--max-body-route-yaw-p95-deg", type=float, default=55.0)
    parser.add_argument("--action-hold-ticks", type=int, default=45,
                        help="minimum ticks to show the preferred non-nominal action before a nominal fallback may take over")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-render", action="store_true", help="run the physical/candidate gate without GIF rendering")
    args = parser.parse_args()
    if args.num_candidates < 2: parser.error("num-candidates must be >= 2")
    args.out.mkdir(parents=True, exist_ok=True)
    planner = PlannerConfig(); boxes = _boxes(args.scene)
    raw_route = _astar(np.array([0., 0.]), np.array([args.goal_x, 0.]), boxes, planner)
    keyframes = _simplify(raw_route, boxes, planner); dense = _densify(keyframes, planner.route_spacing_m)
    segments = _route_segments(dense, keyframes)
    points, obstacle_names = obstacle_pointcloud(args.scene, spacing_m=.06)
    envelope = _calibrate_envelope()
    sampler = RouteFlowSampler(args.windows, args.autoencoder, args.flow, args.device)
    runner = SeedReplayRunner(C.FLAT_SCENE)

    # Preferred semantic action by route interval.  The first segment turns into the upper
    # corridor, the middle segment uses a lateral gait, and the exit segment uses crouch before
    # returning to nominal.  Each preference is backed by K flow candidates and a hard gate.
    preferences = [[6, 5], [4, 5], [2, 5]]
    plan_options: list[list[CandidatePlan]] = []
    evidence: list[dict[str, Any]] = []
    condition_arrays: dict[str, np.ndarray] = {}
    for segment_index, (segment, primitive_choices) in enumerate(zip(segments, preferences)):
        start = segment[0]
        corridor, sdf, command = _segment_condition(segment, start, points, envelope)
        condition_arrays[f"segment_{segment_index}_corridor"] = corridor
        condition_arrays[f"segment_{segment_index}_sdf"] = sdf
        segment_reports: list[dict[str, Any]] = []
        selected: CandidatePlan | None = None
        selected_nominal: CandidatePlan | None = None
        for primitive_id in primitive_choices:
            generated, source_index = sampler.sample(
                primitive_id, corridor, sdf, command, args.num_candidates,
                seed=20260917 + segment_index * 101 + primitive_id,
            )
            candidate_reports = []
            for candidate_index, trajectory in enumerate(generated):
                _, summary = _candidate_screen(trajectory, primitive_id,
                                                args.out / f"segment_{segment_index}_candidates.npz", runner)
                row = {"candidate_index": candidate_index, "accepted": bool(summary["accepted"]),
                       "score": _screen_cost(summary, trajectory), "failed_checks": summary["failed_checks"],
                       "track_err_mean_rad": summary.get("track_err_mean_rad"),
                       "base_z_min_m": summary.get("base_z_min_m"),
                       "exec_path_m": summary.get("exec_path_m"),
                       "motion_progress_ratio": summary.get("motion_progress_ratio"),
                       "source_index": source_index}
                candidate_reports.append(row)
                evidence.append({"segment_index": segment_index, "primitive_id": primitive_id,
                                 "primitive": PRIMITIVE_NAMES[primitive_id], **row})
            viable = [row for row in candidate_reports if row["accepted"]]
            if viable and selected is None:
                if primitive_id in (4, 5):
                    winner = max(viable, key=lambda row: float(row.get("exec_path_m") or 0.0))
                elif primitive_id == 2:
                    winner = min(viable, key=lambda row: float(row.get("base_z_min_m") or 10.0))
                else:
                    winner = min(viable, key=lambda row: float(row["score"]))
                selected = CandidatePlan(segment_index, primitive_id, PRIMITIVE_NAMES[primitive_id],
                                         int(winner["candidate_index"]), generated[int(winner["candidate_index"])],
                                         _motion_from_trajectory(generated[int(winner["candidate_index"])],
                                                                 args.out / "route_candidate.npz", 30., 50.),
                                         winner)
            if viable and primitive_id == 5 and selected_nominal is None:
                winner = max(viable, key=lambda row: float(row.get("exec_path_m") or 0.0))
                selected_nominal = CandidatePlan(segment_index, primitive_id, PRIMITIVE_NAMES[primitive_id],
                                                 int(winner["candidate_index"]), generated[int(winner["candidate_index"])],
                                                 _motion_from_trajectory(generated[int(winner["candidate_index"])],
                                                                          args.out / "route_candidate_nominal.npz", 30., 50.),
                                                 winner)
            segment_reports.append({"primitive_id": primitive_id, "primitive": PRIMITIVE_NAMES[primitive_id],
                                    "source_index": source_index, "candidates": candidate_reports})
        if selected is None:
            raise RuntimeError(f"segment {segment_index}: no Flow candidate passed the physical gate")
        options = [selected]
        # Keep the preferred action visible for a fixed portion of the interval, then let an
        # accepted nominal candidate finish the spatial keyframe.  This is a keyframe-preserving
        # action switch, not an unconditional clip splice: the next keyframe is still locked
        # until measured pelvis distance is within tolerance.
        if selected.primitive_id != 5 and selected_nominal is not None:
            options.append(selected_nominal)
        plan_options.append(options)
        evidence.append({"segment_index": segment_index, "selected": {
            "primitive_id": selected.primitive_id, "primitive": selected.primitive,
            "candidate_index": selected.candidate_index, "screen_summary": selected.screen_summary},
            "fallback": ({"primitive_id": selected_nominal.primitive_id,
                          "primitive": selected_nominal.primitive,
                          "candidate_index": selected_nominal.candidate_index,
                          "screen_summary": selected_nominal.screen_summary}
                         if len(options) > 1 else None),
            "candidate_sets": segment_reports})

    data, execution = execute_plan(args.scene, keyframes, dense, plan_options, args)
    origin = np.asarray(execution["start_world_xy_m"])
    world_keyframes, world_route = keyframes + origin, dense + origin
    report = {
        "experiment": "Stage-2 Flow Matching route-segment candidate generation and physical selection",
        "scene": str(args.scene), "windows": str(args.windows), "flow": str(args.flow),
        "autoencoder": str(args.autoencoder), "obstacles": obstacle_names,
        "planner": {"resolution_m": planner.resolution_m, "inflation_m": planner.inflation_m,
                     "body_radius_m": planner.body_radius_m, "clearance_m": planner.clearance_m},
        "raw_astar_points": int(len(raw_route)), "keyframes": world_keyframes.tolist(),
        "dense_route_points": int(len(dense)), "envelope_semi_m": envelope.tolist(),
        "segments": [{"segment_index": options[0].segment_index,
                       "preferred": {"primitive": options[0].primitive,
                                     "primitive_id": options[0].primitive_id,
                                     "candidate_index": options[0].candidate_index,
                                     "screen_summary": options[0].screen_summary},
                       "fallback": ({"primitive": options[1].primitive,
                                     "primitive_id": options[1].primitive_id,
                                     "candidate_index": options[1].candidate_index,
                                     "screen_summary": options[1].screen_summary}
                                    if len(options) > 1 else None)} for options in plan_options],
        "candidate_evidence": evidence, "execution": execution,
        "accepted": execution["accepted"], "failed_checks": execution["failed_checks"],
        "flow_condition_contract": "each segment supplies corridor [48,7] and sdf [10,10,8] directly to latent Flow Matching",
    }
    np.savez_compressed(args.out / "segment_conditions.npz", **condition_arrays,
                        keyframes=world_keyframes, route=world_route)
    np.savez_compressed(args.out / "executed.npz", **data)
    (args.out / "candidate_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Save every selected generated trajectory for audit and future sequence training.
    for options in plan_options:
        for plan in options:
            np.save(args.out / f"segment_{plan.segment_index}_{plan.primitive}_candidate_{plan.candidate_index}.npy",
                    plan.trajectory)
    # Render the same physical rollout used by the acceptance gate.  The displayed purple
    # corridor is the world-frame route tube; each segment's exact local M_e/SDF condition is
    # retained in segment_conditions.npz and was fed to the Flow model above.
    route_yaw = _route_yaw(world_route)
    render_corridor = np.column_stack([
        world_route, np.full(len(world_route), float(data["base_pos"][0, 2])),
        np.repeat(envelope[None, :], len(world_route), axis=0), route_yaw,
    ])
    if not args.skip_render:
        render(args.scene, data, execution, world_keyframes, world_route, render_corridor,
               boxes, planner, args.out / "flow_route_avoidance.gif", fps=20.0)
    print(json.dumps({"accepted": execution["accepted"], "failed_checks": execution["failed_checks"],
                      "segments": [{"preferred": options[0].primitive,
                                    "fallback": options[1].primitive if len(options) > 1 else None}
                                   for options in plan_options],
                      "keyframes_reached": execution["keyframes_reached"],
                      "primitive_switch_count": execution["primitive_switch_count"],
                      "obstacle_contact_ticks": execution["obstacle_contact_ticks"]}, indent=2))
    return 0 if execution["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
