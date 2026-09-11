"""Goal-reaching RL environment on a *soft* ellipsoid manifold.

The MuJoCo scene is flat ground plus translucent (non-colliding) ellipsoids. The manifold
exists only in the reward and observation:

  * containment: body points leaving the ellipsoid chain are penalized quadratically
    (`w_manifold * (r-1)^2`), and leaving it far terminates the episode;
  * spine: the torso is rewarded for aligning with the nearest primitive's local +z axis,
    which is how a flattened or tilted manifold shapes the body orientation.

Nothing collides with the manifold, so changing it is pure data - no model rebuilds.

Policy (10 Hz) outputs a body-frame velocity command plus a crouch command; the velocity
command drives the kinematic planner and frozen SONIC tracks the resulting reference.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from . import constants as C
from .env import G1FlatEnv
from .geo_env import task_space_obs
from .manifold import EllipsoidManifold, build_scene, update_visuals
from .planner import SonicPlanner
from .reference import CROUCH_DIRECTION, CrouchReference
from .sonic import SonicController

# body points used for manifold containment and visualization
COMPLIANCE_BODIES = ("pelvis", "torso_link", "left_knee_link", "right_knee_link",
                     "left_wrist_yaw_link", "right_wrist_yaw_link")


def roll_pitch(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


@dataclass
class TaskConfig:
    max_episode_steps: int = 200        # policy steps (0.1 s each)
    fall_height: float = 0.42
    tilt_limit_deg: float = 60.0
    goal_margin: float = 0.35           # success once the base passes the end region minus this
    max_lin_vel: float = 1.5
    max_lat_vel: float = 0.6
    max_yaw_rate: float = 1.0
    w_progress: float = 1.0
    w_manifold: float = 30.0             # per second, times min(r - 1, excess_cap) outside
    excess_cap: float = 0.4             # cap on the containment excess used in the penalty
    w_spine: float = 3.0                # per second, penalizes torso misalignment with the axis
    reward_scale: float = 0.2            # scales the whole per-step reward (keeps returns ~O(10))
    w_energy: float = 0.1
    goal_bonus: float = 10.0
    fall_penalty: float = 10.0
    out_penalty: float = 10.0
    manifold_out_radius: float = 1.15    # sustained violation beyond this terminates
    manifold_out_ticks: int = 10        # 0.2 s of sustained violation before terminating
    obs_goal_scale: float = 5.0
    crouch_max: float = 1.3             # 4th action dim in [0, 1] scales this crouch offset
    spawn_crouch: float = 1.2           # the robot starts crouched by this amount (inside a low manifold)
    replan_min_interval: int = 25       # control ticks between command-triggered replans
    replan_command_delta: float = 0.5


class GoalReachEnv:
    CONTROL_PER_POLICY = 5              # 50 Hz control / 10 Hz policy

    def __init__(self, manifold: EllipsoidManifold | None = None, cfg: TaskConfig | None = None):
        self.manifold = manifold or EllipsoidManifold.tunnel()
        self.cfg = cfg or TaskConfig()
        self.scene_path = build_scene(self.manifold)
        self.env = G1FlatEnv(self.scene_path)
        self.controller = SonicController()
        self.planner = SonicPlanner()

        model = self.env.model
        self.torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.compliance_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                                    for name in COMPLIANCE_BODIES]
        self.robot_geoms = [i for i in range(model.ngeom) if model.geom_group[i] == 0]

        self.goal = self.manifold.goal()[:2]
        self.action_dim = 4
        self._qpos_hist: deque[np.ndarray] = deque(maxlen=4)
        self._prev_action = np.zeros(4)
        self._cmd = np.zeros(4)
        self._tick = 0
        self._steps = 0
        self._prev_dist = 0.0
        self._last_replan_tick = -10 ** 9
        self._last_replan_cmd = np.zeros(4)
        self._episode_return = 0.0
        self.crouch_amount = 0.0
        self.reference = CrouchReference.static_stand(np.array([1.0, 0.0, 0.0, 0.0]))

    # -- manifold -----------------------------------------------------------
    def set_manifold(self, manifold: EllipsoidManifold) -> None:
        """Swap the manifold (pure data: no model rebuild, no collision rebuild)."""
        self.manifold = manifold
        self.goal = manifold.goal()[:2]
        update_visuals(self.env.model, manifold)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _qpos36(st: dict) -> np.ndarray:
        qpos = np.empty(7 + 29)
        qpos[0:3] = st["base_pos"]
        qpos[3:7] = st["base_quat"]
        qpos[7:] = st["q_hw"]
        return qpos

    def _distance(self, st: dict) -> float:
        """Distance remaining along x to the end region of the manifold (can go negative)."""
        return float(self.manifold.goal_x() - st["base_pos"][0])

    def manifold_state(self) -> dict:
        """Containment radius, spine alignment and the nearest primitive.

        Uses the same body envelope as the geometric model (pelvis, head, hands), so the
        two worlds agree on what "inside the manifold" means.
        """
        from .geo_env import ENVELOPE_OFFSETS
        st = self.env.state()
        quat = st["base_quat"]
        yaw = float(np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                               1 - 2 * (quat[2] ** 2 + quat[3] ** 2)))
        rot = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                        [np.sin(yaw), np.cos(yaw), 0.0],
                        [0.0, 0.0, 1.0]])
        points = st["base_pos"] + ENVELOPE_OFFSETS @ rot.T
        radii = self.manifold.radii(points)
        worst = int(np.argmax(radii))
        nearest = int(self.manifold.nearest(points[worst:worst + 1])[0])
        primitive = self.manifold.primitives[nearest]
        torso_up = np.asarray(self.env.data.xmat[self.torso_id]).reshape(3, 3)[:, 2]
        return {
            "radius": float(radii.max()),
            "radius_mean": float(radii.mean()),
            "spine": float(np.clip(torso_up @ primitive.axis(), -1.0, 1.0)),
            "primitive": nearest,
            "primitive_ref": primitive,
        }

    def _obs(self, st: dict) -> np.ndarray:
        quat = st["base_quat"]
        rot = C.quat_to_matrix(quat)
        gravity = C.quat_rotate(C.quat_conj(quat), np.array([0.0, 0.0, -1.0]))
        q_isaac = st["q_hw"][C.MUJOCO_TO_ISAACLAB] - C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB]
        dq_isaac = st["dq_hw"][C.MUJOCO_TO_ISAACLAB]
        goal_world = np.array([self.goal[0] - st["base_pos"][0], self.goal[1] - st["base_pos"][1], 0.0])
        goal_body = rot.T @ goal_world
        heading = np.arctan2(goal_world[1], goal_world[0])
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        heading_err = np.arctan2(np.sin(heading - yaw), np.cos(heading - yaw))

        state = self.manifold_state()
        primitive = state["primitive_ref"]
        rel_center = rot.T @ (primitive.center - st["base_pos"])
        axis_body = rot.T @ primitive.axis()
        return np.concatenate([
            rot.T @ st["base_lin_vel"] / 2.0,
            st["base_ang_vel"] / 4.0,
            gravity,
            q_isaac,
            dq_isaac / 20.0,
            goal_body / self.cfg.obs_goal_scale,
            [np.sin(heading_err), np.cos(heading_err)],
            self._prev_action,
            [state["radius"], state["spine"]],
            rel_center, axis_body, primitive.semi,
        ]).astype(np.float32)

    def reduced_obs(self, st: dict | None = None) -> np.ndarray:
        """Task-space observation with the same layout as the geometric model."""
        st = self.env.state() if st is None else st
        quat = st["base_quat"]
        rot = C.quat_to_matrix(quat)
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        velocity = rot.T @ st["base_lin_vel"]  # body frame -> world for yaw-only motion
        return task_space_obs(st["base_pos"][0], st["base_pos"][1], yaw,
                              float(velocity[0]), float(velocity[1]), self.crouch_amount,
                              self.cfg.crouch_max, self.manifold.goal_x(), self.manifold,
                              float(st["base_pos"][2]), self.cfg.obs_goal_scale)

    def _plan_kwargs(self, st: dict) -> dict:
        vx, vy, wz, _ = self._cmd
        rot = C.quat_to_matrix(st["base_quat"])
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        speed = float(np.hypot(vx, vy))
        # aim the reference heading at the goal; tracking the robot's own yaw lets the
        # planner's gait drift accumulate
        goal_dir = np.array([self.goal[0] - st["base_pos"][0], self.goal[1] - st["base_pos"][1]])
        desired_yaw = np.arctan2(goal_dir[1], goal_dir[0]) + 0.5 * wz
        facing = (float(np.cos(desired_yaw)), float(np.sin(desired_yaw)), 0.0)
        if speed > 0.05:
            movement = (
                float(np.cos(yaw) * vx - np.sin(yaw) * vy),
                float(np.sin(yaw) * vx + np.cos(yaw) * vy),
                0.0,
            )
            return dict(mode=2, movement=movement, facing=facing, target_vel=speed, height=-1.0)
        return dict(mode=0, movement=(0.0, 0.0, 0.0), facing=facing, target_vel=-1.0, height=-1.0)

    def _replan(self, st: dict, force: bool = False) -> None:
        context = np.stack(self._qpos_hist)
        self.reference.maybe_replan(self.planner, reserve=16, context=context,
                                    force=force, **self._plan_kwargs(st))

    # -- gym-like API -------------------------------------------------------
    def reset(self, tunnel_semi_z: float | None = None, **manifold_overrides) -> tuple[np.ndarray, dict]:
        if tunnel_semi_z is not None or manifold_overrides:
            height = float(tunnel_semi_z if tunnel_semi_z is not None else self.manifold.primitives[-1].semi[2])
            if len(self.manifold.primitives) == 1:
                self.set_manifold(self.manifold.with_last_semi_z(height))
            else:
                first = self.manifold.primitives[0]
                last = self.manifold.primitives[-1]
                # rebuild with identical span/start so the chain does not drift forward
                self.set_manifold(EllipsoidManifold.tunnel(
                    length=float(last.center[0] - first.center[0]),
                    semi_y=float(first.semi[1]),
                    entry_semi_z=float(first.semi[2]),
                    tunnel_semi_z=height,
                    count=len(self.manifold.primitives),
                    start_x=float(first.center[0]),
                    **manifold_overrides,
                ))
        # a low manifold cannot contain an upright robot, so spawn crouched
        offset_isaac = self.cfg.spawn_crouch * CROUCH_DIRECTION
        self.env.reset(x=float(self.manifold.start_x()), height=0.62,
                       joint_offset=offset_isaac[C.ISAACLAB_TO_MUJOCO])
        self.controller.reset()
        st = self.env.state()
        self._qpos_hist.clear()
        for _ in range(4):
            self._qpos_hist.append(self._qpos36(st))
        self.reference = CrouchReference.static_stand(st["base_quat"])
        self.crouch_amount = self.cfg.spawn_crouch
        self.reference.set_amount(self.crouch_amount)
        self._prev_action = np.zeros(4)
        self._cmd = np.zeros(4)
        self._tick = 0
        self._steps = 0
        self._prev_dist = self._distance(st)
        self._last_replan_tick = -10 ** 9
        self._last_replan_cmd = np.zeros(4)
        self._episode_return = 0.0
        self._replan(st, force=True)
        return self._obs(st), {"distance": self._prev_dist}

    def step(self, action: np.ndarray, tick_callback=None):
        """Advance one policy step (5 control ticks)."""
        cfg = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._cmd = np.array([action[0] * cfg.max_lin_vel,
                              action[1] * cfg.max_lat_vel,
                              action[2] * cfg.max_yaw_rate,
                              action[3]])
        self.crouch_amount = max(0.0, float(action[3])) * cfg.crouch_max
        self.reference.set_amount(self.crouch_amount)
        reward = 0.0
        terminated = truncated = False
        outcome = "running"
        max_tilt = np.radians(cfg.tilt_limit_deg)
        max_radius = 0.0
        outside_ticks = 0

        for _ in range(self.CONTROL_PER_POLICY):
            st = self.env.state()
            self._qpos_hist.append(self._qpos36(st))
            self.controller.append_state(st["q_hw"], st["dq_hw"], st["base_quat"], st["base_ang_vel"])
            _, q_des, _ = self.controller.act(self.reference, st["base_quat"])
            self.env.set_target(q_des)
            self.env.step()
            self.reference.advance()
            self._tick += 1
            if tick_callback is not None:
                tick_callback()

            command_delta = float(np.max(np.abs(self._cmd - self._last_replan_cmd)))
            due = (self._tick - self._last_replan_tick) >= cfg.replan_min_interval
            if command_delta > cfg.replan_command_delta and due:
                self._replan(st, force=True)
                self._last_replan_tick = self._tick
                self._last_replan_cmd = self._cmd.copy()
            elif self.reference.T - self.reference.cursor <= 16:
                self._replan(st)
                self._last_replan_tick = self._tick
                self._last_replan_cmd = self._cmd.copy()

            st = self.env.state()
            distance = self._distance(st)
            reward += cfg.w_progress * (self._prev_dist - distance)
            self._prev_dist = distance

            state = self.manifold_state()
            max_radius = max(max_radius, state["radius"])
            outside = min(max(0.0, state["radius"] - 1.0), cfg.excess_cap)
            reward -= cfg.w_manifold * outside * C.CONTROL_DT
            reward -= cfg.w_spine * (1.0 - max(0.0, state["spine"])) * C.CONTROL_DT
            reward -= cfg.w_energy * float(np.mean(np.square(action))) * C.CONTROL_DT

            roll, pitch = roll_pitch(st["base_quat"])
            fell = st["base_pos"][2] < cfg.fall_height or abs(roll) > max_tilt or abs(pitch) > max_tilt
            success = distance < -cfg.goal_margin and state["radius"] <= 1.0
            outside_ticks = outside_ticks + 1 if state["radius"] > cfg.manifold_out_radius else 0
            outside_far = outside_ticks > cfg.manifold_out_ticks
            if fell:
                reward -= cfg.fall_penalty
                terminated, outcome = True, "fall"
                break
            if outside_far:
                reward -= cfg.out_penalty
                terminated, outcome = True, "out_of_manifold"
                break
            if success:
                reward += cfg.goal_bonus
                terminated, outcome = True, "success"
                break

        self._steps += 1
        if not terminated and self._steps >= cfg.max_episode_steps:
            truncated, outcome = True, "timeout"
        st = self.env.state()
        state = self.manifold_state()
        self._prev_action = action.astype(np.float32)
        reward *= cfg.reward_scale
        self._episode_return += reward
        info = {
            "outcome": outcome,
            "distance": self._distance(st),
            "episode_return": self._episode_return,
            "episode_step": self._steps,
            "base_z": float(st["base_pos"][2]),
            "manifold_radius": float(state["radius"]),
            "manifold_radius_max": float(max_radius),
            "spine_alignment": float(state["spine"]),
            "crouch_amount": float(self.crouch_amount),
            "tunnel_semi_z": float(self.manifold.primitives[-1].semi[2]),
        }
        return self._obs(st), float(reward), terminated, truncated, info
