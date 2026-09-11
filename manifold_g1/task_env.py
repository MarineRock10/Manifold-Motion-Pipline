"""Goal-reaching RL environment: simple manifold, velocity-command policy, frozen SONIC execution.

Policy (10 Hz) outputs a body-frame velocity command; it is mapped to the kinematic
planner (movement/facing/speed), whose 50 Hz output is the motion reference that the
frozen SONIC controller tracks in MuJoCo.

Observation (78-D): body-frame velocities, gravity, joint state relative to the default
pose, goal vector, heading error, previous action, manifold parameters.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from . import constants as C
from .env import G1FlatEnv
from .manifold import ManifoldSpec, build_scene
from .planner import SonicPlanner
from .reference import PlannedReference
from .sonic import SonicController


def roll_pitch(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


@dataclass
class TaskConfig:
    max_episode_steps: int = 200        # policy steps (0.1 s each)
    success_radius: float = 0.35
    fall_height: float = 0.45
    tilt_limit_deg: float = 60.0
    max_lin_vel: float = 1.5
    max_lat_vel: float = 0.6
    max_yaw_rate: float = 1.0
    w_progress: float = 1.0
    w_collision: float = 30.0
    penetration_clip: float = 0.01      # cap the per-tick penetration used in the penalty [m]
    hard_collision_penetration: float = 0.03
    hard_collision_penalty: float = 5.0
    w_energy: float = 0.1
    goal_bonus: float = 10.0
    fall_penalty: float = 10.0
    out_penalty: float = 10.0
    replan_min_interval: int = 10       # control ticks
    replan_command_delta: float = 0.25
    obs_goal_scale: float = 5.0


class GoalReachEnv:
    CONTROL_PER_POLICY = 5              # 50 Hz control / 10 Hz policy

    def __init__(self, spec: ManifoldSpec | None = None, cfg: TaskConfig | None = None):
        self.spec = spec or ManifoldSpec()
        self.cfg = cfg or TaskConfig()
        self.scene_path = build_scene(self.spec)
        self.env = G1FlatEnv(self.scene_path)
        self.controller = SonicController()
        self.planner = SonicPlanner()
        self.obstacle_geoms = self._geom_ids(("wall_left", "wall_right", "ceiling", "wall_back"))
        self.goal = np.array([self.spec.goal_x, 0.0])
        self.action_dim = 3
        self._qpos_hist: deque[np.ndarray] = deque(maxlen=4)
        self._prev_action = np.zeros(3)
        self._cmd = np.zeros(3)
        self._tick = 0
        self._steps = 0
        self._prev_dist = 0.0
        self._last_replan_tick = -10 ** 9
        self._last_replan_cmd = np.zeros(3)
        self._episode_return = 0.0
        self.reference = PlannedReference.static_stand(np.array([1.0, 0.0, 0.0, 0.0]))

    # -- helpers ------------------------------------------------------------
    def _geom_ids(self, names) -> set[int]:
        ids = set()
        for name in names:
            gid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                ids.add(int(gid))
        return ids

    @staticmethod
    def _qpos36(st: dict) -> np.ndarray:
        qpos = np.empty(7 + 29)
        qpos[0:3] = st["base_pos"]
        qpos[3:7] = st["base_quat"]
        qpos[7:] = st["q_hw"]
        return qpos

    def _distance(self, st: dict) -> float:
        return float(np.linalg.norm(st["base_pos"][:2] - self.goal))

    def _penetration(self) -> float:
        penetration = 0.0
        for i in range(self.env.data.ncon):
            contact = self.env.data.contact[i]
            if contact.geom1 in self.obstacle_geoms or contact.geom2 in self.obstacle_geoms:
                penetration += max(0.0, -float(contact.dist))
        return penetration

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
        return np.concatenate([
            rot.T @ st["base_lin_vel"] / 2.0,
            st["base_ang_vel"] / 4.0,
            gravity,
            q_isaac,
            dq_isaac / 20.0,
            goal_body / self.cfg.obs_goal_scale,
            [np.sin(heading_err), np.cos(heading_err)],
            self._prev_action,
            np.array(self.spec.grid) / 5.0,
        ]).astype(np.float32)

    def _plan_kwargs(self, st: dict) -> dict:
        vx, vy, wz = self._cmd
        rot = C.quat_to_matrix(st["base_quat"])
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        speed = float(np.hypot(vx, vy))
        facing = (float(np.cos(yaw + 0.5 * wz)), float(np.sin(yaw + 0.5 * wz)), 0.0)
        if speed > 0.05:
            movement = (
                float(np.cos(yaw) * vx - np.sin(yaw) * vy),
                float(np.sin(yaw) * vx + np.cos(yaw) * vy),
                0.0,
            )
            return dict(mode=2, movement=movement, facing=facing, target_vel=speed)
        return dict(mode=0, movement=(0.0, 0.0, 0.0), facing=facing, target_vel=-1.0)

    def _replan(self, st: dict, force: bool = False) -> None:
        context = np.stack(self._qpos_hist)
        self.reference.maybe_replan(self.planner, reserve=16, context=context,
                                    force=force, **self._plan_kwargs(st))

    # -- gym-like API -------------------------------------------------------
    def reset(self) -> tuple[np.ndarray, dict]:
        self.env.reset()
        self.controller.reset()
        st = self.env.state()
        self._qpos_hist.clear()
        for _ in range(4):
            self._qpos_hist.append(self._qpos36(st))
        self.reference = PlannedReference.static_stand(st["base_quat"])
        self._prev_action = np.zeros(3)
        self._cmd = np.zeros(3)
        self._tick = 0
        self._steps = 0
        self._prev_dist = self._distance(st)
        self._last_replan_tick = -10 ** 9
        self._last_replan_cmd = np.zeros(3)
        self._episode_return = 0.0
        self._replan(st, force=True)
        return self._obs(st), {"distance": self._prev_dist}

    def step(self, action: np.ndarray):
        cfg = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._cmd = np.array([action[0] * cfg.max_lin_vel,
                              action[1] * cfg.max_lat_vel,
                              action[2] * cfg.max_yaw_rate])
        reward = 0.0
        terminated = truncated = False
        outcome = "running"
        max_tilt = np.radians(cfg.tilt_limit_deg)
        collision = False

        for _ in range(self.CONTROL_PER_POLICY):
            st = self.env.state()
            self._qpos_hist.append(self._qpos36(st))
            self.controller.append_state(st["q_hw"], st["dq_hw"], st["base_quat"], st["base_ang_vel"])
            _, q_des, _ = self.controller.act(self.reference, st["base_quat"])
            self.env.set_target(q_des)
            self.env.step()
            self.reference.advance()
            self._tick += 1

            command_delta = float(np.max(np.abs(self._cmd - self._last_replan_cmd)))
            due = (self._tick - self._last_replan_tick) >= cfg.replan_min_interval
            if self.reference.T - self.reference.cursor <= 16 or (command_delta > cfg.replan_command_delta and due):
                self._replan(st)
                self._last_replan_tick = self._tick
                self._last_replan_cmd = self._cmd.copy()

            st = self.env.state()
            distance = self._distance(st)
            reward += cfg.w_progress * (self._prev_dist - distance)
            self._prev_dist = distance

            penetration = self._penetration()
            if penetration > 1e-5:
                collision = True
                reward -= cfg.w_collision * min(penetration, cfg.penetration_clip)
            reward -= cfg.w_energy * float(np.mean(np.square(action))) * C.CONTROL_DT

            roll, pitch = roll_pitch(st["base_quat"])
            fell = st["base_pos"][2] < cfg.fall_height or abs(roll) > max_tilt or abs(pitch) > max_tilt
            success = distance < cfg.success_radius
            outside = (abs(st["base_pos"][1]) > self.spec.width / 2
                       or st["base_pos"][0] < self.spec.start_x - 0.7
                       or st["base_pos"][0] > self.spec.goal_x + 0.7)
            if fell:
                reward -= cfg.fall_penalty
                terminated, outcome = True, "fall"
                break
            if penetration > cfg.hard_collision_penetration:
                reward -= cfg.hard_collision_penalty
                terminated, outcome = True, "collision"
                break
            if success:
                reward += cfg.goal_bonus
                terminated, outcome = True, "success"
                break
            if outside:
                reward -= cfg.out_penalty
                terminated, outcome = True, "out_of_bounds"
                break

        self._steps += 1
        if not terminated and self._steps >= cfg.max_episode_steps:
            truncated, outcome = True, "timeout"
        st = self.env.state()
        self._prev_action = action.astype(np.float32)
        self._episode_return += reward
        info = {
            "outcome": outcome,
            "distance": self._distance(st),
            "collision": collision,
            "episode_return": self._episode_return,
            "episode_step": self._steps,
            "base_z": float(st["base_pos"][2]),
        }
        return self._obs(st), float(reward), terminated, truncated, info
