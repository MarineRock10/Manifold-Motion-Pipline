"""Pure-geometry manifold environment: learn the action distribution without MuJoCo/SONIC.

The robot is reduced to a task-space model whose parameters were *measured* on the SONIC
stack, so a policy learned here can be executed (and validated) by the real stack:

    achievable speed     v_max(h) = 1.00 - 0.45 * h / crouch_max        [m/s]
    pelvis height        z(h)     = 0.76 - 0.16 * h / crouch_max        [m]
    velocity response    first order, tau_v = 0.35 s
    crouch response      first order, tau_h = 0.45 s
    body envelope        pelvis, head (+0.30 m), hands (+-0.32 m lateral, +0.05 m)

Containment uses the same `EllipsoidManifold` as the MuJoCo environment and the action
interface is identical `(vx, vy, wz, crouch)`, so the same policy runs in both worlds and
SONIC's job becomes execution + verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from .manifold import EllipsoidManifold

ENVELOPE_OFFSETS = np.array([
    [0.0, 0.0, 0.0],        # pelvis
    [0.0, 0.0, 0.30],       # head
    [0.0, 0.32, 0.05],      # left hand
    [0.0, -0.32, 0.05],     # right hand
])


def task_space_obs(x, y, psi, vx, vy, crouch, crouch_max, goal_x, manifold, pelvis_height,
                   obs_goal_scale: float = 5.0) -> np.ndarray:
    """Observation shared by the geometric model and the MuJoCo env (identical layout)."""
    rot = np.array([[np.cos(psi), -np.sin(psi), 0.0],
                    [np.sin(psi), np.cos(psi), 0.0],
                    [0.0, 0.0, 1.0]])
    points = np.array([x, y, pelvis_height]) + ENVELOPE_OFFSETS @ rot.T
    radii = manifold.radii(points)
    worst = int(np.argmax(radii))
    nearest = int(manifold.nearest(points[worst:worst + 1])[0])
    primitive = manifold.primitives[nearest]
    goal_world = np.array([goal_x - x, -y, 0.0])
    goal_body = rot.T @ goal_world
    heading = np.arctan2(goal_world[1], goal_world[0])
    heading_err = np.arctan2(np.sin(heading - psi), np.cos(heading - psi))
    rel_center = rot.T @ (primitive.center - np.array([x, y, pelvis_height]))
    axis_body = rot.T @ primitive.axis()
    return np.concatenate([
        [vx, vy],
        [np.sin(psi), np.cos(psi)],
        [crouch / crouch_max],
        goal_body / obs_goal_scale,
        [np.sin(heading_err), np.cos(heading_err)],
        [float(radii.max()), float(np.clip(primitive.axis()[2], -1.0, 1.0))],
        rel_center, axis_body, primitive.semi,
    ]).astype(np.float32)


@dataclass
class GeoConfig:
    dt: float = 0.1                     # policy step = 50 Hz control x 5
    max_episode_steps: int = 200
    goal_margin: float = 0.35
    max_lin_vel: float = 1.5
    max_lat_vel: float = 0.6
    max_yaw_rate: float = 1.0
    crouch_max: float = 1.3
    spawn_crouch: float = 1.2
    speed_upright: float = 1.00         # measured: 0.79 m/s at crouch 0 (0.8 commanded)
    speed_crouch_loss: float = 0.45     # measured: 0.40 m/s at full crouch
    pelvis_upright: float = 0.76
    pelvis_crouch_drop: float = 0.16
    tau_v: float = 0.35
    tau_h: float = 0.45
    w_progress: float = 1.0
    w_manifold: float = 30.0
    excess_cap: float = 0.4
    reward_scale: float = 0.2
    w_spine: float = 3.0
    w_energy: float = 0.1
    goal_bonus: float = 10.0
    fall_penalty: float = 10.0
    out_penalty: float = 10.0
    manifold_out_radius: float = 1.15
    manifold_out_ticks: int = 2
    obs_goal_scale: float = 5.0


class GeometricManifoldEnv:
    """Task-space model of the G1 under SONIC inside an ellipsoid manifold."""

    CONTROL_PER_POLICY = 5
    action_dim = 4

    def __init__(self, manifold: EllipsoidManifold | None = None, cfg: GeoConfig | None = None):
        self.manifold = manifold or EllipsoidManifold.single()
        self.cfg = cfg or GeoConfig()
        self._rng = np.random.default_rng(0)
        self.reset()

    # -- model --------------------------------------------------------------
    def pelvis_height(self) -> float:
        return self.cfg.pelvis_upright - self.cfg.pelvis_crouch_drop * self.crouch / self.cfg.crouch_max

    def speed_limit(self) -> float:
        return self.cfg.speed_upright - self.cfg.speed_crouch_loss * self.crouch / self.cfg.crouch_max

    def _envelope(self) -> np.ndarray:
        """World positions of the envelope points at the current pose."""
        pelvis = np.array([self.x, self.y, self.pelvis_height()])
        heading = np.array([[np.cos(self.psi), -np.sin(self.psi), 0.0],
                            [np.sin(self.psi), np.cos(self.psi), 0.0],
                            [0.0, 0.0, 1.0]])
        return pelvis + ENVELOPE_OFFSETS @ heading.T

    def manifold_state(self) -> dict:
        points = self._envelope()
        radii = self.manifold.radii(points)
        worst = int(np.argmax(radii))
        nearest = int(self.manifold.nearest(points[worst:worst + 1])[0])
        primitive = self.manifold.primitives[nearest]
        return {"radius": float(radii.max()), "primitive": nearest, "primitive_ref": primitive,
                "spine": float(np.clip(primitive.axis()[2], -1.0, 1.0))}

    # -- api ----------------------------------------------------------------
    def set_manifold(self, manifold: EllipsoidManifold) -> None:
        self.manifold = manifold

    def reset(self, **_) -> tuple[np.ndarray, dict]:
        cfg = self.cfg
        self.x = float(self.manifold.start_x())
        self.y = 0.0
        self.psi = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.crouch = cfg.spawn_crouch
        self.steps = 0
        self.episode_return = 0.0
        self.prev_dist = self._distance()
        return self.obs(), {"distance": self.prev_dist}

    def _distance(self) -> float:
        return float(self.manifold.goal_x() - self.x)

    def obs(self) -> np.ndarray:
        return task_space_obs(self.x, self.y, self.psi, self.vx, self.vy, self.crouch,
                              self.cfg.crouch_max, self.manifold.goal_x(), self.manifold,
                              self.pelvis_height(), self.cfg.obs_goal_scale)

    def step(self, action: np.ndarray, tick_callback=None):
        cfg = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        vx_cmd = action[0] * cfg.max_lin_vel
        vy_cmd = action[1] * cfg.max_lat_vel
        wz_cmd = action[2] * cfg.max_yaw_rate
        crouch_cmd = max(0.0, float(action[3])) * cfg.crouch_max

        reward = 0.0
        terminated = truncated = False
        outcome = "running"
        outside_ticks = 0
        max_radius = 0.0
        rollout = action
        for _ in range(self.CONTROL_PER_POLICY):
            dt = C.CONTROL_DT
            self.crouch += (crouch_cmd - self.crouch) * min(1.0, dt / cfg.tau_h)
            limit = self.speed_limit()
            speed = np.hypot(vx_cmd, vy_cmd)
            if speed > limit and speed > 1e-6:          # crouching costs speed
                vx_cmd, vy_cmd = vx_cmd * limit / speed, vy_cmd * limit / speed
            self.vx += (vx_cmd - self.vx) * min(1.0, dt / cfg.tau_v)
            self.vy += (vy_cmd - self.vy) * min(1.0, dt / cfg.tau_v)
            self.psi += wz_cmd * dt
            self.x += (np.cos(self.psi) * self.vx - np.sin(self.psi) * self.vy) * dt
            self.y += (np.sin(self.psi) * self.vx + np.cos(self.psi) * self.vy) * dt
            if tick_callback is not None:
                tick_callback()

            distance = self._distance()
            reward += cfg.w_progress * (self.prev_dist - distance)
            self.prev_dist = distance
            state = self.manifold_state()
            max_radius = max(max_radius, state["radius"])
            outside = min(max(0.0, state["radius"] - 1.0), cfg.excess_cap)
            reward -= cfg.w_manifold * outside * dt
            reward -= cfg.w_spine * (1.0 - max(0.0, state["spine"])) * dt
            reward -= cfg.w_energy * float(np.mean(np.square(rollout))) * dt

            outside_ticks = outside_ticks + 1 if state["radius"] > cfg.manifold_out_radius else 0
            if outside_ticks > cfg.manifold_out_ticks:
                reward -= cfg.out_penalty
                terminated, outcome = True, "out_of_manifold"
                break
            if distance < -cfg.goal_margin and state["radius"] <= 1.0:
                reward += cfg.goal_bonus
                terminated, outcome = True, "success"
                break

        self.steps += 1
        if not terminated and self.steps >= cfg.max_episode_steps:
            truncated, outcome = True, "timeout"
        reward *= cfg.reward_scale
        self.episode_return += reward
        state = self.manifold_state()
        info = {
            "outcome": outcome,
            "distance": self._distance(),
            "episode_return": self.episode_return,
            "episode_step": self.steps,
            "base_z": float(self.pelvis_height()),
            "manifold_radius": float(state["radius"]),
            "manifold_radius_max": float(max_radius),
            "spine_alignment": float(state["spine"]),
            "crouch_amount": float(self.crouch),
            "tunnel_semi_z": float(self.manifold.primitives[-1].semi[2]),
        }
        return self.obs(), float(reward), terminated, truncated, info
