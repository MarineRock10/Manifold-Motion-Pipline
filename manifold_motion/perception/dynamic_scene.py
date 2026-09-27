"""Deterministic MuJoCo moving-obstacle schedules shared by execution and radar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


SUPPORTED_DYNAMIC_EVENTS = (
    "crossing", "appear_disappear", "moving_wall", "route_reopen",
    "projectile", "projectile_grazing", "projectile_overhead",
)


@dataclass(frozen=True)
class DynamicObstacleState:
    event: str
    time_s: float
    active: bool
    center_xy: tuple[float, float]
    center_z: float | None = None


def obstacle_state(event: str | None, time_s: float) -> DynamicObstacleState | None:
    """Return the deterministic world-frame pose for a benchmark event.

    The schedule is deliberately geometry-only and shared by the MuJoCo executor, simulated
    radar, and report renderer.  It is not a learned prediction; the benchmark evaluates the
    planner's response to a synchronized map change.
    """
    if event is None:
        return None
    event = str(event)
    if event not in SUPPORTED_DYNAMIC_EVENTS:
        raise ValueError(f"unsupported dynamic event: {event}")
    t = max(0.0, float(time_s))
    if event == "crossing":
        # Enter from the right, cross the route, then leave to the left.  The crossing station
        # is far enough ahead that a 2.5 Hz radar/D* loop has several updates to bend the
        # physical trajectory before the self-manifold reaches the swept volume.
        alpha = np.clip(t / 3.0, 0.0, 1.0)
        active = t < 3.5
        return DynamicObstacleState(event, t, bool(active), (2.45, float(-1.00 + 2.00 * alpha)))
    if event == "appear_disappear":
        return DynamicObstacleState(event, t, bool(2.0 <= t < 6.0), (1.80, 0.0))
    if event == "moving_wall":
        # A short wall oscillates laterally while remaining inside the static corridor.
        y = float(0.78 * np.sin(2.0 * np.pi * (t - 0.5) / 6.0))
        # It clears after the first sweep. This tests repeated online route repair while
        # retaining a causally reachable exit; a permanently blocking wall is covered by the
        # safety-stop test, not by a locomotion-success metric.
        return DynamicObstacleState(event, t, bool(t < 3.5), (1.80, y))
    if event in {"projectile", "projectile_grazing", "projectile_overhead"}:
        # A torso-height object is launched only after the online estimator has observed an
        # initially clear route.  The robot never receives this schedule as a command: radar
        # sees the moving geom, while the reactive policy estimates relative velocity from
        # consecutive observations and selects its own avoidance action.
        launch_s = 0.55
        flight_s = t - launch_s
        if event == "projectile_grazing":
            active = 0.0 <= flight_s < 3.10
            x = 3.40 - 1.35 * max(0.0, flight_s)
            # A close lateral miss is the nominal visual demo: the learned route/action
            # decision changes the body envelope enough to preserve a measurable gap.
            y = float(0.55 + 0.08 * np.sin(2.4 * max(0.0, flight_s)))
        else:
            active = 0.0 <= flight_s < 4.20
            # A 0.90 m/s centreline launch is within the lateral acceleration envelope of the
            # frozen SONIC side gait. Faster launches are retained for the pressure sweep.
            x = 3.40 - 0.90 * max(0.0, flight_s)
            y = float(0.08 * np.sin(2.4 * max(0.0, flight_s)))
        center_z = 1.48 if event == "projectile_overhead" else 1.02
        return DynamicObstacleState(event, t, bool(active), (float(x), y), center_z)
    # The centre block initially closes the route, then reopens it after the robot has
    # committed to the first safe corridor.  Keeping it in the scene at t=0 makes the initial
    # M_e and the first radar frame causal.
    return DynamicObstacleState(event, t, bool(t < 4.0), (1.80, 0.0))


def apply_dynamic_obstacle(model: mujoco.MjModel, data: mujoco.MjData,
                           event: str | None, time_s: float,
                           *, geom_name: str = "obstacle_dynamic_block") -> dict[str, Any] | None:
    """Apply the schedule to a worldbody box and refresh derived MuJoCo state."""
    state = obstacle_state(event, time_s)
    if state is None:
        return None
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise ValueError(f"dynamic scene is missing geom {geom_name!r}")
    model.geom_pos[geom_id, 0] = state.center_xy[0]
    model.geom_pos[geom_id, 1] = state.center_xy[1]
    # An inactive obstacle is moved below the floor, rather than deleting the geom.  This
    # keeps the MuJoCo model topology and radar geom id stable across all frames.
    center_z = dynamic_half_z(state.event) if state.center_z is None else float(state.center_z)
    model.geom_pos[geom_id, 2] = center_z if state.active else -5.0
    mujoco.mj_forward(model, data)
    return {
        "event": state.event, "time_s": state.time_s, "active": state.active,
        "center_xy_m": [float(state.center_xy[0]), float(state.center_xy[1])],
        "center_z_m": float(center_z),
    }


def dynamic_half_xy(event: str) -> tuple[float, float]:
    return {
        "crossing": (0.18, 0.20),
        "appear_disappear": (0.26, 0.28),
        "moving_wall": (0.10, 0.45),
        "route_reopen": (0.28, 0.34),
        "projectile": (0.12, 0.12), "projectile_grazing": (0.12, 0.12),
        "projectile_overhead": (0.12, 0.12),
    }[str(event)]


def dynamic_half_z(event: str) -> float:
    # Crossing/moving-wall pilots are lateral blockers: keep their top below the nominal
    # torso-clearance threshold so the semantic router requests side/turn motion instead of
    # misclassifying a vertical wall as a low-ceiling crouch.  The dedicated appearance/reopen
    # events retain the taller block used by the obstacle-avoidance stress test.
    if str(event) in {"projectile", "projectile_grazing", "projectile_overhead"}:
        return 0.12
    return 0.25 if str(event) in {"crossing", "moving_wall"} else 0.55


def dynamic_center_z(event: str, state: DynamicObstacleState | None = None) -> float:
    """World-frame obstacle centre height, distinct from its half-height."""
    if state is not None and state.center_z is not None:
        return float(state.center_z)
    return dynamic_half_z(event)
