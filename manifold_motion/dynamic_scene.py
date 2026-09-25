"""Deterministic MuJoCo moving-obstacle schedules shared by execution and radar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


SUPPORTED_DYNAMIC_EVENTS = (
    "crossing", "appear_disappear", "moving_wall", "route_reopen",
)


@dataclass(frozen=True)
class DynamicObstacleState:
    event: str
    time_s: float
    active: bool
    center_xy: tuple[float, float]


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
    model.geom_pos[geom_id, 2] = dynamic_half_z(state.event) if state.active else -5.0
    mujoco.mj_forward(model, data)
    return {
        "event": state.event, "time_s": state.time_s, "active": state.active,
        "center_xy_m": [float(state.center_xy[0]), float(state.center_xy[1])],
    }


def dynamic_half_xy(event: str) -> tuple[float, float]:
    return {
        "crossing": (0.18, 0.20),
        "appear_disappear": (0.26, 0.28),
        "moving_wall": (0.10, 0.45),
        "route_reopen": (0.28, 0.34),
    }[str(event)]


def dynamic_half_z(event: str) -> float:
    # Crossing/moving-wall pilots are lateral blockers: keep their top below the nominal
    # torso-clearance threshold so the semantic router requests side/turn motion instead of
    # misclassifying a vertical wall as a low-ceiling crouch.  The dedicated appearance/reopen
    # events retain the taller block used by the obstacle-avoidance stress test.
    return 0.25 if str(event) in {"crossing", "moving_wall"} else 0.55
