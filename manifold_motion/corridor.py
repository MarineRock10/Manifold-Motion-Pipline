"""Time-aligned safe-corridor and SDF conditions for the dynamic stage.

The first BONES-SEED replay set has no sensor map.  Rather than attaching a misleading
``flat`` token to every motion, this module performs the reverse-data construction described
in the research plan:

1. measure the *executed* G1 body envelope at every replay frame;
2. centre a local ellipsoidal corridor on the reference route;
3. inflate it by a reproducible clearance margin; and
4. rasterise the union of those ellipsoids as a local signed-free-space field.

The resulting condition is labelled ``reverse_synthesized``.  It is geometrically consistent
with the successful execution but it is **not** a real RGB-D/LiDAR map and must not be reported
as one.  The representation deliberately matches the eventual perception interface, so a
real SDF/corridor producer can replace this constructor without changing Stage-2 training.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import constants as C
from .body_envelope import body_points, fit_ellipsoid
from .env import G1FlatEnv


@dataclass(frozen=True)
class CorridorConfig:
    """Geometry and augmentation parameters for one local environment condition."""

    clearance_min_m: float = 0.08
    clearance_max_m: float = 0.28
    sdf_shape: tuple[int, int, int] = (10, 10, 8)
    # Fixed local planning volume, x forward / y lateral / z up, measured from the current root.
    sdf_lower_m: tuple[float, float, float] = (-0.50, -1.20, -0.15)
    sdf_upper_m: tuple[float, float, float] = (2.50, 1.20, 1.65)


class ExecutedEnvelopeEstimator:
    """Evaluate a conservative robot-surface ellipsoid for policy-order replay states."""

    def __init__(self):
        self.env = G1FlatEnv()

    def _set_state(self, q_policy: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray) -> None:
        data = self.env.data
        data.qpos[0:3] = base_pos
        data.qpos[3:7] = base_quat / np.linalg.norm(base_quat)
        data.qpos[self.env.body_qadr] = q_policy[C.ISAACLAB_TO_MUJOCO]
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, data)

    def surface_points(self, q_policy: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
        """Robot surface samples in world coordinates for one policy-order state."""
        self._set_state(np.asarray(q_policy, dtype=np.float64), np.asarray(base_pos, dtype=np.float64),
                        np.asarray(base_quat, dtype=np.float64))
        return body_points(self.env.model, self.env.data)

    def sequence(self, q_policy: np.ndarray, base_pos: np.ndarray,
                 base_quat: np.ndarray) -> np.ndarray:
        q_policy = np.asarray(q_policy, dtype=np.float64)
        base_pos = np.asarray(base_pos, dtype=np.float64)
        base_quat = np.asarray(base_quat, dtype=np.float64)
        if not (q_policy.ndim == 2 and q_policy.shape[1] == 29 and
                base_pos.shape == (len(q_policy), 3) and base_quat.shape == (len(q_policy), 4)):
            raise ValueError("q_policy/base position/base quaternion shapes are inconsistent")
        semis = np.empty((len(q_policy), 3), dtype=np.float32)
        for index, (q, pos, quat) in enumerate(zip(q_policy, base_pos, base_quat)):
            self._set_state(q, pos, quat)
            # Work in the instantaneous pelvis/root axes.  Fitting in world axes would make a
            # turn appear wider merely because it rotated relative to the room.
            local = (body_points(self.env.model, self.env.data) - pos) @ C.quat_to_matrix(self.env.data.qpos[3:7])
            semis[index] = np.asarray(fit_ellipsoid(local, center=np.zeros(3))["semi"], dtype=np.float32)
        return semis


def local_centres(route_pos: np.ndarray, root_quat: np.ndarray, origin: int, future: slice) -> np.ndarray:
    """Executed route centres in the currently executed root-local frame."""
    rotation = C.quat_to_matrix(root_quat[origin])
    return (route_pos[future] - route_pos[origin]) @ rotation


def local_yaws(root_quat: np.ndarray, origin: int, future: slice) -> np.ndarray:
    """Yaw of every future body envelope relative to the origin root axes."""
    origin_inv = C.quat_conj(root_quat[origin])
    values = []
    for quat in root_quat[future]:
        forward = C.quat_rotate(C.quat_mul(origin_inv, quat), np.array([1.0, 0.0, 0.0]))
        values.append(np.arctan2(forward[1], forward[0]))
    return np.asarray(values, dtype=np.float64)


def _margin(seed: int, config: CorridorConfig) -> np.ndarray:
    if not (0.0 <= config.clearance_min_m <= config.clearance_max_m):
        raise ValueError("corridor clearance must satisfy 0 <= min <= max")
    rng = np.random.default_rng(seed)
    # One margin per axis is intentional: a narrow lateral aperture and a low overhead ceiling
    # are exactly the environmental variations that should distinguish side/crouch primitives.
    return rng.uniform(config.clearance_min_m, config.clearance_max_m, size=3)


def corridor_from_window(route_pos: np.ndarray, root_quat: np.ndarray, envelope_semi: np.ndarray,
                         origin: int, future: slice, *, seed: int,
                         config: CorridorConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return ``M_e(t)`` ellipsoids and an egocentric signed-free-space grid.

    ``corridor`` has ``[centre xyz, semi xyz, yaw]`` at every future tick.  Its centreline is
    the *executed* root path, which makes reverse construction geometrically honest; `R_ref`
    stays the generation target and is never used to fake containment.  The SDF is positive
    inside the free corridor union and negative outside.  Its magnitude is an ellipsoid-distance
    approximation in metres, clipped only to keep normalization robust.
    """
    centres = local_centres(route_pos, root_quat, origin, future)
    semi = np.asarray(envelope_semi[future], dtype=np.float64) + _margin(seed, config)[None, :]
    if np.any(semi <= 0.0):
        raise ValueError("corridor semi-axes must be positive")
    corridor = np.concatenate([centres, semi, local_yaws(root_quat, origin, future)[:, None]], axis=1).astype(np.float32)
    return corridor, corridor_sdf(corridor, config)


def corridor_sdf(corridor: np.ndarray, config: CorridorConfig) -> np.ndarray:
    """Rasterise the union of corridor ellipsoids in the fixed local planning volume."""
    shape = tuple(int(x) for x in config.sdf_shape)
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError("sdf_shape must have three dimensions >= 2")
    lower, upper = np.asarray(config.sdf_lower_m), np.asarray(config.sdf_upper_m)
    axes = [np.linspace(lower[i], upper[i], shape[i]) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    centres, semi, yaw = corridor[:, :3], corridor[:, 3:6], corridor[:, 6]
    delta = points[:, None, :] - centres[None, :, :]
    cos, sin = np.cos(yaw)[None, :], np.sin(yaw)[None, :]
    # World/origin-local vectors -> each ellipsoid's yaw-local vectors.
    rotated = np.stack([delta[:, :, 0] * cos + delta[:, :, 1] * sin,
                        -delta[:, :, 0] * sin + delta[:, :, 1] * cos,
                        delta[:, :, 2]], axis=2)
    normalized_radius = np.linalg.norm(rotated / semi[None, :, :], axis=2)
    signed = (1.0 - normalized_radius) * semi.min(axis=1)[None, :]
    # A union is free if a point lies in at least one tube element.
    return np.clip(signed.max(axis=1), -1.0, 1.0).reshape(shape).astype(np.float32)


def _resample_corridor(corridor: np.ndarray, frames: int) -> np.ndarray:
    """Linear 30->50 Hz corridor resampling, with yaw unwrapped before interpolation."""
    corridor = np.asarray(corridor, dtype=np.float64)
    source_t = np.arange(len(corridor), dtype=np.float64)
    target_t = np.linspace(0.0, len(corridor) - 1, frames)
    result = np.stack([np.interp(target_t, source_t, corridor[:, column]) for column in range(6)], axis=1)
    yaw = np.interp(target_t, source_t, np.unwrap(corridor[:, 6]))
    return np.concatenate([result, yaw[:, None]], axis=1)


def execution_corridor_radius(q_policy: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray,
                              corridor: np.ndarray) -> dict[str, float]:
    """Measure the replayed robot surface against its input safe corridor.

    The returned radius is the maximum of the exact mesh-sample ellipsoid implicit value,
    not a base-point proxy.  A value <= 1 means every sampled G1 surface point remained in its
    assigned corridor ellipsoid at that tick.
    """
    q_policy, base_pos, base_quat = (np.asarray(value) for value in (q_policy, base_pos, base_quat))
    if not (len(q_policy) == len(base_pos) == len(base_quat)):
        raise ValueError("execution arrays must have the same length")
    corridor = _resample_corridor(corridor, len(q_policy))
    origin_pos, origin_rotation = base_pos[0], C.quat_to_matrix(base_quat[0])
    estimator = ExecutedEnvelopeEstimator()
    maxima = []
    for q, pos, quat, element in zip(q_policy, base_pos, base_quat, corridor):
        local_points = (estimator.surface_points(q, pos, quat) - origin_pos) @ origin_rotation
        delta = local_points - element[:3]
        cos, sin = np.cos(element[6]), np.sin(element[6])
        aligned = np.column_stack([delta[:, 0] * cos + delta[:, 1] * sin,
                                   -delta[:, 0] * sin + delta[:, 1] * cos,
                                   delta[:, 2]])
        maxima.append(float(np.linalg.norm(aligned / element[3:6], axis=1).max()))
    values = np.asarray(maxima)
    return {"corridor_radius_max": float(values.max()),
            "corridor_radius_p95": float(np.quantile(values, 0.95)),
            "corridor_radius_mean": float(values.mean())}
