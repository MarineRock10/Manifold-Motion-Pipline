"""Small optimization-embedded projection layer for Stage-2 references.

The frozen SONIC controller remains unchanged.  This layer is deliberately placed between
latent Flow decoding and the SONIC physical gate: it unrolls a few projected-gradient steps on
the decoded reference, enforcing the same quantities that matter at execution time (joint
limits, temporal smoothness, handoff continuity and route-corridor consistency).  It is not
claimed to be a learned differentiable MPC solver; the report records the objective before and
after projection so the effect is auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from . import constants as C
from .env import G1FlatEnv


@dataclass(frozen=True)
class ProjectionConfig:
    """Conservative defaults: smooth a decoded clip without erasing its gait semantics."""

    iterations: int = 1
    step_size: float = 0.08
    smooth_weight: float = 0.01
    handoff_weight: float = 0.35
    corridor_weight: float = 0.85
    root_blend: float = 0.85
    max_joint_step: float = 0.005


@lru_cache(maxsize=1)
def _policy_joint_bounds() -> tuple[np.ndarray, np.ndarray]:
    env = G1FlatEnv()
    joint_ids = env.model.actuator_trnid[env.body_act, 0]
    lower = env.model.jnt_range[joint_ids, 0][C.MUJOCO_TO_ISAACLAB].astype(np.float32)
    upper = env.model.jnt_range[joint_ids, 1][C.MUJOCO_TO_ISAACLAB].astype(np.float32)
    return lower, upper


def _objective(q: np.ndarray, root: np.ndarray, corridor: np.ndarray,
               lower: np.ndarray, upper: np.ndarray,
               original_q0: np.ndarray, config: ProjectionConfig) -> dict[str, float]:
    d2 = q[:-2] - 2.0 * q[1:-1] + q[2:] if len(q) > 2 else np.zeros((0, q.shape[1]))
    smooth = float(np.mean(d2 * d2)) if d2.size else 0.0
    violation = np.maximum(lower[None] - q, 0.0) + np.maximum(q - upper[None], 0.0)
    limit = float(np.mean(violation * violation))
    handoff = float(np.mean((q[0] - original_q0) ** 2))
    target = np.asarray(corridor[:len(root), :3], dtype=np.float64)
    corridor_error = float(np.mean((root - target) ** 2)) if len(root) else 0.0
    total = (config.smooth_weight * smooth + 10.0 * limit
             + config.handoff_weight * handoff + config.corridor_weight * corridor_error)
    return {"total": total, "smoothness": smooth, "joint_limit": limit,
            "handoff": handoff, "corridor": corridor_error}


def project_reference(trajectory: np.ndarray, corridor: np.ndarray,
                      config: ProjectionConfig | None = None,
                      handoff_q: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, object]]:
    """Project one decoded ``[T,38]`` reference and return an audit report.

    The root position is route-local and is not consumed by the current SONIC action head, but
    retaining it on the route center makes the dynamic-model target internally consistent.  Joint
    limits and smoothness are applied to the actual 29-DOF reference SONIC receives.
    """
    config = ProjectionConfig() if config is None else config
    value = np.asarray(trajectory, dtype=np.float64)
    corridor = np.asarray(corridor, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 38:
        raise ValueError(f"projection expects [T,38], got {value.shape}")
    if corridor.ndim != 2 or corridor.shape[1] != 7 or len(corridor) != len(value):
        raise ValueError(f"projection corridor must be [{len(value)},7], got {corridor.shape}")
    if config.iterations < 0 or config.step_size <= 0 or config.max_joint_step <= 0:
        raise ValueError("projection iterations/step sizes must be positive")
    lower, upper = _policy_joint_bounds()
    q_original = value[:, :29].copy()
    q = q_original.copy()
    root_original = value[:, 29:32].copy()
    root = root_original.copy()
    original_q0 = q_original[0].copy()
    if handoff_q is not None:
        handoff_q = np.asarray(handoff_q, dtype=np.float64).reshape(29)
        original_q0 = handoff_q.copy()
    before = _objective(q, root, corridor, lower, upper, original_q0, config)
    for _ in range(int(config.iterations)):
        if len(q) > 2:
            d2 = q[:-2] - 2.0 * q[1:-1] + q[2:]
            q[1:-1] -= config.step_size * config.smooth_weight * 2.0 * d2
        q = np.clip(q, lower[None], upper[None])
        # Keep each projected frame close to the decoded Flow proposal.  This makes the layer
        # a feasibility correction rather than a hidden motion retargeter.
        q = q_original + np.clip(q - q_original, -config.max_joint_step, config.max_joint_step)
        q[0] = (1.0 - config.handoff_weight) * q_original[0] + config.handoff_weight * original_q0
        q[0] = np.clip(q[0], lower, upper)
        target = corridor[:, :3]
        root = (1.0 - config.root_blend) * root_original + config.root_blend * target
    after = _objective(q, root, corridor, lower, upper, original_q0, config)
    projected = value.copy()
    projected[:, :29] = q
    projected[:, 29:32] = root
    report = {
        "contract": "projected-gradient joint limits + temporal smoothness + route-center root consistency",
        "config": asdict(config),
        "objective_before": before,
        "objective_after": after,
        "joint_delta_max_rad": float(np.max(np.abs(q - q_original))),
        "root_delta_max_m": float(np.max(np.abs(root - root_original))),
        "handoff_q_used": bool(handoff_q is not None),
    }
    return projected.astype(np.float32), report
