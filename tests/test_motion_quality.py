from types import SimpleNamespace

import numpy as np

from manifold_motion.core import constants as C
from manifold_motion.stage2.flow_route_candidates import (
    _continuous_handoff_phase,
    _gait_balance_metrics,
)


def _motion(frames: int = 86, offset: float = 0.0) -> SimpleNamespace:
    phase = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    q = np.zeros((frames, 29), dtype=np.float64)
    q[:, :12] = np.sin(phase[:, None] + offset)
    dq = np.gradient(q, 0.02, axis=0)
    return SimpleNamespace(T=frames, joint_pos_policy=q, joint_vel_policy=dq)


def test_phase_handoff_stays_near_normalized_cycle_time() -> None:
    previous = _motion()
    current = _motion(offset=0.03)
    previous_phase = 24
    selected = _continuous_handoff_phase(
        current, previous, previous_phase, previous.joint_pos_policy[previous_phase]
    )
    assert abs(selected - previous_phase) <= 4


def test_gait_balance_detects_one_sided_leg_energy() -> None:
    frames = 100
    t = np.linspace(0.0, 4.0 * np.pi, frames)
    balanced_hw = np.zeros((frames, 29), dtype=np.float64)
    balanced_dq_hw = np.zeros_like(balanced_hw)
    balanced_hw[:, :6] = np.sin(t[:, None])
    balanced_hw[:, 6:12] = np.sin(t[:, None] + np.pi)
    balanced_dq_hw[:, :6] = np.cos(t[:, None])
    balanced_dq_hw[:, 6:12] = np.cos(t[:, None] + np.pi)
    balanced = _gait_balance_metrics(
        balanced_hw[:, C.MUJOCO_TO_ISAACLAB],
        balanced_dq_hw[:, C.MUJOCO_TO_ISAACLAB],
    )
    one_sided_dq_hw = balanced_dq_hw.copy()
    one_sided_dq_hw[:, 6:12] *= 0.02
    one_sided = _gait_balance_metrics(
        balanced_hw[:, C.MUJOCO_TO_ISAACLAB],
        one_sided_dq_hw[:, C.MUJOCO_TO_ISAACLAB],
    )
    assert balanced["leg_velocity_energy_imbalance"] < 0.02
    assert one_sided["leg_velocity_energy_imbalance"] > 0.85
