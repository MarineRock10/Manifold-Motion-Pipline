import numpy as np
import torch

from manifold_motion.stage2.composer import EnvironmentSkillComposer
from manifold_motion.stage2.constrained_generator import (
    LocalBodyResidualHead, apply_local_body_residual, family_joint_mask,
)
from manifold_motion.stage2.projection import OptimizationEmbeddedProjection


def test_composer_contract_has_no_primitive_input():
    model = EnvironmentSkillComposer(69, 9, 6, 30, (10, 10, 8), hidden=32)
    output = model(
        torch.zeros(3, 69), torch.zeros(3, 12, 69), torch.zeros(3, 36, 7),
        torch.zeros(3, 10, 10, 8), torch.zeros(3, 36, 3), torch.zeros(3, 6),
        torch.zeros(3, 9),
    )
    assert output.shape == (3, 30)


def test_local_residual_is_zero_initialized_and_body_local():
    head = LocalBodyResidualHead(feature_dim=8, horizon=6, hidden=16)
    base = torch.randn(2, 6, 38)
    masks = torch.as_tensor(np.stack((family_joint_mask("walk_forward"),
                                      family_joint_mask("door_interaction"))))
    assert torch.equal(head(base, torch.randn(2, 8), masks), base)

    trajectory = np.zeros((6, 38), dtype=np.float32)
    corrected, report = apply_local_body_residual(
        trajectory, np.ones((6, 29), dtype=np.float32), "door_interaction", bound_rad=0.05)
    mask = family_joint_mask("door_interaction")
    assert np.allclose(corrected[:, :29][:, mask == 0], 0.0)
    assert np.isclose(np.abs(corrected[:, :29]).max(), 0.05)
    assert report["active_joint_count"] == int(mask.sum())


def test_optimization_projection_is_differentiable_and_clips_joint_limits():
    layer = OptimizationEmbeddedProjection(-np.ones(29), np.ones(29), iterations=2)
    trajectory = torch.full((2, 8, 38), 2.0, requires_grad=True)
    corridor = torch.zeros(2, 8, 7)
    corridor[..., 3:6] = 0.25
    projected = layer(trajectory, corridor)
    assert projected.shape == trajectory.shape
    assert float(projected[..., :29].detach().max()) <= 1.0
    projected.sum().backward()
    assert trajectory.grad is not None
    assert torch.isfinite(trajectory.grad).all()
