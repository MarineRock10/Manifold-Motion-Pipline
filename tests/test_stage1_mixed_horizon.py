import numpy as np
import torch
from manifold_motion.core import constants as C
from manifold_motion.stage1.residual_rl import ResidualConfig
from manifold_motion.stage1.residual_long_rl import StructuredResidualPolicy,eligible


def test_structured_limits_follow_policy_order():
    cfg=ResidualConfig();p=StructuredResidualPolicy(np.eye(29)[:12],cfg)
    hardware=p.joint_limits.numpy()[C.ISAACLAB_TO_MUJOCO]
    np.testing.assert_allclose(hardware[:15],.04)
    np.testing.assert_allclose(hardware[15:],.16)
    x=torch.randn(3,22);base=torch.randn(3,29)
    torch.testing.assert_close(p.action(x,base),base)
    assert torch.all((p.decode(torch.randn(3,12)*100,base)-base).abs()<=p.joint_limits+1e-6)


def test_selection_rejects_safety_tradeoffs_in_either_horizon():
    b=dict(accepted=.95,fallen=.01,target_mae=.22)
    good=dict(b,target_mae=.21)
    assert eligible(good,good,b,b)
    assert not eligible(dict(good,accepted=.94),good,b,b)
    assert not eligible(good,dict(good,accepted=.94),b,b)
    assert not eligible(good,dict(good,fallen=.02),b,b)
    assert not eligible(good,dict(good,target_mae=.23),b,b)
