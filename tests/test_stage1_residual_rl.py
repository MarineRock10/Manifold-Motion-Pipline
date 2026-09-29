import numpy as np
import pytest
import torch
from manifold_motion.stage1.residual_rl import (
    ResidualConfig,ResidualPolicy,dense_reward,group_advantages,make_basis,ppo_update,
)


def test_zero_residual_preserves_imitation_and_bound():
    cfg=ResidualConfig()
    policy=ResidualPolicy(np.eye(29)[:cfg.modes],cfg)
    x=torch.randn(5,22);base=torch.randn(5,29)
    torch.testing.assert_close(policy.action(x,base),base)
    corrected=policy.decode(torch.randn(5,cfg.modes)*100,base)
    assert (corrected-base).abs().max()<=cfg.residual_limit+1e-6


def test_context_difficulty_offset_does_not_change_advantages():
    reward=torch.tensor([[1.,2.,3.,4.],[-9.,-8.,-7.,-6.]])
    advantage=group_advantages(reward)
    torch.testing.assert_close(advantage[0],advantage[1])
    assert advantage[0,0]<0 and advantage[0,-1]>0
    torch.testing.assert_close(group_advantages(reward+77),advantage)
    assert group_advantages(torch.zeros(2,4)).count_nonzero()==0
    with pytest.raises(ValueError):group_advantages(torch.zeros(8,1))


def test_reward_prefers_actual_goal_not_easier_command_tracking():
    result=dict(target_mae=.2,radius=.8,drift_m=.03,fallen=False,nonfoot_contact=False,tracking_mae=.1)
    assert dense_reward(dict(result,target_mae=.15,tracking_mae=.3))>dense_reward(result)
    assert dense_reward(dict(result,fallen=True))<dense_reward(result)
    assert dense_reward(dict(result,radius=1.2))<dense_reward(result)


def test_basis_uses_supplied_train_data_and_is_orthonormal():
    y=np.random.default_rng(3).normal(size=(100,29))
    basis=make_basis(y,12)
    np.testing.assert_allclose(basis@basis.T,np.eye(12),atol=1e-6)
    assert np.all(basis[np.arange(12),np.abs(basis).argmax(1)]>=0)


def test_ppo_has_multiple_updates_nonunit_ratios_and_kl_guard():
    torch.manual_seed(3)
    cfg=ResidualConfig(learning_rate=1e-3,target_kl=.2)
    policy=ResidualPolicy(np.eye(29)[:cfg.modes],cfg)
    opt=torch.optim.Adam(policy.parameters(),lr=cfg.learning_rate)
    x=torch.randn(8,22);base=torch.zeros(8,29)
    raw=torch.randn(8,4,cfg.modes)*cfg.std
    rewards=raw[:,:,0]*3
    updates=ppo_update(policy,opt,x,base,raw,rewards)
    assert len(updates)>1
    assert any(u['ratio_std']>0 for u in updates[1:])
    assert not updates[0]['rolled_back']
    assert policy(x,base)[:,0].mean()>0


def test_excessive_kl_rolls_back_last_step():
    torch.manual_seed(9)
    cfg=ResidualConfig(learning_rate=1.,target_kl=1e-12)
    policy=ResidualPolicy(np.eye(29)[:cfg.modes],cfg)
    opt=torch.optim.Adam(policy.parameters(),lr=cfg.learning_rate)
    x=torch.randn(8,22);base=torch.zeros(8,29)
    raw=torch.randn(8,4,cfg.modes)*cfg.std
    updates=ppo_update(policy,opt,x,base,raw,raw[:,:,0])
    assert updates[0]['rolled_back']
    torch.testing.assert_close(policy(x,base),torch.zeros(8,cfg.modes))
