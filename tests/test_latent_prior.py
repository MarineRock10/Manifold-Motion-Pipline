import torch
from manifold_motion.stage2.latent_prior import StateConditionedSkillVAE, latent_barrier

def test_state_conditioned_prior_and_barrier():
    model=StateConditionedSkillVAE(12,8,latent_dim=4,condition_hidden=8,hidden=16)
    target=torch.randn(3,12); condition=torch.randn(3,8)
    decoded,pm,pl,qm,ql=model(target,condition)
    assert decoded.shape==target.shape and pm.shape==(3,4)
    constrained=latent_barrier(pm+10.0,pm,pl,radius=2.0)
    assert torch.all(torch.linalg.vector_norm((constrained-pm)/torch.exp(0.5*pl),dim=-1)<=2.0001)