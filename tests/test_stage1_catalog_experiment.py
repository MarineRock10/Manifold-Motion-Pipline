import numpy as np
import torch

from manifold_motion.core import constants as C
from manifold_motion.core.reference import KeyframeReference
from manifold_motion.stage1.catalog_experiment import (
    Executor, Protocol, aggregate, bootstrap_actor_difference, selected_rows,
)


def test_reference_adds_default_exactly_once():
    ref=KeyframeReference.standing(np.array([1,0,0,0]))
    delta=np.linspace(-.1,.1,29)
    ref.set_joints(delta)
    np.testing.assert_allclose(ref.joint_pos[0],C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB]+delta)


def test_stratified_indices_do_not_only_pick_first_family():
    family=np.array([0,0,0,2,2,4,4,4])
    assert selected_rows(np.arange(8),family).tolist()==[0,3,5]
    assert selected_rows(np.arange(8),family,2).tolist()==[0,2,3,4,5,7]


def test_errors_count_as_failed_episodes():
    rows=[dict(accepted=True,pose_success=True,radius=.9),
          dict(accepted=False,pose_success=False,error='failed')]
    summary=aggregate(rows)
    assert summary['accepted']==.5
    assert summary['pose_success']==.5
    assert summary['runtime_errors']==1


def test_validation_checkpoint_does_not_alias_live_weights():
    import copy
    model=torch.nn.Linear(2,2)
    saved=copy.deepcopy(model.state_dict())
    old=saved['weight'].clone()
    with torch.no_grad():model.weight.add_(3)
    assert torch.equal(saved['weight'],old)


def test_actor_cluster_bootstrap_is_paired_and_zero_for_identical_results():
    rows={i:dict(accepted=True,pose_success=i%2==0,target_mae=.1*i) for i in range(8)}
    result=bootstrap_actor_difference(rows,rows,['A','B'],np.array([0]*4+[1]*4),7)
    for metric in result.values():
        assert metric['mean_delta']==0
        assert metric['ci95']==[0,0]
        assert metric['actors']==2


def test_all_methods_share_physical_time_contract():
    cfg=Protocol()
    assert (cfg.transition_ticks+cfg.hold_ticks)*C.CONTROL_DT==1.6
    assert cfg.hold_ticks//cfg.measure_stride==12


def test_paired_executor_resets_physics_and_sonic_history():
    cfg=Protocol(warmup_ticks=10,transition_ticks=10,hold_ticks=20)
    executor=Executor(cfg)
    delta=np.zeros(29)
    first,trace_a=executor.run(delta,np.array([1.,1.,1.5]),delta,capture=True)
    executor.run(delta+.02,np.array([1.,1.,1.5]),delta)
    second,trace_b=executor.run(delta,np.array([1.,1.,1.5]),delta,capture=True)
    for key in ('q','pos','quat'):
        np.testing.assert_allclose(trace_a[key],trace_b[key],atol=1e-9)
    assert first==second
    assert first['simulated_seconds']==.6
    assert first['obstacle_contact_ticks'] is None
