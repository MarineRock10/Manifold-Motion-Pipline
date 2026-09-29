"""Mixed-horizon bounded residual RL for the Stage-1 safety selection study.

Each sampled residual is evaluated in both the short response and a 3-second hold.
The long rollout is part of the training reward and the checkpoint eligibility gate,
so a short-horizon fidelity gain cannot silently trade away long-horizon stability.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from manifold_motion.core import constants as C

from manifold_motion.stage1.catalog_experiment import Executor,Protocol,load_data,selected_rows,write_json,aggregate
from manifold_motion.stage1.residual_rl import (
    ResidualConfig,ResidualPolicy,base_predictions,dense_reward,measure,ppo_update,sha256,
    make_basis,
)


VARIANTS=('full','no_imitation','no_geometry','no_long_reward','continued_imitation')


class StructuredResidualPolicy(ResidualPolicy):
    """Smaller corrections for balance-critical joints; identical in all variants."""
    def __init__(self,basis,cfg):
        super().__init__(basis,cfg)
        limits=np.array([.04 if i<15 else .16 for i in range(29)],dtype=np.float32)
        self.register_buffer('joint_limits',torch.from_numpy(limits[C.MUJOCO_TO_ISAACLAB]))

    def decode(self,raw,base):
        return base+self.joint_limits*torch.tanh(raw@self.basis)


def conditions(data,base_path,variant):
    data=dict(data)
    if variant=='no_geometry':
        data['x']=np.zeros_like(data['x'])
    base=base_predictions(base_path,data)
    if variant=='no_imitation':
        base=np.zeros_like(base)  # no pretrained output enters the residual policy
    return data,base


def eligible(short,long,baseline_short,baseline_long):
    # Same eligibility contract for every variant; iteration zero is always allowed.
    return (short['accepted']>=baseline_short['accepted'] and
            long['accepted']>=baseline_long['accepted'] and
            short['fallen']<=baseline_short['fallen'] and long['fallen']<=baseline_long['fallen'] and
            short['target_mae']<=baseline_short['target_mae']+.005 and
            long['target_mae']<=baseline_long['target_mae']+.005)


def run(args):
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    cfg=ResidualConfig(iterations=args.iterations,contexts=args.contexts,candidates=args.candidates,
                       ppo_epochs=args.ppo_epochs,residual_limit=.16,allowed_gate_drop=0.)
    data,base=conditions(load_data(args.catalog),args.base,args.variant)
    train_ids=np.flatnonzero(data['split']==0)
    val_short=selected_rows(np.flatnonzero(data['split']==1),data['primitive'],3)
    val_long=selected_rows(np.flatnonzero(data['split']==1),data['primitive'],1)
    basis=make_basis(data['y'][train_ids],cfg.modes)
    policy=StructuredResidualPolicy(basis,cfg);optimizer=torch.optim.Adam(policy.parameters(),lr=cfg.learning_rate)
    short=Executor(Protocol(warmup_ticks=10,transition_ticks=10,hold_ticks=20))
    long=Executor(Protocol(warmup_ticks=20,transition_ticks=20,hold_ticks=150))
    args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=1,algorithm='mixed_horizon_bounded_residual_ppo',mixed_horizon=True,variant=args.variant,
                  config=asdict(cfg),seed=args.seed,base_path=str(args.base),base_sha256=sha256(args.base),
                  catalog_sha256=sha256(args.catalog),train_ids=train_ids.tolist(),
                  validation_short_ids=val_short.tolist(),validation_long_ids=val_long.tolist(),
                  test_used_for_selection=False,short_executor=asdict(short.cfg),long_executor=asdict(long.cfg),
                  reward='short_dense + 0.7*long_dense; no_long_reward removes only long reward; continued_imitation ignores physics rewards',
                  selection='both horizons: gate >= baseline; fall <= baseline; each target MAE <= baseline + .005',
                  budget='20 iterations x 4 contexts x 2 samples; 2 optimizer updates/iteration. Continued IL matches optimizer/sample counts, not simulator calls.',
                  source_sha256=sha256(__file__),source_dependencies={
                      'executor':sha256(Path(__file__).with_name('catalog_experiment.py')),
                      'ppo':sha256(Path(__file__).with_name('residual_rl.py'))})
    pp=args.out/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:
        raise ValueError('output has another protocol; choose a new --out')
    write_json(pp,protocol)
    if (args.out/'train_complete.json').exists():
        print('Training already completed; use existing artifacts',flush=True);return
    baseline_s,rows_s=measure(policy,data,base,val_short,short)
    baseline_l,rows_l=measure(policy,data,base,val_long,long)
    best_score=baseline_s['dense_reward']+.7*baseline_l['dense_reward'];best_iter=0;best_state=copy.deepcopy(policy.state_dict())
    write_json(args.out/'baseline_short.json',rows_s);write_json(args.out/'baseline_long.json',rows_l)
    log=[dict(iteration=0,validation_short=baseline_s,validation_long=baseline_l,score=best_score)]
    rng=np.random.default_rng(args.seed);begun=time.perf_counter();start=0
    if (args.out/'latest.pt').exists():
        ck=torch.load(args.out/'latest.pt',weights_only=False,map_location='cpu')
        policy.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        start=ck['iteration'];log=ck['log'];best_state=ck['best_state'];best_iter=ck['best_iteration'];best_score=ck['best_reward']
        rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'])
    for it in range(start+1,cfg.iterations+1):
        ids=rng.choice(train_ids,size=cfg.contexts,replace=False)
        x=torch.as_tensor(data['x'][ids]);b=torch.as_tensor(base[ids])
        with torch.no_grad():
            mean=policy(x,b);raw=mean[:,None,:]+cfg.std*torch.randn(cfg.contexts,cfg.candidates,cfg.modes)
            actions=policy.decode(raw,b[:,None,:]).numpy()
        short_rows=[];long_rows=[]
        if args.variant=='continued_imitation':
            updates=[];reward=torch.zeros(cfg.contexts,cfg.candidates)
            for ep in range(cfg.ppo_epochs):
                loss=torch.nn.functional.smooth_l1_loss(policy.action(x,b),torch.as_tensor(data['y'][ids]))
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
                updates.append(dict(epoch=ep+1,supervised_loss=float(loss.detach())))
        else:
            for i,poses in zip(ids,actions):
                for pose in poses:
                    rs,_=short.run(pose,data['semi'][i],data['y'][i]);rl,_=long.run(pose,data['semi'][i],data['y'][i])
                    short_rows.append(rs);long_rows.append(rl)
            weight=0. if args.variant=='no_long_reward' else .7
            reward=torch.tensor([[dense_reward(short_rows[i*cfg.candidates+j])+weight*dense_reward(long_rows[i*cfg.candidates+j])
                                 for j in range(cfg.candidates)] for i in range(cfg.contexts)])
            updates=ppo_update(policy,optimizer,x,b,raw,reward)
        entry=dict(iteration=it,train_short=aggregate(short_rows),train_long=aggregate(long_rows),
                   train_score=float(reward.mean()),updates=updates)
        if it%cfg.validate_every==0 or it==cfg.iterations:
            vs,rs=measure(policy,data,base,val_short,short);vl,rl=measure(policy,data,base,val_long,long)
            is_eligible=eligible(vs,vl,baseline_s,baseline_l)
            score=vs['dense_reward']+.7*vl['dense_reward']
            entry.update(validation_short=vs,validation_long=vl,score=score,eligible=is_eligible)
            if is_eligible and score>best_score:
                best_score=score;best_iter=it;best_state=copy.deepcopy(policy.state_dict())
            write_json(args.out/f'validation_{it:04d}_short.json',rs);write_json(args.out/f'validation_{it:04d}_long.json',rl)
        log.append(entry);write_json(args.out/'history.json',log)
        torch.save(dict(model=policy.state_dict(),optimizer=optimizer.state_dict(),iteration=it,
                        numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),log=log,
                        best_state=best_state,best_iteration=best_iter,best_reward=best_score),args.out/'latest.pt')
        print(f'MIXED {args.variant} seed={args.seed} {it}/{cfg.iterations} train={float(reward.mean()):.4f} '
              f'val={entry.get("score")} eligible={entry.get("eligible")} best={best_iter} elapsed={time.perf_counter()-begun:.0f}s',flush=True)
    torch.save(dict(model=best_state,basis=basis,config=asdict(cfg),best_iteration=best_iter,seed=args.seed,
                    base_sha256=sha256(args.base),base_path=str(args.base),algorithm=protocol['algorithm'],
                    variant=args.variant,input_mean=data['mean'],input_std=data['std']),args.out/'residual.pt')
    write_json(args.out/'train_complete.json',dict(seed=args.seed,best_iteration=best_iter,
               baseline_short=baseline_s,baseline_long=baseline_l,best_score=best_score,
               training_candidates=cfg.iterations*cfg.contexts*cfg.candidates,
               training_physics_episodes=0 if args.variant=='continued_imitation' else 2*cfg.iterations*cfg.contexts*cfg.candidates,
               test_evaluated=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,default=Path('data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz'))
    p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--variant',choices=VARIANTS,default='full')
    p.add_argument('--seed',type=int,default=20260928);p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--contexts',type=int,default=4);p.add_argument('--candidates',type=int,default=2)
    p.add_argument('--ppo-epochs',type=int,default=2);args=p.parse_args()
    if min(args.iterations,args.contexts,args.ppo_epochs)<1 or args.candidates<2:p.error('invalid training budget')
    run(args)


if __name__=='__main__':main()
