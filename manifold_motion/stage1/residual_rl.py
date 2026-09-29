"""Bounded residual PPO for the Stage-1 static-pose pilot.

The imitation model and SONIC are frozen. A low-dimensional residual is trained
with repeated-context, leave-one-out advantages. The simulator, success metrics
and train/validation/test split are unchanged from catalog_experiment.
Training never evaluates test episodes. ``evaluate`` is a separate command.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from manifold_motion.stage1.catalog_experiment import (
    Executor, Protocol, aggregate, bootstrap_actor_difference, load_data,
    predict, selected_rows, write_json,
)
from manifold_motion.stage1.manifold_imitation import _mlp


@dataclass(frozen=True)
class ResidualConfig:
    iterations: int = 60
    contexts: int = 8
    candidates: int = 4
    modes: int = 12
    hidden: int = 64
    residual_limit: float = .35
    std: float = .35
    learning_rate: float = 3e-4
    ppo_epochs: int = 4
    target_kl: float = .03
    mean_anchor: float = .02
    validate_every: int = 10
    validation_per_family: int = 3
    allowed_gate_drop: float = .02


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dense_reward(result):
    """Executed target fidelity, not fidelity to our own (possibly biased) command.

    No binary success bonus; the reporting success threshold is unchanged.
    All terms and weights are fixed before training or test evaluation.
    """
    return float(-5.0 * result['target_mae']
                 - 10.0 * max(0., result['radius'] - .95) ** 2
                 - .25 * (min(result['drift_m'], 1.) / .35) ** 2
                 - 4.0 * int(result['fallen']) - 1.0 * int(result['nonfoot_contact']))


def group_advantages(rewards):
    """Independent samples within each context; subtract other samples' rewards.

    Never normalize a single sample from different contexts as if their absolute
    difficulty were the action's credit. No value net or privileged inference input.
    """
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError('expected [contexts, candidates>=2]')
    baseline=(rewards.sum(1,keepdim=True)-rewards)/(rewards.shape[1]-1)
    advantage=rewards-baseline
    return (advantage / advantage.std(unbiased=False).clamp_min(1e-6)).clamp(-5.,5.)


def make_basis(targets, modes):
    centered=targets-targets.mean(0)
    _,_,v=np.linalg.svd(centered,full_matrices=False)
    basis=v[:modes].astype(np.float32)
    # Resolve SVD sign ambiguity for auditability.
    for row in basis:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return basis


class ResidualPolicy(nn.Module):
    def __init__(self, basis, cfg):
        super().__init__()
        self.cfg=cfg
        self.register_buffer('basis',torch.as_tensor(basis,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(51,cfg.hidden),nn.Tanh(),
                               nn.Linear(cfg.hidden,cfg.hidden),nn.Tanh(),
                               nn.Linear(cfg.hidden,cfg.modes))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,base):
        return self.net(torch.cat([x,base],dim=-1))

    def decode(self,raw,base):
        # Every commanded joint is within residual_limit radians of imitation.
        return base + self.cfg.residual_limit * torch.tanh(raw @ self.basis)

    def action(self,x,base):
        return self.decode(self(x,base),base)


def ppo_update(policy,optimizer,x,base,raw_actions,rewards):
    cfg=policy.cfg
    c,k,d=raw_actions.shape
    xx=x[:,None,:].expand(c,k,x.shape[-1]).reshape(c*k,-1)
    bb=base[:,None,:].expand(c,k,29).reshape(c*k,29)
    actions=raw_actions.reshape(c*k,d).detach()
    advantage=group_advantages(rewards).reshape(-1).detach()
    with torch.no_grad():
        old_mean=policy(xx,bb).detach().clone()
        old_logp=torch.distributions.Normal(old_mean,cfg.std).log_prob(actions).sum(-1)
    info=[]
    for epoch in range(cfg.ppo_epochs):
        mean=policy(xx,bb)
        dist=torch.distributions.Normal(mean,cfg.std)
        ratio=(dist.log_prob(actions).sum(-1)-old_logp).exp()
        surrogate=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
        correction=policy.decode(mean,bb)-bb
        anchor=(correction/cfg.residual_limit).square().mean()
        loss=surrogate+cfg.mean_anchor*anchor
        before=copy.deepcopy(policy.state_dict())
        before_opt=copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
        with torch.no_grad():
            new_mean=policy(xx,bb)
            kl=float(((new_mean-old_mean).square().sum(-1)/(2*cfg.std**2)).mean())
        # A real trust-region guard, including the final optimizer step.
        rolled_back=not np.isfinite(kl) or kl>cfg.target_kl
        if rolled_back:
            policy.load_state_dict(before);optimizer.load_state_dict(before_opt)
        info.append(dict(epoch=epoch+1,loss=float(loss.detach()),kl=kl,
                         ratio_std=float(ratio.std().detach()),rolled_back=rolled_back))
        if rolled_back:
            break
    return info


def base_predictions(path,data):
    ckpt=torch.load(path,weights_only=False,map_location='cpu')
    np.testing.assert_allclose(ckpt['input_mean'],data['mean'])
    np.testing.assert_allclose(ckpt['input_std'],data['std'])
    model=_mlp(torch,22,29,ckpt['hidden'])
    model.load_state_dict(ckpt['model']);model.eval()
    return predict(model,data['x'])


def measure(policy,data,base,ids,executor):
    with torch.no_grad():
        poses=policy.action(torch.as_tensor(data['x'][ids]),torch.as_tensor(base[ids])).numpy()
    rows=[]
    for i,pose in zip(ids,poses):
        result,_=executor.run(pose,data['semi'][i],data['y'][i])
        rows.append(dict(row=int(i),**result,dense_reward=dense_reward(result)))
    summary=aggregate(rows)
    summary['dense_reward']=float(np.mean([r['dense_reward'] for r in rows]))
    return summary,rows


def train(args):
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    cfg=ResidualConfig(iterations=args.iterations,contexts=args.contexts,candidates=args.candidates,
                       ppo_epochs=args.ppo_epochs)
    data=load_data(args.catalog)
    base=base_predictions(args.base,data)
    train_ids=np.flatnonzero(data['split']==0)
    val_ids=selected_rows(np.flatnonzero(data['split']==1),data['primitive'],cfg.validation_per_family)
    basis=make_basis(data['y'][train_ids],cfg.modes)
    policy=ResidualPolicy(basis,cfg)
    optimizer=torch.optim.Adam(policy.parameters(),lr=cfg.learning_rate)
    executor=Executor(Protocol(warmup_ticks=10,transition_ticks=10,hold_ticks=20))
    protocol=dict(version=1,algorithm='bounded_residual_ppo_group_loo',config=asdict(cfg),seed=args.seed,
                  base_path=str(args.base),base_sha256=sha256(args.base),catalog_sha256=sha256(args.catalog),
                  train_ids=train_ids.tolist(),validation_ids=val_ids.tolist(),test_used_for_selection=False,
                  executor=asdict(executor.cfg),reward='-5*target_mae -10*relu(radius-.95)^2 -.25*(min(drift,1)/.35)^2 -4*fall -1*nonfoot',
                  selection='validation dense reward > best, target MAE <= base, gate >= base - .02',
                  limitations=['Static pose only; not navigation','Frozen SONIC','Exploratory revision on previously inspected benchmark'],
                  source_sha256=sha256(__file__))
    args.out.mkdir(parents=True,exist_ok=True)
    pp=args.out/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:
        raise ValueError('existing output has a different protocol; choose a new --out')
    write_json(pp,protocol)
    if (args.out/'train_complete.json').exists():
        print('Training already completed; not modifying saved results',flush=True);return
    rng=np.random.default_rng(args.seed)
    log=[];best_state=copy.deepcopy(policy.state_dict());best_iteration=0;start=0
    if (args.out/'latest.pt').exists():
        latest=torch.load(args.out/'latest.pt',weights_only=False)
        policy.load_state_dict(latest['model']);optimizer.load_state_dict(latest['optimizer'])
        rng.bit_generator.state=latest['numpy_rng'];torch.set_rng_state(latest['torch_rng'])
        log=latest['log'];start=latest['iteration'];best_state=latest['best_state']
        best_iteration=latest['best_iteration'];baseline=log[0]['validation'];best=latest['best_reward']
    else:
        baseline,baseline_rows=measure(policy,data,base,val_ids,executor)
        best=baseline['dense_reward']
        log=[dict(iteration=0,validation=baseline)]
        write_json(args.out/'baseline_validation.json',baseline_rows)
    print(f'BASE seed={args.seed} val={baseline["dense_reward"]:.5f} MAE={baseline["target_mae"]:.5f} gate={baseline["accepted"]:.3f}',flush=True)
    begun=time.perf_counter()
    for iteration in range(start+1,cfg.iterations+1):
        ids=rng.choice(train_ids,size=cfg.contexts,replace=False)
        x=torch.as_tensor(data['x'][ids]);b=torch.as_tensor(base[ids])
        with torch.no_grad():
            mean=policy(x,b)
            raw=mean[:,None,:]+cfg.std*torch.randn(cfg.contexts,cfg.candidates,cfg.modes)
            actions=policy.decode(raw,b[:,None,:]).numpy()
        results=[]
        for i,poses in zip(ids,actions):
            for pose in poses:
                r,_=executor.run(pose,data['semi'][i],data['y'][i]);results.append(r)
        reward=torch.tensor([dense_reward(r) for r in results]).reshape(cfg.contexts,cfg.candidates)
        updates=ppo_update(policy,optimizer,x,b,raw,reward)
        entry=dict(iteration=iteration,train=aggregate(results),dense_reward=float(reward.mean()),updates=updates)
        if iteration%cfg.validate_every==0 or iteration==cfg.iterations:
            summary,rows=measure(policy,data,base,val_ids,executor)
            eligible=(summary['accepted']>=baseline['accepted']-cfg.allowed_gate_drop and
                      summary['target_mae']<=baseline['target_mae'])
            entry.update(validation=summary,eligible=eligible)
            if eligible and summary['dense_reward']>best:
                best=summary['dense_reward'];best_iteration=iteration;best_state=copy.deepcopy(policy.state_dict())
            write_json(args.out/f'validation_{iteration:04d}.json',rows)
        log.append(entry)
        torch.save(dict(model=policy.state_dict(),optimizer=optimizer.state_dict(),iteration=iteration,
                        numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),log=log,
                        best_state=best_state,best_iteration=best_iteration,best_reward=best),args.out/'latest.pt')
        write_json(args.out/'history.json',log)
        v=entry.get('validation',{})
        print(f'RESIDUAL seed={args.seed} {iteration}/{cfg.iterations} train={float(reward.mean()):.4f} '
              f'val={v.get("dense_reward")} best_iter={best_iteration} elapsed={time.perf_counter()-begun:.0f}s',flush=True)
    torch.save(dict(model=best_state,basis=basis,config=asdict(cfg),best_iteration=best_iteration,
                    seed=args.seed,base_sha256=sha256(args.base),base_path=str(args.base),
                    algorithm=protocol['algorithm'],input_mean=data['mean'],input_std=data['std']),args.out/'residual.pt')
    write_json(args.out/'train_complete.json',dict(seed=args.seed,best_iteration=best_iteration,
               baseline_validation=baseline,best_validation_reward=best,training_episodes=cfg.iterations*cfg.contexts*cfg.candidates,
               test_evaluated=False))


def evaluate(args):
    torch.set_num_threads(2)
    protocol=json.loads((args.out/'protocol.json').read_text())
    if sha256(args.catalog)!=protocol['catalog_sha256'] or sha256(args.base)!=protocol['base_sha256']:
        raise ValueError('catalog or imitation checkpoint differs from training')
    ckpt=torch.load(args.out/'residual.pt',weights_only=False,map_location='cpu')
    cfg=ResidualConfig(**ckpt['config']);policy=ResidualPolicy(ckpt['basis'],cfg)
    policy.load_state_dict(ckpt['model']);policy.eval()
    data=load_data(args.catalog);base=base_predictions(args.base,data)
    ids=np.flatnonzero(data['split']==2)
    display=set(selected_rows(ids,data['primitive']).tolist())
    with torch.no_grad():
        proposed=policy.action(torch.as_tensor(data['x'][ids]),torch.as_tensor(base[ids])).numpy()
    executor=Executor(Protocol(**protocol['executor']))
    fingerprint=dict(checkpoint_sha256=sha256(args.out/'residual.pt'),base_sha256=sha256(args.base),
                     test_ids=ids.tolist(),catalog_sha256=sha256(args.catalog),metrics='unchanged from v3')
    fp=args.out/'evaluation_protocol.json'
    if fp.exists() and json.loads(fp.read_text())!=fingerprint:
        raise ValueError('evaluation files belong to another model')
    write_json(fp,fingerprint)
    raw={};summaries={}
    for name,poses in [('il_only',base[ids]),('il_residual_rl',proposed)]:
        path=args.out/(name+'_test.jsonl')
        rows={} if not path.exists() else {r['row']:r for r in map(json.loads,path.read_text().splitlines())}
        with path.open('a') as stream:
            for j,(i,pose) in enumerate(zip(ids,poses)):
                if int(i) in rows:continue
                r,t=executor.run(pose,data['semi'][i],data['y'][i],capture=int(i) in display)
                record=dict(row=int(i),family=str(data['primitive_names'][data['primitive'][i]]),**r,dense_reward=dense_reward(r))
                if int(i) in display:
                    td=args.out/'traces';td.mkdir(exist_ok=True)
                    np.savez_compressed(td/f'{name}_{i:06d}.npz',**t)
                rows[int(i)]=record;stream.write(json.dumps(record)+'\n');stream.flush()
                if (j+1)%100==0 or j+1==len(ids):
                    print(f'TEST seed={ckpt["seed"]} {name} {j+1}/{len(ids)}',flush=True)
        raw[name]=rows
        summary=aggregate(list(rows.values()))
        summary['offline_mae']=float(np.abs(poses-data['y'][ids]).mean())
        summary['dense_reward']=float(np.mean([r['dense_reward'] for r in rows.values()]))
        summary['by_family']={f:aggregate([r for r in rows.values() if r['family']==f]) for f in sorted(set(r['family'] for r in rows.values()))}
        summaries[name]=summary
    meta=json.loads(args.metadata.read_text())
    actors=[str(c['actor_uid']) for c in meta['clips']]
    result=dict(seed=ckpt['seed'],best_iteration=ckpt['best_iteration'],methods=summaries,
                paired_actor_bootstrap=bootstrap_actor_difference(raw['il_residual_rl'],raw['il_only'],actors,data['clip_index'],ckpt['seed']))
    write_json(args.out/'test_summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','evaluate'])
    p.add_argument('--catalog',type=Path,default=Path('data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz'))
    p.add_argument('--metadata',type=Path,default=Path('reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows_metadata.json'))
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260928)
    p.add_argument('--iterations',type=int,default=60)
    p.add_argument('--contexts',type=int,default=8)
    p.add_argument('--candidates',type=int,default=4)
    p.add_argument('--ppo-epochs',type=int,default=4)
    args=p.parse_args()
    if min(args.iterations,args.contexts,args.ppo_epochs)<1 or args.candidates<2:
        p.error('positive iteration/context/epoch counts and at least two candidates required')
    (train if args.command=='train' else evaluate)(args)


if __name__=='__main__':main()
