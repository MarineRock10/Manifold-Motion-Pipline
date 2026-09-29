"""Stage-1 A/B: validation-selected imitation and on-policy SONIC-in-loop PPO.

This is a STATIC POSE experiment, not navigation or dynamic primitive execution.
The physical ruler is the middle-frame corridor semi-axis, centred on the measured
pelvis and aligned with its heading (pitch/roll remain observable). Translation is
not a Stage-1 constraint; drift, falls, contacts and target fidelity are reported
separately. No source pose/root is teleported into the executing simulator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from manifold_motion.core import constants as C
from manifold_motion.core.body_envelope import body_points
from manifold_motion.simulation.keyframe_env import KeyframeEnv
from manifold_motion.stage1.manifold_imitation import _mlp, manifold_features

DEFAULT = C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB].astype(np.float32)


@dataclass(frozen=True)
class Protocol:
    warmup_ticks: int = 20
    transition_ticks: int = 20
    hold_ticks: int = 60
    measure_stride: int = 5
    min_pelvis_z: float = 0.30
    min_upright: float = 0.45
    max_xy_drift: float = 0.35
    target_mae_limit: float = 0.30
    epochs: int = 80
    hidden: int = 256
    rl_iterations: int = 20
    rl_batch: int = 24
    validate_every: int = 5
    validation_per_family: int = 2
    learning_rate: float = 5e-5
    exploration_std: float = 0.08


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def selected_rows(ids, family, per_family=1):
    """Evenly spaced source indices, independent of model predictions or success."""
    rows = []
    for f in sorted(set(family[ids].tolist())):
        group = ids[family[ids] == f]
        indices = np.linspace(0, len(group) - 1, min(per_family, len(group))).astype(int)
        rows.extend(group[indices].tolist())
    return np.asarray(rows, dtype=int)


def load_data(path: Path):
    with np.load(path, allow_pickle=False) as a:
        arrays = {k: a[k] for k in ('corridor', 'target_exec', 'primitive', 'primitive_names', 'split', 'clip_index')}
    x = manifold_features(arrays['corridor'])
    train = arrays['split'] == 0
    mean, std = x[train].mean(0), np.maximum(x[train].std(0), 1e-5)
    frame = arrays['target_exec'].shape[1] // 2
    arrays.update(x=(x - mean) / std, mean=mean, std=std, frame=frame,
                  y=arrays['target_exec'][:, frame, :29] - DEFAULT,
                  semi=arrays['corridor'][:, frame, 3:6])
    for split in range(3):
        if not np.any(arrays['split'] == split):
            raise ValueError('all three actor-disjoint splits are required')
    return arrays


def aggregate(rows):
    fields = ('accepted', 'pose_success', 'fallen', 'nonfoot_contact', 'radius', 'tracking_mae',
              'target_mae', 'request_mae', 'drift_m', 'reward', 'clipped_fraction', 'inference_ms')
    valid = [r for r in rows if 'error' not in r]
    result = {'rows': len(rows), 'runtime_errors': len(rows) - len(valid)}
    for field in fields:
        values = [r[field] for r in valid if field in r]
        result[field] = float(np.mean(values)) if values else None
    # Errors count as failures, never disappear from success denominators.
    for key in ('accepted', 'pose_success'):
        result[key] = sum(bool(r.get(key, False)) for r in rows) / max(1, len(rows))
    return result


class Executor:
    """One reusable model/controller; every episode resets both before warm-up."""
    def __init__(self, cfg: Protocol):
        self.cfg = cfg
        self.key = KeyframeEnv()
        model = self.key.env.model
        joint_ids = model.actuator_trnid[self.key.env.body_act, 0]
        limits = model.jnt_range[joint_ids][C.MUJOCO_TO_ISAACLAB]
        self.low = limits[:, 0] - DEFAULT + 0.005
        self.high = limits[:, 1] - DEFAULT - 0.005
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s + '_ankle_roll_link') for s in ('left','right')]
        self.foot_bodies = set(feet)

    def run(self, delta, semi, target, capture=False):
        cfg, key = self.cfg, self.key
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (29,) or not np.isfinite(delta).all() or np.min(semi) <= 0:
            raise ValueError('invalid target pose or ellipsoid')
        applied = np.clip(delta, self.low, self.high)
        key.reset()
        for _ in range(cfg.warmup_ticks):
            key.step()
        origin = key.state()['base_pos'].copy()
        radii, tracks, targets, contacts, drift = [], [], [], [], []
        fallen = False
        trace = {'q': [], 'pos': [], 'quat': [], 't': []}
        n = cfg.transition_ticks + cfg.hold_ticks
        for tick in range(n):
            # Identical continuous reference entry for every method, including the oracle.
            blend = min(1.0, (tick + 1) / max(1, cfg.transition_ticks))
            key.set_joints(applied * blend)  # delta contract: reference adds DEFAULT once
            key.step()
            state = key.state()
            rot = C.quat_to_matrix(state['base_quat'])
            fallen |= bool(state['base_pos'][2] < cfg.min_pelvis_z or rot[2,2] < cfg.min_upright)
            drift.append(float(np.linalg.norm(state['base_pos'][:2] - origin[:2])))
            nonfoot = False
            for contact in key.env.data.contact:
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if self.floor not in (g1, g2):
                    continue
                other = g2 if g1 == self.floor else g1
                if int(key.env.model.geom_bodyid[other]) not in self.foot_bodies and contact.dist < 0:
                    nonfoot = True
            contacts.append(nonfoot)
            if tick >= cfg.transition_ticks and (tick + 1) % cfg.measure_stride == 0:
                points = body_points(key.env.model, key.env.data) - state['base_pos']
                yaw = np.arctan2(rot[1,0], rot[0,0])
                heading = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0,0,1]])
                radii.append(float(np.linalg.norm((points @ heading) / semi, axis=1).max()))
                q = state['q_hw'][C.MUJOCO_TO_ISAACLAB]
                tracks.append(float(np.abs(q - DEFAULT - applied).mean()))
                targets.append(float(np.abs(q - DEFAULT - target).mean()))
            if capture:
                trace['q'].append(state['q_hw'][C.MUJOCO_TO_ISAACLAB].copy())
                trace['pos'].append(state['base_pos'].copy())
                trace['quat'].append(state['base_quat'].copy())
                trace['t'].append((tick+1)*C.CONTROL_DT)
        radius = max(radii)
        drift_max = max(drift)
        accepted = not fallen and not any(contacts) and drift_max <= cfg.max_xy_drift and radius <= 1
        target_mae, tracking = float(np.mean(targets)), float(np.mean(tracks))
        reward = (1.0 if accepted else 0.0) - 8 * max(0, radius-1) - 3 * target_mae - tracking
        reward -= 8 * int(fallen) + 2 * int(any(contacts)) + 2 * max(0, drift_max-cfg.max_xy_drift)
        result = dict(accepted=accepted, pose_success=accepted and target_mae <= cfg.target_mae_limit,
                      fallen=fallen, nonfoot_contact=any(contacts), radius=radius,
                      tracking_mae=tracking, target_mae=target_mae, request_mae=float(np.abs(applied-target).mean()),
                      drift_m=drift_max, reward=float(reward), clipped_fraction=float(np.mean(applied != delta)),
                      simulated_seconds=n*C.CONTROL_DT, obstacle_contact_ticks=None)
        if capture:
            trace = {k:np.asarray(v) for k,v in trace.items()}
            trace.update(requested_delta=applied, semi=np.asarray(semi), target_delta=np.asarray(target))
        return result, trace


def predict(model, x):
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32)).numpy()


def train_imitation(data, cfg, seed, folder):
    torch.manual_seed(seed)
    model = _mlp(torch, 22, 29, cfg.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    train, val = data['split']==0, data['split']==1
    x, y = torch.as_tensor(data['x'][train]), torch.as_tensor(data['y'][train])
    rng = np.random.default_rng(seed)
    best, best_epoch, snapshot = float('inf'), 0, None
    history = []
    for epoch in range(1, cfg.epochs+1):
        model.train()
        for batch in np.array_split(rng.permutation(len(x)), int(np.ceil(len(x)/128))):
            loss = nn.functional.smooth_l1_loss(model(x[batch]), y[batch])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        model.eval()
        error = float(np.abs(predict(model, data['x'][val])-data['y'][val]).mean())
        history.append({'epoch':epoch, 'validation_mae':error})
        if error < best:
            best, best_epoch, snapshot = error, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(snapshot)
    torch.save({'model':snapshot,'best_epoch':best_epoch,'hidden':cfg.hidden,
                'input_mean':data['mean'],'input_std':data['std']}, folder/'imitation.pt')
    write_json(folder/'imitation_history.json', history)
    print(f'IL seed={seed} best epoch={best_epoch} validation MAE={best:.4f}',flush=True)
    return model


def validation(model, data, ids, executor):
    predictions = predict(model, data['x'][ids])
    rows = [executor.run(p, data['semi'][r], data['y'][r])[0] for p,r in zip(predictions,ids)]
    return aggregate(rows)


def train_rl(initial, data, cfg, seed, folder, label, executor):
    torch.manual_seed(seed+37)
    rng = np.random.default_rng(seed+37)
    model = copy.deepcopy(initial)
    # One static action per reset episode: PPO reduces to a contextual bandit.
    # No mean-probe transition is mixed with a sampled action's log-probability.
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    train_ids = np.flatnonzero(data['split']==0)
    val_ids = selected_rows(np.flatnonzero(data['split']==1), data['primitive'], cfg.validation_per_family)
    baseline = validation(model, data, val_ids, executor)
    best = baseline['reward']
    best_iteration, snapshot = 0, copy.deepcopy(model.state_dict())
    log = [{'iteration':0, 'validation':baseline}]
    started=time.perf_counter()
    for iteration in range(1,cfg.rl_iterations+1):
        ids = rng.choice(train_ids, size=cfg.rl_batch, replace=False)
        x = torch.as_tensor(data['x'][ids])
        with torch.no_grad():
            dist = torch.distributions.Normal(model(x),cfg.exploration_std)
            actions = dist.sample(); old_logp=dist.log_prob(actions).sum(-1)
        results=[executor.run(a, data['semi'][r], data['y'][r])[0] for a,r in zip(actions.numpy(),ids)]
        rewards=torch.tensor([r['reward'] for r in results],dtype=torch.float32)
        advantages=(rewards-rewards.mean())/(rewards.std()+1e-6)
        dist=torch.distributions.Normal(model(x),cfg.exploration_std)
        ratio=(dist.log_prob(actions).sum(-1)-old_logp).exp()
        loss=-torch.minimum(ratio*advantages,ratio.clamp(.8,1.2)*advantages).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),.5); opt.step()
        entry={'iteration':iteration,'train':aggregate(results),'loss':float(loss.detach())}
        if iteration%cfg.validate_every==0 or iteration==cfg.rl_iterations:
            entry['validation']=validation(model,data,val_ids,executor)
            if entry['validation']['reward']>best:
                best=entry['validation']['reward'];best_iteration=iteration
                snapshot=copy.deepcopy(model.state_dict())
        log.append(entry)
        write_json(folder/(label+'_history.json'),log)
        print(f'RL {label} seed={seed} {iteration}/{cfg.rl_iterations} reward={float(rewards.mean()):.3f} '
              f'best_val={best:.3f} selected={best_iteration} elapsed={time.perf_counter()-started:.0f}s',flush=True)
    model.load_state_dict(snapshot)
    torch.save({'model':snapshot,'hidden':cfg.hidden,'best_iteration':best_iteration,'validation_reward':best,
                'input_mean':data['mean'],'input_std':data['std']},folder/(label+'.pt'))
    return model


def bootstrap_actor_difference(rows_full, rows_base, actor_by_clip, clips, seed):
    """Paired actor-cluster bootstrap: overlapping windows are not independent."""
    shared = sorted(set(rows_full)&set(rows_base))
    actors = np.array([actor_by_clip[int(clips[i])] for i in shared])
    unique=np.unique(actors); rng=np.random.default_rng(seed)
    out={}
    for metric in ('pose_success','accepted','target_mae'):
        pairs=[(i,a) for i,a in zip(shared,actors) if metric in rows_full[i] and metric in rows_base[i]]
        a=np.array([v for _,v in pairs]); delta=np.array([float(rows_full[i][metric])-float(rows_base[i][metric]) for i,_ in pairs])
        sums=np.array([delta[a==u].sum() for u in unique]); counts=np.array([(a==u).sum() for u in unique])
        draws=rng.integers(0,len(unique),size=(1000,len(unique)))
        denom=counts[draws].sum(1); estimates=sums[draws].sum(1)/np.maximum(denom,1)
        out[metric]={'mean_delta':float(delta.mean()),'ci95':np.quantile(estimates,[.025,.975]).tolist(),'actors':len(unique)}
    return out


def run(args):
    torch.set_num_threads(2)
    overrides = dict(warmup_ticks=10, transition_ticks=10, hold_ticks=20) if args.profile=='short_response' else {}
    cfg=Protocol(rl_iterations=args.rl_iterations,rl_batch=args.rl_batch, **overrides)
    data=load_data(args.catalog)
    args.out.mkdir(parents=True,exist_ok=True)
    protocol={'version':1, 'config':asdict(cfg), 'seeds':args.seeds,'profile':args.profile,
              'catalog_sha256':hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
              'scope':'static pelvis-centred heading-aligned pose-shape gate, not corridor navigation',
              'frozen_sonic':True,'selection':'validation reward only; checkpoint zero allowed',
              'test_ids':np.flatnonzero(data['split']==2).tolist(),
              'limitations':['source labels do not certify static holdability','no obstacles or SLAM in this Stage-1 test',
                             'family_mean uses oracle family labels','no_imitation removes pretraining, not the common target-fidelity reward']}
    if (args.out/'protocol.json').exists():
        if json.loads((args.out/'protocol.json').read_text())!=protocol:
            raise ValueError('existing experiment has a different protocol; choose a new --out')
    write_json(args.out/'protocol.json',protocol)
    executor=Executor(cfg)
    if args.smoke:
        ids=selected_rows(np.flatnonzero(data['split']==2),data['primitive'])
        started=time.perf_counter()
        rows=[dict(row=int(i),**executor.run(data['y'][i],data['semi'][i],data['y'][i])[0]) for i in ids[:3]]
        print(json.dumps({'elapsed':time.perf_counter()-started,'rows':rows},indent=2));return
    test_ids=np.flatnonzero(data['split']==2)
    display_ids=selected_rows(test_ids,data['primitive'])
    train=data['split']==0
    global_mean=data['y'][train].mean(0)
    family_mean={int(f):data['y'][train&(data['primitive']==f)].mean(0) for f in np.unique(data['primitive'][train])}
    meta=json.loads(args.metadata.read_text())
    actor_by_clip=[str(c['actor_uid']) for c in meta['clips']]
    results={}
    for seed in args.seeds:
        folder=args.out/f'seed_{seed}';folder.mkdir(exist_ok=True)
        def load_or_fit(name, fit):
            path=folder/(name+'.pt')
            if not path.exists(): return fit()
            model=_mlp(torch,22,29,cfg.hidden)
            model.load_state_dict(torch.load(path,weights_only=False,map_location='cpu')['model']);model.eval();return model
        imitation=load_or_fit('imitation',lambda:train_imitation(data,cfg,seed,folder))
        full=load_or_fit('full',lambda:train_rl(imitation,data,cfg,seed,folder,'full',executor))
        torch.manual_seed(seed)
        scratch=_mlp(torch,22,29,cfg.hidden)
        nn.init.zeros_(scratch[-1].weight);nn.init.zeros_(scratch[-1].bias)
        no_il=load_or_fit('no_imitation',lambda:train_rl(scratch,data,cfg,seed,folder,'no_imitation',executor))
        # No geometry: retrain the same architecture on constant normalized M input.
        ng_dir=folder/'no_geometry';ng_dir.mkdir(exist_ok=True)
        if not (ng_dir/'imitation.pt').exists():
            constant=dict(data);constant['x']=np.zeros_like(data['x'])
            train_imitation(constant,cfg,seed,ng_dir)
        ng=_mlp(torch,22,29,cfg.hidden)
        ng.load_state_dict(torch.load(ng_dir/'imitation.pt',weights_only=False)['model']);ng.eval()
        models={'full':full,'no_finetune':imitation,'no_imitation':no_il,'no_geometry':ng}
        predictions={}
        timings={}
        for name,model in models.items():
            x=data['x'][test_ids] if name!='no_geometry' else np.zeros_like(data['x'][test_ids])
            predict(model,x[:1]);started=time.perf_counter();predictions[name]=predict(model,x)
            timings[name]=1000*(time.perf_counter()-started)/len(test_ids)
        predictions['family_mean']=np.stack([family_mean[int(data['primitive'][r])] for r in test_ids])
        predictions['recorded_target']=data['y'][test_ids]
        by_method={};raw_by_method={}
        for name,poses in predictions.items():
            path=folder/(name+'_test.jsonl')
            saved={} if not path.exists() else {int(r['row']):r for r in [json.loads(s) for s in path.read_text().splitlines()]}
            with path.open('a') as stream:
                for j,(row,pose) in enumerate(zip(test_ids,poses)):
                    if int(row) in saved:continue
                    record={'row':int(row),'family':str(data['primitive_names'][data['primitive'][row]]),'inference_ms':timings.get(name,0.)}
                    try:
                        value,trace=executor.run(pose,data['semi'][row],data['y'][row],capture=row in display_ids)
                        record.update(value)
                        if row in display_ids:
                            td=folder/'traces';td.mkdir(exist_ok=True)
                            np.savez_compressed(td/f'{name}_{row:06d}.npz',**trace)
                    except Exception as e:
                        record.update(accepted=False,pose_success=False,error=f'{type(e).__name__}: {e}')
                    saved[int(row)]=record
                    stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                    if (j+1)%25==0 or j+1==len(test_ids):
                        print(f'TEST seed={seed} {name} {j+1}/{len(test_ids)} accepted={aggregate(list(saved.values()))["accepted"]:.3f}',flush=True)
            rows=list(saved.values());summary=aggregate(rows)
            summary['offline_mae']=float(np.abs(poses-data['y'][test_ids]).mean())
            summary['by_family']={f:aggregate([r for r in rows if r['family']==f]) for f in sorted(set(r['family'] for r in rows))}
            by_method[name]=summary;raw_by_method[name]=saved
        comparisons={name:bootstrap_actor_difference(raw_by_method['full'],raw_by_method[name],actor_by_clip,data['clip_index'],seed)
                     for name in by_method if name not in ('full','recorded_target')}
        results[str(seed)]={'methods':by_method,'paired_actor_bootstrap':comparisons}
        write_json(folder/'summary.json',results[str(seed)])
        write_json(args.out/'summary.json',results)
    write_json(args.out/'complete.json',{'seeds':args.seeds,'test_rows_per_method':len(test_ids),
                                       'methods':list(predictions),'display_rows':display_ids.tolist()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,default=Path('data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz'))
    p.add_argument('--metadata',type=Path,default=Path('reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows_metadata.json'))
    p.add_argument('--out',type=Path,default=Path('reports/manifold_motion/stage1_ab_v2'))
    p.add_argument('--seeds',type=int,nargs='+',default=[20260928,20260929,20260930])
    p.add_argument('--rl-iterations',type=int,default=20)
    p.add_argument('--rl-batch',type=int,default=24)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--profile',choices=('static_hold','short_response'),default='static_hold')
    args=p.parse_args()
    if args.rl_iterations<1 or args.rl_batch<2:p.error('iterations >=1 and batch >=2 required')
    run(args)


if __name__=='__main__':main()
