"""Locked evaluation for the mixed-horizon Stage-1 suite.

All 727 development-test rows use the unchanged v3 short-response executor.
Long evaluation uses two source-index-stratified windows per represented family.
Exact zero-residual policies reuse deterministic IL results with explicit provenance.
No test result is used to change a checkpoint or select an action.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from manifold_motion.stage1.catalog_experiment import (
    Executor,Protocol,aggregate,bootstrap_actor_difference,load_data,selected_rows,write_json,
)
from manifold_motion.stage1.residual_rl import ResidualConfig,base_predictions,sha256
from manifold_motion.stage1.residual_long_rl import VARIANTS,StructuredResidualPolicy,conditions

METHODS=('il_only',)+VARIANTS


def run(args):
    torch.set_num_threads(2)
    data=load_data(args.catalog);ids=np.flatnonzero(data['split']==2)
    long_ids=selected_rows(ids,data['primitive'],2)
    display=set(selected_rows(ids,data['primitive']).tolist())
    original=Path(f'reports/manifold_motion/stage1_ab_response_v3/replicate_{args.seed}/seed_{args.seed}')
    seed_dir=args.root/f'seed_{args.seed}';out=seed_dir/'evaluation';out.mkdir(exist_ok=True)
    predictions={'il_only':base_predictions(original/'imitation.pt',data)}
    fingerprints={};ckpts={};base_values={}
    trained_variants=tuple(m for m in METHODS if m!='il_only')
    for variant in trained_variants:
        path=seed_dir/variant/'residual.pt'
        if not (seed_dir/variant/'train_complete.json').exists():raise RuntimeError(f'incomplete training: {variant}')
        ck=torch.load(path,weights_only=False,map_location='cpu');ckpts[variant]=ck
        cfg=ResidualConfig(**ck['config']);policy=StructuredResidualPolicy(ck['basis'],cfg)
        policy.load_state_dict(ck['model']);policy.eval()
        if sha256(ck['base_path'])!=ck['base_sha256']:raise ValueError('base checkpoint changed')
        dd,base=conditions(data,Path(ck['base_path']),variant);base_values[variant]=base
        with torch.no_grad():predictions[variant]=policy.action(torch.as_tensor(dd['x']),torch.as_tensor(base)).numpy()
        fingerprints[variant]=sha256(path)
    configs={'short':Protocol(warmup_ticks=10,transition_ticks=10,hold_ticks=20),
             'long':Protocol(warmup_ticks=20,transition_ticks=20,hold_ticks=150)}
    protocol=dict(seed=args.seed,catalog_sha256=sha256(args.catalog),checkpoint_sha256=fingerprints,
                  short_ids=ids.tolist(),long_ids=long_ids.tolist(),selection='none; frozen validation-selected weights',
                  long_sampling='2 evenly-spaced source indices per represented test family, not success-based',
                  configs={k:asdict(v) for k,v in configs.items()},source_sha256=sha256(__file__))
    pp=out/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('evaluation protocol changed')
    write_json(pp,protocol)
    td=out/'traces';td.mkdir(exist_ok=True)
    results={};raw={};new_episodes=0;reused_episodes=0
    for horizon,eids in [('short',ids),('long',long_ids)]:
        executor=Executor(configs[horizon]);results[horizon]={};raw[horizon]={}
        # Known IL reference records were obtained under precisely these executors.
        legacy={}
        if horizon=='short':
            for label,file in [('il_only','no_finetune_test.jsonl'),('no_geometry','no_geometry_test.jsonl')]:
                legacy[label]={r['row']:r for r in map(json.loads,(original/file).read_text().splitlines())}
        else:
            previous=Path('docs/experiments/stage1_residual_rl_v4/long_hold.json')
            if previous.exists():
                legacy['il_only']={r['row']:r for r in json.loads(previous.read_text())['rows']
                                   if str(r['seed'])==str(args.seed) and r['method']=='il_only'}
        # The cache only reuses exact commanded poses for the same target and geometry.
        cache={}
        for method in METHODS:
            file=out/f'{method}_{horizon}.jsonl'
            saved={} if not file.exists() else {r['row']:r for r in map(json.loads,file.read_text().splitlines())}
            with file.open('a') as stream:
                for j,i in enumerate(eids):
                    i=int(i);pose=predictions[method][i];key=(i,pose.tobytes())
                    if i in saved:
                        cache[key]=saved[i];continue
                    source=None;record=None
                    if key in cache:
                        record=dict(cache[key]);source='exact_action_cache'
                    elif method=='il_only' and i in legacy.get('il_only',{}):
                        record=dict(legacy['il_only'][i]);source='v3_short_or_v4_long_il'
                    elif (method=='no_geometry' and horizon=='short' and
                          np.array_equal(predictions[method],base_values[method])):
                        record=dict(legacy['no_geometry'][i]);source='v3_zero_residual_no_geometry'
                    capture=horizon=='short' and i in display
                    if record is None or capture:
                        measured,trace=executor.run(pose,data['semi'][i],data['y'][i],capture=capture)
                        new_episodes+=1
                        if record is not None:
                            for metric in ('target_mae','accepted','radius','drift_m'):
                                if abs(float(record[metric])-float(measured[metric]))>1e-8:
                                    raise RuntimeError(f'cached rollout mismatch: {method} {i} {metric}')
                        record=measured;source='fresh_rollout'
                        if capture:np.savez_compressed(td/f'{method}_{i:06d}.npz',**trace)
                    else:reused_episodes+=1
                    record.update(row=i,family=str(data['primitive_names'][data['primitive'][i]]),
                                  method=method,horizon=horizon,execution_source=source)
                    cache[key]=record;saved[i]=record
                    stream.write(json.dumps(record)+'\n');stream.flush()
                    if (j+1)%150==0 or j+1==len(eids):
                        print(f'EVAL {args.seed} {horizon} {method} {j+1}/{len(eids)} new={new_episodes} reused={reused_episodes}',flush=True)
            if set(saved)!=set(eids.tolist()):raise RuntimeError('missing or additional evaluation rows')
            rows=list(saved.values());summary=aggregate(rows)
            summary['by_family']={f:aggregate([r for r in rows if r['family']==f]) for f in sorted(set(r['family'] for r in rows))}
            results[horizon][method]=summary;raw[horizon][method]=saved
    meta=json.loads(args.metadata.read_text());actors=[str(c['actor_uid']) for c in meta['clips']]
    paired={h:{m:bootstrap_actor_difference(raw[h][m],raw[h]['il_only'],actors,data['clip_index'],args.seed)
               for m in trained_variants} for h in configs}
    write_json(out/'summary.json',dict(seed=args.seed,methods=results,paired_actor_bootstrap=paired,
               best_iterations={m:ckpts[m]['best_iteration'] for m in trained_variants},
               fresh_rollouts_this_invocation=new_episodes,reused_rollouts_this_invocation=reused_episodes))
    write_json(out/'complete.json',dict(short_rows=len(ids),long_rows=len(long_ids),methods=list(METHODS)))
    print('EVALUATION COMPLETE',args.seed,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--root',type=Path,default=Path('reports/manifold_motion/stage1_mixed_v5'))
    p.add_argument('--catalog',type=Path,default=Path('data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz'))
    p.add_argument('--metadata',type=Path,default=Path('reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows_metadata.json'))
    run(p.parse_args())


if __name__=='__main__':main()
