"""Pool simulator-selected inverse-tracking labels across Stage-1 seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from manifold_motion.stage1.catalog_experiment import Executor, Protocol, load_data, selected_rows, write_json
from manifold_motion.stage1.residual_rl import base_predictions, sha256
from manifold_motion.stage1.simulator_adapter import (
    AdapterConfig,
    LONG,
    SHORT,
    evaluate_horizon,
    fit_ridge,
    predict_residual,
    structured_limits,
    pair_score,
    validation_eligible,
)


def run(args):
    data = load_data(args.catalog)
    cfg = AdapterConfig(ridge_alpha=args.ridge_alpha)
    limits = structured_limits(cfg)
    all_features, all_targets = [], []
    bases = {}
    collection_hashes = {}
    for seed in args.seeds:
        base_path = args.base_root / f'replicate_{seed}' / f'seed_{seed}' / 'imitation.pt'
        collection = args.collection_root / f'seed_{seed}' / 'collection.jsonl'
        if not base_path.exists() or not collection.exists():
            raise FileNotFoundError(f'missing seed {seed} base or collection')
        base = base_predictions(base_path, data)
        rows = {int(r['row']): r for r in map(json.loads, collection.read_text().splitlines())}
        ids = selected_rows(np.flatnonzero(data['split'] == 0), data['primitive'], cfg.train_per_family)
        if set(ids.tolist()) != set(rows):
            raise ValueError(f'collection ids differ for seed {seed}')
        corrections = np.asarray([rows[int(i)]['selected_correction'] for i in ids], dtype=np.float32)
        all_features.append(np.concatenate([data['x'][ids], base[ids]], axis=-1))
        all_targets.append(corrections)
        bases[int(seed)] = (base_path, base)
        collection_hashes[str(seed)] = hashlib.sha256(collection.read_bytes()).hexdigest()
    model = fit_ridge(np.concatenate(all_features), np.concatenate(all_targets), cfg.ridge_alpha)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(model, limits=limits, selected_scale=np.float32(0.0), selected_threshold=np.float32(1.0),
                      seeds=args.seeds, config=json.loads(json.dumps(cfg.__dict__)))
    torch.save(checkpoint, args.out / 'pooled_adapter.pt')
    short_executor, long_executor = Executor(SHORT), Executor(LONG)
    short_ids = selected_rows(np.flatnonzero(data['split'] == 1), data['primitive'], 3)
    long_ids = selected_rows(np.flatnonzero(data['split'] == 1), data['primitive'], 1)
    prepared = {}
    for seed in args.seeds:
        _, base = bases[int(seed)]
        base_short, _ = evaluate_horizon(base[short_ids], data, short_ids, short_executor)
        base_long, _ = evaluate_horizon(base[long_ids], data, long_ids, long_executor)
        prepared[int(seed)] = (base, predict_residual(model, data['x'], base, limits), base_short, base_long)
    records = []
    for scale in args.scales:
        summaries = {}
        eligible_all = True
        score = 0.0
        for seed in args.seeds:
            base, residual, base_short, base_long = prepared[int(seed)]
            short_summary, short_rows = evaluate_horizon(
                base[short_ids] + float(scale) * residual[short_ids], data, short_ids, short_executor
            )
            long_summary, long_rows = evaluate_horizon(
                base[long_ids] + float(scale) * residual[long_ids], data, long_ids, long_executor
            )
            eligible = validation_eligible(short_summary, long_summary, base_short, base_long)
            eligible_all &= eligible
            score += pair_score(short_summary, long_summary)
            summaries[str(seed)] = {'short': short_summary, 'long': long_summary, 'eligible': bool(eligible)}
            write_json(args.out / f'scale_{scale:.2f}_seed_{seed}_short.json', short_rows)
            write_json(args.out / f'scale_{scale:.2f}_seed_{seed}_long.json', long_rows)
        rec = {'scale': float(scale), 'eligible_all_seeds': bool(eligible_all),
               'score_sum': float(score), 'seeds': summaries}
        records.append(rec)
        print(f'POOLED scale={scale:.2f} eligible_all={eligible_all} score={score:.5f}', flush=True)
    eligible = [r for r in records if r['eligible_all_seeds']]
    selected = max(eligible, key=lambda r: r['score_sum']) if eligible else next(r for r in records if r['scale'] == 0.0)
    checkpoint['selected_scale'] = np.float32(selected['scale'])
    torch.save(checkpoint, args.out / 'pooled_adapter_selected.pt')
    protocol = {'version': 1, 'algorithm': 'pooled_safe_inverse_tracking_distillation',
                'seeds': args.seeds, 'catalog_sha256': sha256(args.catalog),
                'base_sha256': {str(s): sha256(bases[int(s)][0]) for s in args.seeds},
                'collection_sha256': collection_hashes, 'test_used_for_selection': False,
                'selection': 'one scale must pass short and long safety gates for every seed',
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.out / 'protocol.json', protocol)
    write_json(args.out / 'validation.json', {'records': records, 'selected': selected,
                                                'training_rows_per_seed': len(all_features[0])})
    write_json(args.out / 'train_complete.json', {'seeds': args.seeds, 'selected_scale': selected['scale'],
                                                  'test_evaluated': False})
    print(f'POOLED COMPLETE scale={selected["scale"]:.2f} eligible_all={selected["eligible_all_seeds"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=Path('data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz'))
    parser.add_argument('--base-root', type=Path, default=Path('reports/manifold_motion/stage1_ab_response_v3'))
    parser.add_argument('--collection-root', type=Path, default=Path('reports/manifold_motion/stage1_sim_adapter_v7'))
    parser.add_argument('--out', type=Path, default=Path('reports/manifold_motion/stage1_sim_adapter_v8_pooled'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[20260928, 20260929, 20260930])
    parser.add_argument('--ridge-alpha', type=float, default=1.0)
    parser.add_argument('--scales', type=float, nargs='+', default=[0.0, 0.25, 0.5])
    args = parser.parse_args()
    if args.ridge_alpha <= 0 or not args.seeds:
        parser.error('positive ridge alpha and at least one seed required')
    run(args)


if __name__ == '__main__':
    main()
