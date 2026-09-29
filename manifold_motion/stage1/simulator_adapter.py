"""Simulator-aware inverse-tracking adapter for the static Stage-1 study.

The imitation checkpoint and SONIC remain frozen.  On a deterministic,
actor-disjoint training subset we execute the imitation command, measure the
achieved pose, try a small fixed set of bounded inverse-error corrections, and
retain only candidates that preserve the baseline's short- and long-horizon
safety.  A regularised linear adapter then distils those selected corrections
from information available at inference time: ``[M_e, imitation_pose]``.

No test row is used while collecting corrections, fitting the adapter, or
selecting its scale.  This is a model-based, simulator-in-the-loop policy
improvement experiment; it is not relabelled PPO and it does not fine-tune the
SONIC ONNX controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from manifold_motion.core import constants as C
from manifold_motion.stage1.catalog_experiment import (
    DEFAULT,
    Executor,
    Protocol,
    aggregate,
    load_data,
    selected_rows,
    write_json,
)
from manifold_motion.stage1.residual_rl import base_predictions, dense_reward, sha256


@dataclass(frozen=True)
class AdapterConfig:
    train_per_family: int = 4
    gains: tuple[float, ...] = (0.0, 0.5, 1.0)
    leg_limit: float = 0.0
    waist_limit: float = 0.04
    arm_limit: float = 0.24
    ridge_alpha: float = 1.0
    validation_scales: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    correction_target: str = 'imitation_request_minus_achieved_pose'


SHORT = Protocol(warmup_ticks=10, transition_ticks=10, hold_ticks=20)
LONG = Protocol(warmup_ticks=20, transition_ticks=20, hold_ticks=150)


def structured_limits(cfg: AdapterConfig) -> np.ndarray:
    """Return anatomical limits in the IsaacLab order used by catalogue poses."""
    hardware = np.array(
        [cfg.leg_limit if i < 12 else cfg.waist_limit if i < 15 else cfg.arm_limit for i in range(29)],
        dtype=np.float32,
    )
    return hardware[C.MUJOCO_TO_ISAACLAB]


def hold_mean_delta(trace: dict[str, np.ndarray], protocol: Protocol) -> np.ndarray:
    """Reproduce Executor's hold-window sampling and return pose delta."""
    q = np.asarray(trace['q'])
    ticks = np.arange(1, len(q) + 1)
    mask = (np.arange(len(q)) >= protocol.transition_ticks) & (ticks % protocol.measure_stride == 0)
    if not np.any(mask):
        raise ValueError('protocol has no hold measurement samples')
    return q[mask].mean(axis=0) - DEFAULT


def with_dense(result: dict) -> dict:
    return dict(result, dense_reward=dense_reward(result))


def candidate_is_safe(short: dict, long: dict, base_short: dict, base_long: dict) -> bool:
    """A correction may repair a failure, but may not break a baseline-safe rollout."""
    if short['fallen'] or long['fallen'] or short['nonfoot_contact'] or long['nonfoot_contact']:
        return False
    if base_short['accepted'] and not short['accepted']:
        return False
    if base_long['accepted'] and not long['accepted']:
        return False
    return True


def pair_score(short: dict, long: dict) -> float:
    return dense_reward(short) + 0.7 * dense_reward(long)


def choose_candidate(candidates: list[dict]) -> dict:
    """Select the highest-reward safe candidate; candidate zero is the fallback."""
    if not candidates or float(candidates[0]['gain']) != 0.0:
        raise ValueError('candidate zero must be the first entry')
    base = candidates[0]
    safe = [c for c in candidates if candidate_is_safe(c['short'], c['long'], base['short'], base['long'])]
    if not safe:
        return base
    # Deterministic tie break prefers the smaller intervention.
    return max(safe, key=lambda c: (pair_score(c['short'], c['long']), -float(c['gain'])))


def summarize(rows: list[dict]) -> dict:
    value = aggregate(rows)
    value['dense_reward'] = float(np.mean([dense_reward(r) for r in rows]))
    return value


def fit_ridge(features: np.ndarray, targets: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1e-5)
    normalized = (features - mean) / std
    design = np.concatenate([normalized, np.ones((len(features), 1))], axis=1)
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return {'feature_mean': mean.astype(np.float32), 'feature_std': std.astype(np.float32),
            'weights': weights.astype(np.float32)}


def predict_residual(model: dict, x: np.ndarray, base: np.ndarray, limits: np.ndarray) -> np.ndarray:
    features = np.concatenate([x, base], axis=-1)
    normalized = (features - model['feature_mean']) / model['feature_std']
    design = np.concatenate([normalized, np.ones((len(features), 1), dtype=normalized.dtype)], axis=1)
    return np.clip(design @ model['weights'], -limits, limits).astype(np.float32)


def normalized_residual_norm(residual: np.ndarray, limits: np.ndarray) -> np.ndarray:
    active = limits > 1e-8
    if not np.any(active):
        return np.zeros(len(residual), dtype=np.float32)
    scaled = residual[:, active] / limits[active]
    return np.sqrt(np.mean(scaled * scaled, axis=1)).astype(np.float32)


def evaluate_actions(actions, data, ids, short_executor, long_executor):
    short_rows, long_rows = [], []
    for row, action in zip(ids, actions):
        s, _ = short_executor.run(action, data['semi'][row], data['y'][row])
        l, _ = long_executor.run(action, data['semi'][row], data['y'][row])
        short_rows.append(dict(row=int(row), **s))
        long_rows.append(dict(row=int(row), **l))
    return summarize(short_rows), summarize(long_rows), short_rows, long_rows


def evaluate_horizon(actions, data, ids, executor):
    rows = []
    for row, action in zip(ids, actions):
        result, _ = executor.run(action, data['semi'][row], data['y'][row])
        rows.append(dict(row=int(row), **result))
    return summarize(rows), rows


def validation_eligible(short, long, baseline_short, baseline_long):
    return (
        short['accepted'] >= baseline_short['accepted']
        and long['accepted'] >= baseline_long['accepted']
        and short['fallen'] <= baseline_short['fallen']
        and long['fallen'] <= baseline_long['fallen']
        and short['target_mae'] <= baseline_short['target_mae']
        and long['target_mae'] <= baseline_long['target_mae']
    )


def run(args):
    torch.set_num_threads(2)
    cfg = AdapterConfig(train_per_family=args.train_per_family, ridge_alpha=args.ridge_alpha)
    data = load_data(args.catalog)
    base = base_predictions(args.base, data)
    limits = structured_limits(cfg)
    train_ids = selected_rows(np.flatnonzero(data['split'] == 0), data['primitive'], cfg.train_per_family)
    val_short_ids = selected_rows(np.flatnonzero(data['split'] == 1), data['primitive'], 3)
    val_long_ids = selected_rows(np.flatnonzero(data['split'] == 1), data['primitive'], 1)
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = {
        'version': 1,
        'algorithm': 'safe_inverse_tracking_distillation',
        'seed': args.seed,
        'config': json.loads(json.dumps(asdict(cfg))),
        'base_path': str(args.base),
        'base_sha256': sha256(args.base),
        'catalog_sha256': sha256(args.catalog),
        'train_ids': train_ids.tolist(),
        'validation_short_ids': val_short_ids.tolist(),
        'validation_long_ids': val_long_ids.tolist(),
        'short_executor': asdict(SHORT),
        'long_executor': asdict(LONG),
        'test_used_for_selection': False,
        'inference_inputs': '[normalized manifold features, imitation output]',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    protocol_path = args.out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('existing output has another protocol; choose a new --out')
    write_json(protocol_path, protocol)
    if (args.out / 'train_complete.json').exists():
        print('Adapter already completed; using existing artifacts', flush=True)
        return

    short_executor, long_executor = Executor(SHORT), Executor(LONG)
    collection_path = args.out / 'collection.jsonl'
    saved = {}
    if collection_path.exists():
        saved = {int(r['row']): r for r in map(json.loads, collection_path.read_text().splitlines())}
    with collection_path.open('a', encoding='utf-8') as stream:
        for number, row in enumerate(train_ids, 1):
            if int(row) in saved:
                continue
            base_short, short_trace = short_executor.run(base[row], data['semi'][row], data['y'][row], capture=True)
            base_long, long_trace = long_executor.run(base[row], data['semi'][row], data['y'][row], capture=True)
            achieved = 0.5 * (hold_mean_delta(short_trace, SHORT) + hold_mean_delta(long_trace, LONG))
            # Compensate only the frozen controller's tracking response.  Using
            # the recorded target here would also distil the imitation model's
            # sample-specific generalisation error and proved seed-dependent.
            direction = base[row] - achieved
            candidates = []
            for gain in cfg.gains:
                correction = np.clip(float(gain) * direction, -limits, limits)
                if gain == 0.0:
                    short_result, long_result = base_short, base_long
                else:
                    action = base[row] + correction
                    short_result, _ = short_executor.run(action, data['semi'][row], data['y'][row])
                    long_result, _ = long_executor.run(action, data['semi'][row], data['y'][row])
                candidates.append({'gain': float(gain), 'correction': correction.tolist(),
                                   'short': with_dense(short_result), 'long': with_dense(long_result)})
            chosen = choose_candidate(candidates)
            record = {'row': int(row), 'family': str(data['primitive_names'][data['primitive'][row]]),
                      'selected_gain': chosen['gain'], 'selected_correction': chosen['correction'],
                      'candidates': candidates}
            saved[int(row)] = record
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            print(f'COLLECT {args.seed} {number}/{len(train_ids)} selected={chosen["gain"]:.2f}', flush=True)

    ordered = [saved[int(i)] for i in train_ids]
    corrections = np.asarray([r['selected_correction'] for r in ordered], dtype=np.float32)
    features = np.concatenate([data['x'][train_ids], base[train_ids]], axis=-1)
    model = fit_ridge(features, corrections, cfg.ridge_alpha)
    predicted = predict_residual(model, data['x'], base, limits)
    base_short, base_short_rows = evaluate_horizon(base[val_short_ids], data, val_short_ids, short_executor)
    # Long selection uses its own fixed stratified subset; do not reuse the short rows.
    base_long, base_long_rows = evaluate_horizon(base[val_long_ids], data, val_long_ids, long_executor)
    write_json(args.out / 'baseline_short.json', base_short_rows)
    write_json(args.out / 'baseline_long.json', base_long_rows)
    validations = []
    best_scale, best_score = 0.0, pair_score(base_short, base_long)
    for scale in cfg.validation_scales:
        short_actions = base[val_short_ids] + float(scale) * predicted[val_short_ids]
        long_actions = base[val_long_ids] + float(scale) * predicted[val_long_ids]
        short_summary, short_rows = evaluate_horizon(short_actions, data, val_short_ids, short_executor)
        long_summary, long_rows = evaluate_horizon(long_actions, data, val_long_ids, long_executor)
        eligible = validation_eligible(short_summary, long_summary, base_short, base_long)
        score = pair_score(short_summary, long_summary)
        validations.append({'scale': float(scale), 'short': short_summary, 'long': long_summary,
                            'eligible': bool(eligible), 'score': float(score)})
        write_json(args.out / f'validation_scale_{scale:.2f}_short.json', short_rows)
        write_json(args.out / f'validation_scale_{scale:.2f}_long.json', long_rows)
        if eligible and score > best_score:
            best_scale, best_score = float(scale), float(score)
        print(f'VALIDATE {args.seed} scale={scale:.2f} short_mae={short_summary["target_mae"]:.5f} '
              f'long_mae={long_summary["target_mae"]:.5f} eligible={eligible}', flush=True)

    checkpoint = dict(model, limits=limits, selected_scale=np.float32(best_scale),
                      seed=np.int64(args.seed), base_path=str(args.base), base_sha256=sha256(args.base),
                      catalog_sha256=sha256(args.catalog), config=asdict(cfg))
    torch.save(checkpoint, args.out / 'adapter.pt')
    selected = sum(float(r['selected_gain']) > 0 for r in ordered)
    write_json(args.out / 'validation.json', {'baseline_short': base_short, 'baseline_long': base_long,
                                               'scales': validations, 'selected_scale': best_scale})
    write_json(args.out / 'train_complete.json', {
        'seed': args.seed,
        'training_rows': len(train_ids),
        'nonzero_policy_improvement_targets': selected,
        'selected_scale': best_scale,
        'best_validation_score': best_score,
        'test_evaluated': False,
    })
    print(f'ADAPTER COMPLETE seed={args.seed} selected_scale={best_scale:.2f} '
          f'nonzero_targets={selected}/{len(train_ids)}', flush=True)


def probe(args):
    """Select a conservative residual-norm gate using validation only."""
    torch.set_num_threads(2)
    out = args.out
    checkpoint = torch.load(out / 'adapter.pt', weights_only=False, map_location='cpu')
    protocol = json.loads((out / 'protocol.json').read_text())
    cfg = AdapterConfig(**protocol['config'])
    data = load_data(args.catalog)
    base = base_predictions(args.base, data)
    limits = np.asarray(checkpoint['limits'], dtype=np.float32)
    residual = predict_residual(checkpoint, data['x'], base, limits)
    norm = normalized_residual_norm(residual, limits)
    short_ids = np.asarray(protocol['validation_short_ids'], dtype=int)
    long_ids = np.asarray(protocol['validation_long_ids'], dtype=int)
    short_executor, long_executor = Executor(SHORT), Executor(LONG)
    base_short, _ = evaluate_horizon(base[short_ids], data, short_ids, short_executor)
    base_long, _ = evaluate_horizon(base[long_ids], data, long_ids, long_executor)
    scales = tuple(float(v) for v in args.scales)
    thresholds = tuple(float(v) for v in args.thresholds)
    records, best = [], None
    for scale in scales:
        for threshold in thresholds:
            mask = norm <= threshold
            short_actions = base[short_ids] + float(scale) * residual[short_ids] * mask[short_ids, None]
            long_actions = base[long_ids] + float(scale) * residual[long_ids] * mask[long_ids, None]
            short_summary, short_rows = evaluate_horizon(short_actions, data, short_ids, short_executor)
            long_summary, long_rows = evaluate_horizon(long_actions, data, long_ids, long_executor)
            eligible = validation_eligible(short_summary, long_summary, base_short, base_long)
            score = pair_score(short_summary, long_summary)
            record = {'scale': scale, 'threshold': threshold, 'active_fraction': float(mask.mean()),
                      'short': short_summary, 'long': long_summary, 'eligible': bool(eligible), 'score': score}
            records.append(record)
            write_json(out / f'probe_scale_{scale:.2f}_threshold_{threshold:.2f}_short.json', short_rows)
            write_json(out / f'probe_scale_{scale:.2f}_threshold_{threshold:.2f}_long.json', long_rows)
            if eligible and (best is None or score > best['score']):
                best = record
            print(f'PROBE {args.seed} scale={scale:.2f} threshold={threshold:.2f} '
                  f'active={mask.mean():.2f} short_mae={short_summary["target_mae"]:.5f} '
                  f'long_mae={long_summary["target_mae"]:.5f} eligible={eligible}', flush=True)
    write_json(out / 'probe.json', {'baseline_short': base_short, 'baseline_long': base_long,
                                    'records': records, 'selected': best})
    if best is not None:
        checkpoint['selected_scale'] = np.float32(best['scale'])
        checkpoint['selected_threshold'] = np.float32(best['threshold'])
        torch.save(checkpoint, out / 'adapter_probe_selected.pt')
        print(f'PROBE SELECTED seed={args.seed} scale={best["scale"]:.2f} '
              f'threshold={best["threshold"]:.2f}', flush=True)
    else:
        print(f'PROBE NO SAFE IMPROVEMENT seed={args.seed}', flush=True)


def _read_jsonl(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    return {int(r['row']): r for r in map(json.loads, path.read_text().splitlines())}


def evaluate(args):
    """Evaluate a frozen adapter on the development-test split exactly once."""
    torch.set_num_threads(2)
    data = load_data(args.catalog)
    base = base_predictions(args.base, data)
    checkpoint = torch.load(args.out / 'adapter.pt', weights_only=False, map_location='cpu')
    limits = np.asarray(checkpoint['limits'], dtype=np.float32)
    scale = float(checkpoint['selected_scale'])
    residual = predict_residual(checkpoint, data['x'], base, limits)
    test_ids = np.flatnonzero(data['split'] == 2)
    long_ids = selected_rows(test_ids, data['primitive'], 2)
    display_ids = set(selected_rows(test_ids, data['primitive'], 1).tolist())
    short_executor, long_executor = Executor(SHORT), Executor(LONG)
    protocol = json.loads((args.out / 'protocol.json').read_text())
    protocol.update({'test_ids': test_ids.tolist(), 'long_test_ids': long_ids.tolist(),
                     'checkpoint_sha256': sha256(args.out / 'adapter.pt'),
                     'test_used_for_selection': False})
    write_json(args.out / 'evaluation_protocol.json', protocol)
    results = {}
    traces = args.out / 'traces'
    traces.mkdir(parents=True, exist_ok=True)
    for horizon, ids, executor in [('short', test_ids, short_executor), ('long', long_ids, long_executor)]:
        actions = base[ids] + scale * residual[ids]
        path = args.out / f'adapter_{horizon}.jsonl'
        saved = _read_jsonl(path)
        with path.open('a', encoding='utf-8') as stream:
            for number, (row, action) in enumerate(zip(ids, actions), 1):
                if int(row) in saved:
                    continue
                capture = int(row) in display_ids
                result, trace = executor.run(action, data['semi'][row], data['y'][row], capture=capture)
                record = dict(row=int(row), family=str(data['primitive_names'][data['primitive'][row]]), **result)
                saved[int(row)] = record
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                if capture:
                    np.savez_compressed(traces / f'adapter_{horizon}_{int(row):06d}.npz', **trace)
                if number % 100 == 0 or number == len(ids):
                    print(f'EVAL ADAPTER seed={args.seed} {horizon} {number}/{len(ids)}', flush=True)
        rows = list(saved.values())
        results[horizon] = {'sim_adapter': aggregate(rows), '_rows_data': rows}

    # Reuse the exact deterministic IL rows from the locked v5 evaluation, and
    # verify row identity before computing paired differences.
    legacy_root = args.legacy_root / f'seed_{args.seed}' / 'evaluation'
    for horizon in ('short', 'long'):
        base_rows = _read_jsonl(legacy_root / f'il_only_{horizon}.jsonl')
        adapter_rows = {r['row']: r for r in results[horizon].pop('_rows_data')}
        expected = set(test_ids.tolist()) if horizon == 'short' else set(long_ids.tolist())
        if set(base_rows) != expected or set(adapter_rows) != expected:
            raise ValueError(f'row identity mismatch for {horizon}')
        from manifold_motion.stage1.catalog_experiment import bootstrap_actor_difference
        meta = json.loads(args.metadata.read_text())
        actors = [str(c['actor_uid']) for c in meta['clips']]
        results[horizon]['il_only'] = aggregate(list(base_rows.values()))
        results[horizon]['paired_actor_bootstrap'] = bootstrap_actor_difference(
            adapter_rows, base_rows, actors, data['clip_index'], args.seed
        )
        results[horizon]['adapter_scale'] = scale
    write_json(args.out / 'test_summary.json', {'seed': args.seed, 'selected_scale': scale, 'methods': results,
                                                 'test_used_for_selection': False})
    print(f'EVALUATION COMPLETE adapter seed={args.seed} scale={scale:.2f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=Path('data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz'))
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--train-per-family', type=int, default=4)
    parser.add_argument('--ridge-alpha', type=float, default=1.0)
    parser.add_argument('--probe', action='store_true', help='probe an existing adapter using validation only')
    parser.add_argument('--evaluate', action='store_true', help='evaluate a frozen adapter on split=2')
    parser.add_argument('--legacy-root', type=Path, default=Path('reports/manifold_motion/stage1_mixed_v5'))
    parser.add_argument('--metadata', type=Path, default=Path('reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows_metadata.json'))
    parser.add_argument('--scales', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    args = parser.parse_args()
    if args.train_per_family < 1 or args.ridge_alpha <= 0:
        parser.error('train-per-family and ridge-alpha must be positive')
    if args.evaluate:
        evaluate(args)
    elif args.probe:
        probe(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
