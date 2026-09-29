"""Build the auditable report for the simulator-aware Stage-1 adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manifold_motion.stage1.catalog_report import load_trace, render_traces
from manifold_motion.stage1.catalog_experiment import load_data, selected_rows, write_json
from manifold_motion.stage1.residual_rl import sha256


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return f'{values.mean():.4f} ± {values.std(ddof=1):.4f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('reports/manifold_motion/stage1_sim_adapter_v9_tracking'))
    parser.add_argument('--legacy-root', type=Path, default=Path('reports/manifold_motion/stage1_mixed_v5'))
    parser.add_argument('--catalog', type=Path, default=Path('data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz'))
    parser.add_argument('--out', type=Path, default=Path('docs/experiments/stage1_simulator_adapter_v9'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    media = args.out / 'media'
    media.mkdir(exist_ok=True)
    seeds = [20260928, 20260929, 20260930]
    summaries = {str(s): json.loads((args.root / f'seed_{s}' / 'test_summary.json').read_text()) for s in seeds}
    data = load_data(args.catalog)
    write_json(args.out / 'summary.json', summaries)
    audit = {}
    for s in seeds:
        folder = args.root / f'seed_{s}'
        audit[str(s)] = {
            'protocol': json.loads((folder / 'protocol.json').read_text()),
            'evaluation_protocol': json.loads((folder / 'evaluation_protocol.json').read_text()),
            'train_complete': json.loads((folder / 'train_complete.json').read_text()),
            'checkpoint_sha256': sha256(folder / 'adapter.pt'),
        }
    write_json(args.out / 'audit.json', audit)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['IL only', 'Safe inverse\ntracking']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, horizon in enumerate(('short', 'long')):
        for col, (metric, title, scale) in enumerate((
            ('target_mae', 'Target MAE ↓ (rad)', 1.0),
            ('accepted', 'Safety gate ↑ (%)', 100.0),
            ('pose_success', 'Pose success ↑ (%)', 100.0),
        )):
            values = []
            errors = []
            for method in ('il_only', 'sim_adapter'):
                v = [summaries[str(s)]['methods'][horizon][method][metric] * scale for s in seeds]
                values.append(np.mean(v)); errors.append(np.std(v, ddof=1))
            axes[row, col].bar(np.arange(2), values, yerr=errors, capsize=4,
                               color=['#98a8b8', '#176683'])
            axes[row, col].set_xticks(np.arange(2), names, rotation=18)
            axes[row, col].set_title(f'{horizon}: {title}')
            axes[row, col].grid(axis='y', alpha=.2); axes[row, col].set_axisbelow(True)
    fig.suptitle('Stage 1: simulator-aware inverse-tracking adapter | mean ± seed SD', fontsize=14)
    fig.savefig(args.out / 'metrics.png', dpi=150); plt.close(fig)

    display = selected_rows(np.flatnonzero(data['split'] == 2), data['primitive'])
    wanted = {'walk_lateral', 'turn_in_place', 'crouch_walk', 'forward_lunge', 'all_fours', 'carry_object'}
    gallery = []
    seed = 20260928
    adapter_trace_root = args.root / f'seed_{seed}' / 'traces'
    legacy_trace_root = args.legacy_root / f'seed_{seed}' / 'evaluation' / 'traces'
    adapter_rows = {int(r['row']): r for r in map(json.loads,
                    (args.root / f'seed_{seed}' / 'adapter_short.jsonl').read_text().splitlines())}
    il_rows = {int(r['row']): r for r in map(json.loads,
                (args.legacy_root / f'seed_{seed}' / 'evaluation' / 'il_only_short.jsonl').read_text().splitlines())}
    for row in display:
        family = str(data['primitive_names'][data['primitive'][row]])
        if family not in wanted:
            continue
        adapter_trace = adapter_trace_root / f'adapter_short_{int(row):06d}.npz'
        il_trace = legacy_trace_root / f'il_only_{int(row):06d}.npz'
        if not adapter_trace.exists() or not il_trace.exists():
            raise FileNotFoundError(f'missing trace for {row}')
        output = media / f'B_{int(row):06d}_{family}.gif'
        render_traces([load_trace(il_trace), load_trace(adapter_trace)],
                      ['IL only / no adapter', 'IL + inverse-tracking adapter'],
                      [il_rows[int(row)], adapter_rows[int(row)]], output,
                      f'{family} / {int(row)}', manifold_first=False)
        gallery.append((family, int(row), str(output.relative_to(args.out))))

    lines = [
        '# Stage 1：SONIC 逆跟踪适配器', '',
        '本实验在冻结 SONIC ONNX 的条件下，使用训练集物理回放测量“IL 请求姿态 − 实际保持姿态”，'
        '通过短时与 3 秒长时候选筛选后，训练一个只读取 `M_e` 与 IL 输出的有界残差适配器。'
        'SEED 目标只用于物理候选奖励与安全筛选，不作为在线残差标签。', '',
        '## 三种子测试结果', '',
        '| 时域 | IL only 目标 MAE | 适配器目标 MAE | IL 安全门控 | 适配器安全门控 | IL 姿态成功 | 适配器姿态成功 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for h, label in [('short', '短时全 727 窗口'), ('long', '长时固定 38 窗口')]:
        lines.append('| '+label+' | '+mean_sd([summaries[str(s)]['methods'][h]['il_only']['target_mae'] for s in seeds])+
                     ' | '+mean_sd([summaries[str(s)]['methods'][h]['sim_adapter']['target_mae'] for s in seeds])+
                     ' | '+mean_sd([100*summaries[str(s)]['methods'][h]['il_only']['accepted'] for s in seeds])+'%'
                     ' | '+mean_sd([100*summaries[str(s)]['methods'][h]['sim_adapter']['accepted'] for s in seeds])+'%'
                     ' | '+mean_sd([100*summaries[str(s)]['methods'][h]['il_only']['pose_success'] for s in seeds])+'%'
                     ' | '+mean_sd([100*summaries[str(s)]['methods'][h]['sim_adapter']['pose_success'] for s in seeds])+'% |')
    lines += ['', '![](metrics.png)', '', '## 冻结协议与 seed 选择', '',
              '- 三个 actor-disjoint seed：20260928 / 20260929 / 20260930。训练集每来源族 4 个窗口，短时和 3 秒长时都参与候选物理筛选。',
              '- 腿部残差固定为 0；腰部最多 ±0.04 rad；手臂最多 ±0.24 rad。在线只使用流形特征和 IL 输出。',
              '- 候选为 0×、0.5×、1.0×请求-实际误差；原本安全的 rollout 不允许因修正变成不安全。缩放值只在 validation 选择，test 未参与。',
              '- 各 seed 选择的缩放：'+', '.join(f'{s} → {summaries[str(s)]["selected_scale"]:.2f}×' for s in seeds)+'。',
              '- 20260929 validation 未通过严格门控，因此选择 0×并等价于 IL；该 seed 没有被隐藏。',
              '- 该实验显示短时 MAE 平均下降，但短时安全率存在约 0.14 个百分点的 test 波动；长时安全率保持。它是 simulator-aware 改善证据，不是已经完成的实机验证或 SOTA 声明。', '',
              '- 当前 split=2 已在 v3–v5 中被查看，因此本页属于 development-test；论文结论仍需新 actor / 新物理场景的未查看 confirmation set。', '',
              '## A/B MuJoCo 回放', '',
              '| 来源 | IL 与逆跟踪适配器同步对照 |', '|---|---|']
    for family, row, path in gallery:
        lines.append(f'| {family} / {row} | [![]({path})]({path}) |')
    lines += ['', '## 复现', '',
              '```bash',
              'seed=20260928',
              'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ./scripts/python.sh -m manifold_motion.stage1.simulator_adapter --seed "$seed" --base reports/manifold_motion/stage1_ab_response_v3/replicate_${seed}/seed_${seed}/imitation.pt --out reports/manifold_motion/stage1_sim_adapter_v9_tracking/seed_${seed}',
              'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ./scripts/python.sh -m manifold_motion.stage1.simulator_adapter --evaluate --seed "$seed" --base reports/manifold_motion/stage1_ab_response_v3/replicate_${seed}/seed_${seed}/imitation.pt --out reports/manifold_motion/stage1_sim_adapter_v9_tracking/seed_${seed}',
              '```', '',
              '[逐 seed 测试摘要](summary.json) · [协议与训练审计](audit.json)']
    text = '\n'.join(lines) + '\n'
    (args.out / 'README.md').write_text(text, encoding='utf-8')
    import markdown
    html = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    (args.out / 'index.html').write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Stage 1 inverse tracking</title>'
        '<style>body{max-width:1400px;margin:28px auto;padding:24px;font-family:Arial,"Microsoft YaHei",sans-serif;line-height:1.65;color:#20354b;background:#f4f7fa}h1,h2{color:#155775}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d5dfe8;padding:10px}th{background:#eaf1f6}img{max-width:100%}pre{overflow:auto;background:#162c42;color:white;padding:14px}</style>' + html + '</html>',
        encoding='utf-8')
    print(f'REPORT COMPLETE {args.out}', flush=True)


if __name__ == '__main__':
    main()
