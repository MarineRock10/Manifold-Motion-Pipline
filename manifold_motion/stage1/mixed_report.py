"""Finalize the six-method Stage-1 A/B matrix; keep unsuccessful ablations visible."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from manifold_motion.stage1.catalog_experiment import load_data,selected_rows,bootstrap_actor_difference,write_json
from manifold_motion.stage1.catalog_report import load_trace,render_traces
from manifold_motion.stage1.mixed_evaluate import METHODS
from manifold_motion.stage1.residual_rl import sha256

LABELS={'il_only':'IL only / no RL','full':'IL + mixed-horizon RL','no_imitation':'No IL pretraining',
        'no_geometry':'No manifold input','no_long_reward':'No long-horizon reward',
        'continued_imitation':'Continued IL (matched updates)'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('reports/manifold_motion/stage1_mixed_v5'))
    p.add_argument('--out',type=Path,default=Path('docs/experiments/stage1_mixed_v5'))
    p.add_argument('--catalog',type=Path,default=Path('data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz'))
    p.add_argument('--metadata',type=Path,default=Path('reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows_metadata.json'))
    p.add_argument('--summary-only',action='store_true');args=p.parse_args();torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=True);media=args.out/'media';media.mkdir(exist_ok=True)
    results={};audit={};raw={}
    for folder in sorted(args.root.glob('seed_*')):
        seed=folder.name.split('_')[1];ev=folder/'evaluation'
        if not (ev/'complete.json').exists():raise RuntimeError(f'evaluation incomplete: {seed}')
        results[seed]=json.loads((ev/'summary.json').read_text())
        audit[seed]={'evaluation':json.loads((ev/'protocol.json').read_text()),'training':{}}
        for m in METHODS[1:]:
            checkpoint=folder/m/'residual.pt'
            if sha256(checkpoint)!=audit[seed]['evaluation']['checkpoint_sha256'][m]:raise RuntimeError('changed weights')
            audit[seed]['training'][m]={k:json.loads((folder/m/f'{k}.json').read_text())
                                         for k in ['protocol','train_complete','history']}
            audit[seed]['training'][m]['checkpoint_sha256']=sha256(checkpoint)
        raw[seed]={h:{m:{r['row']:r for r in map(json.loads,(ev/f'{m}_{h}.jsonl').read_text().splitlines())}
                       for m in METHODS} for h in ['short','long']}
    if len(results)!=3:raise RuntimeError('expected all three seeds')
    data=load_data(args.catalog);meta=json.loads(args.metadata.read_text());actors=[str(c['actor_uid']) for c in meta['clips']]
    comparisons={}
    for h in ['short','long']:
        ids=sorted(raw[next(iter(raw))][h]['il_only'])
        averaged={m:{i:{k:float(np.mean([raw[s][h][m][i][k] for s in raw]))
                           for k in ['accepted','pose_success','target_mae']} for i in ids} for m in METHODS}
        comparisons[h]={m:bootstrap_actor_difference(averaged[m],averaged['il_only'],actors,data['clip_index'],20260928)
                        for m in METHODS[1:]}
    write_json(args.out/'summary.json',dict(seeds=results,seed_mean_actor_bootstrap=comparisons))
    write_json(args.out/'audit.json',audit)
    with gzip.open(args.out/'episodes.jsonl.gz','wt',encoding='utf-8') as stream:
        for s in raw:
            for h in raw[s]:
                for m,rows in raw[s][h].items():
                    for r in rows.values():
                        # Evaluation rows already carry the seed; normalize it here
                        # instead of passing a duplicate keyword to ``dict``.
                        record=dict(r)
                        record['seed']=s
                        stream.write(json.dumps(record)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['IL','IL+RL','No IL','No M','No long r','More IL']
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ri,h in enumerate(['short','long']):
        for ax,k,title,scale in zip(axes[ri],['target_mae','accepted','pose_success'],
                                   ['Target MAE (rad)','Shape / physics gate (%)','Pose success (%)'],[1,100,100]):
            vals=np.array([[results[s]['methods'][h][m][k] for s in results] for m in METHODS])*scale
            ax.bar(np.arange(6),vals.mean(1),yerr=vals.std(1,ddof=1),capsize=3,
                   color=['#98a8b8','#176683','#bdb5aa','#b2a1ab','#ae8b6d','#86a999'])
            ax.set_xticks(np.arange(6),names,rotation=20);ax.set_title(f'{h}: {title}',fontsize=10)
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Stage 1: fixed six-method matrix | mean +/- seed SD',fontsize=14)
    fig.savefig(args.out/'ab_metrics.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for s in audit:
        history=audit[s]['training']['full']['history'];v=[r for r in history if 'validation_long' in r]
        axes[0].plot([r['iteration'] for r in v],[r['validation_short']['target_mae'] for r in v],marker='o',label=s)
        axes[1].plot([r['iteration'] for r in v],[r['validation_long']['accepted']*100 for r in v],marker='o',label=s)
    for ax,title in zip(axes,['Short validation target MAE','Long validation gate (%)']):
        ax.set_title(title);ax.set_xlabel('Iteration');ax.legend();ax.grid(alpha=.2)
    fig.savefig(args.out/'training.png',dpi=150);plt.close(fig)
    display=selected_rows(np.flatnonzero(data['split']==2),data['primitive'])
    wanted={'walk_lateral','turn_in_place','crouch_walk','forward_lunge','all_fours','carry_object'}
    gallery=[];seed='20260928';traces=args.root/f'seed_{seed}'/'evaluation'/'traces'
    for i in display:
        family=str(data['primitive_names'][data['primitive'][i]])
        if family not in wanted:continue
        apath=media/f'A_{i:06d}_{family}.gif';bpath=media/f'B_{i:06d}_{family}.gif'
        if not args.summary_only:
            t=load_trace(traces/f'full_{i:06d}.npz');r=raw[seed]['short']['full'][int(i)]
            render_traces([t,t],['M_e + measured M_self','Validation-selected IL + RL'],[r,r],apath,
                          f'{family} / {i}',manifold_first=True)
            bm=['il_only','full','no_imitation','no_geometry']
            render_traces([load_trace(traces/f'{m}_{i:06d}.npz') for m in bm],[LABELS[m] for m in bm],
                          [raw[seed]['short'][m][int(i)] for m in bm],bpath,f'{family} / {i}')
        if not apath.exists() or not bpath.exists():raise FileNotFoundError('missing media')
        gallery.append((family,int(i),str(apath.relative_to(args.out)),str(bpath.relative_to(args.out))))
        print('A/B ready',family,flush=True)
    def cell(h,m,k):
        scale=1 if k=='target_mae' else 100
        v=np.array([results[s]['methods'][h][m][k] for s in results])*scale
        return f'{v.mean():.4f} ± {v.std(ddof=1):.4f}' if scale==1 else f'{v.mean():.2f} ± {v.std(ddof=1):.2f}%'
    counts={h:results[next(iter(results))]['methods'][h]['il_only']['rows'] for h in ['short','long']}
    lines=['# Stage 1：六组 A/B 与混合时域实验','',
           '本页是当前静态姿态版 Stage 1 的完整实验矩阵，不是完整动态原语、固定世界障碍物通行或 CVPR/SOTA 验证。冻结 SONIC，输入为反向构造的环境流形描述，输出为 29 关节静态目标。','',
           '保留 [v3](../stage1_ab_response_v3/README.md) 和 [v4](../stage1_residual_rl_v4/README.md) 的负结果。本轮不把其他方法输出改名为 RL，不按测试效果挑选展示。','',
           '## 固定协议与验收','',
           '- 三种子：20260928 / 20260929 / 20260930；train 2,610 / validation 651 / development-test 727 个窗口，actor 不重叠。',
           '- 每个 RL 分支 20 轮 × 4 个流形 × 2 个候选；每候选跑短、长两个时域，共 320 次训练物理执行。每批 2 次 PPO 更新。',
           '- 短执行：0.2 秒预热＋0.2 秒过渡＋0.4 秒保持；长执行：0.4 秒预热＋0.4 秒过渡＋3 秒保持。',
           '- 相同的结构化残差界：腿/腰 ±0.04 rad，手臂 ±0.16 rad；PCA 基底仅从 train 目标构建。无模仿分支以站立零增量为基础，同样使用训练奖励和共享基底，因此不等于完全不使用 SEED 监督。',
           '- 验证集分别按来源族均匀取 45 个短时窗口和 15 个长时窗口。选模要求两时域安全门控均不降低、跌倒率均不增加，MAE 不超过基线＋0.005 rad；通过者按 short reward＋0.7×long reward 选择，允许保留第 0 轮。',
           '- 无长时奖励分支仍执行两个时域，以保持物理采样预算相同，只去掉长时奖励项；继续模仿分支匹配优化步数与数据采样，不声称匹配仿真调用数或墙钟耗时。',
           f'- 短时每方法/种子全测 {counts["short"]} 窗；长时每来源族固定两个源索引，共 {counts["long"]} 窗，不能冒充全测试集长时通过率。',
           '- 测试集已在先前版本查看过，本页是开发性复测；论文定稿仍需要新的未查看 actor/物理场景确认集。','',
           '![](training.png)','',
           '## B：完整消融矩阵','', '![](ab_metrics.png)','']
    for h,title in [('short','短时：完整 development-test'),('long','长时：固定分层子集')]:
        lines += [f'### {title}','', '| 方法 | 执行目标 MAE ↓ | 安全门控 ↑ | 姿态成功 ↑ | 跌倒 ↓ |','|---|---:|---:|---:|---:|']
        for m in METHODS:lines.append('| '+LABELS[m]+' | '+' | '.join(cell(h,m,k) for k in ['target_mae','accepted','pose_success','fallen'])+' |')
        c=comparisons[h]['full']
        lines+=['',f'Full−IL：MAE 差 {c["target_mae"]["mean_delta"]:+.5f} rad，actor-cluster 95% CI {c["target_mae"]["ci95"]}；安全通过率差 {c["accepted"]["mean_delta"]:+.5f}，CI {c["accepted"]["ci95"]}。','']
    lines += ['### 验证集实际选择','', '| Seed | Full | No IL | No M | No long reward | Continued IL |','|---|---:|---:|---:|---:|---:|']
    for s,r in results.items():lines.append(f'| {s} | '+' | '.join(str(r['best_iterations'][m]) for m in METHODS[1:])+' |')
    lines+=['','第 0 轮表示原始基础策略被保留，不计作 RL 学习成功。姿态成功 = 安全门控通过且实际平均目标关节误差 ≤0.30 rad；动作名称只说明源数据类别，不代表完整走/蹲/爬任务成功。','',
            '[逐种子/逐来源族和配对区间](summary.json) · [协议、日志、checkpoint SHA256](audit.json) · [全部 episode 记录](episodes.jsonl.gz)','',
            '## A：流形与新执行、同初态同步消融','',
            '固定种子 20260928，六个预先指定来源族各取最早测试窗口。保留失败和相同输出，不人为放大动作。青色为环境形状约束，橙色为实际身体包络；椭球以骨盆为中心，不是固定世界物理障碍物。','',
            '| 来源 | A：流形 / 模型执行 | B：四方法同步对照 |','|---|---|---|']
    for f,i,a,b in gallery:lines.append(f'| {f} / {i} | [![]({a})]({a}) | [![]({b})]({b}) |')
    lines+=['','## 复现','',
            '依赖现有 WSL 环境、SONIC ONNX、本地 SEED 派生 NPZ 和 v3 模仿权重。三个种子最多并行三个进程，单进程两线程；不修改默认在线控制器。','',
            '```bash','seed=20260928',
            'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ./scripts/python.sh -m manifold_motion.stage1.mixed_suite --seed "$seed"',
            'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ./scripts/python.sh -m manifold_motion.stage1.mixed_evaluate --seed "$seed"',
            '# 另外两个种子同样运行，然后：',
            'MUJOCO_GL=egl ./scripts/python.sh -m manifold_motion.stage1.mixed_report','```','',
            '精确相同动作的确定性回放会复用缓存，并记录 execution_source；代表窗口仍重新执行并与旧指标逐项核对。缓存不参与动作选择。',
            '训练模型保存于 `reports/manifold_motion/stage1_mixed_v5/seed_<seed>/<variant>/residual.pt`；不会覆盖 v3/v4。','']
    text='\n'.join(lines);(args.out/'README.md').write_text(text,encoding='utf-8')
    import markdown
    html=markdown.markdown(text,extensions=['tables','fenced_code'])
    (args.out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Stage 1 A/B final matrix</title><style>body{max-width:1440px;margin:28px auto;padding:24px;font-family:Arial,"Microsoft YaHei",sans-serif;line-height:1.65;color:#20354b;background:#f4f7fa}h1,h2{color:#155775}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d5dfe8;padding:10px}th{background:#eaf1f6}img{max-width:100%}pre{overflow:auto;background:#162c42;color:white;padding:14px}</style>'+html+'</html>',encoding='utf-8')
    print('REPORT COMPLETE',args.out,flush=True)


if __name__=='__main__':main()
