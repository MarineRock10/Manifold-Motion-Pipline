"""Audit and visualize the bounded-residual RL revision without relabeling runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from manifold_motion.stage1.catalog_experiment import (
    Executor,Protocol,aggregate,bootstrap_actor_difference,load_data,selected_rows,write_json,
)
from manifold_motion.stage1.catalog_report import load_trace,render_traces
from manifold_motion.stage1.residual_rl import sha256


METHODS=('il_only','il_residual_rl')
NAMES={'il_only':'Imitation only','il_residual_rl':'Imitation + residual PPO'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('reports/manifold_motion/stage1_residual_rl_v4'))
    p.add_argument('--out',type=Path,default=Path('docs/experiments/stage1_residual_rl_v4'))
    p.add_argument('--catalog',type=Path,default=Path('data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz'))
    p.add_argument('--metadata',type=Path,default=Path('reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows_metadata.json'))
    p.add_argument('--long-check',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=True);media=args.out/'media';media.mkdir(exist_ok=True)
    summaries={};histories={};audit={};raw={}
    for folder in sorted(args.root.glob('seed_*')):
        seed=folder.name.split('_')[1]
        if not (folder/'test_summary.json').exists():
            raise RuntimeError(f'incomplete held-out test: {folder}')
        summaries[seed]=json.loads((folder/'test_summary.json').read_text())
        histories[seed]=json.loads((folder/'history.json').read_text())
        protocol=json.loads((folder/'protocol.json').read_text())
        model=torch.load(folder/'residual.pt',weights_only=False,map_location='cpu')
        if model['base_sha256']!=protocol['base_sha256'] or sha256(protocol['base_path'])!=protocol['base_sha256']:
            raise ValueError('base checkpoint provenance mismatch')
        audit[seed]=dict(protocol=protocol,training=json.loads((folder/'train_complete.json').read_text()),
                         evaluation=json.loads((folder/'evaluation_protocol.json').read_text()),
                         residual_checkpoint_sha256=sha256(folder/'residual.pt'),history=histories[seed])
        assert audit[seed]['residual_checkpoint_sha256']==audit[seed]['evaluation']['checkpoint_sha256']
        raw[seed]={m:{r['row']:r for r in map(json.loads,(folder/(m+'_test.jsonl')).read_text().splitlines())} for m in METHODS}
        previous=Path(protocol['base_path']).parent/'no_finetune_test.jsonl'
        if previous.exists():
            old={r['row']:r for r in map(json.loads,previous.read_text().splitlines())}
            keys=('target_mae','accepted','pose_success','radius','drift_m','fallen')
            discrepancy=max(abs(float(raw[seed]['il_only'][i][k])-float(old[i][k]))
                            for i in old for k in keys)
            if discrepancy>1e-8:
                raise RuntimeError('IL baseline differs from v3; investigate executor drift before comparison')
            audit[seed]['baseline_max_abs_metric_difference_vs_v3']=discrepancy
    if len(summaries)!=3:raise RuntimeError('need three completed seeds')
    data=load_data(args.catalog);ids=np.flatnonzero(data['split']==2)
    for seed in summaries:
        for m in METHODS:
            assert set(raw[seed][m])==set(ids.tolist()),'missing or extra test rows'
    mean_rows={m:{int(i):{k:float(np.mean([raw[s][m][int(i)][k] for s in summaries]))
                        for k in ('pose_success','accepted','target_mae')} for i in ids} for m in METHODS}
    actors=[str(c['actor_uid']) for c in json.loads(args.metadata.read_text())['clips']]
    paired=bootstrap_actor_difference(mean_rows['il_residual_rl'],mean_rows['il_only'],actors,data['clip_index'],20260928)
    write_json(args.out/'summary.json',dict(seeds=summaries,seed_mean_paired_actor_bootstrap=paired))
    write_json(args.out/'training_audit.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,metric,title,scale in zip(axes,['target_mae','pose_success','accepted'],
                                   ['Executed target MAE (rad)', 'Short-response pose success (%)','Shape/physics gate (%)'],[1,100,100]):
        values=np.array([[summaries[s]['methods'][m][metric] for s in summaries] for m in METHODS])*scale
        ax.bar([0,1],values.mean(1),yerr=values.std(1,ddof=1),capsize=4,color=['#879bad','#136585'])
        ax.set_xticks([0,1],['Imitation','Imitation + residual RL']);ax.set_title(title,fontsize=10)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Unchanged evaluation metrics | 3 seeds x 727 held-out windows',fontsize=12)
    fig.savefig(args.out/'metrics.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for seed,hist in histories.items():
        values=[r for r in hist if 'validation' in r]
        for ax,key in zip(axes,['dense_reward','target_mae']):
            ax.plot([r['iteration'] for r in values],[r['validation'][key] for r in values],marker='o',label=seed)
    for ax,title in zip(axes,['Validation dense reward (higher better)','Validation target MAE (lower better)']):
        ax.set_title(title);ax.set_xlabel('RL iteration');ax.legend();ax.grid(alpha=.2)
    fig.savefig(args.out/'training.png',dpi=160);plt.close(fig)
    family_names=sorted(summaries[next(iter(summaries))]['methods']['il_only']['by_family'])
    family_delta=[np.mean([summaries[s]['methods']['il_residual_rl']['by_family'][f]['target_mae']-
                           summaries[s]['methods']['il_only']['by_family'][f]['target_mae'] for s in summaries]) for f in family_names]
    fig,ax=plt.subplots(figsize=(9,7),constrained_layout=True)
    ax.barh(family_names,family_delta,color=['#136585' if v<0 else '#c47356' for v in family_delta])
    ax.axvline(0,color='#333',linewidth=1);ax.set_xlabel('Residual RL minus imitation: target MAE (rad); negative is better')
    ax.set_title('All test families, including regressions');fig.savefig(args.out/'family_delta.png',dpi=150);plt.close(fig)
    seed='20260928';folder=args.root/f'seed_{seed}'
    display=selected_rows(ids,data['primitive'])
    wanted={'walk_lateral','turn_in_place','crouch_walk','forward_lunge','all_fours','carry_object'}
    gallery=[]
    for row in display:
        family=str(data['primitive_names'][data['primitive'][row]])
        if family not in wanted:continue
        path=media/f'compare_{row:06d}_{family}.gif'
        if not args.summary_only:
            traces=[load_trace(folder/'traces'/f'{m}_{row:06d}.npz') for m in METHODS]
            render_traces(traces,[NAMES[m] for m in METHODS],[raw[seed][m][int(row)] for m in METHODS],path,
                          f'{family} / row {row} / held-out')
        if not path.is_file():raise FileNotFoundError(path)
        gallery.append((family,int(row),str(path.relative_to(args.out))))
        print('Comparison ready',family,flush=True)
    long_path=args.out/'long_hold.json';long_result=None
    if args.summary_only and long_path.exists():
        long_result=json.loads(long_path.read_text())
    elif args.long_check:
        executor=Executor(Protocol(warmup_ticks=20,transition_ticks=20,hold_ticks=150))
        long_rows=[]
        for seed in summaries:
            for m in METHODS:
                for row in display:
                    trace=load_trace(args.root/f'seed_{seed}'/'traces'/f'{m}_{row:06d}.npz')
                    r,_=executor.run(trace['requested_delta'],trace['semi'],trace['target_delta'])
                    long_rows.append(dict(seed=seed,method=m,row=int(row),**r))
                print('Long hold audit',seed,m,flush=True)
        long_result=dict(warmup_seconds=.4,transition_seconds=.4,hold_seconds=3.,rows=long_rows,
                         summary={m:aggregate([r for r in long_rows if r['method']==m]) for m in METHODS})
        write_json(long_path,long_result)
    def cell(m,key,percent=False):
        values=np.array([s['methods'][m][key] for s in summaries.values()])*(100 if percent else 1)
        return (f'{values.mean():.2f} ± {values.std(ddof=1):.2f}%' if percent else
                f'{values.mean():.4f} ± {values.std(ddof=1):.4f}')
    lines=['# Stage 1：有界残差 RL 优化','',
           '这是对 [v3 负结果](../stage1_ab_response_v3/README.md) 的后续探索性算法修订。旧结果保留，不重命名、不用其他方法的输出冒充 RL。SONIC 仍然冻结。','',
           '**使用状态：研究用候选，未替换默认策略。** 短时目标误差改善不能抵消安全指标退化；特别要查看下方长保持诊断，不能仅挑选短时图表宣称全面提升。','',
           '## 算法与奖励','',
           '- 冻结模仿网络；以其输出为基线，用仅由训练集构造的 12 维 PCA 基底表达关节修正，每个关节修正严格限制在 ±0.35 rad。该基底只是残差参数化，不是已放弃的 latent prior 路线。',
           '- 同一流形采样 4 个独立候选，以其余候选奖励均值为基线；减少不同任务难度混入优势估计。',
           '- 每批执行 4 次 PPO 更新；KL 超过 0.03 时回滚超限的一步。',
           '- 奖励直接使用真实执行后的目标误差，移除可能阻碍误差补偿的指令跟踪惩罚；连续惩罚越界和漂移，单独惩罚跌倒/非脚触地。',
           '- 每种子 60 × 8 × 4 = 1,920 个训练 episode。验证集选模要求实际目标 MAE 不恶化，门控通过率下降不超过预先固定的 2 个百分点；测试指标和阈值不变。','',
           '```text',
           'r = -5 * executed_target_MAE -10 * max(radius - 0.95, 0)^2',
           '    -0.25 * (min(drift, 1)/0.35)^2 -4 * fall -1 * nonfoot_contact',
           '```','',
           '本轮同时修改残差结构、奖励与更新方式，不能将变化单独归因于某一项。旧测试集已被查看过；本轮虽未用其选模，仍应作为开发集上的探索性复测，论文定稿需要新增未看过的 actor/场景确认集。','',
           '## 训练与来源审计','', '![](training.png)','',
           '[完整协议、训练曲线数值与权重 SHA256](training_audit.json)。更新前为零残差，等价于原始模仿策略；零迭代权重允许被保留，绝不保证训练一定提升。','',
           '| Seed | 验证集选择的 RL 迭代 | 训练 episode |','|---|---:|---:|']
    for seed in summaries:
        lines.append(f'| {seed} | {summaries[seed]["best_iteration"]} | {audit[seed]["training"]["training_episodes"]} |')
    lines+=['','## B：同指标、同初态的配对测试','',
            '每方法、每种子 727 个窗口。预热 0.2 秒，过渡 0.2 秒，保持 0.4 秒。成功定义仍为形状/触地/跌倒/漂移门控通过且目标关节平均误差 ≤0.30 rad。','',
            '![](metrics.png)','',
            '| 方法 | 执行目标 MAE ↓ | 短时姿态成功率 ↑ | 门控通过率 ↑ | 跌倒率 ↓ |',
            '|---|---:|---:|---:|---:|']
    for m in METHODS:
        lines.append('| '+NAMES[m]+' | '+' | '.join(cell(m,k,k!='target_mae') for k in ['target_mae','pose_success','accepted','fallen'])+' |')
    lines+=['','种子均值上的 actor 成簇配对 bootstrap（RL−IL，95% CI）：','']
    for metric,r in paired.items():
        lines.append(f'- `{metric}`：差值 {r["mean_delta"]:+.5f}，CI [{r["ci95"][0]:+.5f}, {r["ci95"][1]:+.5f}]；{r["actors"]} 个测试 actor。')
    lines+=['','![](family_delta.png)','',
            '所有来源族都保留在图中，含退化项；MAE 减小与安全通过率提升不是同一结论。各种子、各来源族指标见 [summary.json](summary.json)。','',
            '## A：新执行对比（不是旧 GIF）','',
            '固定展示 seed 20260928，六个预先指定来源族各取最早测试窗口；不按成功筛选。左侧 IL，右侧 IL＋残差 RL。同一输入流形、同一初态，均为新 SONIC/MuJoCo 执行。字幕为全 episode 指标，0.5 倍速。','',
            '流形椭球仍为骨盆中心、航向对齐的形状检查，不是固定世界障碍物。来源名称不是完整动作成功证明，PASS 也不代表完成爬行/绕障任务。','',
            '| 来源 / 样本 | 同步对比 |','|---|---|']
    for family,row,path in gallery:lines.append(f'| {family} / {row} | [![]({path})]({path}) |')
    if long_result:
        lines+=['','## 长保持诊断','',
                '固定全部 19 个测试来源族各一窗 × 3 个种子；0.4 秒预热、0.4 秒过渡、3 秒保持。只是诊断子集，不替代完整长时任务测试。','',
                '| 方法 | 样本 | 门控通过 | 姿态成功 | 跌倒 |','|---|---:|---:|---:|---:|']
        for m,r in long_result['summary'].items():
            lines.append(f'| {NAMES[m]} | {r["rows"]} | {r["accepted"]:.1%} | {r["pose_success"]:.1%} | {r["fallen"]:.1%} |')
        lines+=['','[逐窗诊断记录](long_hold.json)','']
    lines+=['','## 结论边界与下一步','',
            '- 本版本比较相同模仿起点有无残差 RL；还不是新版本完整的去模仿/去几何/去新奖励消融。旧版本不同训练器的对照不能直接充当这些消融。',
            '- 优先下一步：训练和验证都加入混合执行时域（短时＋3 秒保持），采用安全优先的选模约束。当前只对短时奖励训练，不能期待长时稳定性自动改善。随后再做同预算继续模仿、旧奖励＋新更新器、新奖励＋旧更新器等对照，分离各项贡献。',
            '- 将最终方案锁定后，用新的未见 actor/场景确认集验证；不要持续根据已看过的测试集调参。',
            '- 若目标是窄通道侧身、低障碍蹲走，应使用固定世界障碍物、实际身体包络和任务进展评估；不是要求每个输入都拟合某个唯一的源姿态。爬行/跪姿的合法接触也应单独定义，不能沿用只允许脚触地的站立门控。','',
            '## 复现','',
            '使用现有 WSL、SONIC、SEED 派生 NPZ 及 actor 元数据；先完成 v3 的模仿训练。不要将更新后的权重覆盖到旧实验目录。','',
            '```bash','seed=20260928',
            'base=reports/manifold_motion/stage1_ab_response_v3/replicate_${seed}/seed_${seed}/imitation.pt',
            'out=reports/manifold_motion/stage1_residual_rl_v4/seed_${seed}',
            'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ./scripts/python.sh -m manifold_motion.stage1.residual_rl train --seed "$seed" --base "$base" --out "$out"',
            '# 训练/验证确定方案后，才执行测试：',
            'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ./scripts/python.sh -m manifold_motion.stage1.residual_rl evaluate --base "$base" --out "$out"',
            '# 对另外两个种子重复，再生成报告：',
            'MUJOCO_GL=egl ./scripts/python.sh -m manifold_motion.stage1.residual_report --long-check','```','']
    text='\n'.join(lines)
    (args.out/'README.md').write_text(text,encoding='utf-8')
    import markdown
    page=markdown.markdown(text,extensions=['tables','fenced_code'])
    (args.out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Stage 1 residual RL</title><style>body{max-width:1200px;margin:32px auto;padding:20px;font-family:Arial,"Microsoft YaHei",sans-serif;line-height:1.7;color:#21374b;background:#f5f8fc}h1,h2{color:#15536c}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border:1px solid #d8e1ec}th{background:#e6eff5}img{max-width:100%;height:auto}pre{overflow:auto;background:#15283b;color:white;padding:16px}</style>'+page+'</html>',encoding='utf-8')
    print('Report complete',args.out,flush=True)


if __name__=='__main__':main()
