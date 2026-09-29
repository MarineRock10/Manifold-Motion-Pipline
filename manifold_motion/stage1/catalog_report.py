"""Render learned Stage-1 executions and summarize the recorded A/B protocol.

No data-pair GIF is reused: every animation reads a new SONIC rollout trace.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

from manifold_motion.core import constants as C
from manifold_motion.core.body_envelope import body_points, fit_ellipsoid
from manifold_motion.stage1.catalog_experiment import (
    Executor, Protocol, aggregate, load_data, predict, selected_rows, write_json,
)
from manifold_motion.stage1.manifold_imitation import _mlp, manifold_features
from manifold_motion.simulation.env import G1FlatEnv
from manifold_motion.visualization.manifold_action_pairs import (
    _add_ellipsoid, _add_line, _wire_ellipsoid, _render_scene, _panel, _save_gif, FAMILY_ZH,
)

LABELS={'full':'IL + RL (validation selected)', 'no_finetune':'IL only / no RL',
        'no_imitation':'RL only / no IL pretrain', 'no_geometry':'IL / no geometry',
        'family_mean':'Oracle family mean', 'recorded_target':'Recorded target / oracle'}


def load_trace(path):
    with np.load(path) as a:return {k:a[k] for k in a.files}


def render_traces(traces, names, rows, out, title, fps=25, manifold_first=False):
    env=G1FlatEnv()
    physical_groups=env.model.geom_group.copy()
    env.model.geom_group[(env.model.geom_bodyid!=0)&(env.model.geom_group!=1)]=3
    floor=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    env.model.geom_matid[floor]=-1;env.model.geom_rgba[floor]=[.74,.79,.84,1]
    width,height=420,340
    env.model.vis.global_.offwidth=width;env.model.vis.global_.offheight=height
    renderer=mujoco.Renderer(env.model,height=height,width=width)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    top=max(float(np.max(t['pos'][:,2])+t['semi'][2]) for t in traces)
    camera.lookat[:]=[0,0,top/2];camera.distance=max(3.15,top/.63);camera.azimuth=135;camera.elevation=-16
    option=mujoco.MjvOption();mujoco.mjv_defaultOption(option)
    option.geomgroup[:]=0;option.geomgroup[:2]=1
    count=min(len(t['q']) for t in traces)
    frames=[];columns=2 if len(traces)>2 else len(traces)
    display_frames=list(range(0,count,1 if count<=40 else 2))
    try:
        for frame in display_frames:
            panels=[]
            for j,(trace,name,row) in enumerate(zip(traces,names,rows)):
                pos=trace['pos'][frame];quat=trace['quat'][frame]
                rot=C.quat_to_matrix(quat);yaw=math.atan2(rot[1,0],rot[0,0])
                heading=np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1]])
                def draw(scene):
                    for offset in np.arange(-2,2.01,.5):
                        _add_line(scene,np.array([-2,offset,.003]),np.array([2,offset,.003]),(.4,.5,.6,1),.002)
                        _add_line(scene,np.array([offset,-2,.003]),np.array([offset,2,.003]),(.4,.5,.6,1),.002)
                    _wire_ellipsoid(scene,pos,trace['semi'],heading,(.03,.60,.82,1),.005)
                    if manifold_first and j==0:
                        _add_ellipsoid(scene,pos,trace['semi'],heading,(.06,.70,.9,.10))
                        # Rendering suppresses duplicate collision surfaces in group 3.
                        # The envelope sampler needs their original group-0 contract;
                        # changing groups here does not change the already-built scene.
                        render_groups=env.model.geom_group.copy()
                        env.model.geom_group[:]=physical_groups
                        try:
                            local=(body_points(env.model,env.data)-pos)@heading
                        finally:
                            env.model.geom_group[:]=render_groups
                        self_semi=np.asarray(fit_ellipsoid(local,center=np.zeros(3))['semi'])
                        _wire_ellipsoid(scene,pos,self_semi,heading,(.95,.4,.08,1),.006)
                image=_render_scene(env,renderer,camera,option,trace['q'][frame],pos,quat,
                                    hide_robot=bool(manifold_first and j==0),draw=draw)
                status='PASS' if row.get('pose_success') else 'FAIL'
                color='#79dec9' if row.get('pose_success') else '#ffb18d'
                text=[f"{status} | rho={row['radius']:.2f} | target MAE={row['target_mae']:.3f} rad",
                      f"drift={row['drift_m']:.3f} m | fall={int(row['fallen'])} | 0.5x replay"]
                image=_panel(image,name,f"{title} | t={trace['t'][frame]:.2f}s",text,(frame+1)/count,color)
                panels.append(image)
            composite=Image.new('RGB',(width*columns,panels[0].height*math.ceil(len(panels)/columns)),'#101f31')
            for j,im in enumerate(panels):composite.paste(im,((j%columns)*width,(j//columns)*im.height))
            frames.append(composite)
    finally:renderer.close()
    _save_gif(frames,out,fps)
    sheet=Image.new('RGB',(frames[0].width,frames[0].height*3))
    for i,k in enumerate((0,len(frames)//2,len(frames)-1)):sheet.paste(frames[k],(0,i*frames[0].height))
    sheet.save(out.with_suffix('.jpg'),quality=88)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('reports/manifold_motion/stage1_ab_response_v3'))
    p.add_argument('--out',type=Path,default=Path('docs/experiments/stage1_ab_response_v3'))
    p.add_argument('--catalog',type=Path,default=Path('data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz'))
    p.add_argument('--seed',type=int,default=20260928)
    p.add_argument('--long-check',action='store_true')
    p.add_argument('--summary-only',action='store_true',
                   help='Refresh the report from existing media/audits, without new rollouts')
    args=p.parse_args();torch.set_num_threads(2)
    data=load_data(args.catalog)
    args.out.mkdir(parents=True,exist_ok=True)
    media=args.out/'media';media.mkdir(exist_ok=True)
    summaries={}
    for folder in sorted(args.root.glob('replicate_*')):
        if not (folder/'complete.json').is_file():raise RuntimeError(f'incomplete experiment: {folder}')
        summaries.update(json.loads((folder/'summary.json').read_text()))
    if len(summaries)!=3:raise RuntimeError('expected three completed training seeds')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_methods=['full','no_finetune','no_imitation','no_geometry','family_mean','recorded_target']
    plot_labels=['IL + RL*','IL only','RL only','No M (IL)','Family mean','Target oracle']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,metric,title,scale in zip(axes,['offline_mae','target_mae','pose_success'],
                                    ['Requested pose MAE (rad) - lower is better',
                                     'Executed target MAE (rad) - lower is better',
                                     'Short-response pose success (%)'],[1,1,100]):
        values=np.array([[s['methods'][m][metric] for s in summaries.values()] for m in plot_methods])*scale
        ax.bar(np.arange(6),values.mean(1),yerr=values.std(1,ddof=1),capsize=4,
               color=['#174f72','#2b7a9c','#8a9eac','#b1bec8','#66757f','#c7ced3'])
        ax.set_xticks(np.arange(6),plot_labels,rotation=25,ha='right',fontsize=9)
        ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Stage 1 | 3 training seeds x 727 held-out windows per method\nError bars: seed SD; * validation may retain the pre-RL checkpoint',fontsize=12)
    fig.savefig(args.out/'ab_metrics.png',dpi=160);plt.close(fig)
    seed_dir=args.root/f'replicate_{args.seed}'/f'seed_{args.seed}'
    methods=list(summaries[str(args.seed)]['methods'])
    raw={m:{int(r['row']):r for r in [json.loads(s) for s in (seed_dir/(m+'_test.jsonl')).read_text().splitlines()]} for m in methods}
    display=selected_rows(np.flatnonzero(data['split']==2),data['primitive'])
    gallery=[]
    for row in display:
        family=str(data['primitive_names'][data['primitive'][row]])
        trace=load_trace(seed_dir/'traces'/f'full_{row:06d}.npz')
        path=media/f'A_{row:06d}_{family}.gif'
        if not args.summary_only:
            render_traces([trace,trace],['M_e + measured M_self','Learned target -> SONIC'],[raw['full'][int(row)]]*2,path,
                          f'{family} / row {row}',manifold_first=True)
        main_methods=['full','no_finetune','no_imitation','no_geometry']
        traces=[load_trace(seed_dir/'traces'/f'{m}_{row:06d}.npz') for m in main_methods]
        bpath=media/f'B_{row:06d}_{family}.gif'
        if not args.summary_only:
            render_traces(traces,[LABELS[m] for m in main_methods],[raw[m][int(row)] for m in main_methods],bpath,
                          f'{family} / row {row}')
        if not path.is_file() or not bpath.is_file():
            raise FileNotFoundError('missing gallery media; run without --summary-only')
        gallery.append(dict(row=int(row),family=family,A=str(path.relative_to(args.out)),B=str(bpath.relative_to(args.out))))
        print('Verified A/B' if args.summary_only else 'Rendered A/B',family,flush=True)

    # A2 is an explicit geometry intervention on a fixed source window/initial state.
    checkpoint=torch.load(seed_dir/'full.pt',weights_only=False,map_location='cpu')
    model=_mlp(torch,22,29,int(checkpoint['hidden']));model.load_state_dict(checkpoint['model']);model.eval()
    executor=None if args.summary_only else Executor(Protocol(warmup_ticks=10,transition_ticks=10,hold_ticks=20))
    anchor=int(display[0]);interventions=[];c_traces=[]
    for name,scale in [('original',[1,1,1]),('narrow_x',[.7,1,1]),('narrow_y',[1,.7,1]),('low_z',[1,1,.7])]:
        if args.summary_only:
            continue
        corridor=data['corridor'][anchor:anchor+1].copy();corridor[:,:,3:6]*=np.asarray(scale)
        feature=(manifold_features(corridor)-data['mean'])/data['std']
        pose=predict(model,feature)[0];semi=corridor[0,data['frame'],3:6]
        result,trace=executor.run(pose,semi,data['y'][anchor],capture=True)
        interventions.append(dict(name=name,scale=scale,predicted_delta=pose.tolist(),**result));c_traces.append(trace)
    if not args.summary_only:
        render_traces(c_traces,[r['name'] for r in interventions],interventions,media/'A2_geometry_intervention.gif','fixed initial state; input M changed')
        write_json(args.out/'counterfactual.json',{'anchor':anchor,'rows':interventions,
                      'warning':'30% shrink is an out-of-distribution stress test; target MAE refers to the unchanged source pose, not a counterfactual label'})
    elif not (args.out/'counterfactual.json').is_file() or not (media/'A2_geometry_intervention.gif').is_file():
        raise FileNotFoundError('missing A2 audit; run without --summary-only')

    long_result=None
    if args.summary_only and (args.out/'long_hold_audit.json').is_file():
        long_result=json.loads((args.out/'long_hold_audit.json').read_text())['summary']
    elif args.summary_only and args.long_check:
        raise FileNotFoundError('missing long audit; run without --summary-only')
    elif args.long_check:
        # One predeclared seed and every represented test family; no success-based selection.
        long_cfg=Protocol(warmup_ticks=20,transition_ticks=20,hold_ticks=150)
        long_executor=Executor(long_cfg);long_rows=[]
        for method in methods:
            for row in display:
                source=load_trace(seed_dir/'traces'/f'{method}_{row:06d}.npz')
                result,_=long_executor.run(source['requested_delta'],source['semi'],source['target_delta'])
                long_rows.append(dict(method=method,row=int(row),family=str(data['primitive_names'][data['primitive'][row]]),**result))
            print('3-second hold audit',method,flush=True)
        long_result={m:aggregate([r for r in long_rows if r['method']==m]) for m in methods}
        write_json(args.out/'long_hold_audit.json',{'seed':args.seed,'selection':'first test window per family','hold_seconds':3.,
                                                 'warmup_seconds':.4,'transition_seconds':.4,'action_seconds':3.4,
                                                 'summary':long_result,'rows':long_rows})

    # Explicit single-factor contrast for the no-geometry imitation control.
    comparisons={}
    for seed,summary in summaries.items():
        comparisons[seed]=summary['paired_actor_bootstrap']
    write_json(args.out/'summary.json',summaries)
    # Small reviewable evidence is versionable; controller/data weights remain local.
    training_audit={}
    for seed in summaries:
        run_dir=args.root/f'replicate_{seed}';sd=run_dir/f'seed_{seed}'
        training_audit[seed]={
            'protocol':json.loads((run_dir/'protocol.json').read_text()),
            'imitation_history':json.loads((sd/'imitation_history.json').read_text()),
            'full_history':json.loads((sd/'full_history.json').read_text()),
            'no_imitation_history':json.loads((sd/'no_imitation_history.json').read_text()),
        }
    write_json(args.out/'training_audit.json',training_audit)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for seed,audit in training_audit.items():
        history=audit['imitation_history']
        axes[0].plot([r['epoch'] for r in history],[r['validation_mae'] for r in history],label=seed)
        for ax,field in zip(axes[1:],['full_history','no_imitation_history']):
            values=[r for r in audit[field] if 'validation' in r]
            ax.plot([r['iteration'] for r in values],[r['validation']['reward'] for r in values],marker='o',label=seed)
    for ax,title,xlabel in zip(axes,['Imitation: validation MAE (lower better)',
                                   'IL + RL: validation reward (higher better)',
                                   'RL only: validation reward (higher better)'],['Epoch','Iteration','Iteration']):
        ax.set_title(title,fontsize=10);ax.set_xlabel(xlabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(args.out/'training_curves.png',dpi=160);plt.close(fig)
    lines=['# Stage 1：训练与实验 A / B','',
           '这是静态姿态模型的首次三种子完整短时响应实验，不是完整步态、跳跃或导航的成功证明。',
           'SONIC ONNX 冻结；训练和 RL 微调均作用于上层 29 关节目标模型。','',
           '**本轮结论：** 模仿预训练与流形输入带来更好的目标姿态匹配；当前小预算 RL 未在验证集上超越模仿权重。不能据此宣称 RL 有效，也不能据此判定更充分的 RL 一定无效。','',
           '## 协议','',
           '每种子：2,610 个训练窗口、651 个验证窗口、727 个测试窗口。测试集实际覆盖 19 个来源动作族；全库为 21 个。',
           '所有方法从相同站立状态开始：预热 0.2 秒，过渡 0.2 秒，保持 0.4 秒。短时通过率不等于长期稳定率。',
           '几何检查为单帧、骨盆中心、航向对齐的椭球形状适配，保留真实俯仰/翻滚；根平移另外按漂移统计，不是固定世界走廊通过率。',
           'RL 仅从 train 获取奖励，validation 选择 checkpoint（允许保留 iteration 0）。test 不参与选模。',
           '全方法奖励一致，包括目标姿态误差；“无模仿”指去掉模仿预训练，而不是去掉所有数据监督信息。','',
           '几何半径由实际执行姿态的采样网格表面计算，并非连续全表面碰撞安全证明；场景只有地面，障碍物接触指标不适用。',
           '无几何对照在相同训练集、网络结构和训练时长下重新训练，而非只在推理时遮蔽输入。','',
           '训练预算：每种子模仿 80 epochs；每个 RL 分支 20 × 24 = 480 个采样训练 episode，每 5 轮用验证集分层子集选模。此实现每批仅更新一次，是单步 contextual-bandit policy gradient；裁剪 PPO 目标在首次更新前比值为 1，不能等同于充分优化的多 epoch PPO。训练曲线与完整协议保存在 [training_audit.json](training_audit.json)。','',
           '![](training_curves.png)','',
           '## 实验 B：完整测试集','', '![](ab_metrics.png)','',
           '均值 ± 种子标准差；所有失败仍计入分母。Pose success = 几何/身体触地/跌倒/漂移门控通过且平均目标关节误差 ≤ 0.30 rad。','',
           '| 方法 | 离线 MAE ↓ | 执行目标 MAE ↓ | 短时门控通过 ↑ | Pose success ↑ | 跌倒率 ↓ |',
           '|---|---:|---:|---:|---:|---:|']
    def cell(method,metric,percent=False):
        values=np.array([s['methods'][method][metric] for s in summaries.values()])* (100 if percent else 1)
        return f'{values.mean():.2f} ± {values.std(ddof=1):.2f}%' if percent else f'{values.mean():.3f} ± {values.std(ddof=1):.3f}'
    for m in methods:
        lines.append('| '+LABELS[m]+' | '+' | '.join(cell(m,k,k in ('accepted','pose_success','fallen')) for k in ('offline_mae','target_mae','accepted','pose_success','fallen'))+' |')
    lines+=['','对照解释：Full vs IL-only 检查 RL；Full vs RL-only 检查模仿预训练。No-geometry 是 IL-only 的单因素对照，不能把它与 Full 的差异全部归因于几何输入。Family mean 使用真实动作族标签，Recorded target 使用真实目标，二者是特权参考，不是公平部署模型。','',
            '置信区间见 summary.json：按 actor 成簇的 paired bootstrap，避免把同一演员的重叠窗口当作独立样本。inference_ms 为 CPU 批量推理摊销耗时，不是端到端控制延迟。','',
            '### 验证集选模','', '| Seed | Full 选择的 RL 迭代 | RL-only 选择的迭代 |','|---|---:|---:|']
    for seed in summaries:
        sd=args.root/f'replicate_{seed}'/f'seed_{seed}'
        f=torch.load(sd/'full.pt',weights_only=False);n=torch.load(sd/'no_imitation.pt',weights_only=False)
        lines.append(f'| {seed} | {f["best_iteration"]} | {n["best_iteration"]} |')
    lines+=['','如果选择 iteration 0，Full 实际保留模仿权重；对应 GIF 与 IL-only 相同是负结果，不是换皮展示。','',
            '## 实验 A：不同流形的真实执行','',
            f'固定展示 seed={args.seed}，每个测试动作族按源索引取第一窗，未按成功挑选。右侧是模型目标经过 SONIC/MuJoCo 后的新执行，不是 SEED 原动作回放。',
            '动作族名称仅表示源数据类别。模型目前只输出中帧姿态，不意味着已执行跑、跳、爬等完整动作；PASS 仅指上面的姿态指标，不是动作识别或任务成功。',
            'A：左流形、右执行；B：同流形下四方法（Full / IL-only / RL-only / no-geometry IL）的同步执行。FAIL 画面保留。','',
            '| 来源族 | A：流形 → 执行 | B：消融对照 |','|---|---|---|']
    for e in gallery:lines.append(f'| {FAMILY_ZH.get(e["family"],e["family"])} · row {e["row"]} | [![]({e["A"]})]({e["A"]}) | [![]({e["B"]})]({e["B"]}) |')
    lines+=['','## A2：固定初态，仅修改流形','',
            '同一源窗口：原始 / 纵向收缩 30% / 横向收缩 30% / 竖直收缩 30%。这是分布外压力测试，不保证可行；几何未改变之外的状态与模型保持一致。',
            '![](media/A2_geometry_intervention.gif)','']
    if long_result:
        lines+=['## 补充：3 秒保持审查','', '预热 0.4 秒、过渡 0.4 秒、保持 3 秒；动作观测共 3.4 秒。固定同一种子、19 个测试来源族各一窗。此项为诊断子集，不能替代全部测试集的长时稳定评估。完整逐窗记录见 [long_hold_audit.json](long_hold_audit.json)。','',
                '| 方法 | 样本 | 门控通过 | Pose success | 跌倒率 |','|---|---:|---:|---:|---:|']
        for m,r in long_result.items():lines.append(f'| {LABELS[m]} | {r["rows"]} | {r["accepted"]:.1%} | {r["pose_success"]:.1%} | {r["fallen"]:.1%} |')
    lines+=['','## 复现','',
            '需要已配置的 WSL / SONIC / MuJoCo 环境，以及 [数据准备协议](../MANIFOLD_ACTION_CATALOG.md) 中的本地 NPZ 和原始 actor 元数据。下列命令不负责下载受许可约束的 SEED 数据。',
            '权重保存在 `reports/manifold_motion/stage1_ab_response_v3/replicate_<seed>/seed_<seed>/`：`imitation.pt`、`full.pt`、`no_imitation.pt` 和 `no_geometry/imitation.pt`。全量逐窗口测试记录与回放状态保存在同目录。','', '```bash',
            'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ./scripts/python.sh -m manifold_motion.stage1.catalog_experiment \\',
            '  --profile short_response --seeds 20260928 --out reports/manifold_motion/stage1_ab_response_v3/replicate_20260928',
            '# 对 20260929、20260930 重复上述命令并替换 seed 和输出目录。',
            'MUJOCO_GL=egl ./scripts/python.sh -m manifold_motion.stage1.catalog_report --long-check','```','',
            '旧 v1 gate 的绝对角/增量、执行时长与样本覆盖问题使其数字不可与本协议混用。长保持 v2 训练记录作为中断的诊断保留，不作为完成实验。']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')
    import markdown
    page=markdown.markdown('\n'.join(lines),extensions=['tables','fenced_code'])
    (args.out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Stage 1 A/B</title><style>body{max-width:1400px;margin:30px auto;font-family:Arial,"Microsoft YaHei",sans-serif;padding:20px;color:#223345;background:#f7f9fc;line-height:1.7}h1,h2{color:#153c60}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border:1px solid #dce4eb}th{background:#eaf1f7}img{max-width:100%;height:auto}pre{overflow:auto;background:#132637;color:#eef7ff;padding:16px}code{font-size:.9em}</style>'+page+'</html>')
    print('Report complete',args.out,flush=True)


if __name__=='__main__':main()
