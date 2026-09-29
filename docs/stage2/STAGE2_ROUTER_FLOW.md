# Stage 2：路由条件化 Flow Matching 与候选安全筛选

这一阶段把 Stage 1 的连续原语路由真正接入动态生成器。训练/采样过程不再读取窗口中的
`primitive` 标签，而是由冻结的 Stage-1 temporal checkpoint 从 `M_e(t)` 输出 30 维概率
向量 `p(z_p|M_e)`，再和 `M_self(t)`、局部 SDF、状态、history 及命令一起作为条件。

```text
M_e(t) ──> Stage-1 temporal router ──> p(z_p|M_e)
                                      + state/history/M_self/SDF/command
                                      ──> conditional VAE + latent Flow
                                      ──> K candidate R_ref
                                      ──> optional optimization projection
                                      ──> SONIC/MuJoCo hard gate and selection
```

## 首轮可复现实验

训练使用仓库内 `data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz` 的
actor-disjoint 划分（2,610/651/727）。
本轮已验证的权重位于 `models/stage1/temporal_primitive_v11.pt`、
`models/stage2/router_flow_v1/autoencoder.pt` 和 `models/stage2/router_flow_v1/flow.pt`，
因此也可以跳过训练直接运行下面的采样命令。
CPU 训练配置为 latent 32、hidden 256、25 个 VAE epoch 和 35 个 Flow epoch：

```bash
./scripts/python.sh -m manifold_motion.stage2.flow train-ae \
  --windows data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz \
  --router-checkpoint models/stage1/temporal_primitive_v11.pt \
  --out reports/manifold_motion/stage2_router_flow_v1 \
  --epochs 25 --batch-size 64 --latent-dim 32 --hidden 256 --condition-hidden 128 \
  --device cpu --seed 20261001

./scripts/python.sh -m manifold_motion.stage2.flow train-flow \
  --windows data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz \
  --router-checkpoint models/stage1/temporal_primitive_v11.pt \
  --autoencoder models/stage2/router_flow_v1/autoencoder.pt \
  --out reports/manifold_motion/stage2_router_flow_v1 \
  --epochs 35 --batch-size 64 --hidden 256 --condition-hidden 128 \
  --device cpu --seed 20261001
```

在测试窗口生成 16 条候选：

```bash
./scripts/python.sh -m manifold_motion.stage2.flow sample \
  --windows data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz \
  --autoencoder models/stage2/router_flow_v1/autoencoder.pt \
  --flow models/stage2/router_flow_v1/flow.pt \
  --router-checkpoint models/stage1/temporal_primitive_v11.pt \
  --out reports/manifold_motion/stage2_router_flow_sample_v1 \
  --split 2 --index 0 --steps 32 --num-candidates 16 --device cpu --seed 20261001
```

## 首轮筛选结果

同一组 16 条候选使用完全相同的冻结 SONIC/MuJoCo 回放和硬门：跌倒、姿态、关节范围、
跟踪、运动进度和走廊半径都在候选选择前检查。

| 版本 | 候选数 | 通过数 | 选中 | 说明 |
|---|---:|---:|---:|---|
| raw Flow | 16 | 1 | 4 | 直接对 Flow 解码结果做物理筛选 |
| Flow + optimization projection | 16 | 15 | 4 | 先做 1 次受限平滑/关节限位/走廊中心投影，再做同一物理筛选 |

投影候选 4 的审计值：目标函数 `0.000856 → 0.000196`，最大关节改变量
`0.00193 rad`，最大根位置修正 `0.05 m`；它没有替代 SONIC 回放，也没有把失败候选
直接标成成功。结果文件分别为：

![Router-conditioned Flow selected execution](../experiments/stage2_router_flow_v1/router_flow_selected.gif)

- `reports/manifold_motion/stage2_router_flow_selection_v1/selection.json`
- `reports/manifold_motion/stage2_router_flow_selection_projected_v1/selection.json`
- `reports/manifold_motion/stage2_router_flow_selection_projected_v1/selected_executed.npz`

多动作族扩展见[多动作 router/Flow 物理验证报告](STAGE2_ROUTER_FLOW_MULTIFAMILY.md)，
其中包含 5 个动作族的可预览 GIF、自动路由审计和每候选失败原因。

## 重要边界

这是一条正式接线和首轮候选筛选实验，不是最终 SOTA 声明。当前 Flow checkpoint 仍需
在更多动作族、多个场景和多随机种子上验证；下一步是将候选筛选接到长时序在线路线段，
并补充 raw/投影、oracle/router、单候选/多候选的消融矩阵。
