# Stage 2 第一项：Stage-1 概率路由接入动态轨迹模型

Stage 2 不再把数据集中的 primitive 标签直接传给动态模型。新增
`manifold_motion.stage2.stage1_router_bridge`，加载 Stage 1 的时序原语 checkpoint，从
`M_e(t)` 预测 30 维动作族概率，并用该概率向量替换 Stage 2 条件中的 oracle one-hot。

当前训练链路为：

```text
M_e(t) → Stage-1 temporal primitive probabilities
      + measured state + history + SDF/corridor/command
      → Stage-2 conditional trajectory mean
      → later Flow Matching / candidate safety gate
```

## 首轮结果

训练数据使用 2,610 train、651 validation、727 test 窗口；目标是 `target_exec`，即
冻结 SONIC 执行参考。模型没有读取 `primitive` 标签。

| split | joint MAE | root position MAE | root rotation-6D MAE |
|---|---:|---:|---:|
| train | 0.0742 rad | 0.0187 m | 0.0213 |
| validation | 0.1087 rad | 0.0249 m | 0.0359 |
| test | 0.1379 rad | 0.0303 m | 0.0421 |

路由概率平均熵为 `1.1651`，说明动态模型收到的是有不确定性的预测分布，而不是被
强制喂入真实动作类别。

## 复现

```bash
./scripts/python.sh -m manifold_motion.stage2.router_conditioned_mean \
  --windows reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows.npz \
  --stage1-checkpoint reports/manifold_motion/stage1_temporal_primitive_v11/temporal_primitive.pt \
  --out reports/manifold_motion/stage2_router_conditioned_mean_v1 \
  --model-target-field target_exec --epochs 35 --root-weight 2.0
```

模型权重和完整训练曲线在本地 `reports/manifold_motion/stage2_router_conditioned_mean_v1/`，
GitHub 只保留本页和可复现入口，不提交二进制 checkpoint。

## 尚未声称的内容

这是 Stage 2 的第一项接线和确定性均值基线，还不是完整 Flow Matching，也没有把候选
轨迹送入在线 D*/ESDF 重规划和长时物理验收。下一项是：在这个无 oracle primitive 的
条件上训练 residual/latent Flow，并接入已有的身体流形、碰撞和根轨迹门控。
