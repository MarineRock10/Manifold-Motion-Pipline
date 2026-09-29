# Stage 1：SONIC 条件残差适配器

当前仓库包含 SONIC 的 ONNX 推理图（`model_encoder.onnx` / `model_decoder.onnx`），没有可
反向传播的原始训练图。因此本实验不声称“重训 SONIC 权重”，而是采用可部署的
zero-initialized、bounded residual adapter：冻结 SONIC 基础动作，只训练一个读取状态、
历史、`M_e`/SDF、command 和 primitive 条件的低秩残差。残差为零时与原 SONIC 位级一致，
可以安全回滚。

## 已有动作对齐训练

训练入口是 `manifold_motion.stage2.train_sonic_adapter`，数据必须包含真实冻结 SONIC
输出 `base_action` 和接受的教师 `target_action`，不会从关节参考凭空伪造动作标签。

| split | frozen base MSE | adapter MSE | change |
|---|---:|---:|---:|
| train (828) | 1.8150 | 1.7421 | -4.0% |
| validation (582) | 1.5555 | 1.4740 | -5.2% |
| test (2744) | 5.7329 | 5.6831 | -0.9% |

已有权重和审计记录位于本地 `reports/manifold_motion/targeted_sonic_adapter_v2/`；权重不放入
GitHub。`zero_init_base_parity_max_abs = 0.0`，且使用按动作族的局部身体 mask。

## 与本次 Stage 1 的关系

本页适配器不替换 Stage 1 时序原语头，也不把确认集结果泄漏进 SONIC 训练。当前推荐部署
顺序是：`M_e(t) → temporal primitive → frozen SONIC → bounded residual adapter → safety gate`。
若需要真正微调 SONIC 本体，必须补充其训练图、动作对齐的实机/仿真数据以及完整的物理
验收矩阵；在这些条件出现前，adapter 是唯一可审计的增量路径。

## 复现

```bash
./scripts/python.sh -m manifold_motion.stage2.train_sonic_adapter \
  --dataset reports/manifold_motion/seed_failure_supplement_v1/adapter_dataset_v2.npz \
  --out reports/manifold_motion/targeted_sonic_adapter_v2 --epochs 25 --device cpu
```
