# Stage 1：训练与实验 A / B

这是静态姿态模型的首次三种子完整短时响应实验，不是完整步态、跳跃或导航的成功证明。
SONIC ONNX 冻结；训练和 RL 微调均作用于上层 29 关节目标模型。

**本轮结论：** 模仿预训练与流形输入带来更好的目标姿态匹配；当前小预算 RL 未在验证集上超越模仿权重。不能据此宣称 RL 有效，也不能据此判定更充分的 RL 一定无效。

## 协议

每种子：2,610 个训练窗口、651 个验证窗口、727 个测试窗口。测试集实际覆盖 19 个来源动作族；全库为 21 个。
所有方法从相同站立状态开始：预热 0.2 秒，过渡 0.2 秒，保持 0.4 秒。短时通过率不等于长期稳定率。
几何检查为单帧、骨盆中心、航向对齐的椭球形状适配，保留真实俯仰/翻滚；根平移另外按漂移统计，不是固定世界走廊通过率。
RL 仅从 train 获取奖励，validation 选择 checkpoint（允许保留 iteration 0）。test 不参与选模。
全方法奖励一致，包括目标姿态误差；“无模仿”指去掉模仿预训练，而不是去掉所有数据监督信息。

几何半径由实际执行姿态的采样网格表面计算，并非连续全表面碰撞安全证明；场景只有地面，障碍物接触指标不适用。
无几何对照在相同训练集、网络结构和训练时长下重新训练，而非只在推理时遮蔽输入。

训练预算：每种子模仿 80 epochs；每个 RL 分支 20 × 24 = 480 个采样训练 episode，每 5 轮用验证集分层子集选模。此实现每批仅更新一次，是单步 contextual-bandit policy gradient；裁剪 PPO 目标在首次更新前比值为 1，不能等同于充分优化的多 epoch PPO。训练曲线与完整协议保存在 [training_audit.json](training_audit.json)。

![](training_curves.png)

## 实验 B：完整测试集

![](ab_metrics.png)

均值 ± 种子标准差；所有失败仍计入分母。Pose success = 几何/身体触地/跌倒/漂移门控通过且平均目标关节误差 ≤ 0.30 rad。

| 方法 | 离线 MAE ↓ | 执行目标 MAE ↓ | 短时门控通过 ↑ | Pose success ↑ | 跌倒率 ↓ |
|---|---:|---:|---:|---:|---:|
| IL + RL (validation selected) | 0.194 ± 0.001 | 0.218 ± 0.000 | 96.70 ± 0.14% | 84.27 ± 0.21% | 0.00 ± 0.00% |
| IL only / no RL | 0.194 ± 0.001 | 0.218 ± 0.000 | 96.70 ± 0.14% | 84.27 ± 0.21% | 0.00 ± 0.00% |
| RL only / no IL pretrain | 0.309 ± 0.001 | 0.305 ± 0.001 | 96.42 ± 0.00% | 56.67 ± 0.14% | 0.00 ± 0.00% |
| IL / no geometry | 0.254 ± 0.000 | 0.270 ± 0.000 | 94.64 ± 0.36% | 61.49 ± 2.03% | 0.00 ± 0.00% |
| Oracle family mean | 0.206 ± 0.000 | 0.235 ± 0.000 | 87.35 ± 0.00% | 70.98 ± 0.00% | 0.00 ± 0.00% |
| Recorded target / oracle | 0.000 ± 0.000 | 0.117 ± 0.000 | 89.41 ± 0.00% | 89.41 ± 0.00% | 0.41 ± 0.00% |

对照解释：Full vs IL-only 检查 RL；Full vs RL-only 检查模仿预训练。No-geometry 是 IL-only 的单因素对照，不能把它与 Full 的差异全部归因于几何输入。Family mean 使用真实动作族标签，Recorded target 使用真实目标，二者是特权参考，不是公平部署模型。

置信区间见 summary.json：按 actor 成簇的 paired bootstrap，避免把同一演员的重叠窗口当作独立样本。inference_ms 为 CPU 批量推理摊销耗时，不是端到端控制延迟。

### 验证集选模

| Seed | Full 选择的 RL 迭代 | RL-only 选择的迭代 |
|---|---:|---:|
| 20260928 | 0 | 15 |
| 20260929 | 0 | 20 |
| 20260930 | 0 | 0 |

如果选择 iteration 0，Full 实际保留模仿权重；对应 GIF 与 IL-only 相同是负结果，不是换皮展示。

## 实验 A：不同流形的真实执行

固定展示 seed=20260928，每个测试动作族按源索引取第一窗，未按成功挑选。右侧是模型目标经过 SONIC/MuJoCo 后的新执行，不是 SEED 原动作回放。
动作族名称仅表示源数据类别。模型目前只输出中帧姿态，不意味着已执行跑、跳、爬等完整动作；PASS 仅指上面的姿态指标，不是动作识别或任务成功。
A：左流形、右执行；B：同流形下四方法（Full / IL-only / RL-only / no-geometry IL）的同步执行。FAIL 画面保留。

| 来源族 | A：流形 → 执行 | B：消融对照 |
|---|---|---|
| 向前行走 · row 0 | [![](media/A_000000_walk_forward.gif)](media/A_000000_walk_forward.gif) | [![](media/B_000000_walk_forward.gif)](media/B_000000_walk_forward.gif) |
| 向前慢跑 · row 283 | [![](media/A_000283_jog_forward.gif)](media/A_000283_jog_forward.gif) | [![](media/B_000283_jog_forward.gif)](media/B_000283_jog_forward.gif) |
| 背手行走 · row 423 | [![](media/A_000423_hands_back_walk.gif)](media/A_000423_hands_back_walk.gif) | [![](media/B_000423_hands_back_walk.gif)](media/B_000423_hands_back_walk.gif) |
| 侧向行走 · row 628 | [![](media/A_000628_walk_lateral.gif)](media/A_000628_walk_lateral.gif) | [![](media/B_000628_walk_lateral.gif)](media/B_000628_walk_lateral.gif) |
| 曲线行走 · row 792 | [![](media/A_000792_walk_curve.gif)](media/A_000792_walk_curve.gif) | [![](media/B_000792_walk_curve.gif)](media/B_000792_walk_curve.gif) |
| 原地转向 · row 1031 | [![](media/A_001031_turn_in_place.gif)](media/A_001031_turn_in_place.gif) | [![](media/B_001031_turn_in_place.gif)](media/B_001031_turn_in_place.gif) |
| 蹲走 · row 1278 | [![](media/A_001278_crouch_walk.gif)](media/A_001278_crouch_walk.gif) | [![](media/B_001278_crouch_walk.gif)](media/B_001278_crouch_walk.gif) |
| 蹲起转换 · row 1442 | [![](media/A_001442_crouch_transition.gif)](media/A_001442_crouch_transition.gif) | [![](media/B_001442_crouch_transition.gif)](media/B_001442_crouch_transition.gif) |
| 侧向闪避 · row 1656 | [![](media/A_001656_dodge_lateral.gif)](media/A_001656_dodge_lateral.gif) | [![](media/B_001656_dodge_lateral.gif)](media/B_001656_dodge_lateral.gif) |
| 向前弓步 · row 1817 | [![](media/A_001817_forward_lunge.gif)](media/A_001817_forward_lunge.gif) | [![](media/B_001817_forward_lunge.gif)](media/B_001817_forward_lunge.gif) |
| 侧向跳步 · row 2072 | [![](media/A_002072_side_hop.gif)](media/A_002072_side_hop.gif) | [![](media/B_002072_side_hop.gif)](media/B_002072_side_hop.gif) |
| 高跳来源片段 · row 2225 | [![](media/A_002225_high_jump.gif)](media/A_002225_high_jump.gif) | [![](media/B_002225_high_jump.gif)](media/B_002225_high_jump.gif) |
| 箱跳来源片段 · row 2349 | [![](media/A_002349_box_jump.gif)](media/A_002349_box_jump.gif) | [![](media/B_002349_box_jump.gif)](media/B_002349_box_jump.gif) |
| 上台阶来源片段 · row 2431 | [![](media/A_002431_step_up_box.gif)](media/A_002431_step_up_box.gif) | [![](media/B_002431_step_up_box.gif)](media/B_002431_step_up_box.gif) |
| 下台阶来源片段 · row 2526 | [![](media/A_002526_step_down_box.gif)](media/A_002526_step_down_box.gif) | [![](media/B_002526_step_down_box.gif)](media/B_002526_step_down_box.gif) |
| 跪姿转换 · row 2634 | [![](media/A_002634_kneel.gif)](media/A_002634_kneel.gif) | [![](media/B_002634_kneel.gif)](media/B_002634_kneel.gif) |
| 四肢支撑 · row 2856 | [![](media/A_002856_all_fours.gif)](media/A_002856_all_fours.gif) | [![](media/B_002856_all_fours.gif)](media/B_002856_all_fours.gif) |
| 按压 / 拨杆动作 · row 3381 | [![](media/A_003381_button_lever.gif)](media/A_003381_button_lever.gif) | [![](media/B_003381_button_lever.gif)](media/B_003381_button_lever.gif) |
| 搬运姿态 · row 3570 | [![](media/A_003570_carry_object.gif)](media/A_003570_carry_object.gif) | [![](media/B_003570_carry_object.gif)](media/B_003570_carry_object.gif) |

## A2：固定初态，仅修改流形

同一源窗口：原始 / 纵向收缩 30% / 横向收缩 30% / 竖直收缩 30%。这是分布外压力测试，不保证可行；几何未改变之外的状态与模型保持一致。
![](media/A2_geometry_intervention.gif)

## 补充：3 秒保持审查

预热 0.4 秒、过渡 0.4 秒、保持 3 秒；动作观测共 3.4 秒。固定同一种子、19 个测试来源族各一窗。此项为诊断子集，不能替代全部测试集的长时稳定评估。完整逐窗记录见 [long_hold_audit.json](long_hold_audit.json)。

| 方法 | 样本 | 门控通过 | Pose success | 跌倒率 |
|---|---:|---:|---:|---:|
| IL + RL (validation selected) | 19 | 100.0% | 89.5% | 0.0% |
| IL only / no RL | 19 | 100.0% | 89.5% | 0.0% |
| RL only / no IL pretrain | 19 | 94.7% | 52.6% | 0.0% |
| IL / no geometry | 19 | 89.5% | 47.4% | 0.0% |
| Oracle family mean | 19 | 68.4% | 57.9% | 0.0% |
| Recorded target / oracle | 19 | 57.9% | 57.9% | 5.3% |

## 复现

需要已配置的 WSL / SONIC / MuJoCo 环境，以及 [数据准备协议](../MANIFOLD_ACTION_CATALOG.md) 中的本地 NPZ 和原始 actor 元数据。下列命令不负责下载受许可约束的 SEED 数据。
权重保存在 `reports/manifold_motion/stage1_ab_response_v3/replicate_<seed>/seed_<seed>/`：`imitation.pt`、`full.pt`、`no_imitation.pt` 和 `no_geometry/imitation.pt`。全量逐窗口测试记录与回放状态保存在同目录。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ./scripts/python.sh -m manifold_motion.stage1.catalog_experiment \
  --profile short_response --seeds 20260928 --out reports/manifold_motion/stage1_ab_response_v3/replicate_20260928
# 对 20260929、20260930 重复上述命令并替换 seed 和输出目录。
MUJOCO_GL=egl ./scripts/python.sh -m manifold_motion.stage1.catalog_report --long-check
```

旧 v1 gate 的绝对角/增量、执行时长与样本覆盖问题使其数字不可与本协议混用。长保持 v2 训练记录作为中断的诊断保留，不作为完成实验。
