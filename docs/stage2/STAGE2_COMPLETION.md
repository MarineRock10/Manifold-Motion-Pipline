# Stage 2 完成验收包

本页是当前冻结 SONIC + MuJoCo 主线的 Stage 2 收口结果。它把环境流形、机器人自身流形、
状态/历史、在线地图更新、增量规划、Flow 候选、物理影子回放和连续执行放进同一套验收口径。
每个“通过”都来自 MuJoCo/几何硬门，而不是 GIF 文件名或预先写入的动作标签。

> 范围：真实 RGB-D/LiDAR/SLAM 的 P1 接入仍作为部署适配层单独进行；这里使用同步的 MuJoCo
> 雷达、滑动栅格和动态障碍夹具验证 Stage 2 本身。

## 总体结果

机器可读汇总：[acceptance.json](../experiments/stage2_completion_v1/acceptance.json)。

| 子实验 | 验证内容 | 结果 | 可视化/审计 |
|---|---|---:|---|
| 环境流形反事实 | 同一路线下宽阔、低顶棚、窄通道和中心障碍分别改变 `M_e`；动作由 aperture/曲率重新路由 | 通过；低位切 crouch，窄通道切侧身，中心障碍触发转向 | [反事实 GIF](../demo_gallery/media/wide_vs_low_counterfactual.gif) · [报告](../stage2/STATUS.md#环境流形因果反事实替代写死动作表) |
| 自身流形安全门 | `M_self(t)` 随任务姿态收缩，并经过 G1 表面窄相位 clearance 检查 | 通过；窄通道 0 接触，侧身 yaw P95 21.2° | [侧身 GIF](../demo_gallery/media/side_passage.gif) · [投影报告](../stage2/STATUS.md#online-semantic-rerouting-and-incremental-planning) |
| 在线状态/历史与投影 | 真实当前状态、12 帧历史、局部 SDF、`M_e/M_self` 生成候选，短 rollout 后才换参考 | 4/4 场景通过，0 接触 | [在线闭环 GIF](../demo_gallery/media/online_closed_loop.gif) · [在线报告](../stage2/ONLINE_COMPOSER.md) |
| 在线 composer 长路线 | 连续雷达更新触发 D* Lite、动作族迟滞和 SONIC 参考切换 | 3/3 路线通过，8–10/8–10 关键帧，0 接触 | [Composer 报告](../stage2/ONLINE_COMPOSER.md) |
| 增量动态规划 | 障碍出现、穿越、移动和消失；增量 ESDF + D* Lite 对比全体素 A* | 8/8 更新，4 次路线变化，中位加速 85.3× | [动态重规划 GIF](../experiments/stage2_completion_v1/moving_obstacle_replanning.gif) · [JSON](../experiments/stage2_completion_v1/incremental_dynamic_report.json) |
| 动态交叉障碍 | 障碍穿过机器人规划路线，在线重建 `M_e(t)` 并重新选择动作 | 8/8 关键帧，0 接触，7 次切换 | [GIF](../demo_gallery/media/repaired_autonomous_dynamic.gif) |
| 投射物擦身/高空 | 由相对位置和速度驱动 hazard head，选择侧闪或蹲下，经过当前状态物理门 | 两个场景均 8/8，0 接触 | [擦身](../demo_gallery/media/repaired_autonomous_projectile_grazing.gif) · [高空](../demo_gallery/media/repaired_autonomous_projectile_overhead.gif) |
| 连续 router Flow | 每个 A* 区间动态生成候选，在一次不重置 rollout 中执行 | 3/3 关键帧，0 接触，accepted | [Flow 长时序报告](../stage2/STAGE2_ROUTER_FLOW_LONG_HORIZON.md) |

## CVPR 单种子物理预检

冻结的 4 方法 × 26 场景矩阵现已完成 104/104 行：92 行为 exact MuJoCo physics，12 行为
明确标注的 held-out proxy geometry。总成功数为 57/104；动态子集为 15/16。该数字只用于
发现协议和实现缺口，不能代替多种子论文表。唯一动态失败行仍然保留：B2 在
route-reopen 触发自身流形安全停止。Ours-4 moving-wall 去除动态场景中会立刻过时的整程
probe 后在 117.3 秒完成；实时 state/history、projection、shadow gate 和所有物理门未移除。

完整的机器可读快照见
[cvpr_primary_physical_seed31000_summary.json](../experiments/results/cvpr_primary_physical_seed31000_summary.json)。

## 关键结论

1. `M_e → primitive` 已不再是按路线段编号写死：低顶棚改变垂直 aperture 后进入 crouch，窄通道改变横向 aperture 后进入侧身，动态障碍改变路线后触发在线重新规划。
2. `M_self` 是实机部署前的独立安全门。椭球只是保守 broad phase，最终检查使用 G1 表面与障碍的窄相位距离；通过该门之前不会提交新 SONIC 参考。
3. 连续任务保持同一个 MuJoCo 状态和历史，不通过重置或瞬移拼接 GIF；关键帧、路线进度、接触和切换都写入报告。
4. 当前冻结 SONIC 的能力边界仍然明确：crawl 记为 unsupported，jump 记为 partial，不会因为 router 分类结果而伪装成可执行动作。

## 一键复核

```bash
# 核心 Stage 2 回归
./scripts/python.sh -m pytest -q \
  tests/test_composer_constraints.py tests/test_dynamic_scene.py \
  tests/test_incremental_planner.py tests/test_online_composer.py \
  tests/test_online_perception_semantics.py tests/test_reactive_policy.py \
  tests/test_stage2_router_bridge.py tests/test_stage2_router_flow_benchmark.py

# 动态规划基准
./scripts/python.sh -m manifold_motion.evaluation.incremental_dynamic_benchmark \
  --out reports/manifold_motion/incremental_dynamic_benchmark_final

# 在线 composer 和 MuJoCo 长路线
./scripts/run_stage2_online_composer_long.sh

# 静态/动态/投射物自主场景（需要本地训练 checkpoint）
./scripts/run_trained_autonomous_suite.sh

# 汇总所有硬门
./scripts/python.sh -m manifold_motion.evaluation.stage2_final_acceptance \
  --root . --out docs/experiments/stage2_completion_v1/acceptance.json
```

最终验收不等于真实机器人部署：下一阶段只剩把真实 SLAM 位姿、雷达时间戳和外参接入
相同的 `online_perception` 接口，并保持这里的 `M_self`/接触/路线进度硬门不变。
