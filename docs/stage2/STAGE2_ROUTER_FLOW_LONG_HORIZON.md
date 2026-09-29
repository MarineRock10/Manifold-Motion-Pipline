# Stage 2：router Flow 连续长时序路线

本实验把多候选 router Flow 从独立窗口接到连续 A* 路线。每个路线段都重新构造局部
`M_e(t)`、`M_self(t)` 和 SDF，由 Stage 1 temporal router 输出动作概率，再生成候选并
经过冻结 SONIC/MuJoCo 物理筛选。通过后才进入一次不重置的连续执行。

![Continuous router Flow route](../experiments/stage2_router_flow_long_horizon_v1/flow_route_avoidance.gif)

## 结果

| 指标 | 结果 |
|---|---:|
| A* keyframes reached | 3/3 |
| simulator reset | 0 |
| obstacle contact ticks | 0 |
| primitive switches | 2 |
| physical accepted | true |
| route segments | 3 |

这一次路线的三个局部环境分布都选择了 `walk_nominal`，但它们不是预先写入的动作：
动作族来自每段的 Stage-1 概率，候选通过率和最终切换由物理门决定。该结果证明了
“router Flow → 连续 SONIC 执行”的接口和关键帧保持是通的，同时也暴露出当前静态路线
走廊偏宽，尚未充分激活动作切换。

审计产物：

- `reports/manifold_motion/stage2_router_flow_long_horizon_v1/report.json`
- `reports/manifold_motion/stage2_router_flow_long_horizon_v1/candidate_evidence.json`
- `reports/manifold_motion/stage2_router_flow_long_horizon_v1/segment_conditions.npz`

## 复现

```bash
MUJOCO_GL=egl ./scripts/python.sh -m manifold_motion.stage2.flow_route_candidates \
  --scene data/g1_flat/scene_long_avoidance.xml \
  --windows data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz \
  --autoencoder models/stage2/router_flow_v1/autoencoder.pt \
  --flow models/stage2/router_flow_v1/flow.pt \
  --router-checkpoint models/stage1/temporal_primitive_v11.pt \
  --out reports/manifold_motion/stage2_router_flow_long_horizon_v1 \
  --num-candidates 4 --max-ticks 650 --device cpu
```

下一步是把同一入口接到在线雷达/滑动栅格更新，使用移动障碍改变局部 `M_e(t)`，并在
窄通道与低顶棚场景验证 router 是否从 nominal 自动切换到侧身或低位动作。
