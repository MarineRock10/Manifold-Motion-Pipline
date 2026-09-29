# Stage 2：多动作族 router/Flow 物理验证

这是 Stage 2 首轮单窗口实验之后的多动作扩展。测试样本来自 actor-disjoint test split；
系统先用 Stage 1 从 `M_e(t)` 预测动作概率，再按照预测概率的最高置信度自动挑选不同
动作族。没有用 `primitive` 真值决定动作或筛选候选，真值只用于报告中的 `match` 审计。

## 自动路由与结果

每个动作族生成 6 条候选，并对原始 Flow 与 optimization-embedded projection 两个版本
分别通过冻结 SONIC/MuJoCo 硬门。`passed` 是独立候选数，不是把同一条轨迹重复计数。

| Stage-1 路由族 | 数据真值 | 路由置信度 | router match | raw passed | projected passed | GIF |
|---|---|---:|:---:|---:|---:|---|
| `jog_forward` | `jog_forward` | 0.984 | ✓ | 0/6 | 1/6 | [GIF](../experiments/stage2_router_flow_multifamily_v1/jog_forward.gif) |
| `kneel` | `kneel` | 0.980 | ✓ | 6/6 | 6/6 | [GIF](../experiments/stage2_router_flow_multifamily_v1/kneel.gif) |
| `crouch_transition` | `button_lever` | 0.972 | ✗ | 4/6 | 3/6 | [GIF](../experiments/stage2_router_flow_multifamily_v1/crouch_transition.gif) |
| `all_fours` | `all_fours` | 0.965 | ✓ | 0/6 | 0/6 | — |
| `carry_object` | `step_down_box` | 0.953 | ✗ | 2/6 | 3/6 | [GIF](../experiments/stage2_router_flow_multifamily_v1/carry_object.gif) |
| `forward_lunge` | `forward_lunge` | 0.946 | ✓ | 1/6 | 1/6 | [GIF](../experiments/stage2_router_flow_multifamily_v1/forward_lunge.gif) |

汇总为 raw `13/36`、projection `14/36`；6 个族中 5 个至少有一个 projected candidate
通过。`all_fours` 的 0/6 是当前结果的一部分：失败原因主要是走廊半径、腿部跟踪和
跌倒门，而不是被删除或改写。完整的每候选失败清单在
[版本化 benchmark.json](../experiments/stage2_router_flow_multifamily_v1/benchmark.json)；
本地训练运行也会写入 `reports/manifold_motion/stage2_router_flow_multifamily_v1/benchmark.json`。

## 复现

```bash
./scripts/python.sh -m manifold_motion.stage2.router_flow_benchmark \
  --windows data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz \
  --router-checkpoint models/stage1/temporal_primitive_v11.pt \
  --autoencoder models/stage2/router_flow_v1/autoencoder.pt \
  --flow models/stage2/router_flow_v1/flow.pt \
  --out reports/manifold_motion/stage2_router_flow_multifamily_v1 \
  --max-cases 6 --num-candidates 6 --steps 32 --device cpu --threads 2 --seed 20261002
```

本实验仍是第一轮多族物理检查，不是最终泛化结论。下一轮应增加每族多个 actor、多个
随机种子，并将这些候选接入连续长时序路线，而不是只对独立窗口做回放。
