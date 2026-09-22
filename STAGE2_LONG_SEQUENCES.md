# Stage-2 长时序演示

这些任务在同一次 MuJoCo rollout 中跨越多个环境流形区间，不在路段之间 reset。

| 任务 | 流形触发的动作序列 | 验收 |
|---|---|---|
| `long_low_cycle` | `crouch → walk_nominal → crouch` | 625 physics ticks，12.5 s，11/11 keyframes，0 contact |
| `long_combo` | `crouch → walk_lateral_reverse` | 507 physics ticks，10.14 s，10/10 keyframes，0 contact |

两项任务的最终验收都包含 `robot_self_manifold_safety`：白色 `M_r^safe` 先做椭球广相，
再对每个执行帧的 G1 表面采样点做障碍物窄相，默认最小 clearance 为 0.02 m。当前
`long_low_cycle` 的最小精确 clearance 约 0.107 m，`long_combo` 约 0.074 m；广相
出现的保守重叠会继续进入窄相，不会直接被当作无碰撞证明。

每个动作切换记录在 `primitive_switches` 中，并包含 phase handoff、切换前关节 RMS
和当前目标关键帧。GIF 和 JSON 位于
`reports/manifold_motion/stage2_long_sequence_v1/`。

```bash
./run_stage2_long_sequence_demo.sh
```

这些场景仍是 MuJoCo 合成几何，用于验证长时序动作切换和连续执行；真实点云接入后可
复用同一执行器和物理门。
