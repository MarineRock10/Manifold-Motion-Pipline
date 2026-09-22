# Stage-2 多场景演示

这些演示使用当前的 `M_e(t) → primitive → Flow candidates → projection → SONIC/MuJoCo`
链路。它们都是 MuJoCo fixture 几何，不代表真实点云感知。

运行：

```bash
./run_stage2_diverse_demo.sh
```

结果目录：`reports/manifold_motion/stage2_diverse_demo_v2/`

| 场景 | 由流形触发的动作变化 | GIF |
|---|---|---|
| short low | `walk_nominal → crouch → walk_nominal` | `low_short/manifold_adaptive.gif` |
| left offset block | `walk_lateral_reverse → walk_nominal` | `block_left/manifold_adaptive.gif` |
| right offset block | `walk_lateral_reverse → walk_nominal`，末段 `walk_turn` | `block_right/manifold_adaptive.gif` |

统一验收报告：`comparison_report.json`。每个场景同时保存 `report.json`、候选审计和
`executed.npz`，可以检查动作切换是否由走廊的高度、侧向余量和路线曲率触发。
