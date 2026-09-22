# ORCS 与当前 SONIC 管线的兼容性结论

## 结论

ORCS（[lok-i/orcs](https://github.com/lok-i/orcs)）不是一个可以直接替换
`gear_sonic_deploy/policy/release/` 中 ONNX 文件的控制器发布。它发布的是
任务专用的 PyTorch/rsl_rl checkpoint：冻结 SONIC 基座，在 decoder 上挂零初始化
LoRA adapter，再用 PPO 训练任务条件。公开 release v0.1.0 的模型是：

| checkpoint | 任务 | 大小（约） |
|---|---|---:|
| `Orcs-Dodge-AdaptSonic` | 躲避飞球 | 65 MB |
| `Orcs-PerLoco-Grail-AdaptSonic` | 感知地形 locomotion | 66 MB |
| `Orcs-PerLoco-OmRe-AdaptSonic` | OmniRetarget locomotion | 66 MB |
| `Orcs-Uolm-AdaptSonic` | 物体交互 | 68 MB |

这些权重依赖 ORCS/MJLab 的任务观测分组、rsl_rl 的
`SonicWithAdapterModel`、对应的 SONIC base checkpoint，以及固定的
`mocke`/`rsl_rl` commit。它们没有当前项目的 `R_ref -> SONIC ONNX -> MuJoCo`
接口，因此不能安全地把某个 `.pt` 当作当前 ONNX 控制器替换。

## 可以直接借鉴的部分

1. **冻结基座，先训练小 adapter。** 初始 adapter 为零时应与 SONIC bit-exact，
   只让 LoRA 承担环境条件带来的残差，避免破坏已有走路能力。
2. **条件单独进入 augmentation stream。** 当前项目对应的条件应是
   `M_e(t) / corridor / SDF + command + state/history`，而不是把点云直接
   拼进 SONIC 的固定输入。
3. **先 privileged oracle，再做感知 student。** 在真实点云接入前，用仿真中的
   完整 corridor/SDF 训练并通过 MuJoCo 硬门控；之后再蒸馏到雷达/SLAM 可观测量。
4. **每次更新都保留冻结 SONIC 回退。** 任何 adapter 都必须同时检查摔倒、脚/手
   接触、根高度/姿态、障碍接触、自身椭球余量和路线进度。

## 对本仓库的执行边界

当前先完成 Stage-2 条件轨迹模型训练；SONIC 本体仍保持冻结，避免把“条件模型
没有学会路线跟踪”和“控制器被微调破坏”混在一起。只有在部署条件模型在多个
MuJoCo 场景上通过硬门控后，才建立 ORCS 风格的 `corridor/state/history ->
LoRA residual` 训练环境，并导出与当前 SONIC 版本匹配的 adapter。ORCS 的公开
checkpoint 可作为训练配方和超参参考，但不作为未经转换的替换权重。

来源：ORCS README、`src/orcs/release.json`（公开 release v0.1.0，仓库状态截至
2026-09-14）。
