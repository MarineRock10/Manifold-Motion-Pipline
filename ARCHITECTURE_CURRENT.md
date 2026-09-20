# G1 流形分层运动生成：当前可复现架构

本文件对应仓库中已经通过 MuJoCo/GEAR-SONIC 物理门控的实现，而不是只描述未来规划。
配套总图是 [`latest_g1_manifold_architecture.svg`](latest_g1_manifold_architecture.svg)，
PNG 预览为 [`latest_g1_manifold_architecture.png`](latest_g1_manifold_architecture.png)。

## 1. 当前运行时闭环

```text
环境点云 / MuJoCo 几何
        ↓
A* 路线 + keyframes + safe corridor
        ↓
时变环境流形 Mₑ(t)
  corridor [48,7] + SDF [10,10,8] + route yaw
        ↓
Stage 1：按自由半轴与路线曲率路由原语
  walk_nominal / crouch / strict side-on / walk_turn
        ↓
Stage 2 条件包 Xₜ = (Mₑ, pₜ, sₜ, Hₜ, cₜ)
  sₜ ∈ R⁶⁹；Hₜ ∈ R¹²×⁶⁹
        ↓
Latent Flow Matching + AE
  K 条 R_refᵏ ∈ R⁴⁸×³⁸
        ↓
optimization-embedded projection
  joint limits + smoothness + handoff + route consistency
        ↓
每条候选分别经过 frozen SONIC + MuJoCo G1
        ↓
接触 / 跌倒 / roll / tracking / keyframe / route / body-yaw 硬门
        ↓
选择可行 R*，连续执行；到每个路段边界读取真实 sₜ/Hₜ 重条件化
```

当前“在线”语义是段边界在线重条件化：先进行一次不渲染 probe rollout，从真实执行结果
提取状态和历史，再为每个路段重采样候选，最后从一个连续 MuJoCo 状态完成最终回放。它
不是每个 50 Hz tick 都重新解 Flow 的 MPC；这样保留了现有 SONIC 接口和计算预算，同时
消除了使用固定 SEED anchor 状态的主要误差来源。

## 2. 接口与张量契约

| 接口 | 形状 | 来源 / 含义 |
|---|---:|---|
| `corridor` | `48×7` | 路线局部坐标中的中心、半轴和 yaw |
| `sdf` | `10×10×8` | 与训练窗口一致的固定网格安全走廊 SDF |
| `s_t` | `69` | `q_exec(29), dq_exec(29), gravity(3), local base velocity(3), foot/hand/non-foot contacts(5)` |
| `H_t` | `12×69` | 最近 0.4 s 的真实执行 history，按 30 Hz 训练契约重采样 |
| `target_ref` | `48×38` | `q(29) + root position(3) + root rotation 6D(6)` |
| command | `9` | 目标位移、朝向和路线局部命令 |

`R_ref` 是 SONIC 的参考输入；`R_exec` 只用于物理筛选和反向构建数据。当前 SONIC
动作头消费关节参考、关节速度和基座姿态，不消费 root position，因此 projection 中的
root position 修正用于保持动态目标与路线一致，最终运动仍由关节/姿态和 MuJoCo 真实执行
决定。

## 3. Stage 1：原语路由

`manifold_motion/stage2_manifold_adaptive.py` 从每个路线段的实测 `Mₑ` 计算：

- 垂直自由半轴低于阈值：选择 `crouch`；
- 横向自由半轴低于阈值：选择 `walk_lateral_reverse`，窄通道版本要求隔离位移接近
  ±90°，并在最终回放中要求身体 yaw 仍接近侧身；
- 路线曲率超过阈值：临时启用 `walk_turn`；
- 其他段：选择 `walk_nominal`。

这不是按 segment index 写死动作。原语选择来自几何流形和路线曲率；候选仍必须通过
冻结 SONIC 的物理回放。当前 crawl/jump 没有纳入自动路由，因为冻结控制器的低接触能力
门尚未通过。

由于 pilot Flow 数据在 walk→crouch 和在线侧向状态上的覆盖有限，候选集合保留一个
训练支持 anchor：p2/p5 使用训练状态下的 Flow anchor，p4 使用经过物理验证的 SEED
侧身 exemplar；其余候选使用真实 `s_t/H_t` 条件生成。锚点不是绕过物理门，而是和在线
候选一起被筛选，并在报告中以 `candidate0_training_support_anchor` 标注。

## 4. Stage 2：动态生成与投影

`RouteFlowSampler.condition()` 先组装 `(Mₑ, p, s, H, c)`，再使用训练 split 的
normalizer。`sample()` 通过 32 步 latent Flow integration 解码 K 条候选。

`manifold_motion/stage2_projection.py` 在解码之后、SONIC 之前执行保守的 projected-gradient
修正。目标由以下项组成：

```text
J(R) = λ_smooth J_smooth
     + λ_limit  J_limit
     + λ_handoff J_handoff
     + λ_corridor J_corridor
```

默认只做 1 次投影，最大单关节修正为 0.005 rad，避免 projection 变成隐藏的动作重定向器。
每条候选同时保存 raw Flow 和 projected 版本，便于消融比较。

## 5. 物理门与验收

候选筛选与最终回放使用同一个 frozen SONIC/MuJoCo 执行接口，拒绝条件包括：

- 命名障碍接触、跌倒、极端 roll；
- 关节跟踪误差、原语交接 discontinuity；
- keyframe 未到达、终点误差和路线偏差；
- 严格侧身段的 body-route yaw mismatch。

当前完整回归结果：

| 场景 | 原语结果 | keyframes | 接触 | route deviation P95 |
|---|---|---:|---:|---:|
| `wide` | 全程 `walk_nominal` | 6/6 | 0 | 0.148 m |
| `low` | `walk_nominal → crouch` | 6/6 | 0 | 0.134 m |
| `narrow` | 严格 `walk_lateral_reverse` | 6/6 | 0 | 0.097 m |
| `center` | `walk_turn + side/nominal` | 10/10 | 0 | 0.171 m |

汇总结果保存在 `reports/manifold_motion/stage2_online_projection_v1/comparison_report.json`。

## 6. 可复现资产布局

为了让 clone 后的用户不依赖本机的 `reports/` 和 `data/` 忽略目录，本次版本把最小
复现实验资产纳入 Git LFS：

```text
data/g1_flat/                                  # G1 MJCF + meshes + four scenario scenes
reports/manifold_motion/seed_windows_corridor_stage2_v2/
  seed_stage2_windows.npz                       # 25 MB, Stage-2 windows
reports/manifold_motion/stage2_flow_corridor_v1/
  autoencoder.pt                                # Flow AE
  flow.pt                                       # latent Flow
reports/manifold_motion/stage2_mean_primitive{2,4,5,6}_v1/
  conditional_mean.pt                           # verified primitive anchors
gear_sonic_deploy/policy/release/
  observation_config.yaml                        # tracked config
```

GEAR-SONIC 的 encoder/decoder ONNX 不在 Git 中重复分发；使用仓库已有的
`download_from_hf.py` 从 `nvidia/GEAR-SONIC` 下载，并在运行前检查文件存在。这样既避免
把近百 MB 控制器二进制复制进代码历史，也保留了明确的外部来源。

## 7. 一键复现

安装依赖、下载 SONIC ONNX 后，在 WSL 中执行：

```bash
python3 -m pip install -r requirements.txt
python3 download_from_hf.py --stage2-only
export PYTHONPATH="$PWD:/home/$USER/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl
./run_stage2_online_projection.sh
```

脚本会依次生成 `wide/low/narrow/center` 四个目录、GIF、`executed.npz`、候选审计和
`comparison_report.json`。PowerShell 包装器为 `run_stage2_online_projection.ps1`。

如果模型资产被放在其他位置，可通过命令行显式传入 `--windows`、`--autoencoder`、
`--flow`；场景必须保持与 `data/g1_flat/` 同一 G1 MJCF/mesh 相对结构。

