# 总体架构

> 本文是**唯一**的架构文档：整个系统要做什么、分几层、接口怎么定、哪些是冻结的、
> 九个阶段各自的验收标准。
> 配套：[`PIPELINE.md`](PIPELINE.md)（**当前实现**的技术路线与实测数字）。
> 历史设计稿（`G1_Manifold_Hierarchical_...Design(1).md`、`Robot_aware_Manifold_to_Motion_..._v4.md`、
> `MINIMAL_GOAL_AND_PATH.md`、`EXECUTION_ORDER.md`）已并入本文，见 git。
>
> 日期：2026-09-16。

---

## 1. 目标

$$
Environment \rightarrow Safe\ Corridor \rightarrow Manifold \rightarrow Primitive
\rightarrow Dynamic\ Motion \rightarrow SONIC \rightarrow G1
$$

让 G1 在**时变椭球流形** `M_t` 和任务条件 `c_t` 下，生成**物理可行、且始终待在流形内**的运动。执行层是**冻结的** SONIC 全身体控制器（ONNX；不解冻、不微调）。

**关键设计判断：不直接从点云生成全身关节动作，而是把问题分成"运动意图"和"动态展开"两层。**

$$
\underbrace{p(R_{t:t+H}\mid M_t,s_t,c_t,H_t)}_{\text{要学的总分布}}
=
\sum_{z_p}
\underbrace{p_\theta(R_{t:t+H}\mid M_t,z_p,s_t,c_t,H_t)}_{\text{Stage 2：动态生成}}
\;
\underbrace{p_\phi(z_p\mid M_t)}_{\text{Stage 1：原语选择}}
$$

其中 `M_t` 环境流形、`z_p` 运动原语、`s_t` 当前状态、`H_t` 历史、`c_t` 命令、`R_{t:t+H}` 未来运动。

**为什么分层**：直接学 `M → R` 信息不足（同一个流形，站着和蹲着该做的事完全不同，见 §7），而且"姿态对不对"与"动起来跟不跟得上"是两个不同的可行性问题，混在一起训练会互相掩盖。

**贯穿全系统的约束：生成的动作必须是执行层真能做到的。** 这一条决定了训练信号必须来自真控制器**实测的落点**，而不是对控制器的建模（见 §8.3）。

---

## 2. 全链路

**✅ = 已实现并在跑（Phase 1–3）　◻ = 设计已定，未实现**

```
RGB-D / LiDAR / Point Cloud
              │
              ▼
      3D Occupancy / SDF                                    ◻ Stage 0
              │
              ▼
      3D A* / Trajectory Opt.                    ┌──────────────────────────┐
              │                                   │ Robot Capability         │
              ▼                                   │ Manifold  M^R            │
        Safe Corridor                             │ （从 SONIC 已验证运动     │
              │                                   │   反向构建，§5）          │
              ▼                                   └────────────┬─────────────┘
   Environment Manifold  M_1 … M_T                            │
              │                                                │
              └──────────────────┬─────────────────────────────┘
                                 ▼
                    M* = M^E ∩ M^R      （环境允许 ∩ 机器人真能做）
                                 │
                                 ▼
        ┌────────────────────────────────────────────┐
        │ Stage 1：Motion Primitive   p_φ(z_p | M)   │  ✅ 姿态部分已实现
        │ 单个局部流形 → 姿态（z_p == q）              │     （步态 k 未做）
        │ 已实现：克隆 + SONIC 在环                    │     ◻ 二阶模仿未做
        │ 未实现：步态 k、能耗/平滑项、二阶模仿         │
        └────────────────┬───────────────────────────┘
                         ▼
                  q = 29 维驻留姿态        ✅
                         │
        ┌────────────────┼────────────────┐
        │                │                │
     State s_t      History H_t      Command c_t      ◻ Stage 2 才消费
        │                │                │
        └────────────────┼────────────────┘
                         ▼
        ┌────────────────────────────────────────────┐
        │ Stage 2：Dynamic Motion Generator          │  ◻ 未实现（Phase 6–9）
        │ Conditional Flow Matching                  │
        │ v_θ(x, τ | M, z_p, s, H, c)                │
        └────────────────┬───────────────────────────┘
                         ▼
                候选运动 R^(1…N)                          ◻
                         ▼
        Safety / Stability / Task / Energy / Smoothness  ◻ 仅部分项有实现（门控）
                         ▼
                        R* = argmin J(R^(i))              ◻
                         ▼
                      SONIC  →  G1                        ✅ 唯一可信的落点来源
```

**当前真正端到端跑通的只有这一段**：`M → q → SONIC → 实测落点`。上游（环境流形）和下游（动态生成）都是设计。

---

## 3. 环境流形生成（Stage 0）

### 3.1 感知 → 地图

输入 RGB-D / LiDAR / 点云，建立 3D Occupancy 与 SDF/ESDF。局部几何必须能给出**高度、宽度、深度、方向、安全距离**——这五项正是流形参数的来源。

### 3.2 全局路径

用成熟的传统规划，不学：

$$
\tau^* = \arg\min_\tau \; w_d J_d + w_c J_{collision} + w_s J_{smooth} + w_k J_{curvature}
$$

3D A\* 搜可行拓扑路径，轨迹优化做平滑与动态可执行化。

### 3.3 Safe Corridor → 流形序列

沿 `τ*` 生成连续局部安全走廊 `C = {E_1 … E_T}`，每个局部空间用椭球描述：

$$
(x-c_t)^\top A_t (x-c_t) \le 1
\;\Longrightarrow\;
M_t = (h_t, w_t, d_t, R_t, \theta_t, clearance_t, \dots)
$$

**Stage 1 用单个 `M_t`；Stage 2 用连续序列 `{M_t}`。**

---

## 4. Stage 1：流形 → 姿态（**已实现**）

| | Stage 1 | Stage 2 |
|---|---|---|
| 问题 | **摆成什么姿态 / 用什么步态** | **怎么动起来** |
| 输入 | 单个 `M`（+ 任务条件） | `M, z_p, s_t, H_t, c_t` |
| 输出 | `z_p` 原语（含姿态参数） | `R_{t:t+H}`：T 帧关节轨迹 |
| 方法 | 模仿 + RL + 二阶模仿 | Conditional Flow Matching + N 候选 + J(R) 选择 |
| 时间性 | **无**：一个驻留姿态/步态 | **有**：整段运动 |
| **状态** | **✅ 姿态部分已实现** | ❌ 未实现 |

Stage 1 输出**一个**驻留姿态，它的**时间展开是 Stage 2 的工作**。这条边界是刻意的：它让"姿态可行性"和"动态可行性"可以分别训练、分别验收、分别失败。

### 4.1 设计意图：原语 `z_p` 的完整形态

**（目标形态，尚未完全实现）**

```python
z_p = [k, α_height, α_pitch, α_stance, v, …]
# k ∈ {walk, run, crouch, crawl, side-step, lean, turn}
# 其余是动作强度、姿态、速度、步态参数的连续量
```

离散类型 + 连续参数的混合，才能表达"低矮通道里小步快走的蹲行"这类组合；one-hot 会把强度信息丢掉。

### 4.2 Stage 1 网络（设计）

```
单个流形 M
     │
     ▼
几何特征编码器
     ├─ 椭球尺度 (semi)
     ├─ 朝向 (axis)
     ├─ clearance
     ├─ 局部方向
     └─ 曲率
     │
     ▼
PointNet++ / 3D CNN / Transformer
     │
     ▼
特征融合 MLP
     │
     ├──────────────┐
     ▼              ▼
Primitive Head   Parameter Head
walk/crouch/…    height/pitch/stance/v/…
     │              │
     └──────┬───────┘
            ▼
           z_p
```

### 4.3 当前实现（**这一段是真在跑的**）

上游给的是**解析椭球**，不是点云，所以几何编码器不需要 PointNet——**退化成 15 维观测 + MLP**：

```
M（椭球：center, semi, quat） + 当前姿态 q
     │
     ▼
15 维观测（pose_policy.observation，唯一定义）
     ├─ pose[:3] / 1.4            3   当前姿态的前三个关节
     ├─ pose mean, std            2   整体形变程度
     ├─ 包含度 r                  1   几何上离流形表面多远
     ├─ (center − pelvis) / 0.5   3   流形相对身体在哪
     ├─ semi                      3   多大
     └─ axis                      3   脊柱该朝哪
     │
     ▼
MLP  body: 15 → 256 → 128        （40891 参数，BC 与微调同一结构）
     │
     ▼
mu: 128 → 29                     ← Parameter Head
     │
     ▼
q = tanh(mu) × 1.4               ← 29 维驻留姿态（action_to_pose）
```

| 设计中的东西 | 当前实现 |
|---|---|
| Primitive Head（学 `k`） | **未实现**——没有步态分类，策略只输出姿态 |
| Parameter Head 输出 `α_height, α_pitch, …` | 输出 **29 维关节角本身**。不用低维参数是因为走路帧的大部分构型在肩 roll / 肘 / 腕 / 髋 pitch 上，任何低维基都盖不住，而**风格差异正好住在那里** |
| 几何编码器 / PointNet | 退化为 15 维观测 + MLP（`M` 已是解析椭球） |
| 模仿 + RL + 二阶模仿 | 实现为 **① 克隆（模仿）+ ③ 在环微调（RL）**；二阶模仿（`q̇`/`q̈`/contact）**未做**——静态姿态没有时间维度，它是 Stage 2 的事 |

**为什么先只做姿态**：`M → q` 是无时间维度的最简闭环，它的可行性判据（r ≤ 1、运动学门控）能独立验证。步态 `k` 需要时间维度才有意义（"crouch-walk" 是一个运动，不是一个姿态），所以归 Stage 2。

**训练信号必须来自实测**：`M → q` 的监督来自真控制器记录的运动（①），微调的奖励来自真控制器**实测的落点**（③），不是任何模型的预测。原因见 §8.3。

### 4.4 原语 `z_p` 与 `q` 的关系

当前 `z_p` **等价于 `q`**（29 维姿态）。将来加入 `k` 后：

$$
z_p = (k,\; q) \qquad\text{或}\qquad z_p = (k,\; \alpha_{height}, \alpha_{pitch}, \dots) \to q
$$

接口已按这个方向留好：`z_p` 是 Stage 1 的输出、Stage 2 的输入，中间怎么分解不影响上下游。

### 4.5 Stage 1 的奖励与模仿

**设计（完整版，含时间维度）**：

$$
r_t = w_{safe}r_{safe} + w_{task}r_{task} + w_{stable}r_{stable}
     + w_{energy}r_{energy} + w_{smooth}r_{smooth} + w_{clear}r_{clear}
$$

$$
r_{safe} = -\max(0, d_{safe} - d_{robot}), \qquad
r_{stable} = -\|\theta_{base}\|^2 - \beta\|\omega_{base}\|^2
$$
$$
r_{energy} = -\sum_i |\tau_i \dot q_i|, \qquad
r_{smooth} = -\|\ddot q\|^2 - \eta\|j_q\|^2
$$

二阶模仿损失（只模仿位置会丢失动态风格，所以 `q` / `q̇` / `q̈` / contact 一起）：

$$
L_{IM} = \lambda_q L_q + \lambda_{\dot q} L_{\dot q} + \lambda_{\ddot q} L_{\ddot q} + \lambda_c L_{contact}
$$

训练顺序：`监督/模仿 → RL 微调 → 二阶模仿精修`。

**当前实现（静态姿态，无时间维度，只有一阶）**：

$$
L = \mathrm{MSE}\big(\tanh(a)\cdot 1.4,\; q_{demo}\big)              \quad\text{① 克隆}
$$

$$
r = \underbrace{w_{gate}\cdot\text{运动学门控}}_{\text{违规就扣}} + w_{inside}\cdot\text{余量}
    + w_{outside}\cdot(\text{超出量}) + w_{track}\cdot\text{跟踪} + w_{imitation}\cdot\text{距示范}
    + \text{success}
$$

| 设计中的项 | 当前实现 |
|---|---|
| `r_{energy}`（能耗）、`r_{smooth}`（平滑） | **未实现**——两者都需要 `q̇`/`q̈`，静态姿态没有 |
| 二阶模仿 `L_{q̇}, L_{q̈}, L_{contact}` | **未实现**，同上 |
| `r_{stable}` | 实现为**运动学门控**：质心偏离支撑 / 抬脚 / 超关节极限 → 直接判违规（比软惩罚更强，因为不可保持的姿态不该有部分分） |
| 训练顺序 | **一阶**：① 克隆 → ③ 在环 RL，没有第三段精修 |

**门控比奖励更重要的原因**：一个会摔倒的姿态，无论它对包络内性做了什么，都不该得到分。"装进流形"只在**机器人真能保持的姿态**上才是目标——否则策略会学会用不可执行的姿态骗奖励（这一点实测过，见 §8.3）。

---

## 5. Robot Capability Manifold `M^R`

**（设计目标；当前只有一个具体实例，见文末）**

除了**环境**流形，还要从 SONIC 已验证能执行的运动**反向**构建**机器人能力**流形：

$$
R(t) \;\longrightarrow\; M^R(t)
$$

最终有效空间是两者的交：

$$
M^*_t = M^E_t \cap M^R
$$

- `M^E`：环境允许怎么动；
- `M^R`：G1 实际能怎么动。

**这使模型不只是"理解环境"，而是理解"这个环境对 G1 意味着什么动作"。**

**当前实现的实例**：`capability.py` 做了一次这个反向统计，但只覆盖了**一个轴**——从记录的运动里统计"给定走廊（clearance、半宽），这个动作可行吗、多快"，得到 `capability.json`（clearance / half_width / speed 的可行域）。它是完整 `M^R` 的一个切片：只回答了"能不能走、多快"，没有回答"该摆什么姿态"。

`M* = M^E ∩ M^R` 这个交集的**用法**目前也还没落地：现在的流程直接把记录的运动流形当训练输入，没有做"环境允许 ∩ 机器人能做"的裁剪。

---

## 6. 接口

改接口必须说明理由并等确认——接口一改，上下游全部作废。标 ✅ 的是**现在冻结、已在跑**的；
标 ◻ 的是**为将来留好、尚未使用**的。

### 6.1 流形 `M` ✅

```python
M = {"center": [x,y,z], "semi": [sx,sy,sz], "quat": [w,x,y,z]}
```

- 椭球，**骨盆坐标系**；`center` 是身体相对骨盆的位置（**不是**骨盆本身：在骨盆下方约 0.11 m，因为身体大部分是腿）；
- `semi` 是**拟合**椭球的半轴，**不是**各轴最大偏移。两者是不同的形状：用盒子半轴当椭球半轴，站姿在**自己的** scale-1.0 流形下算出 r = 1.145，"装进流形"变成不可满足的目标；
- 包含性判据（**全项目唯一**）：

$$
r = \left\| \frac{p - c}{semi} \right\|_\infty^{\text{ell}} \;=\; \max_i \left\| \frac{p_i - c}{semi} \right\|_2 , \quad r \le 1 \iff \text{在内部}
$$

对**所有身体表面点**取最大。

### 6.2 姿态 `q` ✅ / 原语 `z_p` ◻

- **`q`（现在就在用）**：29 维关节向量，**policy(IsaacLab) 顺序**，相对默认站姿的**增量**；
- **`z_p`（预留）**：设计上是 `(k, 参数)` 的混合，当前**等价于 `q`**——Stage 1 只输出姿态，没有步态 `k`。Stage 2 实现时再决定怎么分解（见 §4.4）；
- 观测：15 维，单一定义在 `pose_policy.observation`：
  `pose[:3]/1.4 (3) + mean,std (2) + r (1) + (center−pelvis)/0.5 (3) + semi (3) + axis (3)`；
- 动作映射：`q = tanh(a) × 1.4`，单一定义在 `pose_policy.action_to_pose`。用 `tanh` 不用 `clamp`——clamp 会饱和并失去"回来"的梯度。

### 6.3 运动 `R` ✅（只用到 T=1）

```python
R = {"joint_pos": [T,29], "joint_vel": [T,29],
     "root_pos": [T,3], "root_quat": [T,4]}          # 全部 policy 顺序
```

SONIC 读的是**运动参考**：逐帧关节位置/速度 + 根位姿。**一个关键帧就是 T=1 的 `R`**——这是静态姿态能直接接到 SONIC 上的原因（`reference.KeyframeReference`）。T>1 的用法（Stage 2 的输出）还没实现。

### 6.4 执行层 ✅

- SONIC ONNX，50 Hz 控制 / 200 Hz 物理（4 子步）；
- `q_target = default_angles[isaaclab→mujoco] + action × g1_action_scale`，PD 力矩控制；
- **唯一可信的"到达姿态"是 MuJoCo 里的实测值**，不是任何模型的预测。

---

## 7. 为什么 State / History 必须进 Stage 2

同一个低矮流形 `M`：

- 机器人**站着**（`s = s_stand`）→ 需要 `stand → crouch → crawl`；
- 机器人**已经蹲着**（`s = s_crouch`）→ 应该 `crouch → crawl / crouch-walk`。

所以 `M → R` 信息不足，真正需要的是 `(M, z_p, s, H, c) → R`。History 还负责**动态连续性**：`run → decelerate → crouch → crouch-walk` 这种过渡只有看历史才一致，否则每个窗口都会重新决定一次"我现在该是什么步态"，接起来会跳。

---

## 8. 四条必须记住的约束（重复试错代价最高）

### 8.1 判据是椭球，不是盒子
`r = ‖(p−c)/semi‖`。盒子半轴当椭球半轴用，角点会到 r = 1.73。所有 `r` 都必须出自这条判据，**写入端也必须存椭球**（详见 PIPELINE §3）。

### 8.2 训练分布与评估分布必须同尺
示范流形带 `MARGIN = 1.05`，评估的 scale 1.0 不带——差 5%，实测吻合到三位小数（预测 1.029，实测 1.030）。

### 8.3 "包络内性"和"控制器跟不跟得上"必须一起看
加大倾斜扰动让策略学会用 **waist pitch** 凑姿态，而 waist pitch 正是冻结控制器基本不跟的通道——结果每一行都变坏（median 0.975 → 1.200）。**一个姿态通道如果控制器不认，训练压力再大也没用。**

### 8.4 控制器有它不认的维度

实测（扫通道后把到达关节拟合回方向）：

| 通道 | 跟不跟 |
|---|---|
| crouch（髋/膝/踝） | 跟，但饱和：要 1.6–2.0 只给到 ≈1.2 |
| twist（腰 yaw + 肩 yaw） | 跟 |
| lean（腰 pitch） | **基本不跟** |
| arms（肩 roll、肘） | 深姿态下**不跟**：手臂关键帧不被采纳 |

这限定了任何策略能要求什么。要摆手臂/躯干必须换通道——SONIC release 支持
`vr_3point_local_target` / `vr_3point_local_orn_target`（显式手/头目标）。

---

## 9. 反向数据构建（Stage 2 不从零 RL）

Stage 2 **不建议一开始从零 RL**。核心策略：

> 用已经被 SONIC 验证能执行的大量 G1 motion，反向构建动态训练数据。

```
① Motion Library        收集 walk / run / crouch / crawl / side-step / turn /
                        accel / decel / stop / recovery，覆盖速度、方向、约束
        │
② SONIC 验证           每条 R_i → SONIC → G1，记录 tracking error / stability /
                        contact / energy / collision / success / failure reason
        │
        └──► D_valid = {R_i | success_i = 1}        只留能执行的
        │
③ 动态 Manifold 反向提取  对每条有效轨迹，**逐时间片**提取：
                        R_i(t) → M_i(t)、s_t = (q, q̇, base, contact)、z_{p,t}
        │
④ 滑窗数据集            X_t = (M_t, z_{p,t}, s_t, H_t, c_t)
                        Y_t = R_{t:t+H}
                        D_dynamic = {(X_t, Y_t)}
        │
⑤ 数据增强             时间缩放 / 速度缩放 / 局部旋转 / 起始状态扰动 /
                        流形尺度扰动 / 安全距离扰动 / 障碍物随机化
                        ⚠ 增强后必须**重新校验** manifold feasibility 与 SONIC 可执行性
```

**② 和 ③ 是这套架构的关键**：它把"哪些运动是可行的"变成**实测过的事实**，而不是训练时才发现的问题。**不能只给整条轨迹一个 `walk`/`crouch` 标签——必须时间滑窗**，否则 Stage 2 永远学不会步态过渡。

这套流程与当前 ①②（数据采集 → 拆帧）是同一条思路的静态版：先记录 SONIC 真的做了什么，再从记录里反推流形。当前实现的 `M_t` 提取（`recompute_envelopes.py`）就是 ③ 的原型。

---

## 10. Stage 2 的设计要点

### 10.1 条件输入

$$
X_t = (M_t, z_{p,t}, s_t, H_t, c_t)
$$

$$
s_t = (q_t, \dot q_t, base_t, v_t, \omega_t, contact_t)
$$
$$
H_t = \{s_{t-K:t}, M_{t-K:t}, z^p_{t-K:t}\}
$$
$$
c_t = (v_x, v_y, \omega_z, goal, \dots)
$$

### 10.2 为什么用 Flow Matching

动态任务天然是**连续轨迹分布**：同一个 primitive（如 crouch）可以有多种合理展开——crouch-walk、crouch-turn、crouch-stop、crouch-accelerate。用一个确定性回归会平均成一个不存在的动作。

$$
x_0 \sim \mathcal{N}(0,I), \quad x_1 = R, \quad
\frac{dx_\tau}{d\tau} = v_\theta(x_\tau, \tau \mid X)
$$

从噪声轨迹积分到目标运动轨迹。

### 10.3 网络

```
M_t ─┐
z_p ─┤
s_t ─┼─► Condition Encoder ─► h_cond ─┐
H_t ─┤                                 ▼
c_t ─┘                        Temporal Transformer ◄── noisy trajectory x_τ
                                       │
                                       ▼
                                   Flow Head ─► v_θ(x, τ | X) ─► ODE ─► R
```

第一版直接预测 `R = {q, q̇, base, contact}_{t:t+H}`。后续可压缩到低维 motion latent。

### 10.4 多模态 + 最优选择

不同初始噪声 `x_0^(1…N)` → `R^(1…N)`，因此学的是 `p(R|M,z_p,s,H,c)` 而**不是单条确定性轨迹**。然后

$$
J(R) = w_t J_{task} + w_s J_{safe} + w_b J_{stable} + w_e J_{energy} + w_m J_{smooth} + w_c J_{clear}
$$

$$
R^* = \arg\min_i J(R^{(i)})
$$

```text
Flow Matching → Multiple Motions → Feasibility + Cost → Optimal Motion R*
```

---

## 11. 一个典型案例：直走 / 侧身 / 低矮通过

| 环境 | 流形 | 原语 |
|---|---|---|
| 开阔走廊 | `h↑, w↑` | `walk` |
| 宽度受限 | `h` 正常, `w↓` | `side-step` |
| 高度受限 | `h↓, w` 正常 | `crouch` |
| 极低空间 | `h ≪ h_normal` | `crawl` |

**不是硬编码规则**，而是 `p(z_p|M)`；具体轨迹再由 Stage 2 按状态和目标生成。

---

## 12. 阶段与验收

每个阶段落三样：**可复现命令、落盘产物、量化验收数字**。没有数字等于没做完。

| Phase | 内容 | 验收 | 状态 |
|---|---|---|---|
| **Phase 0** G1 + SONIC 基础 | G1 稳定、SONIC 执行、state/action 接口统一、轨迹可记录 | 关键帧能站住/保持 | ✅ L0 |
| **Phase 1** 静态包络 | `G1 → Envelope`：椭球、pelvis-relative 坐标、包含度计算 | 站姿 r = 1.000 | ✅ |
| **Phase 2** Primitive | `M → z_p`：先 height→crouch，再加 width/depth/orientation | 克隆拟合 ≥90% | ✅ **L2 94.9%** |
| **Phase 3** Primitive RL + 二阶模仿 | 安全/稳定/能耗/任务/平滑 | 实测落点 inside | ✅ **L3 success 1.00, r 0.82**（一阶；二阶未做） |
| — 度量口径 | 椭球判据 + 训练/评估同尺 | 记录姿态 r = 1.000 | ✅ L4 |
| — 姿态通道 | 超出当前通道的流形（倾斜/极窄） | — | ❌ **L5 卡住**（见 §8.4） |
| **Phase 4** SONIC Motion Library | walk/run/crouch/crawl/side-step/turn 等已验证 | 每类有成功样本 | 部分（walk 有，其余未系统收集） |
| **Phase 5** 反向 Manifold | `R(t) → M(t)` | 提取的 M 能装下原运动 | 部分（`recompute_envelopes` 是原型） |
| **Phase 6** Dynamic Dataset | `(M_t,z_{p,t},s_t,H_t,c_t) → R_{t:t+H}` | 滑窗数据集落盘 | 未开始 |
| **Phase 7** Flow Matching | 先单 primitive，再多 | 生成的 R 可执行 | 未开始 |
| **Phase 8** 多模态生成 | `R^(1…N)` | N 条样本彼此不同且都可执行 | 未开始 |
| **Phase 9** 最优选择 | `R* = argmin J(R)` | J(R*) ≤ 所有候选 | 未开始 |

**当前在 Phase 3 与 Phase 4 之间**：静态姿态这条闭环（`M → q`）已经通了并且数字可信；
下一步是 §8.4 那个姿态通道问题，它同时卡住 Phase 3 的倾斜/极窄流形和 Phase 4 的
crouch/crawl 步态。

### 12.1 硬规矩

1. **顺序不可跳**：上一阶段验收数字达标才开工下一阶段；
2. **每个阶段必须有量化验收数字**；
3. **接口先冻结**（§6），要改先说明理由并等确认；
4. **任何"更快/更好"的改动先给 A/B 对照数字**，再改默认值，没有对照不许动默认；
5. **不确定就做 ≤10 分钟能出数字的最小实验**，不做大重构；
6. **长任务（>2 分钟）跑完立刻汇报**：结论 + 数字 + 产物路径。

### 12.2 优先级

$$
Success > Safety > Stability > Naturalness > Diversity
$$
