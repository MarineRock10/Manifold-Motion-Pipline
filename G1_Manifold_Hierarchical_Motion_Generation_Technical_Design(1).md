# G1 人形机器人：基于流形的分层运动生成系统

> 技术设计文档 v1.0｜目标平台：Unitree G1 + SONIC

## 1. 项目目标

最终系统：

$$
Environment\rightarrow Safe\ Corridor\rightarrow Environment\ Manifold
\rightarrow Motion\ Primitive\rightarrow Dynamic\ Motion\ Distribution
\rightarrow SONIC\rightarrow G1
$$

系统不直接从点云生成全身关节动作，而是把问题分成"运动意图"和"动态运动展开"两层。

核心概率分解：

$$
p(R_{t:t+H}|M_t,s_t,c_t,H_t)
=
\sum_{z_p}
p_\theta(R_{t:t+H}|M_t,z_p,s_t,c_t,H_t)
p_\phi(z_p|M_t)
$$

其中 $M_t$ 是环境流形，$z_p$ 是运动原语，$s_t$ 是当前 G1 状态，$H_t$
是历史状态，$c_t$ 是任务命令，$R_{t:t+H}$ 是未来动态运动。

------------------------------------------------------------------------

## 2. 总体架构

``` text
RGB-D / LiDAR / Point Cloud
              |
              v
      3D Occupancy / SDF
              |
              v
      3D A* / Trajectory Opt.
              |
              v
        Safe Corridor
              |
              v
   Environment Motion Manifold
        M_1 ... M_T
              |
              v
+-----------------------------------+
| Stage 1: Motion Primitive         |
| p_phi(z_p | M)                    |
| 单个局部流形 -> 动作原语/姿态参数   |
| Imitation + RL + 二阶模仿           |
+----------------+------------------+
                 |
                 v
          Motion Primitive z_p
                 |
        +--------+--------+
        |        |        |
      State    History   Command
        |        |        |
        +--------+--------+
                 |
                 v
+-----------------------------------+
| Stage 2: Dynamic Motion Generator |
| Conditional Flow Matching         |
| v_theta(x,t | M,z_p,s,H,c)         |
+----------------+------------------+
                 |
                 v
       Candidate Motions R^(1...N)
                 |
                 v
      Safety / Stability / Task
       / Energy / Smoothness
                 |
                 v
                R*
                 |
                 v
               SONIC
                 |
                 v
                G1
```

------------------------------------------------------------------------

# 3. 环境流形生成

## 3.1 环境感知与地图

输入：

-   RGB-D
-   LiDAR
-   Point Cloud

建立：

-   3D Occupancy
-   SDF / ESDF

局部几何需要能够提供高度、宽度、深度、方向和安全距离。

## 3.2 全局路径

采用成熟的传统规划：

$$
\tau^*=\arg\min_\tau
w_dJ_d+w_cJ_{collision}+w_sJ_{smooth}+w_kJ_{curvature}
$$

3D A\* 用于搜索可行拓扑路径，Trajectory Optimization
用于平滑和动态可执行化。

## 3.3 Safe Corridor

沿 \$ au\^\*\$ 生成连续局部安全走廊：

$$
C=\{E_1,E_2,\ldots,E_T\}
$$

每个局部空间用椭球/局部几何原语描述：

$$
(x-c_t)^TA_t(x-c_t)\le1
$$

从而得到：

$$
M_t=(h_t,w_t,d_t,R_t,\theta_t,clearance_t,\ldots)
$$

Stage 1 使用单个 $M_t$；Stage 2 使用连续序列 $\{M_t\}_{t=1}^T$。

------------------------------------------------------------------------

# 4. Robot Motion Capability Manifold

除了环境流形，还需要从 SONIC 已验证运动反向构建机器人能力流形：

$$
R(t)\rightarrow M^R(t)
$$

最终有效空间可以理解为：

$$
M_t^*=M_t^E\cap M^R
$$

含义：

-   $M^E$：环境允许怎么动；
-   $M^R$：G1 实际能怎么动。

这使模型不只是"理解环境"，而是理解"这个环境对 G1 意味着什么动作"。

------------------------------------------------------------------------

# 5. Stage 1：Manifold → Motion Primitive

## 5.1 定义

输入：

$$
M=(h,w,d,R,\theta,clearance,\ldots)
$$

输出：

$$
z_p
$$

不建议把原语限制为 one-hot。建议：

$$
z_p=[k,\alpha_{height},\alpha_{pitch},
\alpha_{stance},v,\ldots]
$$

其中 $k$ 可以是：

-   walk
-   run
-   crouch
-   crawl
-   side-step
-   lean
-   turn

连续参数描述动作强度、姿态、速度和步态参数。

## 5.2 当前实验的定位

目前已经验证的：

$$
Ellipsoid\ Geometry\rightarrow Static\ Robot\ Posture
$$

例如改变椭球高度：

$$
h\downarrow\Rightarrow crouch\ amount\uparrow
$$

并存在不可行区域：

$$
h<h_{min}\Rightarrow no\ feasible\ posture
$$

这应正式作为 Stage 1 Primitive Generator 的基础实验，而不是独立的 toy
demo。

------------------------------------------------------------------------

# 6. Stage 1 网络

``` text
Single Manifold M
       |
       v
Geometric Feature Encoder
       |
       +-- ellipsoid scale
       +-- orientation
       +-- clearance
       +-- local direction
       +-- curvature
       |
       v
PointNet++ / 3D CNN / Transformer
       |
       v
Feature Fusion MLP
       |
       +------------------+
       |                  |
       v                  v
Primitive Head       Parameter Head
walk/crouch/...      height/pitch/stance/speed/...
       |                  |
       +--------+---------+
                v
               z_p
```

------------------------------------------------------------------------

# 7. Stage 1 强化学习 + 二阶模仿

## 7.1 为什么不能只模仿

模仿学习只能回答：

> 数据中机器人做过什么。

RL 可以进一步回答：

> 在当前空间约束下，什么动作更安全、更稳定、更高效。

因此采用：

$$
Imitation+RL
$$

## 7.2 奖励函数

$$
r_t=
w_{safe}r_{safe}
+w_{task}r_{task}
+w_{stable}r_{stable}
+w_{energy}r_{energy}
+w_{smooth}r_{smooth}
+w_{clear}r_{clear}
$$

安全：

$$
r_{safe}=-\max(0,d_{safe}-d_{robot})
$$

稳定：

$$
r_{stable}=-\|\theta_{base}\|^2
-\beta\|\omega_{base}\|^2
$$

能耗：

$$
r_{energy}=-\sum_i|\tau_i\dot q_i|
$$

平滑：

$$
r_{smooth}=-\|\ddot q\|^2-\eta\|j_q\|^2
$$

其中：

$$
j_q=\frac{d\ddot q}{dt}
$$

## 7.3 二阶模仿

模仿损失：

$$
L_{IM}=
\lambda_qL_q+
\lambda_{\dot q}L_{\dot q}+
\lambda_{\ddot q}L_{\ddot q}+
\lambda_cL_{contact}
$$

其中：

$$
L_q=\|q-\hat q\|_2^2
$$

$$
L_{\dot q}=\|\dot q-\hat{\dot q}\|_2^2
$$

$$
L_{\ddot q}=\|\ddot q-\hat{\ddot q}\|_2^2
$$

必要时加入：

$$
L_{jerk}=\|j_q-\hat j_q\|_2^2
$$

推荐训练：

$$
Supervised/Imitation
\rightarrow RL\ Fine-tuning
\rightarrow Second-order\ Imitation\ Refinement
$$

------------------------------------------------------------------------

# 8. Stage 2：Dynamic Motion Generator

## 8.1 条件输入

$$
X_t=(M_t,z_{p,t},s_t,H_t,c_t)
$$

其中：

$$
s_t=(q_t,\dot q_t,base_t,v_t,\omega_t,contact_t)
$$

历史：

$$
H_t=\{s_{t-K:t},M_{t-K:t},z^p_{t-K:t}\}
$$

命令：

$$
c_t=(v_x,v_y,\omega_z,goal,\ldots)
$$

输出：

$$
R_{t:t+H}
$$

## 8.2 为什么使用 Flow Matching

动态任务天然是连续轨迹分布：

$$
p(R_{t:t+H}|X_t)
$$

同一个 primitive 可能有多种合理动态展开，例如：

-   crouch-walk
-   crouch-turn
-   crouch-stop
-   crouch-accelerate

采用 Conditional Flow Matching：

$$
x_0\sim\mathcal N(0,I),\qquad x_1=R
$$

学习：

$$
v_\theta(x_\tau,\tau|X)
$$

满足：

$$
\frac{dx_\tau}{d\tau}=v_\theta(x_\tau,\tau|X)
$$

从噪声轨迹积分到目标运动轨迹。

------------------------------------------------------------------------

# 9. Stage 2 网络

``` text
M_t -----------+
z_p -----------+
State s_t -----+--> Condition Encoder --> h_cond
History H_t ---+                         |
Command c_t ---+                         |
                                           v
                                    Temporal Transformer
                                           ^
                                           |
                                   noisy trajectory x_tau
                                           |
                                           v
                                       Flow Head
                                           |
                                           v
                                v_theta(x,t | X)
                                           |
                                           v
                                      ODE / Flow
                                           |
                                           v
                                     Motion R
```

推荐第一版直接预测：

$$
R=\{q,\dot q,base,contact\}_{t:t+H}
$$

后续可以进一步压缩到低维 motion latent。

------------------------------------------------------------------------

# 10. 多模态动作生成与最优选择

使用不同初始噪声：

$$
x_0^{(1)},\ldots,x_0^{(N)}
$$

得到：

$$
R^{(1)},\ldots,R^{(N)}
$$

因此真正学习：

$$
p(R|M,z_p,s,H,c)
$$

而不是单条确定性轨迹。

然后评价：

$$
J(R)=
w_tJ_{task}
+w_sJ_{safe}
+w_bJ_{stable}
+w_eJ_{energy}
+w_mJ_{smooth}
+w_cJ_{clear}
$$

最终：

$$
R^*=\arg\min_iJ(R^{(i)})
$$

这样形成：

``` text
Flow Matching
      |
      v
Multiple Motions
      |
      v
Feasibility + Cost
      |
      v
Optimal Motion R*
```

------------------------------------------------------------------------

# 11. SONIC 反向数据构建

Stage 2 不建议一开始从零 RL。

核心策略：

> 使用已经被 SONIC 验证能够执行的大量 G1 motion，反向构建动态训练数据。

## 11.1 Motion Library

收集：

-   walk
-   run
-   crouch
-   crawl
-   side-step
-   turn
-   acceleration
-   deceleration
-   stop
-   recovery
-   不同速度、方向和环境约束

## 11.2 SONIC 验证

每条：

$$
R_i\rightarrow SONIC\rightarrow G1
$$

记录：

-   tracking error
-   stability
-   contact
-   energy
-   collision
-   success/failure
-   failure reason

保留高质量样本：

$$
D_{valid}=\{R_i|success_i=1\}
$$

------------------------------------------------------------------------

# 12. 动态 Manifold 反向提取

对于有效轨迹：

$$
R_i(t)
$$

每个时间片提取机器人周围局部包络：

$$
R_i(t)\rightarrow M_i(t)
$$

每个时间片同时提取：

$$
s_t=(q_t,\dot q_t,base_t,contact_t)
$$

并得到原语：

$$
z_{p,t}
$$

因此每个滑窗样本：

$$
X_t=(M_t,z_{p,t},s_t,H_t,c_t)
$$

$$
Y_t=R_{t:t+H}
$$

最终：

$$
D_{dynamic}
=
\{(M_t,z_{p,t},s_t,H_t,c_t,R_{t:t+H})\}
$$

**不能只给整条轨迹一个 walk/crouch 标签。必须进行时间滑窗。**

------------------------------------------------------------------------

# 13. 数据增强

在保持可执行性的前提下，对 SONIC motion 做：

-   时间缩放；
-   速度缩放；
-   局部旋转；
-   起始状态扰动；
-   流形尺度扰动；
-   安全距离扰动；
-   环境障碍物随机化。

原则：

> 数据增强后必须重新检查 manifold feasibility 和 SONIC execution
> validity。

------------------------------------------------------------------------

# 14. 为什么 State / History 必须进入 Stage 2

同一个低矮流形：

$$
M
$$

如果机器人站立：

$$
s=s_{stand}
$$

可能需要：

$$
stand\rightarrow crouch\rightarrow crawl
$$

如果机器人已经蹲伏：

$$
s=s_{crouch}
$$

则应该继续：

$$
crouch\rightarrow crawl/crouch-walk
$$

因此：

$$
M\rightarrow R
$$

信息不足。

真正需要：

$$
(M,z_p,s,H,c)\rightarrow R
$$

History 还可以解决：

$$
run\rightarrow decelerate\rightarrow crouch\rightarrow crouch-walk
$$

这样的动态连续性问题。

------------------------------------------------------------------------

# 15. 一个典型案例：直走 / 侧身 / 低矮通过

环境流形不同：

### 开阔走廊

$$
M=(h\uparrow,w\uparrow)
$$

得到：

$$
z_p=walk
$$

### 宽度受限

$$
M=(h\ normal,w\downarrow)
$$

得到：

$$
z_p=side-step
$$

### 高度受限

$$
M=(h\downarrow,w\ normal)
$$

得到：

$$
z_p=crouch
$$

### 极低空间

$$
M=(h\ll h_{normal})
$$

得到：

$$
z_p=crawl
$$

最终不是硬编码规则，而是：

$$
p(z_p|M)
$$

动态阶段再根据状态和目标生成具体轨迹。

------------------------------------------------------------------------

# 16. 工程训练顺序

## Phase 0：G1 + SONIC 基础

验收：

-   G1稳定；
-   SONIC执行；
-   state/action接口统一；
-   trajectory可记录。

## Phase 1：静态包络

$$
G1\rightarrow Envelope
$$

完成椭球、pelvis-relative坐标和碰撞/包含度计算。

## Phase 2：Primitive

$$
M\rightarrow z_p
$$

先 height→crouch，再加入 width/depth/orientation。

## Phase 3：Primitive RL + 二阶模仿

$$
M\rightarrow z_p
$$

优化安全、稳定、能耗、任务和二阶平滑。

## Phase 4：SONIC Motion Library

完成 walk/run/crouch/crawl/side-step/turn 等验证。

## Phase 5：反向 Manifold

$$
R(t)\rightarrow M(t)
$$

## Phase 6：Dynamic Dataset

$$
(M_t,z_{p,t},s_t,H_t,c_t)
\rightarrow R_{t:t+H}
$$

## Phase 7：Flow Matching

先单 primitive，再多 primitive。

## Phase 8：多模态生成

$$
R^{(1)},...,R^{(N)}
$$

## Phase 9：最优选择

$$
R^*=\arg\min J(R)
$$

## Phase 10：动态走廊

$$
M_1,...,M_T
$$

## Phase 11：A\* + TrajOpt + Safe Corridor

形成环境到连续流形。

## Phase 12：完整闭环

$$
Environment
\rightarrow M_{1:T}
\rightarrow z_p
\rightarrow Flow
\rightarrow R^*
\rightarrow SONIC
\rightarrow G1
$$

------------------------------------------------------------------------

# 17. 四团队分工

## A：G1 / SONIC / Control

负责：

-   MuJoCo G1；
-   SONIC接口；
-   trajectory tracking；
-   contact；
-   stability；
-   execution evaluation。

验收：任意生成 reference 都能被统一接口 replay，并输出 tracking/success
指标。

## B：Motion Intelligence

负责：

-   Primitive Encoder；
-   Primitive Decoder；
-   RL；
-   二阶模仿；
-   Flow Matching；
-   candidate generation；
-   trajectory ranking。

验收：Stage 1 和 Stage 2 分别可以独立训练、推理和评估。

## C：Manifold / Planning

负责：

-   Occupancy/SDF；
-   3D A\*；
-   TrajOpt；
-   Safe Corridor；
-   dynamic manifold。

验收：从地图自动生成连续 $M_{1:T}$。

## D：Data / Scaling / Evaluation

负责：

-   SONIC motion library；
-   replay；
-   manifold extraction；
-   sliding-window dataset；
-   augmentation；
-   benchmark；
-   大规模并行仿真。

验收：自动完成 $R\rightarrow M\rightarrow z_p\rightarrow sample$。

------------------------------------------------------------------------

# 18. 推荐工程目录

``` text
project/
├── configs/
├── env/
│   ├── mujoco_env.py
│   ├── g1_env.py
│   └── sonic_env.py
├── manifold/
│   ├── ellipsoid.py
│   ├── envelope.py
│   ├── sdf.py
│   ├── corridor.py
│   └── encoder.py
├── planner/
│   ├── astar3d.py
│   ├── trajopt.py
│   └── safe_corridor.py
├── primitive/
│   ├── encoder.py
│   ├── decoder.py
│   ├── policy.py
│   ├── reward.py
│   ├── imitation.py
│   └── train_rl.py
├── dynamic/
│   ├── condition_encoder.py
│   ├── temporal_transformer.py
│   ├── flow_matching.py
│   ├── sampler.py
│   └── trajectory_generator.py
├── dataset/
│   ├── motion_loader.py
│   ├── sonic_replay.py
│   ├── manifold_extractor.py
│   ├── primitive_labeler.py
│   ├── window_sampler.py
│   └── dataset_builder.py
└── evaluation/
    ├── safety.py
    ├── stability.py
    ├── tracking.py
    ├── energy.py
    └── benchmark.py
```

------------------------------------------------------------------------

# 19. 数据接口

## Manifold

``` python
{
    "center": [x, y, z],
    "scale": [h, w, d],
    "rotation": [rx, ry, rz],
    "clearance": ...,
    "direction": ...,
    "curvature": ...
}
```

## State

``` python
{
    "q": ...,
    "dq": ...,
    "base_pose": ...,
    "base_vel": ...,
    "base_omega": ...,
    "contact": ...
}
```

## Primitive

``` python
{
    "type": ...,
    "posture": ...,
    "speed": ...,
    "latent": ...
}
```

## Dynamic sample

``` python
{
    "manifold": M,
    "primitive": z_p,
    "state": s,
    "history": H,
    "command": c,
    "future_motion": R
}
```

------------------------------------------------------------------------

# 20. 关键实验

1.  **单变量静态**：$h\rightarrow crouch$。
2.  **多维静态**：$(h,w,d,\theta)\rightarrow z_p$。
3.  **多动作**：walk/crouch/crawl/side-step。
4.  **State ablation**：比较有无 $s_t$。
5.  **History ablation**：比较有无 $H_t$。
6.  **二阶模仿 ablation**：比较有无 $\ddot q/j_q$。
7.  **生成模型比较**：MLP regression、CVAE、Diffusion、Flow Matching。
8.  **多模态**：不同 $N$ 下的 diversity / validity / success。
9.  **完整走廊**：直走、狭窄、低矮、转弯、高度/宽度突变。

------------------------------------------------------------------------

# 21. 核心评价指标

Primitive：

$$
Accuracy,\quad E_\alpha,\quad Success_{primitive}
$$

Dynamic：

$$
E_q=\frac1T\sum_t\|q_t-\hat q_t\|
$$

以及：

-   collision rate；
-   fall rate；
-   tracking error；
-   energy；
-   smoothness；
-   task success；
-   minimum clearance；
-   generation latency；
-   trajectory diversity。

------------------------------------------------------------------------

# 22. 最终论文级贡献

### Contribution 1：Robot-aware Motion Manifold

$$
R\rightarrow M^R
$$

从 SONIC 验证运动反向构建 G1 的运动能力流形。

### Contribution 2：Manifold-conditioned Primitive Generator

$$
p_\phi(z_p|M)
$$

实现：

$$
Spatial\ Constraint\rightarrow Motion\ Intention
$$

### Contribution 3：Primitive-conditioned Dynamic Flow

$$
p_\theta(R|M,z_p,s,H,c)
$$

使用 Conditional Flow Matching 学习连续、多模态动态运动分布。

### Contribution 4：Optimal Motion Selection

$$
\{R_i\}\rightarrow J(R_i)\rightarrow R^*
$$

将生成分布与安全、稳定、任务、能耗和运动平滑性结合。

### Contribution 5：SONIC-executable G1 Motion

$$
R^*\rightarrow SONIC\rightarrow G1
$$

验证生成结果具有实际机器人可执行性。

------------------------------------------------------------------------

# 23. 最终统一表述

整个系统可以概括成：

$$
\boxed{
Environment
\rightarrow
Manifold
\rightarrow
Primitive
\rightarrow
Flow\ Matching
\rightarrow
Optimal\ Motion
\rightarrow
SONIC
\rightarrow
G1
}
$$

其中：

> **Stage 1 解决"空间告诉机器人应该采用什么运动原语"；Stage 2
> 解决"给定原语、当前状态、历史状态和任务目标，机器人接下来具体如何连续运动"。**

最终目标不是让一个大模型直接学会所有 locomotion，而是建立：

$$
\boxed{
Geometry
\rightarrow
Intention
\rightarrow
Dynamic\ Distribution
\rightarrow
Execution
}
$$

这一分层运动智能框架。
