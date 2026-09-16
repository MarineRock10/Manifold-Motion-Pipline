# 最小目标与实现路径

> 配套：[`STATIC_MANIFOLD_PIPELINE.md`](STATIC_MANIFOLD_PIPELINE.md)（已跑通并验证的静态链路）、
> `G1_Manifold_Hierarchical_Motion_Generation_Technical_Design(1).md`（完整架构）。
> 本文定义**当前的最小目标**和到达它的路径。

---

## 1. 最小目标

**把整条 pipeline 端到端跑通，内容只要"走路"。**

```
手写走廊 M_1..M_T  →  z_p = walk(v, heading)  →  R（走路运动）  →  SONIC  →  G1  →  指标
   环境→流形            Stage 1: M→原语          Stage 2: 生成运动      执行      评估
```

不追求任何一格做到最好，只要求：

1. **每一格都存在且真的在跑**（不是空壳、不是写死的 return）；
2. **格与格之间的接口先定死**（`M` / `z_p` / `R` 的数据格式，见 §3），这样后面把某一格换成更强的模型时，其余部分不用动；
3. **端到端有指标**：走得动吗、跟踪误差多大、身体是否一直在流形里、速度多少。

第一轮每格都用最简版本（Stage 1 常数速度、Stage 2 用 planner 生成步态），**先把管子接通**，再逐格升级。

---

## 2. 每一格怎么填：最简版 → 升级版

| 格 | 第一轮（最简，先跑通） | 之后升级 | 现有资产 |
|---|---|---|---|
| 环境 → 流形 `M_1..M_T` | **手写走廊**：一列椭球沿路径排布，高度/宽度可调 | 点云/SDF → 3D A* → Safe Corridor | `manifold.py` 的 `build_scene`/`update_visuals` 还在；`tunnel()` 在 git 历史里 |
| Stage 1 `M→z_p` | **常数速度**（或查表：局部高度→速度档） | 学 `p(z_p|M)`（RL / 监督） | `torch_env.py` 的批量环境可直接改造（旧的 `static_fit` 训练器已删除） |
| Stage 2 `(M,z_p,s,H,c)→R` | **planner_sonic 生成步态**（已接通，10 Hz 重规划 + 8 帧 crossfade） | Conditional Flow Matching（用第一轮采到的数据训） | `planner.py` + `PlannedReference`（已恢复）；采集端 `clip.py` |
| 执行 | **冻结 SONIC + MuJoCo**（已有） | 换 `vr_3point` 接口 / 非冻结 WBC | `env.py` + `sonic.py` + `keyframe_env.py` |
| 评估 | 逐 tick 记录 + 汇总指标 | 加 `J(R)` 排序、多候选 | `clip.py` 的记录器（命令 vs 实际、接触、包络） |

**关键点**：Stage 2 从 planner 换成 flow matching 时，**`R` 的数据格式不变**——所以现在就能把链路跑通，生成模型后补，pipeline 不用改。

---

## 3. 先定死的接口

```python
M_t   = {"center": [x,y,z], "semi": [h,w,d], "quat": [w,x,y,z]}   # 走廊的局部流形（椭球）
z_p   = {"type": "walk", "speed": v, "heading": ψ}                 # 第一轮只有一种原语
R     = {"t": [H], "q": [H,29], "dq": [H,29], "base_pos": [H,3],
         "base_quat": [H,4], "contact": [H,2]}                     # 与 SONIC 参考同构
metrics = {"track_err":…, "containment_r":…, "speed":…, "fell":…, "success":…}
```

`R` 与 `SonicController` 吃的参考缓冲（`ReferenceBuffer`：joint_pos/joint_vel/root_pos/root_quat）是同一套字段，所以"planner 版"和"flow matching 版"对执行层是同一个东西。

---

## 4. 现状盘点

| 已有 | 说明 |
|---|---|
| 执行层 ✅ | 冻结 SONIC + MuJoCo 闭环，50 Hz，关键帧参考与计划参考都能跑 |
| 运动生成（占位版）✅ | `planner.py` + `PlannedReference`（10 Hz 重规划、8 帧 crossfade） |
| 采集/评估 ✅ | `clip.py`：逐 tick 记命令 vs 实际、接触、骨盆锚定包络，落 npz+json |
| 静态流形/姿态 ✅ | 身体模型、包络、pelvis-relative 包含度、静态姿态策略 |
| 可视化 ✅ | 原生 viewer（椭球 + 曲线），键控流形参数 |

| 缺 | 需要做什么 |
|---|---|
| **走廊流形** | 恢复 `EllipsoidManifold.tunnel()`（椭球链 + 可视化） |
| **走廊内的包含度** | 走路时逐 tick 判定：取沿路径的局部流形，身体包络 pelvis-relative 落在其内 |
| **episode runner** | 走廊 → z_p → planner → SONIC → 记录 → 指标汇总，一条命令跑一个 episode |
| **走廊扫描** | 高度/宽度网格各跑一遍，输出"能不能走通"的表（pipeline 跑通的证据） |

---

## 5. 步骤（按依赖排序）

1. **走廊**：恢复椭球链几何 + 场景可视化（`tunnel()` 在 git 历史，`build_scene`/`update_visuals` 在仓库里）。
2. **走廊包含度**：实现"沿路径的局部流形 + 走路中的包含度判定"，与静态那套同口径（pelvis-relative、极点 landmark）。
3. **episode runner**：串起 §1 的五个格子，逐 tick 记录（复用 `clip.py` 的记录器），产出 metrics。
4. **走廊扫描**：高度（如 1.7 / 1.4 / 1.1 m）× 宽度，各跑几个 seed，得到成败表。**这张表就是"整条 pipeline 跑通"的证据。**
5. **升级（各自独立，接口不变）**：
   a. Stage 1 学习化（M→速度档，先查表再 RL）；
   b. Stage 2 换 conditional flow matching（用第 4 步采到的走路数据训，N 候选 + `J(R)` 排序）；
   c. 多原语（引入蹲、转、侧移）。

---

## 6. 这一轮明确不做

点云/SDF/3D A*/TrajOpt（走廊手写）、多原语（只有 walk）、二阶模仿（没有示范数据）、真实机器人、flow matching（放到第 5 步）。

---

## 7. 成本（实测）

| 动作 | 耗时 |
|---|---|
| 录制 4 s 运动片段 | 6.6 秒（≈0.6× 实时） |
| 静态策略训练 600 轮 | ~5 分钟 |
| SONIC 验证 10 个流形 | 15 秒 |
| 身体模型标定 162 样本 | 2.5 分钟 |

→ 算力不是瓶颈；瓶颈在**走廊几何与包含度判定的接口设计**（第 1–2 步），以及"哪些走廊真的走得通"这件事本身要用执行证据回答。
