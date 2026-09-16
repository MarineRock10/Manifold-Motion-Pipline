# 当前实现：技术路线与实测

> 本文是**唯一**的实现文档：三段管线各自怎么做的、为什么这么做、实测数字是多少、
> 哪些坑踩过、哪些还没解决。所有数字都可用文末命令重跑复现。
> 配套：[`ARCHITECTURE.md`](ARCHITECTURE.md)（分层、接口、约束）。
> 取代了 `STATIC_MANIFOLD_PIPELINE.md`、`VISUALIZATION.md`、`manifold_g1/README.md`（历史见 git）。
>
> 日期：2026-09-16。

---

## 1. 三段管线

```
① 数据                          ② BC                        ③ SONIC 在环微调
采集 → 能力图 → 拆帧示范集       克隆 (M, s) → q              真控制器在环，扰动流形
                               bc/bc_policy.pt              sonic_rl/policy_sonicrl.pt
```

| 阶段 | 命令 | 产物 | **它自己的指标** | 数字 |
|---|---|---|---|---|
| ① 数据 | `collect` → `capability build` → `demos build` → `dataset build` | `dataset/pose_demos.npz` | SONIC 跟踪误差 | **0.054 rad**；137 clips / 18705 窗口 / 10336 姿态对 |
| ② BC | `python3 -m manifold_g1.bc train` | `bc/bc_policy.pt` | 克隆姿态在**记录包络**内的比例 | **9336/9840 = 94.9%**；val 姿态误差 0.087 rad |
| ③ 在环 | `python3 -m manifold_g1.sonic_rl run` | `sonic_rl/policy_sonicrl.pt` | **实测到达**姿态在**扰动包络**内的比例 | success **1.00**，r **0.82**，30 轮 gate 6 |
| 门禁 | `verify_sonic` | `bc/verify_sonic.json` | 姿态交给真 SONIC 后实测的 r 分布 | **6/10 inside**，r(sonic) 中位 **0.976** |

**三个指标不可互换**（这是踩过最多次的坑）：拿 ③ 的流形测 ② 得到的是泛化不是 ② 的质量；早期由此得出"BC 只有 3/10"的结论是错的（那 3/10 是 `report_specs` 泛化集，不是同分布）。

---

## 2. ① 数据

### 2.1 采集

`clip.py` 驱动 SONIC 走一段运动，逐 tick 记录：`q_act`（实测关节）、`q_cmd`（SONIC 给的指令）、根位姿速度、接触、命令，以及**每 `envelope_stride` tick 一次的包络**。

包络的写法是**关键**，也是本文档 §3 一整节的来历：

```python
rel = 身体表面点 − 骨盆                        # 骨盆系
env_semi = fit_ellipsoid(rel)["semi"]          # 椭球半轴（不是各轴 max|rel|）
```

### 2.2 示范集：拆帧

`demos.py` 把 clip 拆成 `(M, q)` 对：

```
pose demo = (M, q)      # q 是 29 维完整姿态（policy 顺序，相对默认站姿的增量）
                        # M 是该姿态自己的拟合椭球 × MARGIN(1.05，留 5% 余量)
```

保留全部 29 关节而不是投影到几个手设计通道，因为走路帧的大部分构型在肩 roll / 肘 / 腕 / 髋 pitch 上，任何小基都盖不住——**风格差异正好住在那里**。

### 2.3 数据结构事实

| | |
|---|---|
| 流形数 | **9840** |
| 示范姿态数 | **10336** |
| 单姿态流形 | **9466**（374 个 ≥2，10 个 ≥4） |

**这不是缺陷，是阶段边界**：② 的目标就是克隆示范的**姿态意图**，`M → 单个 q` 正是它该学的；多样性是 ③ 在扰动流形上探索出来的。数据侧真正值得记的是：不同流形之间的姿态差异存在（每关节 std 均值 8.6°，与流形参数相关 **+0.42**），所以 `M → q` 有信号可学——这是 ② 成立的前提。

体检命令：`dataset show`（速度/包络/朝向覆盖）、`demo_spread`（流形内分歧）。

---

## 3. 度量口径：两个已修的 bug（这段决定了所有 r 怎么读）

### 3.1 包络是盒子、判据是椭球

`envelope_semi` 曾存各轴 `max|rel|`（**盒子**），而包含性判据是**椭球**。盒子角点按椭球算是 `sqrt(3)=1.73`。后果不微妙：

- 记录姿态在**自己的**包络下算出 **r = 1.18–1.57（中位 1.40）**——"装进流形"连产生包络的那条姿态都做不到；
- 站姿在家族 scale 1.0 下是 **1.145**，所以 `report_specs` 的 `h1.00_w1.00_d1.00` 那行站着不动也永远判 outside。

**修法**：两个写入端（`recompute_envelopes.py`、`clip.py`）都改用 `fit_ellipsoid`；`family.BASE_SEMI/BASE_CENTER` 改成与示范集同一把尺。核验：记录姿态 **r = 1.0000**，站姿 **r = 1.0002**。

精确说明（因为我自己先说错过）：`fit_ellipsoid` 是**均匀**膨胀（一个标量乘三轴），aspect ratio 不变；逐轴自由优化只能再小 11%（中位体积比 0.893），所以均匀够用。它不是一个全局常数——膨胀系数逐 tick 变化（1.18–1.57，中位 1.40）。

我先前说"fit 比均匀膨胀小 31%"是**错的**：那次比较用了两个不同 tick 集合的 k，是噪声。二者本来就是同一个操作。

### 3.2 中心是位置，不是尺度

`ManifoldSpec.build` 曾把中心也按 height/width/depth 缩放。中心是**位置**（身体在骨盆下方 0.11 m），缩放等于把椭球从身体上挪开，spec 越极端偏得越多。实测代价：站姿 r 偏 0.005。

### 3.3 训练与评估必须同尺

示范流形带 `MARGIN=1.05`，而 `report_specs` 的 scale 1.0 不带。策略在自己流形里 0.98 的姿态，家族口径下读 `0.98 × 1.05 = 1.029`——**预测 1.029，实测 1.030**。`family.MARGIN` 现在是唯一定义，`verify_sonic` 因此从 2/10 升到 6/10。

---

## 4. ② BC：克隆

### 4.1 为什么必须先从示范开始

从这个任务从头 RL 会崩：奖励稀疏（整身进椭球并保持住才算成功）、动作 29 维、流形池大部分难到随机探索几乎碰不到成功。四次训练都在前 ~100 轮冲到峰值然后漂到没有任何示范覆盖的姿态、掉出所有流形、再也回不来——**没有可用梯度**的特征。

示范集正好是 `(M, s) → q` 的监督信号：先训它，把策略放进姿态空间的正确区域，RL 只需在区域内精修。

### 4.2 实现要点

- 观测由**真环境**（`torch_env.TorchPrimitiveEnv.obs`）构造，不是另写一份——否则训练和部署会悄悄分叉；
- 标签是记录姿态本身；
- 损失 `mse(action_to_pose(mu), q_target)`，`action_to_pose = tanh(a) × 1.4`。

**教训**：曾把克隆训成 `mse(tanh(mu)*1.4, q)` 而环境里用的是 `clamp(mu)*1.4` 的等效语义，两者拟合的尺度不同，环境收到 mu≈2.2 被 clamp 到 1，姿态和克隆目标完全不是一回事——RL 微调在好的 warm start 之后立刻崩。`action_to_pose` 现在只有 `pose_policy.py` 一处定义。

### 4.3 实测

```
val 姿态误差 0.087 rad        （对示范的偏差，同分布内）
r_cloned 0.85 vs r_demo 0.95  （克隆姿态 vs 示范姿态在各自流形里的 r）
全量 9336/9840 = 94.9%        （拟合自己的训练流形）
```

---

## 5. ③ SONIC 在环微调

### 5.1 为什么不用模型

早期路线学一个残差模型 `Δ̂(q)` 预测 SONIC 对指令姿态实际做什么，再用它做 RL（几何模型能跑几千轮/秒）。**这条路线已删除**：那个偏差（优于忽略残差 51%）不等于预测正确，目标变成了"我以为控制器会做什么"。

而真控制器**够快**：一个姿态约 0.3 s 物理，16 流形 × 6 tick ≈ 30 s 一轮，30 轮全量 879 s。既然可以直接跑真的，就没有理由用一个会骗人的代理。

### 5.2 实现（`sonic_rl.py`）

**刻意简单**：

- **串行、不批量**：一次一个姿态过 `KeyframeEnv`。没有环境状态与 autograd 的别名问题、没有 in-place 版本冲突、没有 GPU 张量要同步。微调是几千个 episode，不是几百万；
- **运动学门控，包络奖励**：会摔倒的姿态（质心偏离支撑、抬脚）或需要超出关节极限的姿态**一分不得**，无论它对包络内性做了什么。**只有在机器人真能保持的姿态上，装进流形才是目标**；
- **奖励用实测到达的姿态**，不是预测的；
- **扰动**：每个 episode 取一条示范，把它**自己的**包络重新塑形（半轴 ±8%、中心 ±2 cm），示范本身固定。这才使微调意味着"这条记录的动作，适配这个新形状的流形"，而不是"一个新流形、从头摆"。

奖励权重：`w_gate=12`（门控，通过才给包含度）、`w_containment=4`、`w_outside=8`、`w_track=2`、`w_imitation=1.5`、`success_bonus=4`。每步保持 0.6 s。

### 5.3 关键修复：给部署用的动作打分

PPO 用**采样**动作探索，但部署用**分布均值**。早期训练日志报的是采样姿态的包含度，而每次评估量的是均值的——这个落差让一次运行看起来在改善（r 0.90→0.84）而独立检查里并不比 BC 好。现在 `mean_probe` 给均值动作也打分，且**日志报的就是它的结果**。

同一个陷阱在几何路线里出现过五次，所以这条记在这。

### 5.4 实测

```
30 轮 × 16 流形 × 6 tick，扰动 8%，repeats 3，879 s
末 10 轮：success 1.00，r(achieved) 0.82，gate 6/480
```

一致性评估（同一个检查、同一起点，真控制器）：

```
                 BC            policy_sonicrl
mean r(achieved) 0.87          0.88
inside           3/4           3/4
mean track       0.060 rad     0.065 rad
```

**③ 不一定要超过 ②**——它的职责是把 ② 的结果约束到它自己的指标上（实测落点），并在此过程中学会适应流形形变。现在两者持平，符合预期。

---

## 6. 未解决：卡在 L5

`report_specs` 的 **t=±10**（倾斜）和极窄两行是最大失败（r 1.12–1.33，其余多在 1.0 附近）。三步查清了：

1. **训练分布里根本没有倾斜。** 记录包络存了 `quat`，但那是走路时骨盆的 **yaw**（中位 87°），只有 0.5% 带真正的矢状倾斜；`_reshape` 也只扰动半轴和中心，不扰动朝向。所以 t=±10 是纯外推。
2. **倾斜流形几何上可解。** 站姿在 t=+10/−10 下是 r 1.13/1.22，但把身体整体**俯仰 +7°/−8°** 就能到 **0.97/0.98**。所以"塞得进去"有解。
3. **策略的幅度不够。** 它在 +10 与 −10 之间的姿态只差 0.065 rad（≈3.7°），而几何要 7–8°。方向对，幅度差一半。

**然后"加大倾斜扰动"这条路试过了，是错的：**

| 训练扰动 | r(sonic) 中位 | inside |
|---|---|---|
| 无倾斜（当前发布） | **0.975** | **7/10** |
| tilt 6°，30 轮 | 1.000 | 5/9 |
| tilt 14°，20 轮 | **1.200** | **1/7** |

tilt14 那次**每一行都坏了**，不只倾斜行：waist 跟踪误差从 0.02–0.07 跳到 **0.26–0.31 rad**。原因清楚——策略学会用 **waist pitch** 去凑，而 waist pitch 正是冻结控制器基本不跟的通道（ARCHITECTURE §6.4）。它在生产跟踪层执行不了的动作。所以默认值已改回 0。

**结论**：倾斜需要的是一个**控制器能执行的姿态通道**，不是同方向加更多压力。可选方向：

1. **换执行通道**——SONIC release 支持 `vr_3point_local_target` / `vr_3point_local_orn_target`（显式手/头目标），那才是能真正摆手臂和躯干的通道；
2. **收窄家族**——把 `report_specs` 收到策略实际可执行的范围内，把"做不到"写进文档而不是当失败。

---

## 7. 可视化（三个 viewer，各对应一段）

| | 命令 | 显示什么 |
|---|---|---|
| ① 数据 | `python3 -m manifold_g1.view_data replay` | 记录回放，椭球是当时的包络，叠加"命令 vs 实际" |
| ② BC | `python3 -m manifold_g1.show_bc` | 克隆在**它被训练的流形**上：记录姿态（青）vs 克隆姿态（绿）vs SONIC 实测落点（机器人） |
| ③ 在环 | `python3 -m manifold_g1.show_rl` | 微调策略在**扰动分布**上；按键实时改流形（`u/i` 高、`j/k` 宽、`,/.` 深、`z/x` 偏移、`t/y` 倾斜），每次按键重解姿态并重跑 SONIC |

共同前提：**机器人实体永远是 SONIC 执行后的结果**，不是运动学摆位——骨盆高度存在自由关节里，纯摆位会让"蹲下"看起来腿在折叠而身体不降，那是假象。

**看 `show_bc` 的关键**：它按 `spread`（流形内记录姿态的分歧）排序，小 spread 的在前面——只有在那些行上"流形决定姿态"，克隆才可判。屏幕上：
- `r recorded` = 记录姿态在自己包络里的 r（数据质量，应为 1.00）
- `r clone` = 克隆姿态在同一包络里的 r（**克隆质量**）
- `r achieved` = SONIC 实测落点的 r（**②的真正指标**）

`show_rl` 的 `r on recorded envelope` 是与**未扰动**原包络之比，和 `r achieved` 一比就知道"这次形变帮了多少"。

---

## 8. 数值报告（不开窗口）

```bash
python3 -m manifold_g1.dataset show                    # ① 数据体检
python3 -m manifold_g1.demo_spread                     # ① 示范集内部分歧
python3 -m manifold_g1.bc eval --policy reports/manifold_g1/bc/bc_policy.pt --device cpu
python3 -m manifold_g1.sonic_rl eval --manifolds 10 --steps 4 --device cpu \
    --policy reports/manifold_g1/sonic_rl/policy_sonicrl.pt
python3 -m manifold_g1.verify_sonic --policy reports/manifold_g1/sonic_rl/policy_sonicrl.pt
python3 -m manifold_g1.eval_session --clips 3 --per-clip 4 --device cpu   # 包络口径核查
```

`eval_session` 打印两种半径（椭球 / 盒子）作重建核对：椭球列应读 1.000（记录姿态正好在包络表面），盒子列 ~0.71（盒子角点效应，只作形状对照）。

---

## 9. 代码地图（`manifold_g1/`）

| 模块 | 职责 |
|---|---|
| `constants.py` | 关节序置换、默认角、KP/KD/动作尺度、力矩上限（全部镜像 `gear_sonic_deploy`） |
| `env.py` | MuJoCo 平地场景、PD 力矩控制、状态读出（200 Hz 物理 / 4 子步） |
| `sonic.py` | ONNX 编解码 + 部署的观测拼接（10 帧历史、零填充） |
| `reference.py` | `KeyframeReference`：T=1 的运动参考。`set_joints` 收完整 29 维姿态 |
| `planner.py` | `planner_sonic.onnx` 的薄封装（运动学生成器），采集时用来生成步态 |
| `ppo.py` | 最小 PPO（clipped surrogate + GAE），actor-critic MLP，逐维学习 log_std |
| `keyframe_env.py` | **执行层**：MuJoCo + 冻结 SONIC，收姿态、报实测落点 |
| `manifold.py` / `family.py` | 椭球几何与场景；流形家族（height/width/depth/offset/tilt + `MARGIN`） |
| `pose_policy.py` | **接口唯一定义**：15 维观测 + `tanh` 动作→姿态 |
| `demos.py` | 示范集：拆帧、`demo_key`/`demo_values`、`load_demos` |
| `torch_env.py` | 批量环境（观测与半径），供 BC 与验证共用 |
| `bc.py` | ② 克隆 |
| `sonic_rl.py` | ③ 在环微调 |
| `kinematics.py` | GPU 批量前向运动学（逐 geom 对过 MuJoCo） |
| `body_model.py` / `body_envelope.py` | 姿态→身体表面点；表面采样与 `fit_ellipsoid` |
| `viewer.py` | 三个 viewer 共用：位姿点云、`settle` 松弛循环、`SETTLE_STEPS` |
| `paths.py` | 所有工件路径的唯一定义 |
| `collect.py` / `capability.py` / `calibrate.py` / `clip.py` / `dataset.py` | ① 数据各步 |
| `recompute_envelopes.py` | 重写存盘包络到椭球口径（§3.1 的修正所在） |
| `verify_sonic.py` / `eval_session.py` / `demo_spread.py` | 门禁与核查工具 |
| `view_data.py` / `show_bc.py` / `show_rl.py` | 三个 viewer |

**已删除**（历史见 git，备份 `~/sonic_backup_20260916_1637/`）：`primitive.py`、`primitive_torch.py`（环境部分存为 `torch_env.py`）、`residual.py`、`residual_data.py`、`compare.py`、`static_fit.py`（`BodyModel` 抽为 `body_model.py`）。
