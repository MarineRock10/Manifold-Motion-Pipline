# 可视化使用说明

每个阶段配一个 viewer，只留三个：

| 想看 | 命令 | 评的是什么指标 |
|---|---|---|
| **① 数据集**：录下来的动作本身 | `view_data` | 跟踪误差、包络、接触（数据质量） |
| **② BC 之后**：克隆在它自己的流形上 | `show_bc` | **请求姿态的包络内性**，经 SONIC 实测 |
| **③ RL 之后**：微调策略 + 按键调流形 | `show_rl` | **实际到达姿态的包络内性**（微调的奖励） |

共同前提：**机器人（实体）永远是 SONIC 执行后的结果**，不是运动学摆位——骨盆高度存在自由关节里，纯摆位会让"蹲下"看起来腿在折叠而身体不降，那是假象。

三个阶段的指标**不可互换**，不要拿 ② 的数字去判 ③。详见最后一节。

两条核查用的命令行工具（不开窗口）也留了：`eval_session`（包络形状）、`demo_spread`（示范集内部分歧）。

---

## ① `view_data` —— 采集的数据

```bash
# 回放：机器人按记录的关节角播放，椭球是它当时的包络，叠加显示"命令 vs 实际"
python3 -m manifold_g1.view_data replay --clip reports/manifold_g1/clips/ep0000_seed1000.npz

# 整批数据总览（轨迹、速度直方图、包络分布、朝向覆盖）-> overview.png
python3 -m manifold_g1.view_data plots

# 单个 episode 时序（跟踪误差、速度、包络、接触与命令）-> ep*.png
python3 -m manifold_g1.view_data episode --clip reports/manifold_g1/clips/ep0000_seed1000.npz
```

**按键**：`P` 暂停 · `N`/`M` 前后退 0.5 s · `E` 关/开椭球 · `T` 关/开轨迹 · `R` 重头。

屏幕上：`cmd vel` 是发给 SONIC 的速度指令，`track` 是 `|q_cmd − q_act|`——**这个数就是数据质量的上限**，后面两个阶段不可能比它更准。底部 `contact L/R` 是左右脚触地。

**先看这里**：如果 `track` 普遍 > 0.1 rad，或者包络曲线乱跳，先修数据，不要训策略。

---

## ② `show_bc` —— 克隆在它被训练的流形上

```bash
# 默认：随机抽 400 个 recorded 流形打分，窗口马上开
python3 -m manifold_g1.show_bc

# 看全部 9859 个（慢，几分钟；但汇总数字就是整个训练分布，和 `bc eval` 一致）
python3 -m manifold_g1.show_bc --sample 0

# 只看 recorded 姿态内部一致的流形（能真正判克隆的那些）
python3 -m manifold_g1.show_bc --max-spread 0.02
```

BC 的训练对是 **(记录的包络, 记录的姿态)**，所以唯一的诚实评法是拿它和这个对子比，在数据覆盖的流形上比。

### 画面里是什么

| 元素 | 颜色 | 含义 |
|---|---|---|
| **机器人（实体）** | — | 被请求姿态交给 SONIC 后实际到达的结果 |
| **记录姿态点云** | 青 | 目标：包络是从它身上量出来的 |
| **克隆姿态点云** | 绿 | 克隆对同一包络给出的答案（按 `S` 切到"记录姿态"时青点云单独显示） |
| **半透明椭球** | 蓝 | 当前流形（骨盆系，跟随骨盆） |

右上角三个数：

| 字段 | 含义 |
|---|---|
| `r recorded` | 记录姿态在这个包络里的内性。**这是数据质量**：> 1 说明这个包络根本装不下产生它的身体，那行不能用来判克隆 |
| `r clone` | 克隆姿态在同一包络里的内性。**这是克隆质量** |
| `r achieved` | SONIC 实际留下的姿态的内性——②的真正指标 |

### 按键

`N`/`M` 上下一个流形 · `1`–`9` 跳 · `S` 切换请求"克隆姿态 ↔ 记录姿态" · `R` 重新保持 · `P` 暂停。

启动时终端打印抽样规模和两个通过率，以及每个流形的 `spread`（记录姿态之间的分歧）。

### 实测（全量 9859 个流形）

```
记录姿态装进自己的包络（椭球判据）：  9859/9859  (100%)
克隆姿态装进同一包络（椭球判据）：    9341/9859  (95%)
```

**这解释了之前"BC 只有 3/10"的困惑**：那个 3/10 是在 `family.report_specs()` 那 10 个手挑流形上测的，属于分布外泛化，不是 BC 自己的指标。BC 在它被训练的分布上是 95%。判 BC 用 `r clone`，判泛化才用 `report_specs`。

不过要注意：**这一列的 100% / 95% 是在椭球判据下、对着盒子包络量出来的**（见下一节的形状问题）。它作为"克隆有没有学会 M → q"的相对比较是有效的（分母一致、流形一致），但绝对值不能读成"离边界还有多远"。

### 一个必须知道的数据事实

`--sample 0` 的打分同时暴露了示范集的结构：**9859 个流形里 9480 个只有 1 个记录姿态**（379 个 ≥ 2，10 个 ≥ 4）。也就是说每条记录几乎只贡献一对 (M, q)。

后果：`spread` 这一列在多数行是 0（只有一个姿态，无从分歧）。克隆在单姿态流形上"拟合"得很好，但它学的是 **M → 单个 q**，不是 M → q 的分布。想要多样性（同一 M 下的多种风格），数据侧得先补——见 `demo_spread`：

```bash
python3 -m manifold_g1.demo_spread    # 每个流形内记录姿态的分歧
```

### ⚠️ 包络是"盒子"，流形是"椭球"——这两个形状不一样

这是验证 viewer 时量出来的、会影响所有 r 数值的问题：

| | 形状 | 站姿的 r |
|---|---|---|
| 录下来的 `envelope_semi` | 各轴 `max|rel|`，即**轴对齐盒子** | 盒子判据下正好 **1.000**（0 个点越界） |
| pipeline 里的 `EllipsoidManifold` | **椭球**，其 r=1 等值面内切于那个盒子 | **1.35**（盒子的角点在 r = 1.73） |

后果：用椭球判据去评盒子包络时，"r > 1"**不等于**"跑到记录的身体包络外面了"，只等于"跑到内切椭球外面了"——而**任何一个记录姿态都无法满足椭球判据**。站姿在它自己的包络下就是 1.35，正姿表面约 **39%** 的点在 r > 1。

所以：

- 所有的 `r achieved ≈ 0.8` 不要读成"几乎贴边"；它对应的盒子内性大约在 0.6 左右；
- `w_outside` 惩罚追的是一个**盒子数据描述不出来的形状**；
- 修法不在 viewer 里，而在测量时就把包络定义成椭球：把盒子半轴按"能容纳身体"的系数放大（站姿是 **1.35**），`recompute_envelopes` 是改这个的地方。

`eval_session` 同时打印两种半径，就是为了让这个差可见：

```bash
python3 -m manifold_g1.eval_session --clips 3 --per-clip 4 --device cpu
#   box 列：记录姿态 12/12 都是 1.000（验证重建正确）
#   ellipsoid 列：记录姿态 1.3~1.5，克隆 1.5~2.2
```

---

## ③ `show_rl` —— 微调后，按键调流形

```bash
python3 -m manifold_g1.show_rl
python3 -m manifold_g1.show_rl --policy reports/manifold_g1/sonic_rl/policy_sonicrl.pt
python3 -m manifold_g1.show_rl --perturb 0.12 --device cpu
```

默认用 SONIC 在环微调的 `sonic_rl/policy_sonicrl.pt`，找不到就退回 BC。

采样的分布**就是微调的分布**：随机抽一条 recorded 包络，按训练同款规则扰动（各半轴 ±`--perturb`，中心 ±2 cm），再把策略姿态交给 SONIC 保持。

### 按键：形状参数是这一屏的主角

每按一次形状键，流形当场重建、策略当场重解、SONIC 当场重跑：

| 键 | 调什么 | 范围 |
|---|---|---|
| `u` / `i` | 压扁 / 抬高（height） | 0.55–1.30 |
| `j` / `k` | 收窄 / 放宽（width） | 0.50–1.30 |
| `,` / `.` | 前后压扁 / 加深（depth） | 0.45–1.30 |
| `z` / `x` | 椭球左右平移 | ±0.10 m |
| `t` / `y` | 中轴后倾 / 前倾 | ±25° |
| `N` / `M` | 换一条 recorded 包络 + 重新扰动 | |
| `S` | 切换请求"策略姿态 ↔ 记录示范姿态" | |
| `R` | 重抽扰动 | `P` 暂停 |

### 屏幕上

```
0.27,0.40,1.03,0.06,-0.01,-0.12   perturb 8%  h 1.00 w 1.00 d 1.00 offset +0.00 tilt +0deg
requesting the policy pose   holding 2.4s
                        r requested 1.085   r achieved 0.942   r on recorded envelope 1.091
                        joint error 0.026 (arms 0.031)   pelvis z 0.79
```

| 字段 | 含义 |
|---|---|
| `r requested` | 几何模型对策略姿态的判断（不经 SONIC） |
| `r achieved` | **SONIC 实际留下的结果——微调真正最大化的就是这个** |
| `r on recorded envelope` | 同一姿态在**未扰动**的原包络下的内性；和 `r achieved` 比，就是"这次形变帮了多少忙" |
| `joint error (arms …)` | 请求 vs 实际；SONIC 对手臂最差 |
| `pelvis z` | 骨盆高度，压扁时应真的降低（证明接了物理） |

启动时会打印记录里的微调摘要（30 轮、末 10 轮 success 1.00、r 0.82、gate 6），和屏幕上的实测对照着看。

### 一个实测用法：压扁（连按 `u`）

蹲幅随流形**单调增加**（条件化成立），但 height 0.75 以下守不住——那是策略的蹲深权限边界，不是 bug。

---

## 三个阶段的指标不能互换
这是踩过最多次的坑，写在文档里当提醒：

| 阶段 | 它自己的指标 | 怎么测 |
|---|---|---|
| ① 数据 | SONIC 跟踪误差、包络覆盖 | `view_data` / `dataset show` |
| ② BC | **请求姿态**在 recorded 包络内的比例 | `show_bc` 的 `r clone`、`bc eval` |
| ③ RL 在环 | **实际到达姿态**在扰动包络内的比例 | `show_rl` 的 `r achieved`、`sonic_rl/train_log.csv` |

- 拿 ③ 的流形去测 ②，测到的是**泛化**，不是 ② 的质量。之前由此得出"BC 只有 3/10"的结论是错的。
- 拿 `primitive_torch/policy.pt`（几何 RL，用残差模型预测 SONIC）的数字去比 ③ 也不对：那个阶段的奖励是**预测的**落点，不是实测的。现在有效的主线是 ①数据 → ②BC → ③SONIC 在环微调。

### 数值报告（不开窗口）

```bash
# ③ 的准入检查：把姿态交给真 SONIC，量实际到达（r 的分布，不只是通过率）
python3 -m manifold_g1.verify_sonic --policy reports/manifold_g1/sonic_rl/policy_sonicrl.pt

# ② 的评估：与示范的偏差（分布内）
python3 -m manifold_g1.bc eval --policy reports/manifold_g1/primitive_torch/bc_policy.pt

# 包络形状核查：盒子 vs 椭球两种半径并排（见上一节）
python3 -m manifold_g1.eval_session --clips 3 --per-clip 4 --device cpu

# 数据体检 / 示范集内部分歧
python3 -m manifold_g1.dataset show
python3 -m manifold_g1.demo_spread
```
