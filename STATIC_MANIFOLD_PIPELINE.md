# 静态流形 → 姿态：当前主线与已废弃路线

> 本文档取代 2026-09-15 版。那一版描述的是**四个手设计通道**（crouch / lean / twist / arms）
> + 纯几何训练 + "模型预测 vs 真机实测" 的路线，现已废弃；本文说明现在实际在跑的那条，
> 以及旧路线的代码还剩什么。更新：2026-09-16。

---

## 1. 现在的主线（三段，可复现）

```
① 数据                   ② BC                     ③ SONIC 在环微调
采集 → 能力图 → 拆帧      克隆 (M, s) → q           真控制器在环，扰动流形
示范集                    bc_policy.pt              policy_sonicrl.pt
```

| 阶段 | 命令 | 产物 | 指标 | 当前数字 |
|---|---|---|---|---|
| ① 数据 | `collect` → `capability build` → `demos build` → `dataset build` | `dataset/pose_demos.npz`、`dataset/health.json` | SONIC 跟踪误差 | 0.054 rad（105 clips / 18705 窗口 / 10376 姿态对） |
| ② BC | `python3 -m manifold_g1.bc train` | `reports/manifold_g1/bc/bc_policy.pt` | 克隆姿态在 recorded 包络内的比例 | **9336/9840 (94.9%)**；val 姿态误差 0.086 rad |
| ③ 在环微调 | `python3 -m manifold_g1.sonic_rl run` | `reports/manifold_g1/sonic_rl/policy_sonicrl.pt` | **实际到达**姿态在扰动包络内的比例 | success 1.00、r 0.82、30 轮 gate 6 |

每个定义只有一个来源，这是防止训练与部署悄悄分叉的关键：

| 定义 | 唯一来源 |
|---|---|
| 观测布局（15 维）+ 动作映射（`tanh`） | `pose_policy.py` |
| 示范集读取（`load_demos`）+ `POSE_DIM` | `demos.py` |
| 批量环境（obs / 半径，供 BC 与 verify 复用） | `torch_env.py` |
| 流形几何 + 场景构建 | `manifold.py`、`family.py` |
| 真机执行（MuJoCo + 冻结 SONIC） | `keyframe_env.py`、`sonic.py` |

三个可视化：`view_data`（数据）、`show_bc`（②）、`show_rl`（③）。见 `VISUALIZATION.md`。

---

## 2. 废弃的路线：几何 RL + 残差模型

早期思路：几何模型能跑几千轮/秒，先用它训练，再拿真机验证，用"模型预测 vs 实测"的差量化可执行性。
具体做法是学一个**残差模型** `Δ̂(q)` 预测 SONIC 对指令姿态实际做什么，再让 PPO 优化"我认为
控制器会到达的落点"。

**为什么废弃**：那个偏差（优于忽略残差 51%）不等于预测正确——微调目标变成了"我以为控制器会做什么"。
而 BC 本身已经足够好，把真控制器直接放进环里也够快（一轮 16 流形 × 8 tick ≈ 40 s，总计 879 s）。
`sonic_rl.py` 从头到尾没有 import 过那条路径。

**已删除**（完整备份在 `~/sonic_backup_20260916_1637/`）：

| 删除的东西 | 是什么 |
|---|---|
| `primitive.py` | numpy 版几何环境 + 它自己的 PPO 训练循环 |
| `primitive_torch.py` | 批量 torch 环境 + 几何 RL 训练器（环境部分移到 `torch_env.py`） |
| `residual.py` / `residual_data.py` | 残差模型 `Δ̂(q)` 及其配对数据 |
| `compare.py` | 为对比几何 RL 与 BC 而写 |
| `primitive_torch/policy*.pt` | 几何 RL 的三个产物 |
| `sonic_residual.pt` / `residual_pairs.npz` | 残差模型权重与配对数据 |

**保留**：`TorchPrimitiveEnv` 的环境部分（观测布局与半径计算）改名为 `torch_env.py`，因为
`bc.py` 和 `verify_sonic.py` 一直在用它——删掉整块会把 BC 一起弄坏（这个坑踩过一次：曾单方面重构
`bc.py`，导致它从 8/10 掉到 3/10）。产物目录 `reports/manifold_g1/primitive_torch/` 随之更名
`bc/`，因为它现在只装 `bc_policy.pt`。

`static_fit.py` 已删除：只有 `BodyModel`（29 关节姿态 → 身体表面点）有调用者，已抽成
`body_model.py`；它的 `StaticFitEnv` / `BatchedStaticEnv` / `PoseSlew` 与整个 CLI 属于旧的
4 通道路线，无调用者。

---

## 3. 本轮修掉的度量错误：包络的"盒子 vs 椭球"

**症状**：记录姿态在**自己的**包络下算出 r = 1.18–1.57（中位 1.40）。"把身体装进流形"这个目标，
连产生该包络的那条姿态都做不到；站姿在家族 scale 1.0 的流形下是 r = 1.145，所以
`report_specs` 里的 `h1.00_w1.00_d1.00` 那一行，站着不动也永远判 outside。

**原因**：`envelope_semi` 存的是各轴 `max|rel|`，那是**盒子**；而所有包含性判断用的是
**椭球**公式 `r = ||(p−c)/semi||`。盒子角点按椭球算就是 `sqrt(3) = 1.73`。两个形状混用。

**修法**（已做，2026-09-16）：

1. `recompute_envelopes.py` 和 `clip.py` 都改用 `fit_ellipsoid(rel)`：把各轴半轴乘上
   `max_i ||rel_i/semi||` 这一个标量，即"膨胀到刚好包住身体"。核验：记录姿态 r = 1.0000。
2. `family.BASE_SEMI/BASE_CENTER` 改成与**示范集同一把尺**（`demos.pose_envelope` 在站姿下的
   拟合结果），所以 `report_specs` 与策略训练时见过的流形形状一致。核验：站姿 r = 1.0002。
3. （顺带）`clip.py` 里那段从未被读取的 `_mesh_envelope` 删除。

**关于修法的两个精确说明**，因为我自己先说错过：

- `fit_ellipsoid` 是**均匀**膨胀（一个标量乘三个轴），不是逐轴重拟合；aspect ratio 不变。
  它近似最优：逐轴自由优化只能再小 11%（中位体积比 0.893）。要取代它也不是一个全局常数——
  膨胀系数逐 tick 变化（1.18–1.57，中位 1.40）。
- 我先前说"fit 比均匀膨胀小 31%"是**错的**：那次比较用了两个不同 tick 集合的 k（1.408 vs 1.387），
  是噪声不是结论。二者本来就是同一个操作。

**影响范围**（这点也修正过）：示范集用的是 `demos.pose_envelope` → `fit_ellipsoid`，
**从来就是拟合椭球**，所以 ② 的训练数据没受影响（BC 通过率修前 95%、修后 94.9%，一致）。
受影响的只有 clip 存下来的包络，也就是 `verify_sonic` / `eval_session` 的判据。

**修完之后的读数**：

| | 修前 | 修后 |
|---|---|---|
| 记录姿态在自己的包络下 | 1.40 中位 | **1.000** |
| 站姿在 scale 1.0 家族流形下 | 1.145 | **1.000** |
| `verify_sonic` r(sonic) | 1.001 中位 | 1.031 中位 |
| `sonic_rl eval`（示范流形上）BC | — | **5/6 inside，r 0.86** |

`verify_sonic` 剩下的 3% 不是策略问题：示范集流形带 `MARGIN=1.05`，而 `report_specs` 的
scale 1.0 没有余量，所以同一姿态在家族口径下正好高 5%——预测 0.98×1.05 = 1.029，实测 1.030。

### 3.2 示范集几乎是一流形一姿态

9840 个流形里 **9480 个只有 1 个记录姿态**（374 个 ≥ 2，10 个 ≥ 4）。克隆学的是 `M → 单个 q`，
不是 `q` 的分布；`demo_spread` 里多数流形的 spread 是 0.000，因为无从分歧。

**这不是缺陷，是阶段边界**：② 的目标就是克隆示范的姿态意图，`M → q` 正是它该学的；
多样性是 ③ 在扰动流形上探索出来的，不是从数据里读出来的。数据侧真正值得记的是：
不同流形之间的姿态差异是存在的（每关节 std 均值 8.6°，与流形参数相关 +0.42），
所以 `M → q` 有信号可学——这才是 ② 成立的前提。

### 3.3 泛化缺口定位在"倾斜"这一维，而且**几何上可行、策略上不足**

`report_specs` 的 t=±10 两行一直是最大的失败（r 1.12–1.33，其余多在 1.0 附近）。分三步查清了：

1. **训练分布里根本没有倾斜。** 记录包络存了 `quat`，但那是走路时骨盆的 **yaw**（中位 87°），
   只有 0.5% 带真正的矢状倾斜（x 分量 > 0.1）；而 `_reshape` 只扰动半轴和中心，不扰动朝向。
   所以 t=±10 是纯外推。→ 已加 `tilt_deg`/`roll_deg` 扰动（默认 6°/3°）。
2. **倾斜流形是几何可行的，不是无解。** 站姿在 t=+10/-10 下是 r 1.13/1.22，但把身体整体
   **俯仰 +7°/-8°** 就能到 r 0.97/0.98。所以"塞得进去"这件事有解。
3. **策略的反应幅度不够。** 它在 +10 与 -10 之间的姿态只差 0.065 rad（≈3.7°），而几何要求 7–8°。
   方向是对的，幅度差一半。

结论：这不是 bug，是**能力缺口**，修法是训练分布覆盖更大的倾斜。用 6° 扰动微调后
t=±10 基本没变（+10 1.12→1.12，-10 1.17→1.23），说明 6° 不足以覆盖 10°——
扰动必须至少覆盖评估范围。这是 ③ 下一步该做的事，不是 ①② 的问题。

---

## 4. 复现全部数字

```bash
python3 -m manifold_g1.dataset show                      # ① 数据体检
python3 -m manifold_g1.demo_spread                       # ① 示范集内部分歧
python3 -m manifold_g1.bc eval --policy reports/manifold_g1/bc/bc_policy.pt --device cpu
python3 -m manifold_g1.sonic_rl eval --manifolds 10 --steps 4 --device cpu \
    --policy reports/manifold_g1/sonic_rl/policy_sonicrl.pt
python3 -m manifold_g1.verify_sonic --policy reports/manifold_g1/sonic_rl/policy_sonicrl.pt
python3 -m manifold_g1.eval_session --clips 3 --per-clip 4 --device cpu   # §3.1 的核查
```
