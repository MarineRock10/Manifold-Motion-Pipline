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
| ② BC | `python3 -m manifold_g1.bc train` | `reports/manifold_g1/bc/bc_policy.pt` | 克隆姿态在 recorded 包络内的比例 | **9341/9859 (95%)**；val 姿态误差 0.086 rad |
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

`static_fit.py` 只在被需要的那一半上存活：`BodyModel`（29 关节姿态 → 身体表面点）仍被
`sonic_rl`、两个 viewer、`verify_sonic`、`demos` 依赖；它的 `train` / `report` / `verify` /
`diagnostics` 子命令属于旧的 4 通道路线，已无调用者。

---

## 3. 仍未解决的两件事（写在前面，避免误读所有 r）

### 3.1 包络是盒子，流形是椭球

`envelope_semi` 是各轴 `max|rel|`，即**轴对齐盒子**：recorded 姿态装进它正好 1.000（已核验
12/12、max 1.0000、零越界点）。但 `EllipsoidManifold` 是**椭球**，其 r=1 等值面内切于那个盒子。
站姿在**自己的**包络下就是 r=1.35，盒子角点 r=1.73，正姿表面约 39% 的点在 r>1。

后果：所有 `r ≈ 0.8` 不要读成"快贴边了"；`w_outside` 惩罚追的形状是数据描述不出来的。
修法在 `recompute_envelopes.py`（测量时拟合椭球，站姿系数约 1.35），不在 viewer。
`eval_session` 同时打印两种半径，让这个差可见。

### 3.2 示范集几乎是一流形一姿态

9859 个流形里 **9480 个只有 1 个记录姿态**（379 个 ≥ 2，10 个 ≥ 4）。克隆学的是 `M → 单个 q`，
不是 `q` 的分布；`demo_spread` 里多数流形的 spread 是 0.000，因为无从分歧。这是"同一流形多风格"
目前没有立足点的**数据侧**原因，不是模型侧。

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
