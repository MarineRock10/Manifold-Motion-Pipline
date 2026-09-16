# 可视化使用说明

pipeline 每个阶段都能看到东西，按"看哪一段"选：

| 想看 | 命令 | 经 SONIC？ |
|---|---|---|
| **手动调流形看反应**（推荐） | `shape_manifold` | ✅ 控制器驱动 + 物理 |
| **策略在一批流形上的表现** | `show_bc` | ✅ |
| **策略姿态形状**（快速） | `show_pose` | ❌ 纯运动学 |
| **采集的数据** | `view_data` | 回放模式是 |

---

## 1. `shape_manifold` —— 按键实时改流形（推荐）

```bash
python3 -m manifold_g1.shape_manifold
python3 -m manifold_g1.shape_manifold --policy reports/manifold_g1/primitive_torch/bc_policy.pt
```

默认用 RL 微调后的策略（`primitive_torch/policy.pt`），找不到就退回 BC。

**这是唯一能手动拖流形的 viewer**，每按一次键走一遍完整闭环：

```
按键 → 流形重建 → 策略重新求解姿态 → SONIC 关键帧 → PD 力矩 → 物理 → 实际落点
```

### 按键

| 键 | 调什么 | 范围 |
|---|---|---|
| `u` / `i` | 压扁 / 抬高（height） | 0.55–1.30 |
| `j` / `k` | 收窄 / 放宽（width） | 0.50–1.30 |
| `,` / `.` | 前后压扁 / 加深（depth） | 0.45–1.30 |
| `n` / `m` | 椭球左右平移 | ±0.10 m |
| `t` / `y` | 中轴后倾 / 前倾 | ±25° |
| `S` | 切换请求"策略姿态 ↔ 记录的示范姿态" | |
| `R` | 从当前流形重新保持 | |
| `P` | 暂停 | |

### 屏幕上的文字

```
manifold: h 0.80  w 1.00  d 1.00  offset +0.00  tilt +0deg
requesting the policy pose   held 1.2s
                                    r requested 1.00   r achieved 1.04
                                    joint error 0.061 (arms 0.072)   pelvis z 0.55
```

| 字段 | 含义 |
|---|---|
| `r requested` | 几何模型对策略姿态的判断（不经 SONIC） |
| `r achieved` | **SONIC 实际留下的结果** ← 这个才有意义 |
| `joint error` | 请求 vs 实际的平均关节误差；`(arms …)` 是手臂分组，SONIC 对手臂最差 |
| `pelvis z` | 骨盆高度，蹲下时应真的降低（证明接了物理） |

### 一个实测用法：连按 `u`（压扁）

| height | 膝 | 髋 | r(achieved) |
|---|---|---|---|
| 1.00 | 7.8° | −5.6° | 0.93 |
| 0.85 | 17.7° | −7.3° | 0.92 |
| 0.75 | 23.0° | −8.7° | 0.99 |
| 0.65 | 26.5° | −9.8° | 1.10 |

蹲幅随流形**单调增加**（条件化成立），但 **height 0.75 以下守不住**——那是策略的蹲深权限边界，不是 bug。

---

## 2. `show_bc` —— 策略在一批流形上

```bash
# 默认用 BC 策略（未微调）
python3 -m manifold_g1.show_bc --count 24

# 看 RL 微调后的策略
python3 -m manifold_g1.show_bc --policy reports/manifold_g1/primitive_torch/policy.pt --count 24

# 只看最难的（需要摆姿态的）流形 / 最简单的（应几乎不动）
python3 -m manifold_g1.show_bc --policy reports/manifold_g1/primitive_torch/policy.pt --band hard
python3 -m manifold_g1.show_bc --band easy
```

### 画面里是什么

| 元素 | 含义 |
|---|---|
| **机器人（实体）** | SONIC 实际到达的姿态（策略输出 → 关键帧 → SONIC → PD 力矩 → 物理） |
| **青色点云** | 被**请求**的姿态（`mesh_points_pose` 采样的点） |
| **半透明椭球** | 当前流形（骨盆系，跟随骨盆位置） |

青点云是"模型想让它摆成什么样"，机器人是"SONIC 让它成了什么样"——两者的差就是执行层的代价。

### 按键

| 键 | 作用 |
|---|---|
| `N` / `M` | 下一个 / 上一个流形 |
| `1`–`9` | 跳到第 n 个 |
| `S` | 切换请求"策略姿态 ↔ 记录姿态"（机器人始终执行策略姿态） |
| `R` | 重新保持；`P` 暂停；鼠标转视角 |

启动时终端会打印每个流形的 `r(clone)` / `r(recorded)` / `pose err`，以及"clone 拟合自己训练流形的比例"。

### 建议的观察顺序

1. **`--band easy`**：机器人应**几乎不动**。如果它在瞎摆，说明没学会"什么时候不用动"；
2. **`--band hard`**：应**明显蹲下去**。两档对比就是"是否条件化"的直接证据；
3. **按 `S` 反复切换**：青点云在两种姿态间跳、机器人不变，看差多少。

---

## 3. `show_pose` —— 策略姿态（轻量版）

```bash
python3 -m manifold_g1.show_pose --with-demos --torch
```

**纯运动学摆位**（`qpos` 直写），不经 SONIC、不跑物理——快、画面干净，但**会骗人**：骨盆高度存在自由关节里，蹲下时腿折叠而身体不降。适合快速看姿态形状，不适合判断可行性。

按键：`N`/`M` 切流形 · `1`–`9` 跳 · `S` 切换"策略姿态 ↔ 数据姿态" · `P` 冻结视角。

---

## 4. `view_data` —— 采集的数据

```bash
# 回放：机器人按记录的关节角播放，椭球是它当时的包络，叠加显示"命令 vs 实际"
python3 -m manifold_g1.view_data replay --clip reports/manifold_g1/clips/ep0000_seed1000.npz

# 整批数据总览（轨迹、速度直方图、包络分布、朝向覆盖）-> overview.png
python3 -m manifold_g1.view_data plots

# 单个 episode 时序（跟踪误差、速度、包络、接触与命令）-> ep*.png
python3 -m manifold_g1.view_data episode --clip reports/manifold_g1/clips/ep0000_seed1000.npz
```

**`replay` 按键**：`P` 暂停 · `N`/`M` 前后退 0.5 s · `E` 关/开椭球 · `T` 关/开轨迹 · `R` 重头。

---

## 5. 数值报告（不开窗口）

```bash
# SONIC 在环：BC vs 微调后的逐流形对比（真控制器，同一起点）
python3 -m manifold_g1.sonic_rl eval --manifolds 10 --steps 4 \
    --policy /tmp/srl_run/policy_sonicrl.pt

# SONIC 门禁：把姿态交给真 SONIC，量实际到达（这是准入检查）
python3 -m manifold_g1.verify_sonic --policy reports/manifold_g1/primitive_torch/policy.pt

# BC 评估：与示范的偏差（同分布内）
python3 -m manifold_g1.bc eval --policy reports/manifold_g1/primitive_torch/bc_policy.pt

# 数据体检
python3 -m manifold_g1.dataset show
```

### 三个策略产物与它们的数字

| 文件 | 是什么 | 训练池 | SONIC 门禁 |
|---|---|---|---|
| `primitive_torch/bc_policy.pt` | BC（克隆示范） | 73% | 3/10 |
| `primitive_torch/policy.pt` | BC + 几何 RL | **98%** | **7/10** |
| `sonic_rl/policy_sonicrl.pt` | BC + SONIC 在环 | — | r 0.85 / 跟踪 0.055 |

**注意 `show_bc` 的默认是 BC 版**，要看微调效果必须加 `--policy .../policy.pt`。

---

## 一个容易看混的地方

`show_bc` 和 `shape_manifold` 都会显示"请求姿态"和"实际姿态"，但：

| | 谁在动 | 说明 |
|---|---|---|
| **机器人（实体）** | SONIC 执行后 | 永远是被执行的那一侧 |
| **青点云** | 按 `S` 切换 | 可能是策略姿态，也可能是记录的示范姿态 |

屏幕上 `requesting the policy pose` / `requesting the recorded pose` 那行字就是提示青点云代表哪个。
