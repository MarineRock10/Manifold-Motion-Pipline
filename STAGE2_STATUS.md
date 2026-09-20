# Stage-2 实验状态（2026-09-19）

所有执行均在 WSL 的 `/home/xiyuan/Manifold-Motion-Pipline` 中完成，使用冻结的
GEAR-SONIC ONNX 控制器和 MuJoCo G1。`accepted` 的含义是经过 SONIC 回放、关节范围、
跌倒/姿态、跟踪、必要时走廊与命名障碍物接触的硬门；不是仅由姿态图或文件名判定。

## 朝向、相位与窄走廊修复（当前推荐版本）

当前推荐产物为 `reports/manifold_motion/stage2_manifold_optimized_v2`。旧低顶棚实验使用
p2 窗口 669，其隔离执行位移方向约为 -149°；执行器为把该倒退位移对齐路线而让机体
转约 180°，所以出现“背着走”。新版本扫描 p2 数据后选用窗口 590，并加入不可绕过的
路线切向进度与朝向硬门：实际低顶棚条件下候选向前约 0.60 m、方向误差约 9°。

长时序问题还来自两个执行错误：每个路线段重置动作相位，以及 `_rolling_reference`
把 Flow 生成的时变 root orientation 压成常量。G1 SONIC 虽不读取 root position，却把
root orientation 作为必需编码输入。现版本保留完整根朝向周期、只施加一个路线 yaw
对齐，并对 50 tick 预览做循环展开；同原语跨段使用最近姿态对齐，不再重复播放入口
过渡。最终回归结果：

- 宽场景：全程 `walk_nominal`，6/6 关键帧，0 障碍接触。
- 低顶棚：`walk_nominal -> crouch`，机器人正向通过，6/6 关键帧，0 顶棚接触；路线偏差
  P95 0.099 m，身体—路线 yaw 误差 P95 15.0°，pelvis 最低 0.437 m。
- 1.05 m 窄通道：流形横向自由半轴从 0.700 m 收缩到 0.345 m，自动切换
  `walk_lateral_reverse`，6/6 关键帧，0 侧墙接触。
- 中心障碍：10/10 关键帧，0 障碍接触，路线偏差 P95 0.165 m，平均关节跟踪误差
  0.079 rad；曲率只在第 0/3/7 段激活 `walk_turn`。

`comparison_report.json` 的全部反事实检查为 true。三联 GIF 为
`stage2_manifold_optimized_v2/manifold_driven_actions.gif`，单独绕障 GIF 为
`stage2_manifold_optimized_v2/center/manifold_adaptive.gif`。一键复现使用
`./run_stage2_manifold_optimized.sh`。

## 严格侧身横移（当前窄通道推荐版本）

用户明确要求窄通道中机器人“身体侧着、横着走”，因此不能把旧 p4
`walk_lateral_reverse` 的斜向前进当作侧身。SEED 窗口 1375 的实测位移为约
`(0.67,-0.22)`，身体几乎沿通道；新版本筛选窗口 1450，冻结 SONIC 回放的局部位移为
约 `(-0.17,-0.96)`，方向约 -100°，再用该位移反推路线对齐 yaw，机器人实际身体 yaw
中位数约 101°，横向源位移约 0.96 m。

`reports/manifold_motion/stage2_manifold_side_on_v3` 加入了两道独立硬门：候选位移方向
必须接近 ±90°，最终执行身体—通道方向必须接近 90°。1.05 m 窄通道结果为：

- `side_on_semantics_passed=true`，侧身执行 283 tick；
- 身体 yaw 中位数 103.4°，相对 90° 的 yaw 误差 P95 21.2°；
- 6/6 关键帧、0 侧墙接触、路线偏差 P95 0.103 m；
- 宽、低顶棚、窄通道、中心障碍共 22/22 个验收检查通过。

中心障碍的 p4 段属于路线转弯/侧避，不强行套用平行窄通道的 90°约束，而使用曲率自适应
的斜向侧避；这不是窄通道侧身指标的放宽。复现入口为
`./run_stage2_manifold_side_on.sh`。

| 顺序 | 产物 | 当前可复现实证 | 状态 |
| --- | --- | --- | --- |
| 1. 固定状态流形反事实 | `reports/manifold_motion/stage2_counterfactual_transition_v4` | 同一 test walk 状态、历史和命令下，normal M 路由 `walk_nominal`（0.783）并通过；low M 只改变 M 即路由 `low_transition`（0.998），骨盆均值降至 0.578 m。该跨状态 low 回放因走廊半径 2.808 被拒绝。 | 选择因果性已验证；严格跨原语安全执行仍是未通过的负例。 |
| 2. 独立 crouch 测试 | `stage2_residual_crouch_actor_test200_v2/selection` | A247 独立测试演员的 crouch 窗口通过；骨盆均值 0.596 m，进度 25.2%。 | 通过 |
| 3. 站立到下蹲过渡 | `seed_replay_low_transition_actor_split_v1`、`stage2_low_transition_actor_test_scan_v1/index_0` | 46 条演员隔离 `crouch_ff_start` 中 34 条通过 SONIC；新 p3 动态模型的 A145 测试窗口通过，骨盆均值 0.451 m。 | 通过 |
| 4. 在线感知接口 | `perception_online_normal_contract_v2` | 从含墙 MuJoCo 场景采样 9,278 个世界点；运行时转换到根坐标，并输出与训练一致的“安全走廊并集 SDF”（原始离障 SDF 另存为诊断）。路由 walk 后在含墙场景执行通过，障碍接触率为 0。 | 通过（场景点云集成夹具，不是实机相机包） |
| 5. 连续多原语 | `stage2_sequence_transition_crouch_v2` | `low_transition → crouch` 合成 107 个 30 Hz 帧，在一次无重置 MuJoCo 回放中通过；交接前 RMS 0.467 rad、12 帧平滑桥接；终段进度 24.6%（阈值 20%）、障碍接触率 0。 | 通过 |

视觉产物：`reports/manifold_motion/stage2_sequence_transition_crouch_v2/sequence.gif`。
青色为模型参考，橙色为 SONIC 实际执行；场景包含低顶棚与侧墙。

## 小幅环境流形动作演示

新增 `manifold_motion/stage2_manifold_behavior_demo.py` 和
`reports/manifold_motion/stage2_manifold_behavior_v1`。该实验只启用当前可稳定执行的
`walk_nominal`、`crouch`、`walk_lateral_reverse`，把路由、动态参考、冻结 SONIC 与物理
障碍硬门放进同一流程：

- 宽通道：`(M,c)` 路由 `walk_nominal`，置信度 96.3%（只看 M 为 78.3%）；执行通过，
  骨盆均值 0.745 m，运动进度 70.9%。
- 低顶棚：路由 `crouch`，置信度 99.5%（只看 M 为 82.2%）；执行通过，骨盆均值
  0.502 m，障碍接触率 0。把完全相同的正常直行动作放入该场景时，低顶棚接触率为
  100%，被硬门拒绝。
- 前方封堵：路由 `walk_lateral_reverse`，置信度 90.8%（只看 M 为 50.4%，说明路线
  指令 c 对“转向还是侧移/后退”的消歧是必要的）；执行通过。相同正常直行动作会与
  前挡墙接触并被拒绝。

三联 GIF 为 `stage2_manifold_behavior_v1/manifold_behavior.gif`。这是小幅动作的可视化
能力演示；低顶棚面板使用训练分布内的 crouch 条件窗口，不替代上文 A247 独立测试演员
的泛化指标。紫色透明体是当前流形，橙色轨迹是 SONIC/MuJoCo 实际执行。

## 长时序 A* 绕障与关键帧保持（新增）

新增 `manifold_motion/stage2_long_horizon_avoidance.py` 和物理场景
`data/g1_flat/scene_long_avoidance.xml`。这不是把多个 GIF 拼接起来：A* 先读取场景中
命名为 `obstacle_*` 的真实 MuJoCo box，按机体半径 0.46 m 与 0.12 m clearance 膨胀，
再生成绕过中央障碍的路线和时变安全走廊。机器人从同一个 MuJoCo 状态开始，在一次连续
rollout 中闭环执行；到达当前 pelvis 关键帧之前不会释放下一个关键帧。

路线段之间按实测航向选择 `walk_turn`，对齐后切回 `walk_nominal`。切换只替换 SONIC
参考，不重置 simulator，也不瞬移 root。最终 v2 实验（关键帧容差 0.22 m）结果：

- 3/3 个关键帧到达；末端误差 0.215 m；无 simulator reset。
- `walk_turn` 24 tick、`walk_nominal` 323 tick，共 7 次闭环 primitive switch；最大切换前
  关节 RMS 0.209 rad。
- 命名物理障碍接触 0 tick；最低 pelvis 高度 0.714 m；最大 roll 6.73°。
- 实际轨迹相对 A* 中心线的 P95 偏差 0.163 m；平均关节跟踪误差 0.107 rad。

可视化结果为 `reports/manifold_motion/stage2_long_horizon_avoidance_v2/long_horizon_avoidance.gif`，
报告为同目录 `report.json`。GIF 中红色是实际碰撞体、红色细线是膨胀边界、青色是 A* 中心线、
橙色是实际 pelvis 轨迹、黄色/绿色点是待达/已达关键帧。Windows 复制产物位于
`artifacts/stage2_long_horizon_avoidance_v2/`。

复现实验：

```bash
PYTHONPATH=.:/home/xiyuan/.local/share/sonic-manifold-g1 MUJOCO_GL=egl \
python3 -m manifold_motion.stage2_long_horizon_avoidance \
  --keyframe-tolerance-m 0.22 \
  --out reports/manifold_motion/stage2_long_horizon_avoidance_v2
```

这里的“绕障”是场景几何规划 + SONIC 闭环执行的物理证据；此前短 demo 中的前方封堵
只做了局部后退/侧移，不能替代本实验的完整绕行路线。走廊仍是由场景点云构造的
`reverse_synthesized` 条件，不应误称为真实 RGB-D/LiDAR 感知结果。

## 路线分段 Flow Matching 多候选（新增）

> 纠正（2026-09-18）：这一节的 v1 结果使用了按分段编号写死的
> `walk_turn / walk_lateral_reverse / crouch` 偏好，只能证明多原语连续执行，不能证明
> 障碍流形导致了动作选择。它已被下面的“环境流形因果反事实”实验取代，不再作为
> `M_e -> primitive` 的因果证据。

`manifold_motion/stage2_flow_route_candidates.py` 把上面的 A* 路线按 3 个关键帧区间拆开，
将每段自己的 `corridor[48,7]` 与 `sdf[10,10,8]` 直接写入 Stage-2 条件向量，并用已训练的
latent Flow Matching 为不同 `z_p` 生成候选。最终 v1 每个候选集合采样 6 条，共执行 36 次
独立 SONIC/MuJoCo 候选筛选；候选筛选不修改轨迹，连续任务仍执行命名障碍接触、关键帧、
跟踪、跌倒/滚转、路线偏差和动作交接硬门。

连续任务使用如下模型动作序列：

1. 第一段：`walk_turn` Flow candidate 2，保持 45 tick 后切换到同一段通过筛选的
   `walk_nominal` candidate 0 到达关键帧。
2. 第二段：`walk_lateral_reverse` candidate 0，保持 45 tick 后切换到该段的
   `walk_nominal` candidate 0。
3. 第三段：`crouch` Flow candidate 5，保持 45 tick 后恢复到该段的
   `walk_nominal` candidate 0 并到达终点。

三个非普通动作都来自动态模型候选，不是渲染标签。最终物理结果：3/3 关键帧到达、5 次
原语切换、0 障碍接触、末端误差 0.211 m、路线偏差 P95 0.165 m、平均关节跟踪误差
0.105 rad、最低 pelvis 0.507 m、最大 roll 7.73°，`accepted=true`。最大原语交接前关节
RMS 为 0.662 rad，低于本实验 0.75 rad 的显式硬门。

主要产物：

- `reports/manifold_motion/stage2_flow_route_avoidance_v1/flow_route_avoidance.gif`
- `reports/manifold_motion/stage2_flow_route_avoidance_v1/report.json`
- `reports/manifold_motion/stage2_flow_route_avoidance_v1/candidate_evidence.json`
- `reports/manifold_motion/stage2_flow_route_avoidance_v1/segment_conditions.npz`
- `reports/manifold_motion/stage2_flow_route_avoidance_v1/executed.npz`

Windows 副本位于 `artifacts/stage2_flow_route_avoidance_v1/`。这里的 nominal candidate 0 是
条件均值锚点，其余 candidate 是 latent Flow 随机样本；保留均值锚点能防止一次随机采样把
已验证的行走能力从候选集中抹掉。最终选择仍由实测物理结果决定。

```bash
PYTHONPATH=.:/home/xiyuan/.local/share/sonic-manifold-g1 MUJOCO_GL=egl \
python3 -m manifold_motion.stage2_flow_route_candidates \
  --num-candidates 6 --max-ticks 1200 \
  --out reports/manifold_motion/stage2_flow_route_avoidance_v1
```

## 环境流形因果反事实（替代写死动作表）

新增 `stage2_manifold_adaptive.py` 与三个对照场景 `scene_manifold_wide_long.xml`、
`scene_manifold_low_long.xml`、`scene_manifold_narrow_long.xml`。三次实验使用完全相同的
直线路线、阈值、Flow 模型、候选数和随机种子；低场景只增加一个 MuJoCo 物理低顶棚，
窄场景只增加两面物理侧墙。每段原语由实测 `M_e` 自由半轴和
路线曲率确定，不读取分段编号：垂直半轴低于 1.12 m 选 `crouch`，横向半轴低于 0.42 m
选侧移，否则选普通行走。

最终三个实验均通过全部物理门：6/6 关键帧、0 障碍接触。宽场景的 6 段垂直自由半轴
均为 1.338 m，因此全程 `walk_nominal`；低场景第 1 段为 1.338 m，先普通行走，接近
顶棚后 5 段降为 0.990 m，自动切到 `crouch`。平均 pelvis 高度由 0.777 m 降到
0.625 m（下降 0.152 m），最大 roll 分别为 4.46° 和 4.90°。窄场景横向自由半轴从
0.700 m 降到 0.405 m，动作从第 2 段开始自动切为 `walk_lateral_reverse`，最大 roll
4.49°。顶棚影响区按完整机器人水平包络膨胀，而非只检测 pelvis 点，因此不会在头/手
尚未离开顶棚时提前站起。

同一执行器还在 `scene_long_avoidance.xml` 上完成中心障碍绕行：A* 把直线改成 10 段
弯折路线，只有曲率超过 0.28 rad 的第 0、3、7 段允许 `walk_turn`，障碍邻域横向自由
半轴收缩时选择 `walk_lateral_reverse`。连续执行到达 10/10 关键帧，`walk_turn` 19 tick、
侧身步态 266 tick、普通行走 535 tick；0 障碍接触，路线偏差 P95 0.160 m，平均关节
跟踪误差 0.084 rad，最大 roll 3.52°。

主要产物：

- `reports/manifold_motion/stage2_manifold_counterfactual_v1/manifold_driven_actions.gif`
- `reports/manifold_motion/stage2_manifold_counterfactual_v1/comparison_report.json`
- `reports/manifold_motion/stage2_manifold_counterfactual_v1/center/manifold_adaptive.gif`
- `reports/manifold_motion/stage2_manifold_counterfactual_v1/{wide,low,narrow,center}/report.json`

WSL 一键复现：`./run_stage2_manifold_counterfactual.sh`。Windows PowerShell 入口为
`powershell -ExecutionPolicy Bypass -File .\run_stage2_manifold_counterfactual.ps1`。

## Crawl / jump 能力门

- `seed_replay_crawl_capability_v1` 与 `seed_replay_crawl_capability_actor_split_v2`：共 87 条
  crawl/all-fours 源轨迹（后者额外覆盖 19 位演员）在冻结 SONIC 下均未通过，因而没有把
  它们用于训练或宣称 G1 已会爬。要支持该原语，需要提供/训练 low-contact SONIC 控制器，
  而非只扩充流形或 Flow Matching 数据。
- `crawl_initial_height_probe_v1`：针对三条代表性 crawl 轨迹，以 0.35 / 0.45 / 0.55 /
  0.70 m 的根初始高度重放，12/12 仍因翻滚或跌倒拒绝。这排除了“仅因站立高度初始化
  不当而失稳”的解释；当前冻结控制器没有四足接触平衡能力。
- 追加控制变量诊断（2026-09-18）：使用 NVIDIA 官方
  `GR00T-WholeBodyControl` 源码（commit `7f15131`）的带接触代理 G1 MJCF，对三条
  SEED crawl（`crawl_ff_start`、`idle_crawl_start`、`crawl_ff_loop`）以同一冻结 ONNX
  重放，结果与当前 MJCF 完全相同，均因 `fall_or_extreme_roll` / `roll_limit` 拒绝；
  最大滚转为 82.2°–111.1°。再从站立姿态以 1 / 2 / 3 秒平滑进入 crawl，6/6 仍失败。
  因而已排除接触网格版本、根初始高度和入口突变三种解释。官方训练指南确认可以从
  `sonic_release/last.pt` 在 Isaac Lab 中续训，但推荐 64+ GPU 才能在合理时间收敛；本机
  仅有 RTX 5060 8 GB，且未安装 Isaac Lab，不能把数天以上的低置信单卡试跑伪称为成功。
  crawl 仅能在具备多 GPU 训练资源后，以这 87 条失败源轨迹及更多低接触数据进行官方
  SONIC 微调，并重新导出 ONNX 后再进入 Stage-2 数据门。
- `seed_replay_jump_high_partial_v1`：完整 metadata 的 `high_jump` 演员隔离 12/4/4 清单
  `data/seed_jump_high_actor_split_v1.csv` 已全部提取、回放并通过（20/20）。独立 test 动态
  jump 模型位于 `stage2_mean_jump_high_actor_split_v1`；其 test index 0 的物理验证通过：
  抬升 0.253 m、连续腾空 15 tick、成功落地、无跌倒。该样本在原始反向合成走廊下会被
  `corridor_violation` 拒绝，因此 jump 的通过结论来自无障碍物的严格物理起跳/落地门，
  并不声称已解决跳跃穿越狭窄走廊的条件建模。

## 关键实现改动

- `manifold_motion/perception_runtime.py`：世界点云、路径、规划朝向和时变包络 → `condition.npz`。
- `manifold_motion/perception_corridor.py`：将模型输入 SDF 与训练的安全走廊 SDF 语义对齐，
  并保留独立的 obstacle-clearance SDF。
- `manifold_motion/stage2_sequence.py`：无重置连续拼接、根坐标重基、显式交接差异和终段进度硬门。
- `manifold_motion/seed_replay.py`：命名障碍接触与严格 jump 起跳/落地检查。

早期“固定 normal 窗口状态 + 低通道 M”的 p3 跨状态反事实仍保留为负例；它测试的是
单帧状态突变，不等同于本次随路线演化、提前进入低姿态的连续实验。本次通过没有放宽
接触、跌倒、滚转、跟踪或路线偏差门槛。

## 在线真实状态/history 与 optimization-embedded 投影（当前阶段）

本阶段的入口是 `run_stage2_online_projection.sh`（PowerShell 包装为
`run_stage2_online_projection.ps1`），产物位于
`reports/manifold_motion/stage2_online_projection_v1/`，Windows 副本位于
`artifacts/stage2_online_projection_v1/`。

每个场景先进行一次不渲染的探测回放：从冻结 SONIC + MuJoCo 实际执行轨迹提取
`q_exec(29)、dq_exec(29)、gravity(3)、局部基座线速度(3)、足/手/非足接触(5)`，形成与
训练集完全一致的 69-D state；在每个实际原语边界回取最近 0.4 s 的 12 帧作为 history，
再将 `(M_e, primitive, state, history, command)` 送入 Flow Matching 重新生成候选。最终
回放仍从单一 MuJoCo 状态开始，并由同一组接触、关键帧、跟踪、跌倒、路线和 yaw 硬门验收。

Flow 候选后新增 `manifold_motion/stage2_projection.py`。它是一个保守的
optimization-embedded projected-gradient 层：对 29 个 SONIC 关节做物理范围投影与小步
时间二阶平滑，对 root-local 目标位置做路线中心一致性修正，并输出投影前后的
`J_smooth + J_limit + J_handoff + J_corridor` 审计值。投影不改 SONIC 控制器，MuJoCo
执行结果仍是最终权威；当前默认仅 1 次、最大 0.005 rad 的关节修正，避免把投影层变成
隐藏的动作重定向器。原始 Flow 候选保存为各场景的 `*_raw_flow.npy`，投影后候选单独
保存，便于做消融。

四个场景的在线探测回放与最终回放均通过：

- 宽：6/6 关键帧、0 障碍接触，路线偏差 P95 0.148 m；全程 `walk_nominal`。
- 低顶棚：6/6、0 接触；`walk_nominal -> crouch`，路线偏差 P95 0.134 m。
- 1.05 m 窄通道：6/6、0 接触；282 tick 严格侧身，身体—通道 yaw 误差 P95 20.9°，
  路线偏差 P95 0.097 m。
- 中心障碍：10/10、0 接触；曲率段触发 `walk_turn`，路线偏差 P95 0.171 m。

汇总验收文件为 `stage2_online_projection_v1/comparison_report.json`；四个可直接查看的
GIF 为各场景目录中的 `manifold_adaptive.gif`。由于当前 pilot Flow 数据在 walk→crouch
和在线侧向状态上的覆盖仍然稀疏，每个在线条件集合明确保留一个训练支持锚点候选，其他
候选才是实时 state/history 条件生成；锚点与在线候选共同经过同一物理筛选，报告中以
`candidate0_training_support_anchor` 标注，不能误解为所有候选都已完成分布外泛化。
