# 当前算法路线与验收口径

本仓库当前已经通过 MuJoCo + 冻结 GEAR-SONIC 完成了 Stage-2 的**验证型闭环**：
环境流形 `M_e(t)` → 原语路由 → latent Flow 候选 → 保守 projection → SONIC/MuJoCo
物理门 → 连续执行。四个固定场景（wide/low/narrow/center）已经有可复现实验，但这
不等同于在真实感知输入和未见环境上的泛化。

## 已完成

- Stage 1 静态包络、原语和物理可执行性门控；
- Stage 2 的 actor-disjoint SEED 滑窗、latent Flow Matching、多候选筛选；
- execution-target mean ablation：`target_exec` 可作为 SONIC tracking-error 对照；
- 低顶棚下蹲、1.05 m 通道严格侧身、中心障碍 A* 绕行；
- 路段边界的真实 state/history 重条件化；
- 关节范围、平滑、交接和路线一致性的保守投影层。

## 当前明确边界

1. `M_e` 仍主要由 MuJoCo 几何或 `R_exec` 反向合成，真实 RGB-D/LiDAR 接入另行推进；
2. 在线重条件化目前发生在路段边界，不是每个控制周期的 MPC；
3. SONIC 不直接消费 root position，root progress 只能通过关节、姿态和路线 command 间接产生；
4. pilot Flow 数据覆盖稀疏，在线版本保留了训练支持的条件均值候选；
5. crawl 在冻结 SONIC 下不具备稳定低接触能力，jump 只完成了无障碍起跳/落地门。

## 后续顺序（跳过感知 P1）

| 顺序 | 工作 | 必须通过的验收 |
| --- | --- | --- |
| P0 | 统一文档、数据契约和报告口径 | clone 后不会把 Stage-2 误读为“未实现” |
| P2 | 扩充动态窗口、环境/演员隔离和 transition 元数据 | 无同演员泄漏；每类原语和过渡都有统计 |
| P3 | 去除手工 raw anchor 的泛化消融 | wide/low 已通过；窄通道侧向候选若后退则必须被方向门拒绝 |
| P4 | 滚动时域 state/history 重规划 | 连续执行、无 reset、窗口切换无明显相位跳变 |
| P5 | root progress / yaw command 闭环 | 路线进度、身体航向和关键帧误差同时受控 |
| P6 | 短时域物理 rollout 优化层 | projection 前后成本下降，且不能掩盖物理失败 |
| P7 | crawl/jump 能力边界与控制器升级入口 | 不支持的原语不会被错误路由；新控制器可重新接入 |

所有新增结果都要同时保存：复现命令、JSON 报告、候选审计、GIF/NPZ（如适用）以及
anchor/无 anchor 的 A/B 对照。`accepted=true` 只代表全部物理硬门通过，不代表已经有
真实传感器泛化能力。
