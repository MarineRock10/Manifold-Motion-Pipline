# Stage 1 failure clusters

该报告只做失败归因，不回看确认集来重新选模型。

| cluster | rows | families |
|---|---:|---|
| accepted | 12 | button_lever, dodge_lateral, door_interaction, forward_lunge, hands_back_walk, jog_forward, kneel, ladder, side_hop, step_up_box, walk_curve, walk_lateral |
| root_drift | 2 | box_jump, high_jump |

## 定向补充清单

| priority | family | reason |
|---:|---|---|
| 1 | high_jump | confirmation root drift exceeded 0.35 m |
| 1 | box_jump | confirmation root drift exceeded 0.35 m |
| 2 | jog_forward | temporal head confused with side_hop |
| 2 | dodge_lateral | temporal head confused with side_hop |
| 2 | button_lever | temporal head confused with door_interaction |
