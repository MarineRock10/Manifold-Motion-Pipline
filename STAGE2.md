# Stage 2: SEED → SONIC → Dynamic Motion

This is the executable dynamic-data path for the selected BONES-SEED G1 subset. It keeps the three
signals separate:

- `R_ref`: the 29-joint policy-order reference that frozen SONIC consumes;
- `R_exec`: the MuJoCo state actually achieved by the robot;
- `M^E`: a time-aligned safe-corridor ellipsoid sequence plus an egocentric SDF. For SEED it is
  reverse-synthesized around `R_exec`, explicitly marked as simulation data rather than a
  sensor-derived map.
- `M^R_safe(t)`: the measured robot self-manifold, fitted from the G1 surface mesh at every
  frame and transformed into the route frame. `M^R_task(t)` is a separate primitive-conditioned
  target: crouch caps vertical size and side gait caps route-lateral size. The renderer draws blue
  `M^E`, white `M^R_safe`, and colored `M^R_task` (orange=crouch, green=side, yellow=nominal).
  `executed.npz` stores both manifolds and their measured axes; `report.json` stores per-primitive
  means/minima/maxima. For deployment, `robot_manifold_safe` is now part of the
  hard gate: its ellipsoid is queried first, then every sampled G1 surface point is checked
  against every physical obstacle. The configurable minimum exact clearance is 2 cm by default;
  failure adds `self_manifold_obstacle_clearance` and rejects the trajectory.

The flow model learns `p(R_ref | M^E, z_p, s_t, H_t, c_t)`. `R_exec` is never substituted for
the generated target; it is retained to test whether the generated reference remains executable.

## Preconditions

The selected archive is already in:

```text
data/seed_stage2_pilot/
  seed_stage2_pilot_manifest_v004.csv
  g1/csv/...
```

From the WSL checkout, use the installed local SONIC runtime:

```bash
cd /home/xiyuan/Manifold-Motion-Pipline
export PYTHONPATH="$HOME/.local/share/sonic-manifold-g1${PYTHONPATH:+:$PYTHONPATH}"
```

## 1. Replay and gate the source clips

Run a stable, per-primitive pilot selection. It saves both passing and failing records; only
passing summaries are consumed by the next command.

```bash
python3 -m manifold_motion.seed_replay \
  --strata all_fours crouch low_transition walk_lateral_reverse walk_nominal walk_turn \
  --per-stratum-limit 20 --jobs 4 \
  --out reports/manifold_motion/seed_replay_stage2_v2
```

`crawl` is intentionally excluded from the initial run: the first calibrated five crawl clips all
fell under this frozen SONIC configuration. Revisit it only after adding a controller/reference
interface that can track that morphology.

Each `summaries/<motion_id>.json` records falls, roll, joint tracking error, foot/hand/non-foot
contacts, reference joint-range violations, and reference-versus-executed planar-path progress.
The default gate accepts a locomotion clip only when it makes at least 20% of a reference path of
0.25 m or longer; a stable robot that remains essentially in place is a failed task completion.

## 2. Build actor-disjoint training windows

```bash
python3 -m manifold_motion.seed_windows \
  --replay-root reports/manifold_motion/seed_replay_stage2_v2 \
  --out reports/manifold_motion/seed_windows_stage2_v2
```

The default output `seed_stage2_windows.npz` contains 30 Hz windows with a 0.4-second execution
history and a 1.6-second future horizon:

| key | contents |
|---|---|
| `state`, `history` | executed state `s_t, H_t` (policy-order joints, velocity, gravity, local base velocity, contacts) |
| `primitive` | `z_p` category index |
| `corridor` | `48 × 7` local safe ellipsoids: centre xyz, body-clearance semi-axes xyz, yaw |
| `sdf` | `10 × 10 × 8` local signed-free-space grid; positive denotes free corridor interior |
| `manifold` | legacy flat token retained only for compatibility |
| `command` | horizon goal derived from `R_ref` for offline supervision |
| `target_ref` | future SONIC reference to generate, 48 × 38 |
| `target_exec` | future achieved trajectory for execution evaluation, 48 × 38 |

Splits are actor-disjoint by stable hash, never random windows from the same actor on both sides.

## 3. Train latent Conditional Flow Matching

```bash
python3 -m manifold_motion.stage2_flow train-ae \
  --windows reports/manifold_motion/seed_windows_stage2_v2/seed_stage2_windows.npz \
  --out reports/manifold_motion/stage2_flow_v2 --epochs 100

python3 -m manifold_motion.stage2_flow train-flow \
  --windows reports/manifold_motion/seed_windows_stage2_v2/seed_stage2_windows.npz \
  --out reports/manifold_motion/stage2_flow_v2 \
  --autoencoder reports/manifold_motion/stage2_flow_v2/autoencoder.pt --epochs 200
```

The VAE first embeds the raw future trajectory; Flow Matching then learns its conditional vector
field in latent space. The 29 generated joint coordinates use a logit of their normalized MJCF
range, so every generated `R_ref` is mathematically within the active G1 joint ranges. This is an
output parameterization, not post-hoc clipping.

### CPU-safe deploy-condition adaptation

The `deploy` P1 condition is not identical to a held-out SEED window: it contains a local
time-aligned corridor/SDF from the probabilistic 3-D grid and a short route command. The
resource-bounded adaptation uses the existing support-preserving 6,508-window generalisation
archive, a 128/256 MLP, batch 64, CPU execution, and a small root-pose loss weight:

```bash
./run_stage2_deploy_training.sh
# optional: STAGE2_EPOCHS=80 STAGE2_ROOT_WEIGHT=2 ./run_stage2_deploy_training.sh
```

It writes three per-primitive conditional means under
`reports/manifold_motion/stage2_mean_deploy_cpu_v2/` (crouch, side, nominal walk). The script
does not mark a checkpoint deployable by training loss; `run_deploy_perception_demo.sh` must run
the same model through the SONIC/MuJoCo hard gate. The full A* route remains in the perception
artifact, while Stage-2 receives only a 0.9 m receding-horizon corridor. This alignment is
important: passing the full multi-metre route to a 1.6-second model causes an out-of-distribution
root trajectory.

### P1 deploy condition into the continuous main chain

`run_deploy_perception_demo.ps1` (or `.sh` inside WSL) is the end-to-end acceptance command. It
first builds the simulated radar -> probabilistic 3-D sliding voxel grid -> A* route and saves
`condition.npz`, `slam_grid.npz`, and the two map PNGs. It then passes that condition into
`stage2_manifold_adaptive`, not into the old isolated one-window replay. The continuous executor
keeps primitive switching, measured state/history, optimization-embedded projection, SONIC, the
50 Hz MuJoCo rollout, and the self-manifold hard gate in the main chain.

The deploy route uses the 3-D map for occupancy and vertical M_e evidence, a flat-ground XY
projection for this fixture, and an explicit detour margin. A crouch is enabled only when an
auditable overhead obstacle is present; side-wall returns cannot falsely trigger a low posture.
The accepted run uses `online-condition-iterations=1`; the turn helper keeps its verified phase
because the pilot corpus has no reliable online turn history. `receding-horizon-ticks=0` is
intentional until the turn/side online candidates are retrained.

The current CPU-safe acceptance result is 11/11 keyframes, zero obstacle contacts, zero runtime
self-manifold stops, minimum exact surface clearance 0.223 m, terminal error 0.218 m, route
deviation P95 0.120 m, and body-route yaw P95 15.1 degrees. The official replay is now a
synchronized split-screen GIF: left is the accepted MuJoCo G1 rollout, while right is the same
tick's executed trace, active primitive, measured `M_r^safe`, world-frame `M_e`, radar returns,
and 3-D probability slices. Its first 5 map frames replay the P1 scout updates. During Stage-2,
the simulated radar continues at 2.5 Hz (`--online-perception-scan-ticks 20`): each scan updates
the global sliding voxel grid, reruns local 3-D A*, and directly overrides the route lookahead yaw.
The current accepted online run adds 19 live scans (5 initial + 19 online = 24 map updates).
Intermediate legacy keyframes are progress-gated after a replan, while the terminal goal remains
a strict Euclidean position gate. `online_perception.npz` stores every map snapshot, route, radar
return batch, pose, and update tick used by the synchronized renderer.
The output is `artifacts/deploy_perception_demo/deploy_sonic_mujoco_comprehensive.gif` (the
uncomposed MuJoCo-only file remains `manifold_adaptive.gif`); the corresponding report is
`stage2_continuous_report.json`.

### What remains after online perception

The accepted online loop currently uses the new local A* route to control route yaw, while the
physically screened primitive clips still come from the initial P1 route decomposition. The next
algorithmic milestone is therefore online semantic replanning: rebuild the current `M_e` horizon,
reroute the primitive token when its aperture class changes, generate/project candidates from the
actual 69-D state/history, and commit a replacement only after a short-horizon MuJoCo safety gate.
After that, replace full-grid A* with incremental ESDF + D* Lite/LPA* (the current safe CPU demo
runs radar/SLAM/A* at 2.5 Hz), replace ground-truth MuJoCo odometry with timestamped SLAM poses,
and fine-tune SONIC on root-velocity/turn/side/crouch transition data before hardware deployment.

Six accepted long-horizon examples are collected by `run_stage2_long_gallery.ps1` (or `.sh`) in
`reports/manifold_motion/stage2_long_sequence_gallery_v1/`: wide nominal walking, long crouch,
long side gait, online center-block avoidance, crouch-walk-crouch, and crouch-to-side gait.

For an execution-aware conditional-mean baseline, `train-mean` also accepts
`--model-target-field target_exec`. That checkpoint uses the SONIC-achieved window target for
the mean proposal (with its own target normalizer); it is an ablation for the frozen controller's
tracking error, not a claim that SONIC has acquired direct root-position control.

## 4. Sample, execute, and select generated references

```bash
python3 -m manifold_motion.stage2_flow sample \
  --windows reports/manifold_motion/seed_windows_stage2_v2/seed_stage2_windows.npz \
  --autoencoder reports/manifold_motion/stage2_flow_v2/autoencoder.pt \
  --flow reports/manifold_motion/stage2_flow_v2/flow.pt \
  --split 2 --num-candidates 8 --out reports/manifold_motion/stage2_sample_v2

python3 -m manifold_motion.stage2_select \
  --sample reports/manifold_motion/stage2_sample_v2/sample.npz \
  --out reports/manifold_motion/stage2_selection_v2

python3 -m manifold_motion.stage2_validate \
  --sample reports/manifold_motion/stage2_selection_v2/selected_sample.npz \
  --out reports/manifold_motion/stage2_validation_v2
```

`stage2_select` evaluates every stochastic candidate with frozen SONIC/MuJoCo and applies the
same hard gate as the validator. It selects only from candidates that pass; it then ranks viable
candidates by tracking, exact mesh-corridor margin, execution progress, and reference smoothness.
If none pass, it writes the least-bad diagnostic candidate but exits with status 2.

The final validator returns status 0 only for a reference that passes the SONIC/MuJoCo gate. It
tests G1 mesh samples against the exact input corridor and requires execution progress. A sample
that stays in joint range but falls, rolls too far, loses support, creates non-foot ground
contacts, leaves its corridor, or remains essentially in place is a failed dynamic model
output—not a usable motion.

## 5. Open the MuJoCo GUI

WSL2 with WSLg exposes the native MuJoCo window on Windows. Use an evaluated sample, for example
the normal-walking independent test:

```bash
cd /home/xiyuan/Manifold-Motion-Pipline
export PYTHONPATH="$HOME/.local/share/sonic-manifold-g1:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3 -m manifold_motion.stage2_viewer \
  --sample reports/manifold_motion/stage2_mean_walk80_test2/sample.npz \
  --executed reports/manifold_motion/stage2_mean_walk80_validation_test2/executed.npz
```

The robot shown is the actual `q_exec/base_pos/base_quat` replay from SONIC/MuJoCo. Overlays are
cyan=model `R_ref`, green=held-out SEED reference, orange=actual execution, and translucent
purple=the conditioned reverse-synthesized corridor. `P` pauses, `N/M` step, `C` toggles the
corridor, `1/2/3` toggle trails, and `Q` closes the viewer. A rejected sample can be viewed for
diagnosis but must not be called deployable.

If a Remote Desktop session shows a white MuJoCo window or `WARN: COPY MODE`, render an animated
GIF instead; this bypasses the remote OpenGL surface entirely:

```bash
python3 -m manifold_motion.stage2_render \
  --sample reports/manifold_motion/stage2_mean_walk80_test2/sample.npz \
  --executed reports/manifold_motion/stage2_mean_walk80_validation_test2/executed.npz \
  --out /mnt/c/Users/Xiyuan\ Wang/Downloads/DeltaForce-Locker-desktop/stage2_walk_effect.gif \
  --width 640 --height 360 --fps 20
```

For a visible training-effect check, compare three executions on the same held-out window:

```powershell
.\run_stage2_comparison.ps1
```

The generated `stage2_training_comparison.gif` shows, left to right, the held-out SEED reference,
the early Flow checkpoint, and the trained conditional walk model. The labels report progress,
tracking error, and corridor margin.

### Residual Flow around a passing baseline

The deterministic conditional mean is retained as an executable anchor while a residual Flow
Matching field learns variation around it:

```bash
python3 -m manifold_motion.stage2_flow train-residual-flow \
  --windows reports/manifold_motion/seed_windows_walk80_v1/seed_stage2_windows.npz \
  --mean-model reports/manifold_motion/stage2_mean_walk80_v1/conditional_mean.pt \
  --out reports/manifold_motion/stage2_residual_flow_walk80_v1 --epochs 80

python3 -m manifold_motion.stage2_flow sample \
  --windows reports/manifold_motion/seed_windows_walk80_v1/seed_stage2_windows.npz \
  --sampler residual_flow \
  --mean-model reports/manifold_motion/stage2_mean_walk80_v1/conditional_mean.pt \
  --residual-flow reports/manifold_motion/stage2_residual_flow_walk80_v1/residual_flow.pt \
  --split 2 --index 2 --num-candidates 8 \
  --out reports/manifold_motion/stage2_residual_walk80_test2
```

Candidate 0 is the exact mean anchor; subsequent candidates are stochastic residual proposals.
Run `run_stage2_acceptance.ps1` from PowerShell to reproduce the five-window actor-disjoint
acceptance report. The current run is 5/5 accepted, with 8 candidates evaluated per window.

For a visual effect comparison that cannot be obscured by similar robot poses, run
`run_stage2_effect_dashboard.ps1`. It uses the same held-out test window on both sides and shows
SEED (green), generated reference (cyan), and actual SONIC/MuJoCo execution (orange) in root-frame
coordinates. Early Flow is rejected at 9.9% progress / corridor 2.365; the selected residual
candidate passes at 20.1% / corridor 0.982.

## 6. Perception SDF handoff

`manifold_motion.perception_corridor` is the runtime adapter for a real local point cloud:

```bash
python3 -m manifold_motion.stage2_perception_smoke
```

It emits the same `corridor [T,7]` and `sdf [10,10,8]` keys expected by Stage 2. The smoke test
uses a synthetic wall only to verify geometry and serialization. A real obstacle experiment still
requires time-aligned RGB-D/LiDAR points and a matching MuJoCo obstacle scene; feeding an
out-of-distribution wall to the flat-ground checkpoint is intentionally rejected by validation.

## Current boundary

This implements the robot-execution and trajectory-generation path, including a geometrically
consistent reverse corridor/SDF. It does **not** yet make claims about RGB-D/LiDAR obstacle
navigation: the next integration replaces this reverse constructor with a time-aligned local
SDF/safe-corridor encoding from the perception and planning stack, preserving the same stored
`corridor` and `sdf` keys.

## Generalisation, guarded rolling horizon, and capability policy

The pilot archive has 2,446 actor-disjoint windows. `stage2_generalization.py` creates a larger
6,508-window training archive by adding support-preserving corridor widening and small state/history
perturbations only to the training split; validation/test windows remain pristine. It also stores
`transition_phase`, `environment_bucket`, and `augmentation_id` for leakage audits. These variants
are synthetic conditioning augmentation, not sensor-derived obstacle data.

For the no-handcrafted-anchor A/B test:

```bash
./run_stage2_no_handcrafted_anchor.sh
```

`--disable-anchor` removes the hand-selected raw SEED exemplar while retaining the learned
conditional-mean proposal. This is the meaningful no-handcrafted-anchor ablation: it passes on the
wide and low-ceiling scenes, while the narrow side-step case is currently rejected when its
conditional Flow proposals point backwards. `--disable-learned-anchor` removes the additional
training-support mean candidate, and `--pure-stochastic-flow` is the strictest diagnostic; with the
current sparse pilot model both are expected to expose the missing side-transition coverage rather
than being hidden by a relaxed directional gate.

`--receding-horizon-ticks N` refreshes a future reference from the actual 69-D state and 12-frame
history. `--receding-horizon-shadow` is the safe experiment mode: candidates are generated and
audited while the last verified reference remains active. Direct commit mode is deliberately not
the default until each refresh can receive a current-state physical rollout gate.

Executions now log `route_progress_error_m`, `route_velocity_cmd_mps`, and `yaw_command_rad`.
These are the command channels required for a controller that directly tracks root progress; the
frozen SONIC action head currently consumes joint pose/velocity and base orientation, not root
position. The projection layer also includes velocity, acceleration, jerk, handoff, joint-limit,
and corridor terms. `stage2_capability.py` records crawl as unsupported and jump as partial, so
neither is routed automatically under the frozen controller.

## Additional visual demos

The original four regression scenes are complemented by three more accepted fixture scenes:

| scene | route-conditioned action sequence | evidence |
|---|---|---|
| `scene_manifold_low_short.xml` | `walk_nominal → crouch → walk_nominal` | 6/6 keyframes, 0 contact |
| `scene_manifold_block_left.xml` | side gait → nominal walk | 7/7 keyframes, 0 contact |
| `scene_manifold_block_right.xml` | side gait → nominal walk + one turn segment | 7/7 keyframes, 0 contact |

They are still MuJoCo fixture geometries, not RGB-D/LiDAR results. To render all three GIFs and
write their candidate/routing audit:

```bash
./run_stage2_diverse_demo.sh
```

The output is `reports/manifold_motion/stage2_diverse_demo_v2/`, with one
`manifold_adaptive.gif` per scene and a `comparison_report.json`.

For a stricter deployment margin, increase the hard-gate threshold on a single run:

```bash
python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_narrow_105cm.xml \
  --out reports/manifold_motion/narrow_deploy_gate \
  --self-manifold-clearance-m 0.05
```

The run is rejected if the exact G1 surface has less than that clearance, even when MuJoCo has
not yet reported a contact. This is the intended pre-deployment behavior.

For repeated environment changes in one continuous task, run
`./run_stage2_long_sequence_demo.sh`. The accepted long-sequence reports are documented in
`STAGE2_LONG_SEQUENCES.md`.

The extended gallery moves beyond the original four regression fixtures:

```bash
./run_stage2_extended_gallery.sh
```

| task | environment-driven sequence | accepted evidence |
|---|---|---|
| alternating chicane | nominal walk with repeated curvature turn helpers; no false side-gait trigger from a unilateral obstacle | 17/17 keyframes, 9 turn intervals, 0 obstacle contacts |
| low-side-turn compound | crouch → side gait → turn → nominal recovery | 18/18 keyframes, 3 turn intervals, 0 obstacle contacts |
| repeated gate cycle | crouch → recovery → crouch → side → crouch → recovery | 15/15 keyframes, 0 obstacle contacts |
| long slalom | nominal walk → three bilateral side intervals → nominal walk, with nine latched turns | 18/18 keyframes, 9 turn intervals, 0 obstacle contacts |

Every row uses the same aperture/curvature router, Flow candidate generator, projection layer and
continuous SONIC/MuJoCo safety gate. The scene files contain geometry only; they do not contain a
per-segment primitive schedule. Full reports are written to
`reports/manifold_motion/stage2_extended_gallery_v2/`.

The v2 scheduler uses turn hysteresis: a curvature helper has a minimum hold, releases below a
lower yaw threshold, and is not re-armed by the small body-yaw oscillation of a side gait. The
slalom is planned with the validated 0.42 m footprint radius; its exact surface clearance is
4.92 cm at runtime. The low-side compound also records an explicit `low_transition` helper
around crouch boundaries (36 ticks) while retaining the continuous state and physical gate.

The long-sequence GIFs now show three distinct layers: blue `M_e(t)` is the environment safe
corridor, white is the measured safe self-manifold `M_r^safe`, and the colored ellipsoid is the
task-conditioned `M_r^task` (orange=crouch, green=side gait, yellow=nominal, violet=turn).
`M_r^safe` is used by the deployment gate: its broad-phase ellipsoid is followed by an exact
G1-surface/obstacle narrow phase with a 2 cm clearance requirement. The arrays are stored in
`executed.npz` as `robot_manifold_safe`, `self_manifold_obstacle_clearance_m`, and
`self_manifold_environment_radius`; the report contains `robot_self_manifold_safety`.
The same surface-distance query also runs inside the 50 Hz continuous executor. If the current
state falls below the threshold, the executor stops before issuing the next control step and
records `runtime_self_manifold_clearance_stop`; this is separate from the terminal audit.

For the current accepted fixtures, the measured route-frame means are approximately:

| primitive | safe tangent | safe lateral | safe vertical | task lateral/vertical |
|---|---:|---:|---:|---:|
| crouch | 0.80 m | 0.53–0.61 m | 0.69 m | ≤ `0.90 M_e,z` |
| side gait | 0.73–0.80 m | 0.53 m | 1.05 m | ≤ `0.72 M_r,y`, `0.90 M_e,y` |
| nominal walk | 0.60 m | 0.59 m | 1.05 m | measured safe shape |

These values are generated frame-by-frame, not a fixed calibration ellipse. They are simulation
fixture measurements until the real point-cloud perception adapter is connected.

## Online semantic rerouting and incremental planning

The deploy chain now updates more than route yaw. Every live sensor update produces
`route -> M_e(48x7) + SDF -> debounced primitive_id`. If that primitive differs from the
currently executing gait, Stage 2 decodes and projects live-condition candidates. A switch is
committed only after a short shadow rollout cloned from the current MuJoCo `qpos/qvel/mocap`
and the current ten-frame SONIC history passes fall, roll, tracking, contact and exact
self-manifold-clearance gates.

The ground-bound route layer uses an XY body-height projection of the full 3-D map. The map and
vertical corridor remain 3-D; the route planner is incremental truncated ESDF + D* Lite in global
grid coordinates. Reports include `planning_ms`, `expanded_vertices`, `changed_cost_cells`,
`online_semantic_updates` and `online_semantic_switch_count`.

```bash
PYTHONPATH=. python3 tests/test_incremental_planner.py
PYTHONPATH=. python3 -m manifold_motion.incremental_dynamic_benchmark \
  --out reports/manifold_motion/incremental_dynamic_benchmark
```

The accepted moving-obstacle benchmark covers obstacle appearance, crossing and disappearance.
Its JSON compares incremental latency with the previous full voxel A* and its GIF displays both
the previous and updated routes.

## Real SLAM/time/extrinsic ingress

`manifold_motion.real_slam.TimeSynchronizedSlamAdapter` buffers timestamped `world_T_base`
poses, interpolates at the sensor header timestamp, applies calibrated `base_T_sensor`, rejects
stale/high-covariance packets, and emits the existing world-frame `RadarScan` contract. The
transport-independent unit test is:

```bash
PYTHONPATH=. python3 tests/test_real_slam.py
```

`configs/real_slam_example.json` defines frames, time limits and the 4x4 extrinsic. An optional
ROS2 transport is provided in `manifold_motion.ros2_slam_bridge`; the default topics are
`/slam/odometry` (`nav_msgs/Odometry`) and `/radar/points`
(`sensor_msgs/PointCloud2`). The current WSL acceptance uses simulated radar and ground-truth
odometry; a real sensor has not been claimed as tested until those topics and the calibrated
extrinsic are supplied on the target machine.

## Transition supplement and SONIC adapter boundary

The transition builder creates explicit actor/clip-provenance windows for `walk <-> turn`,
`walk <-> side` and `walk <-> crouch`:

```bash
PYTHONPATH=. python3 -m manifold_motion.stage2_transition_dataset \
  --windows reports/manifold_motion/seed_windows_corridor_stage2_v2/seed_stage2_windows.npz \
  --out reports/manifold_motion/seed_transition_stage2_v1/transitions.npz
PYTHONPATH=. python3 -m manifold_motion.stage2_transition_gate \
  --transitions reports/manifold_motion/seed_transition_stage2_v1/transitions.npz \
  --out reports/manifold_motion/seed_transition_stage2_v1/physical_gate.json
```

`manifold_motion.sonic_adapter` follows the reusable ORCS rule: frozen SONIC base,
zero-initialized low-rank residual, and a separate augmentation stream containing environment,
command and state/history. The zero adapter is bit-exact with the base by test. ORCS release
checkpoints remain task-specific PyTorch/rsl_rl weights and are not substituted for the current
ONNX files without conversion and parity verification.

The resource-bounded warm-start can be reproduced with:

```bash
SONIC_ADAPTER_THREADS=4 SONIC_ADAPTER_TRANSITIONS=24 \
  SONIC_ADAPTER_EPOCHS=10 ./run_sonic_adapter_warmstart.sh
```

The 24 selected clips are stratified by transition type and original source split. The current
pilot contains 960 train, 240 validation and 720 held-out test action rows. The zero residual has
exact base parity (`max_abs=0.0`); the best validation checkpoint reduces held-out action MSE from
0.5753 to 0.5493. This is a supervised initialization only. It is not enabled in the deployment
controller until privileged PPO/distillation and the complete MuJoCo acceptance matrix pass.

The simulation paper protocol, baselines, ablations and statistics are specified in
`CVPR_EXPERIMENTS.md` and `configs/cvpr_simulation_protocol.json`.
