# manifold_g1

Single-environment harness for the "Manifold → Motion" line of work: a MuJoCo G1 on
flat ground driven by the **frozen SONIC** controller, all in Python.

This is the first milestone: the closed loop `reference → SONIC encoder → SONIC
decoder → MuJoCo G1` runs in-process at 50 Hz control / 200 Hz physics, with no
DDS, no C++ deploy binary and no Isaac Lab.

## Run

```bash
python3 -m manifold_g1.run --mode stand --seconds 5          # static stand reference
python3 -m manifold_g1.run --mode walk  --seconds 10 --video out.mp4   # planner-driven walk

python3 -m manifold_g1.train --iterations 60 --rollout-steps 512 --eval-every 15   # PPO on the goal task
python3 -m manifold_g1.train --eval-only --resume reports/manifold_g1/ppo/policy.pt --eval-episodes 20
python3 -m manifold_g1.train --viz --iterations 60 --rollout-steps 512             # live 3D dashboard
```

Or via make: `make g1-stand`, `make g1-walk`, `make g1-train`, `make g1-eval`,
`make g1-train-viz`, `make g1-train-video`.

Outputs (metrics JSON + per-tick npz + optional mp4 + PPO logs/checkpoints) go to
`reports/manifold_g1/`, which is git-ignored.

## Layout

| File | Role |
|---|---|
| `constants.py` | Joint-order permutations, default angles, KP/KD/action scales, quaternion helpers, observation-config parser. All values mirror `gear_sonic_deploy`. |
| `env.py` | MuJoCo flat scene, torque PD control, state readout (200 Hz physics, 4 substeps per control tick). |
| `sonic.py` | ONNX Runtime encoder + decoder with the deploy's observation assembly (g1 encoder mode, 10-frame history, zero padding). |
| `planner.py` | `planner_sonic.onnx` wrapper (kinematic motion generator, 27 modes). |
| `reference.py` | 50 Hz reference buffers: static stand or planner-generated with 8-frame crossfade replanning. |
| `loop.py` | Open-loop control loop, metrics, logging, optional viewer / offscreen video. |
| `manifold.py` | Procedural manifold geometry (currently an axis-aligned corridor with parameters L, W, H). |
| `task_env.py` | `GoalReachEnv`: gym-like single-env task, velocity-command action, progress/collision/energy rewards, fall/success termination. |
| `ppo.py` | Minimal PPO (clipped surrogate + GAE), actor-critic MLP with learned action std. |
| `train.py` | Training/eval CLI, CSV logging, checkpointing, live 3D dashboard hooks. |
| `live_viz.py` | Live dashboard: two 3D MuJoCo renders (manifold / robot) plus metric curves. |
| `run.py` | Stand/walk demo CLI. |

## Assets

- Robot model: `data/g1_flat/scene_flat.xml` (+ `g1_29dof_with_hand.xml`). The G1 meshes
  are missing from this repo's Git LFS server, so they were downloaded from the
  official Unitree description (`unitreerobotics/unitree_ros`,
  `robots/g1_description/meshes`) into `data/g1_flat/meshes/`.
- Controller: `gear_sonic_deploy/policy/release/{model_encoder,model_decoder}.onnx`
  and `planner/target_vel/V2/planner_sonic.onnx`, downloaded from HuggingFace
  `nvidia/GEAR-SONIC` (needs a proxy on this machine).
- `data/` is git-ignored, so the generated manifold scenes live there too.

## Manifold and RL interface (L1-L3)

- **Manifold**: corridor of length L and width W whose ceiling is high at the entrance
  (`height_start`) and drops to `height_goal` from `ramp_end_x` onward. The free space is
  drawn as an ellipsoid chain (manifold primitives); the box hull is what actually collides.
  The policy observes `(L, W, height_start, height_goal)` plus the local ceiling clearance.
- **Policy**: 10 Hz, action = `[vx, vy, wz, crouch]`:
  - `vx, vy, wz` are body-frame velocity commands mapped to the kinematic planner
    (mode 2 walking; heading aimed at the goal to suppress gait drift);
  - `crouch ∈ [-1, 1]` scales a sagittal crouch offset (hips/knees/ankles in joint space,
    `crouch_max` = 1.3) applied to the planner's walking reference — this is how the policy
    lowers the body when the manifold gets low. `CrouchReference` re-applies the offset after
    every replan.
- **Observation** (82-D): body-frame linear/angular velocity, gravity, joint positions
  relative to the default pose, joint velocities, goal vector in body frame, heading error,
  previous action, base height, normalized ceiling margin, manifold parameters.
- **Reward**: progress + goal bonus (10) − collision (capped penetration, 30/m) − energy (0.1)
  − fall penalty (10) − hard-collision penalty (5).
- **Termination**: fall (height < 0.45 m or tilt > 60°), success (< 0.40 m), hard collision,
  out of bounds; time limit 200 policy steps (20 s).
- **Curriculum** (`--curriculum`): the goal-end ceiling starts at `--height-goal` and drops by
  `--curriculum-step` whenever the rolling success rate exceeds `--curriculum-success`,
  down to `--height-goal-min`. Achievable body heights (measured): pelvis 0.76 m walking
  upright, 0.60 m at full crouch; a ceiling below ~1.00 m therefore requires crouching.

## Verified

| Scenario | Result |
|---|---|
| `--mode stand` (3 s) | base height 0.757–0.762 m, roll/pitch < 1°, no fall |
| `--mode walk` (10 s) | 11.6 m forward, vx ≈ 1.22 m/s, height 0.73–0.78 m, roll RMS 2.3°, no fall |
| Scripted policy (full forward) | reaches goal in 2.5 s, episode return ≈ +11.6 |
| PPO (`make g1-train`, 60 iters) | greedy eval 20/20 success, 0 falls, 0 collisions, 2.8 s to goal (mean) |
| Crouch authority (scripted) | pelvis 0.76 m upright → 0.60 m at full crouch, stable |
| Low-ceiling task (scripted, H_goal 0.95 m) | upright: ceiling contact; crouch ≈1.2: passes clean |
| Speed | ~3x faster than real time on CPU (10.4 s of sim in 3.9 s wall, incl. video) |

## Live 3D training dashboard

Enable while training (`--viz` opens a window; `--viz-video <path>` records it headless):

```bash
make g1-train-viz                                        # live window
python3 -m manifold_g1.train --viz-video reports/manifold_g1/viz/training.mp4 --iterations 60
```

The dashboard shows, at ~2 Hz while training runs (raise `--viz-every` to reduce the
rendering overhead, which slows training by roughly 2-4x):

- **top-left (3D MuJoCo render)** — the manifold with the robot hidden: the corridor and the
  translucent **ellipsoid chain** that represents the manifold primitives. As L3 sweeps the
  height/width, this view shows the ellipses shrinking.
- **top-right (3D MuJoCo render)** — the G1 training *inside* the manifold, camera tracking
  the pelvis, so you watch the behavior evolve from random fumbling to goal-reaching.
- **bottom row (live curves)** — episode return, rolling success rate, mean `|vx|` command,
  the manifold compliance radius `r` (max normalized free-space-ellipse radius over the
  tracked body points; `r > 1` means a body point left the manifold), and the L3 panel:
  achieved pelvis height versus the manifold's goal-end ceiling height (green = success,
  red = failed). The L3 signal is that green points must sit lower as the ceiling drops.

## Curriculum gates

The project advances by training capability, not by module; each gate must pass before
the next one starts.

| Gate | Environment | Manifold | Capability | Status |
|---|---|---|---|---|
| L0 | 1x MuJoCo, flat | none | fixed reference → frozen SONIC (stand/walk) | done |
| L1 | 1x MuJoCo | fixed corridor (L, W, H) | PPO velocity policy → planner → SONIC, goal reaching | done |
| L2 | 1x MuJoCo | fixed | PPO converged (eval success > 90%) | done (20/20 greedy) |
| L3 | 1x MuJoCo | ceiling height swept | manifold height → body height (crouch channel) | in progress |
| L4 | 1x MuJoCo | one new parameter per stage (W, θ, κ, obstacles) | walk/crouch generalization | |
| L5 | batched 8→4096 | randomized p(M) | single-mode large-scale training | |
| L6 | batched | randomized | multimodal π(M, s, c, z) | |
| L7 | batched | complex 3D (ellipsoids, slopes, sparse footholds) | multimodal generalization | |
| L8 | real robot + LiDAR | point-cloud manifold | sim2real | |

Principles: (1) correctness before throughput — one environment first; (2) deterministic
before distribution — `M → a` before `M → p(a|M)`; (3) simple manifold before complex
manifold; (4) Success > Safety > Stability > Naturalness > Diversity.

## Not implemented yet

- Batched / vectorized environments (L5).
- Manifold parameter randomization and curriculum (L3/L4).
- Motion-latent / multimodal policy (L6).
- Complex 3D manifolds (ellipsoids, slopes, sparse footholds) and LiDAR input (L7/L8).
- Reference tracking reward / termination terms ported from the Isaac Lab env.
