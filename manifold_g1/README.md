# manifold_g1

Single-environment harness for the "Manifold → Motion" line of work: a MuJoCo G1 on flat
ground driven by the **frozen SONIC** controller, in-process at 50 Hz control / 200 Hz
physics — no DDS, no C++ deploy binary, no Isaac Lab.

The active pipeline has three stages. Each is judged by its own metric, and they are not
interchangeable:

```
① data                         ② clone (BC)                  ③ in-the-loop fine-tune
collect → capability →          (M, s) → q  from the          perturb each recorded envelope,
pose demos                      recorded (envelope, pose)     measure the pose SONIC REACHES
                                                                                           
10376 poses / 9859 manifolds    bc/bc_policy.pt               sonic_rl/policy_sonicrl.pt
track err 0.054 rad             fits 9341/9859 (95%)          success 1.00, r 0.82
```

Everything is measured on the real controller in the loop, not on a model of it: the
controller is fast enough to run serially (one pose ≈ 0.3 s of physics, a full fine-tune 879 s),
so there is no residual/execution model anywhere in the live path.

## Run

```bash
# ① data (only needed to rebuild the dataset)
python3 -m manifold_g1.collect --episodes 8 --jobs 4 --seconds 20
python3 -m manifold_g1.capability build
python3 -m manifold_g1.demos build
python3 -m manifold_g1.dataset build && python3 -m manifold_g1.dataset show

# ② the clone
python3 -m manifold_g1.bc train --epochs 200
python3 -m manifold_g1.bc eval --policy reports/manifold_g1/bc/bc_policy.pt

# ③ SONIC in the loop
python3 -m manifold_g1.sonic_rl run --iterations 30 --manifolds 16 --steps 6
python3 -m manifold_g1.sonic_rl eval --manifolds 10 --steps 4 \
    --policy reports/manifold_g1/sonic_rl/policy_sonicrl.pt

# see it
python3 -m manifold_g1.view_data replay          # ① the recording
python3 -m manifold_g1.show_bc                   # ② the clone on its training manifolds
python3 -m manifold_g1.show_rl                   # ③ the fine-tune, keys reshape the manifold
```

See `VISUALIZATION.md` for viewer keys and what each number on screen means.

## Layout

| File | Role |
|---|---|
| `constants.py` | Joint-order permutations, default angles, KP/KD/action scales, effort limits, observation-config parser. All values mirror `gear_sonic_deploy`. |
| `env.py` | MuJoCo flat scene, PD torque control, state readout (200 Hz physics, 4 substeps per control tick). |
| `sonic.py` | ONNX Runtime encoder + decoder with the deploy's observation assembly (g1 encoder mode, 10-frame history, zero padding). |
| `reference.py` | Pose directions by joint name and `KeyframeReference`: a T = 1 motion reference. `set_joints` takes the full 29-joint pose, which is what the policy outputs. |
| `keyframe_env.py` | MuJoCo + SONIC runner that holds a pose as a keyframe and reports where the robot actually ended up. The execution layer for every stage. |
| `ppo.py` | Minimal PPO (clipped surrogate + GAE), actor-critic MLP with learned per-dimension action std. |
| `pose_policy.py` | **The interface, single source of truth**: the 15-D observation layout and the `tanh` action → pose map. Both trainers and every viewer use this one definition. |
| `torch_env.py` | `TorchPrimitiveEnv`: batched observation and containment on the GPU. Exists for the observation layout and the containment ruler that BC and the verifier share. |
| `demos.py` | The demonstration set: `(M, s) → q` cut from recorded motion, plus `load_demos`, the one reader every consumer uses. |
| `manifold.py` | `Primitive` / `EllipsoidManifold`: soft conditioning geometry — analytic containment `r` and a visual ellipsoid, no collision hull, no model rebuild. |
| `family.py` | `ManifoldSpec`: the manifold family (height/width/depth/offset/tilt) and the standing envelope it scales. |
| `bc.py` | Behaviour cloning: supervises `(M, s) → q` on the recorded pairs. |
| `sonic_rl.py` | The in-the-loop fine-tune: real controller, real physics, reward from the pose actually reached, with manifold perturbation. |
| `kinematics.py` | Batched GPU forward kinematics, verified geom-by-geom against MuJoCo. Lets the containment ruler run on device. |
| `body_envelope.py` | Body-surface sampling, `fit_ellipsoid`, and the extremity-landmark calibration. |
| `static_fit.py` | Live part: `BodyModel` (29-joint pose → body surface points). Its `train`/`report`/`verify`/`diagnostics` subcommands belong to a retired 4-channel route and have no callers. |
| `dataset.py`, `capability.py`, `calibrate.py`, `clip.py`, `collect.py` | The data stage: scene/motion collection, the capability map, and the sliding-window dataset. |
| `recompute_envelopes.py` | Rewrites stored clip envelopes onto the body-surface ruler. **The place to fix the box-vs-ellipsoid mismatch** (see below). |

## The containment ruler, and one measurement bug that is now fixed

Containment is `r = ||(p - c) / semi||` against an ellipsoid. Until 2026-09-16 the stored
envelopes were the per-axis extents `max|rel|` - a **box** - while the test is an **ellipsoid**,
and a box corner sits at r = 1.73 under an ellipsoid norm. The result: the recorded pose scored
**1.18-1.57 (median 1.40) against its own envelope**, and the standing body sat at **r = 1.145**
against the family's scale-1.0 manifold, so "fit inside the manifold" was unsatisfiable by the
poses the envelopes were measured from.

Both writers now store a fitted ellipsoid (per-axis extents inflated by the single scalar
`max_i ||rel_i / semi||`, the smallest uniform inflation containing the body), and
`family.BASE_SEMI`/`BASE_CENTER` are derived from the same fit the demonstrations use. Verified:
the recorded pose is at r = 1.0000 and the standing pose at r = 1.0002 in scale 1.0.

Two things worth knowing about the fix:

* it is a **uniform** scale, so the shape is unchanged. A per-axis optimum is only 11% smaller,
  and no global constant would do - the factor varies per tick (1.18-1.57, median 1.40).
* the demonstration set was **never affected**: `demos.py` builds each manifold from the pose via
  `fit_ellipsoid` rather than reading the stored envelope, so the clone's training data was
  always correct (95% before the fix, 94.9% after - the same number).

`verify_sonic` still reads ~3% above 1, and that is arithmetic rather than a policy failure: the
demonstration manifolds carry `MARGIN = 1.05` while `report_specs` scale 1.0 carries none, so a
pose fitted at 0.98 reads 0.98 x 1.05 = 1.029. Measured, it reads 1.030.

## The demonstration set is almost one pose per manifold

9480 of 9840 manifolds hold a **single** recorded pose (374 have >= 2, 10 have >= 4), so the clone
learns `M -> one q`. That is the stage's definition - it is supposed to learn the demonstrated
*intent*, not a distribution - and multi-style exploration is the fine-tune's job, not the data's.
What the data does provide is variation *between* manifolds: per-joint std 8.6 deg, correlated
+0.42 with the manifold parameters, which is the signal `M -> q` needs.

## Known limits of the frozen controller

Measured by sweeping the pose channels and fitting the *achieved* joints back onto the commanded
directions. These bound what any policy here can ask for:

| channel | tracked? |
|---|---|
| crouch (hips/knees/ankles) | yes, but saturates: held ≈1.2 of the 1.6–2.0 requested |
| twist (waist yaw + shoulder yaw) | yes |
| lean (waist pitch) | weakly: the waist pitch is largely ignored |
| arms (shoulder roll, elbows) | **no** in the deep regime: the arm keyframe is not adopted |

The arm result is why a *narrow* manifold is hard: the hands are the widest part of the body and
the channel that would tuck them is the one the controller declines. Placing the hands would
need `vr_3point_local_target` / `vr_3point_local_orn_target`, which the SONIC release also
accepts.

Ground-contact poses were probed directly: double-knee kneeling is stable but buys only ~2 cm of
head height over the deep crouch (the torso stays upright, and folding the torso is the channel
the controller will not follow); single-knee support falls over (roll −94°).

## Retired: geometric RL + the residual model

Early on the plan was to train against a learned model of the controller — a residual `Δ̂(q)`
predicting what SONIC would do with a commanded pose — because the geometric path runs thousands
of iterations per second. That is gone. The residual being better than ignoring it (51%) never
made it correct, the objective became "what I think the controller does", and running the real
controller serially turned out to be cheap enough to just do. `sonic_rl.py` never imported it.

Removed: `primitive.py`, `primitive_torch.py` (its environment half survives as `torch_env.py`),
`residual.py`, `residual_data.py`, `compare.py`, `reports/manifold_g1/primitive_torch/policy*.pt`,
`sonic_residual.pt`, `residual_pairs.npz`. Backup: `~/sonic_backup_20260916_1637/`.

## Assets

- Robot model: `data/g1_flat/scene_flat.xml` (+ `g1_29dof_with_hand.xml`). The G1 meshes are
  missing from this repo's Git LFS server, so they were downloaded from the official Unitree
  description (`unitreerobotics/unitree_ros`, `robots/g1_description/meshes`).
- Controller: `gear_sonic_deploy/policy/release/{model_encoder,model_decoder}.onnx` from
  HuggingFace `nvidia/GEAR-SONIC` (`python3 download_from_hf.py`).
- `data/` and `reports/` are git-ignored, so trained policies live only on disk — back them up
  separately from git.

## Full write-up

[`STATIC_MANIFOLD_PIPELINE.md`](../STATIC_MANIFOLD_PIPELINE.md) collects the current pipeline and
what was retired; [`VISUALIZATION.md`](../VISUALIZATION.md) covers the three viewers.

## Roadmap

The project advances by training capability, not by module.

| Gate | Manifold | Capability | Status |
|---|---|---|---|
| L0 | none | keyframe → frozen SONIC (stand/hold) | done |
| L1 | static ellipsoid | pose policy fits the body inside, model and SONIC agree | done, then replaced by the in-loop route |
| L2 | recorded-envelope family | clone `(M, s) → q` on the recorded pairs | done — 95% of training manifolds |
| L3 | perturbed envelopes | fine-tune on the pose SONIC reaches, generalise in the manifold | done — success 1.00, r 0.82 |
| L4 | same | fix the box/ellipsoid ruler so `r` means what it says | **next** |
| L5 | same | more than one recorded pose per manifold, so the clone can learn a distribution | data-side blocker |
| L6+ | randomized p(M) | multimodal π(M, s, c, z) | |

Principles: (1) correctness before throughput — one environment first; (2) deterministic
before distribution — `M → a` before `M → p(a|M)`; (3) simple manifold before complex
manifold; (4) Success > Safety > Stability > Naturalness > Diversity.
