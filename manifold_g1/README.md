# manifold_g1

Single-environment harness for the "Manifold → Motion" line of work: a MuJoCo G1 on flat
ground driven by the **frozen SONIC** controller, in-process at 50 Hz control / 200 Hz
physics — no DDS, no C++ deploy binary, no Isaac Lab.

The active stage is the **static manifold → pose distribution** pipeline. One robot, one
manifold *family*, one policy that maps every manifold in it to its own optimal pose:

```
body_envelope.py              StaticFitEnv (geometry)            frozen SONIC
────────────────              ─────────────────────             ────────────
hold keyframes in      →      PPO learns p(pose | M)       →     hold the learned pose as a
MuJoCo+SONIC, fit the         for a family of manifolds:          keyframe; measure the real
standing envelope and         how much to crouch, lean            containment and compare with
its extremity landmarks       and twist to fit inside             the model's prediction
over a (crouch, lean,
twist) grid
```

## Run

```bash
python3 -m manifold_g1.body_envelope --crouch 0,0.4,0.8,1.2 --lean 0,0.5,1.0 \
    --twist="-1.2,-0.8,-0.4,0,0.4,0.8,1.2"                    # calibrate (≈1 min)
python3 -m manifold_g1.static_fit train  --iterations 300 --envs 32   # learn p(pose | manifold)
python3 -m manifold_g1.static_fit report                       # manifold -> mean/std table
python3 -m manifold_g1.static_fit verify                       # MuJoCo + SONIC check
python3 -m manifold_g1.static_fit view                         # native viewer
```

or `make g1-envelope`, `make g1-static-train`, `make g1-static-report`, `make g1-static-verify`,
`make g1-static-view`.

Training is headless (terminal + `train_log.csv`). `view` opens the **native MuJoCo viewer**
(orbit/zoom with the mouse) and shows the translucent ellipsoid plus live curves.

Viewer keys (MuJoCo's viewer already owns `Space`, `+`/`-`, the arrows, `Tab` and `[` `]`,
which cycles cameras): `u` / `i` squash / raise the height, `j` / `k` pinch / widen,
`n` / `m` shift the manifold sideways, `R` reset, `P` pause.

## Layout

| File | Role |
|---|---|
| `constants.py` | Joint-order permutations, default angles, KP/KD/action scales, effort limits, observation-config parser. All values mirror `gear_sonic_deploy`. |
| `env.py` | MuJoCo flat scene, PD torque control, state readout (200 Hz physics, 4 substeps per control tick). |
| `sonic.py` | ONNX Runtime encoder + decoder with the deploy's observation assembly (g1 encoder mode, 10-frame history, zero padding). |
| `reference.py` | Pose directions by joint name (crouch, lean, twist) and `KeyframeReference`: a T = 1 motion reference = default pose + pose offsets. What the policy outputs is exactly what SONIC is asked to hold. |
| `keyframe_env.py` | Minimal MuJoCo + SONIC runner that holds a keyframe, returns real landmark positions, and reshapes the (visual-only) manifold in place. |
| `ppo.py` | Minimal PPO (clipped surrogate + GAE), actor-critic MLP with learned per-dimension action std. |
| `manifold.py` | `Primitive` / `EllipsoidManifold`: soft conditioning geometry — analytic containment `r` and a visual ellipsoid, no collision hull, no model rebuild. |
| `body_envelope.py` | Body model calibration: the standing envelope (fit to mesh-vertex samples) and the extremity landmarks over a (crouch, lean, twist, arms) grid, reached with the same protocol deployment uses (`--protocol slew`). |
| `static_fit.py` | The pipeline: `BodyModel`, `ManifoldSpec` family, `StaticFitEnv` (one stream) and `BatchedStaticEnv` (many manifolds in parallel), PPO training, `report`, SONIC `verify`, viewer. |

The dynamic corridor/goal stage (`geo_env.py`, `task_env.py`, `train.py`, `sim2sim.py`,
`planner.py`, `loop.py`, `run.py`, `verify_geo.py`) is retired while the static stage is
built out; it lives in the git history.

## The static task

### Manifold family

`ManifoldSpec(height, width, offset)` reshapes the standing envelope: `height` scales the
vertical semi-axis (and its centre), `width` pinches the lateral semi-axis, `offset` shifts the
manifold sideways. Training samples the *feasible* part of that box — a manifold is kept only
if some pose on the calibration grid keeps every landmark inside with room to spare
(`best r ≤ 0.95`, while success needs `r ≤ 0.98`).

The family is coupled: a manifold shifted towards the robot's **right** arm cannot be pinched
as hard, because that arm hangs ~2 cm wider than the left (measured, not assumed), so the
sampler draws a wider `width` for negative offsets.

### Pose

Three dimensions, all held by SONIC as a keyframe:

| Pose | Joints (by name) | Tracked by frozen SONIC? |
|---|---|---|
| `crouch ∈ [0, 1.3]` | hips flex, knees bend, ankles dorsiflex | yes, but saturates: it holds ≈1.2 of the 1.6–2.0 the keyframe can ask for |
| `twist ∈ [-1.2, 1.2]` | waist yaw (+ shoulder yaw) | yes |
| `lean ∈ [0, 0.5]` | waist pitch, hips/ankles compensate | weakly: the waist pitch is largely ignored |
| `arms ∈ [0, 0.4]` | shoulder roll in, elbows fold | **no**: in the deep regime the arm keyframe is not adopted at all |

The commanded box is therefore smaller than the calibrated grid. This was measured, not assumed:
sweeping one channel at a time and fitting the *achieved* joint angles back onto the pose
directions gives, for a commanded `lean = 1.0`, an achieved lean of ≈0.02 (residual 0.74 rad),
and for `arms = 1.0` a residual of 2.3 rad — the arms simply stay where the controller wants
them. Beyond ~1.3 the crouch saturates too (commanded 2.0 → achieved ≈1.3).

Two consequences worth keeping in mind:

- the manifold family has to stay within what those two channels can deliver — a *narrow* or
  *shallow* manifold is only fittable by tucking the arms, which this interface cannot do. To
  pose the limbs against the manifold, the interface would have to change: the SONIC release
  also accepts `vr_3point_local_target` / `vr_3point_local_orn_target`, i.e. explicit hand and
  head targets, which is the channel that could actually place the hands;
- the calibration protocol must match deployment. The frozen controller is *path dependent*:
  the same keyframe reached by slewing (what the viewer and `verify` do) and by jumping from a
  fresh reset gives different body configurations. `body_envelope.py --protocol slew|jump`
  selects the protocol; `slew` is what the model is calibrated with.

Why twist is in the mix: the hands are held forward, so a waist yaw **pulls one hand in and
pushes the other out**. That is useless for a centred manifold (both hands are already as close
to the body as they get) but exactly what a laterally *shifted* and *pinched* manifold needs —
twist out of the way of the near side, so the far hand clears. Measured: without the twist
channel the tightest manifolds sit at `r = 1.12`, with it `r = 0.98`.

### Ground-contact poses (measured, not implemented)

Kneeling is the obvious way past the crouch limit and was probed directly:

| pose | result |
|---|---|
| both knees (hips −1.2, knees 2.6, ankles −0.2) | **stable** — knees down to 0.09 m, head 0.898 m, drift 0.06 m |
| both knees, deeper (hips −1.5, knees 2.8) | stable — head 0.880 m |
| one knee (other foot planted) | **falls over** — roll −94°, drift 2.1 m |
| one hand reaching for the ground | the hand stops at 0.26 m: the shoulder keyframe is not tracked |

So double-knee kneeling works but buys only ~2 cm over the deep crouch (head 0.898 vs 0.897 m),
because the torso stays upright — the *height* comes from folding the torso, and that is the
channel the frozen controller will not follow. Single-knee support and hand support are out of
reach for the same reason. The kneeling pose is still a legitimate manoeuvre for the *planner*
to command (no joint limits are violated); it is not something the RL pose space can exploit.

### Manifold orientation and the spine

`ManifoldSpec(... , tilt_deg)` tilts the ellipsoid's axis in the sagittal plane, and the reward
now carries a spine term (`-w_spine * (1 - cos(spine, axis))`) instead of the hard-coded
`spine_alignment = 1.0`: the spine is taken as the pelvis → top-of-head direction, which the
body model already provides.

Measured behaviour: the policy does tilt the spine towards a tilted manifold — a forward-tilted
manifold gets +4.4 deg of spine tilt against +2.3 deg for the upright one — but the response is
small, because `lean` is the weakly tracked channel, and the policy prefers to answer a tilt with
crouch and twist (which it can actually execute). Tilted manifolds are consequently the hardest
entries in the verification table (r_real 1.04–1.23 against r_model 0.99–1.07).

### Interface

- **Action**: 3-D, `[crouch, lean, twist]`, mapped into the calibrated pose box (crouch and
  lean are one-sided, twist is signed). The pose responds as a first-order lag
  (`tau = 0.45 s`), measured from the SONIC stack.
- **Training**: `N` manifolds are trained in parallel (`--envs`, default 16), with a curriculum
  on the pose magnitude a manifold demands (`--need-start` → `--need-max`). The body model and
  the containment are vectorized over poses, so a parallel step costs ~29 µs per manifold
  instead of ~500 µs, and 300 iterations take about 80 s.
- **Observation** (11-D): the current pose, containment `r`, the manifold centre relative to
  the pelvis, its semi-axes, and its z-axis alignment — nothing else, so the same policy runs
  in the geometric model and in MuJoCo + SONIC.
- **Reward**: `+1/s` per unit of containment margin while inside (`r ≤ 0.98`, capped at
  `margin_cap` so there is no reason to ball up), a steep barrier `−25/s` per unit of violation
  past the margin, `−0.4/s` per unit of pose (economy), `−5/s` per unit of spine-axis
  misalignment, `+5` on 1 s of sustained containment.
- **Body model**: 8 landmark points (pelvis, torso, fingertips, knees, ankles). Each landmark
  is the *extremity* of its body — for each body the calibration stores the local-frame point
  of the farthest mesh vertex from the pelvis, so a landmark tracks the real surface in any
  pose (`head` is the top of the torso mesh, not a fixed offset). Containment is evaluated
  pelvis-relative, which removes the robot's forward drift from the comparison.

## Verified

| Scenario | Result |
|---|---|
| Keyframe stand (frozen SONIC) | base height 0.757–0.762 m, roll/pitch < 1°, no fall |
| Pose authority | head 1.292 m upright → 1.087 m at the executable `(crouch, lean) = (1.3, 0.32)` |
| Pose authority (keyframe reach) | the keyframe itself can ask for much more (head 0.85 m, hands to |x| 0.11 m) — the controller just will not hold it |
| Twist authority | one hand 0.229 m → −0.13 m across the yaw range, the other pushed out to 0.44 m |
| Training | 600 PPO iterations in ≈4 min (32 manifolds in parallel, 2048 transitions/update), success 0.98 |
| Static policy, SONIC check | `r_real` within 0.01–0.05 of `r_model` on the executable manifolds; 6 of 10 fully inside (`r ≤ 1`), the tilted ones are the residual |
| Spine alignment | active reward; +2.3 deg upright vs +4.4 deg for a +12 deg tilted manifold |
| Kneeling | double-knee stable (head 0.898 m), single-knee falls, hand support not tracked |

The learned `p(pose | manifold)` (`reports/manifold_g1/static_fit/greedy_table.json`), next to
the task's ground-truth optimum (least posing that still fits):

| manifold (h, w, offset) | learned (crouch, lean, twist) | `r` | task optimum | `r_opt` |
|---|---|---|---|---|
| 1.00, 0.98, +0.00 | 0.01, +0.44, +0.05 | 0.96 | 0.00, +0.25, +0.20 | 0.97 |
| 0.80, 0.98, +0.00 | 1.14, +0.07, +0.06 | 0.92 | 0.55, +0.15, +0.00 | 0.98 |
| 0.76, 0.98, +0.00 | 1.17, +0.11, +0.05 | 0.96 | 1.15, +0.00, +0.00 | 0.98 |
| 0.95, 0.90, +0.04 | 0.02, +0.38, +0.36 | 0.96 | 0.00, +0.35, +0.30 | 0.97 |
| 1.00, 0.98, −0.04 | 1.12, +0.04, −0.16 | 0.95 | 0.00, +0.40, −0.20 | 0.97 |
| 0.90, 0.90, +0.05 | 0.02, +0.38, +0.40 | 0.97 | 0.00, +0.20, +0.50 | 0.97 |

Two honest caveats, both visible above:

- the twist **sign** is manifold-determined (positive offset → positive twist, negative → negative),
  but its magnitude runs a little large, and on loose manifolds the policy keeps a safety crouch
  (1.1–1.2) where the optimum is none — a consequence of a flat inside reward plus exploration
  noise, which makes a deeper pose the safer bet. A sharper effort term or a state-dependent
  `log_std` would tighten it;
- the tightest manifolds end up `r_real ≈ 0.99–1.01` against `r_model ≈ 0.96–0.98`: at the
  boundary the model is slightly optimistic, because the real crouch pose differs from the
  calibration's by a few centimetres (leg stance asymmetry up to 5 cm in a deep crouch).

### Deploying a policy pose

The action is a *target*; the keyframe is slewed to it with the body model's first-order lag
(`PoseSlew`, `tau = 0.45 s`). Applying the raw action to the keyframe instead makes the policy
bang-bang between standing and full crouch — the robot cannot reach the commanded pose within
one tick while the observation claims it has.

## Assets

- Robot model: `data/g1_flat/scene_flat.xml` (+ `g1_29dof_with_hand.xml`). The G1 meshes are
  missing from this repo's Git LFS server, so they were downloaded from the official Unitree
  description (`unitreerobotics/unitree_ros`, `robots/g1_description/meshes`).
- Controller: `gear_sonic_deploy/policy/release/{model_encoder,model_decoder}.onnx` from
  HuggingFace `nvidia/GEAR-SONIC` (`python3 download_from_hf.py`).
- `data/` and `reports/` are git-ignored.

## Full write-up

[`STATIC_MANIFOLD_PIPELINE.md`](../STATIC_MANIFOLD_PIPELINE.md) in the repository root collects the
whole pipeline in one document: body model and its calibration protocol, the manifold family, the
reward design with the failure mode behind each weight, the verification results, and the measured
limits of the frozen controller.

## Roadmap

The project advances by training capability, not by module.

| Gate | Environment | Manifold | Capability | Status |
|---|---|---|---|---|
| L0 | 1× MuJoCo, flat | none | keyframe → frozen SONIC (stand/hold) | done |
| L1 | 1× MuJoCo | fixed corridor | PPO velocity policy → planner → SONIC, goal reaching | retired with the dynamic stage |
| L2 | 1× MuJoCo | static ellipsoid (height) | PPO pose policy fits the body inside | done (model and SONIC agree) |
| L3 | 1× MuJoCo | height + width + offset family | one policy, one optimal pose distribution per manifold | done (3-D pose, SONIC-verified) |
| L4 | 1× MuJoCo | tilted / rotated manifolds | orientation-aware posing (torso roll) | |
| L5+ | batched | randomized p(M) | multimodal π(M, s, c, z) | |

Principles: (1) correctness before throughput — one environment first; (2) deterministic
before distribution — `M → a` before `M → p(a|M)`; (3) simple manifold before complex
manifold; (4) Success > Safety > Stability > Naturalness > Diversity.
