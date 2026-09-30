# Architecture-aligned Stage 1 / Stage 2 experiment

This experiment is the executable interpretation of the project architecture figure.  It
separates the discrete primitive decision from the continuous dynamic reference generation:

```text
P1 simulated radar / 3-D probability grid / SDF
        ↓
environment manifold M_e(t)
        ↓
Stage 1: p(z_p | M_e)                  (discrete primitive)
        ↓
Stage 2: p(R_{t:t+H} | M_e(t), z_p, s_t, H_t, c_t)
        ↓
projection + SONIC + MuJoCo
```

`M_self` is not the behavior input in this audit.  It is retained as a measured, pose-dependent
body envelope and used only by the post-generation physical safety gate.  This is why the main
visualization does not draw a large robot-centered ellipse over the G1.

![Architecture-aligned Stage 1 and Stage 2 experiment](architecture_figure_v2/architecture_stage1_stage2.gif)

## What is held fixed

All four cases reuse the same held-out state, the same 12-frame history, the same command, the
same conditional-mean Stage-2 checkpoint, and the same inference seed.  Only the environment
corridor `M_e(t)` and its coherent SDF are replaced.  The top half of each card shows the
environment corridor and the Stage-1 primitive distribution; the middle chart shows the selected
Stage-2 knee references `R_ref` against the corresponding measured `R_exec`; the bottom band is
the authoritative SONIC/MuJoCo execution from the corresponding physical pilot.  The separate
fixed-`z_p` counterfactual is quantified in JSON rather than being overlaid on a different
physical rollout.

The report is machine-readable at
[`architecture_report.json`](architecture_figure_v2/architecture_report.json).  The controlled
continuous Stage-2 effect is the `fixed_zp_effect_vs_wide` field: it holds `z_p=walk_nominal`
fixed and measures how the decoded reference changes when only `M_e` changes.

| Case | Environment change | learned Stage-1 top / physical deployed `z_p` | Stage-2 fixed-`z_p` joint RMS | Stage-2 fixed-`z_p` root RMS |
|---|---|---|---:|---:|
| wide | nominal aperture | walk_nominal / walk_nominal | 0.000 rad | 0.000 m |
| low | vertical aperture contracts | crouch / crouch | 0.0489 rad | 0.0275 m |
| narrow | lateral aperture contracts | walk_lateral_reverse / walk_lateral_reverse | 0.0498 rad | 0.0118 m |
| center | route bends around a block | walk_lateral_reverse / walk_lateral_reverse | 0.0506 rad | 0.0480 m |

The physical pilot executions remain separately gate-checked; all four cards report a passing
`M_self` safety gate.  The calibrated learned router now matches the deployment decision in
all 4 audited conditions.  During execution its neural proposal is accepted only if it agrees
with the geometry/self-manifold gate; an out-of-distribution disagreement is recorded and falls
back to the auditable safe route.  Across all 28 physical route segments in this audit the
learned proposal passes the gate, so the Stage-2 token is actually produced by Stage 1 rather
than selected from a segment index or a prewritten action sequence.

The table is not a claim of real-SLAM generalization: the current P1 source is simulated
perception and the pilot corridors are reverse-synthesized from the checked-in SEED-derived
windows.

## Reproduce

From WSL at the repository root:

```bash
export PYTHONPATH="$PWD:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl
./scripts/python.sh -m manifold_motion.evaluation.architecture_experiment \
  --out docs/experiments/architecture_figure_v2
```

To reproduce the calibration and physical closed loop first run:

```bash
./scripts/python.sh -m manifold_motion.planning.calibrate_primitive_router \
  --windows reports/manifold_motion/seed_windows_corridor_stage2_v2/seed_stage2_windows.npz \
  --pilot-root reports/manifold_motion/stage2_manifold_counterfactual_v1 \
  --base-router reports/manifold_motion/primitive_router_geometry_v1/router.pt \
  --out reports/manifold_motion/primitive_router_geometry_v2
cp reports/manifold_motion/primitive_router_geometry_v2/router.pt \
  models/stage1/primitive_router_geometry_v2.pt
./scripts/run_stage2_stage1_closed_loop.sh
```

The commands use CPU-safe checkpoints already produced by the project. They do not modify the
SONIC controller. The calibrated router retains 88.7% accuracy on the original held-out SEED
windows and reaches 100% (28/28) agreement on the physical deployment conditions.
