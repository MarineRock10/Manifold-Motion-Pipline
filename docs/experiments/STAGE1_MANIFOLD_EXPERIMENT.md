# Stage 1 CVPR experiment: manifold to primitive

> **Audit notice (2026-09-28).** The numerical pilot claims below are historical and
> superseded. The original gate mixed absolute joints with pose deltas, used different
> physical durations for imitation and PPO, and truncated to the first action family.
> CPU validation checkpoints also aliased live parameters before the clone fix.
> Do not cite those pilot acceptance values as valid paired comparisons.
> The new `catalog_experiment` protocol uses a shared executor, all 727 held-out rows,
> three independent seeds, validation-only checkpoint selection, and actor-cluster
> uncertainty estimates. Its short-response gate is explicitly distinguished from
> a separate 3-second hold diagnostic. The completed results are in the
> [Stage-1 A/B report](stage1_ab_response_v3/README.md), including the negative RL result.
> The subsequent [bounded-residual RL revision](stage1_residual_rl_v4/README.md)
> reports a small execution-fidelity gain together with a gate-acceptance trade-off;
> it does not replace or relabel the historical negative result.
> The [mixed-horizon six-method matrix](stage1_mixed_v5/README.md) is the current
> reproducible Stage-1 completion and records the negative RL result with long-horizon
> validation, continued-IL and no-long-reward controls.
> The [simulator-aware inverse-tracking adapter](stage1_simulator_adapter_v9/README.md)
> is a subsequent model-based policy-improvement experiment; it corrects frozen SONIC's
> measured request-to-hold error with a bounded local residual and reports its safety trade-off.

## Current baseline

### Temporal primitive completion (confirmation protocol)

The remaining Stage-1 temporal head is implemented in
`manifold_motion.stage1.temporal_primitive`.  It deliberately removes three leakage paths from
the static catalogue: the recorded primitive label is not an input, future `self_manifold` is not
an input, and the target pose is not an input.  The model sees only the 36-frame environment
ellipsoid sequence `M_e(t)` and predicts all 29 joints for all 36 frames.  An auxiliary family
head is scored, but is never used to select a target at test time.

The confirmation archive is made once, before training, by stable actor hashing:

```bash
./scripts/python.sh -m manifold_motion.stage1.temporal_primitive make-confirmation \
  --catalog data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz \
  --metadata reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows_metadata.json \
  --out data/manifold_action_catalog_v1/manifold_action_catalog_confirmation_v1.npz
./scripts/python.sh -m manifold_motion.stage1.temporal_primitive train \
  --catalog data/manifold_action_catalog_v1/manifold_action_catalog_confirmation_v1.npz \
  --out reports/manifold_motion/stage1_temporal_primitive_v11 --eval-split 3
MUJOCO_GL=egl ./scripts/python.sh -m manifold_motion.stage1.temporal_primitive physical \
  --catalog data/manifold_action_catalog_v1/manifold_action_catalog_confirmation_v1.npz \
  --checkpoint reports/manifold_motion/stage1_temporal_primitive_v11/temporal_primitive.pt \
  --out reports/manifold_motion/stage1_temporal_primitive_v11 --eval-split 3
./scripts/python.sh -m manifold_motion.stage1.temporal_primitive report \
  --out reports/manifold_motion/stage1_temporal_primitive_v11
```

The [temporal report](stage1_temporal_primitive_v11/README.md)
uses 1,587 training, 1,151 validation and 523 actor-held-out confirmation windows.  The model
gets 0.1920 rad confirmation joint MAE (global sequence mean: 0.2227 rad), 39.4% family top-1,
and 12/14 physical confirmation rollouts accepted.  The [failure clusters](stage1_failure_clusters_v1/failure_clusters.json)
identify high-jump/box-jump root drift and several family confusions; those supplements are
fixed before the next SEED extraction rather than selected from the confirmation results.

This completes the executable Stage-1 temporal experiment, not the entire project: root
translation/rotation prediction, online routing and dynamic obstacle interaction remain Stage 2;
SONIC remains frozen in this experiment.

The first reproducible Stage 1 run uses the reverse-synthesized manifold/action catalogue. The
input is a static descriptor of each `M_e(t)` window; the target is the middle 29-DoF joint pose
of the executed `target_exec` trajectory, expressed relative to the standing pose. Actor-disjoint
splits are inherited from the catalogue.

```bash
./scripts/python.sh -m manifold_motion.stage1.manifold_imitation \
  --catalog data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz \
  --out reports/manifold_motion/stage1_manifold_imitation_v1 \
  --target-field target_exec --epochs 80 --batch-size 128 --hidden 256 --device cpu
```

The current run uses 2,610 train, 651 validation and 727 test windows. Test joint-angle MAE is
0.198 rad. The two fixed baselines are 0.255 rad for a global mean pose and 0.206 rad for an
action-family mean pose. The complete per-family result is
[`stage1_manifold_imitation_v1.json`](results/stage1_manifold_imitation_v1.json).

This is an imitation result, not yet the final physical claim. The model predicts a pose target;
the next gate sends its deterministic mean through SONIC and MuJoCo and reports tracking,
containment, contact and self-manifold clearance with the same test rows.

The physical gate is now wired and can be run on a bounded pilot while the full sweep is being
prepared:

```bash
./scripts/python.sh -m manifold_motion.stage1.manifold_physical_gate \
  --catalog data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz \
  --checkpoint reports/manifold_motion/stage1_manifold_imitation_v1/manifold_imitation.pt \
  --out reports/manifold_motion/stage1_physical_gate_v1.json --max-rows 24 --steps 30
```

The latest bounded 24-row pilot accepted 24/24 imitation outputs with mean spatial-union corridor
radius 0.868 and p95 radius 0.992. It is an executor pilot, not paper statistics; the full
held-out gate must run every test row before making a physical Stage-1 claim. The paired report is
[`stage1_physical_gate_v1.json`](results/stage1_physical_gate_v1.json).

The first 24-row paired warm-start pilot is also recorded in
[`stage1_warm_paired_gate_pilot_v1.json`](results/stage1_warm_paired_gate_pilot_v1.json).
Warm-start SONIC reduced final joint tracking error to 0.074 rad, but its corridor radius rose to
0.746 and acceptance fell to 23/24, versus 0.680 and 24/24 for direct imitation. This is a
failure-directed result: controller tracking and geometric containment are not interchangeable,
so the next fine-tune adds an explicit containment/barrier term instead of selecting a checkpoint
by tracking error alone.

The first containment-weighted retry (before correcting the mean-probe state handling) is
[`stage1_barrier_paired_gate_pilot_v1.json`](results/stage1_barrier_paired_gate_pilot_v1.json).
It is retained as a debugging artifact, not a valid comparison, because the mean probe was
evaluated after the sampled action had already changed the simulator state.

The valid same-state-probe rerun is
[`stage1_barrier_snapshot_paired_gate_pilot_v1.json`](results/stage1_barrier_snapshot_paired_gate_pilot_v1.json).
It gives 23/24 acceptance, radius 0.750 and tracking error 0.077 rad for warm-start SONIC,
versus 24/24, radius 0.680 and tracking error 0.203 rad for direct imitation. The current
selection therefore remains direct imitation; the next fine-tune must be a constrained residual
update whose safety is selected on held-out physical rows.

The SONIC-in-the-loop fine-tune now accepts the same catalogue directly. It samples only the
actor-disjoint training split, uses the recorded middle-frame pose as the imitation anchor, and
measures the achieved pose with the real SONIC/MuJoCo loop rather than a learned controller model:

```bash
./scripts/python.sh -m manifold_motion.stage1.sonic_rl run \
  --catalog data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz \
  --catalog-split 0 \
  --imitation-checkpoint reports/manifold_motion/stage1_manifold_imitation_v1/manifold_imitation.pt \
  --learning-rate 5e-5 --w-containment 12 --w-outside 30 \
  --w-track 1.5 --w-imitation 2.0 \
  --iterations 30 --manifolds 16 --steps 6 \
  --out reports/manifold_motion/stage1_sonic_catalog_v1
```

This is the main warm-start path. To measure the no-imitation ablation, omit
`--imitation-checkpoint`; keep every other argument, seed and catalogue split unchanged. For a
bounded wiring smoke, use `--iterations 1 --manifolds 2 --steps 1`. The smoke is not an
improvement claim; the paper run must use paired held-out rows and report the frozen-SONIC and
fine-tuned policies under identical seeds.

Evaluate a saved fine-tuned policy on the actor-disjoint test split without requiring the legacy
BC checkpoint:

```bash
./scripts/python.sh -m manifold_motion.stage1.sonic_rl eval \
  --catalog data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz \
  --catalog-split 2 \
  --imitation-checkpoint reports/manifold_motion/stage1_manifold_imitation_v1/manifold_imitation.pt \
  --policy reports/manifold_motion/stage1_sonic_catalog_v1/policy_sonicrl.pt \
  --skip-bc --manifolds 727 --steps 6
```

The earlier bounded pilot (8 iterations × 8 windows × 4 steps, fresh PPO and no deterministic
mean probe) reached 0.375 best training success and 20/24 held-out rows inside the gate (mean
radius 0.887, p95 1.186). This is recorded in
[`stage1_sonic_catalog_pilot_v1.json`](results/stage1_sonic_catalog_pilot_v1.json) as wiring
evidence only. It is deliberately **not** called fine-tuning improvement: no BC checkpoint was
available in this checkout, so the policy started from random weights. The next run must warm
start from the imitation policy and compare frozen SONIC versus fine-tuned SONIC on identical
test rows. The warm-start smoke is covered by an atanh action-adapter unit test; its short pilot
remains a pilot until the paired baseline is evaluated on the full test split.

The paired gate can be reproduced directly with:

```bash
./scripts/python.sh -m manifold_motion.stage1.manifold_physical_gate \
  --catalog data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz \
  --checkpoint reports/manifold_motion/stage1_manifold_imitation_v1/manifold_imitation.pt \
  --sonic-policy reports/manifold_motion/stage1_sonic_warm_pilot_v1/policy_sonicrl.pt \
  --imitation-checkpoint reports/manifold_motion/stage1_manifold_imitation_v1/manifold_imitation.pt \
  --max-rows 24 --steps 4 \
  --out reports/manifold_motion/stage1_warm_paired_gate_pilot_v1.json
```

## Experiment A: normal manifold-conditioned behavior

For each held-out source clip, render the same initial G1 state under several corridor apertures
and route headings. The figure must show `M_e`, measured `M_self`, predicted primitive/pose,
executed SONIC pose, and the physical gate. The primary qualitative grid is:

| condition | expected response |
|---|---|
| wide/tall | nominal walk or neutral stance |
| low vertical semi-axis | crouch or a validated low transition |
| narrow lateral semi-axis | side gait with body yaw aligned to the corridor |
| heading change | turn/curve primitive |
| object-carry self-manifold | wider lateral margin and reduced arm swing |

The scene schedule and random seeds are fixed before rendering. A GIF is evidence of behavior;
the quantitative table uses every held-out row and keeps failed gates in the denominator.

## Experiment B: paired ablations

The same catalogue rows, source actors and simulator seeds are used for all variants:

| variant | removed component | purpose |
|---|---|---|
| full Stage 1 | imitation + SONIC physical gate | reference method |
| no imitation | initialize from the standing/global-mean pose and train only the physical objective | tests whether manifold labels alone are sufficient |
| no SONIC fine-tune | imitation output sent directly to frozen SONIC | separates data imitation from controller adaptation |
| no geometry input | replace `M_e` by a global mean descriptor | tests environment causality |
| family mean | one pose per family | simple non-neural baseline |

The primary metrics are joint MAE, executed tracking RMS, minimum `M_self` clearance, obstacle
contact ticks, fall rate, task success and inference latency. Report paired per-row differences
and bootstrap confidence intervals; do not compare only the best GIF.

## Boundary for the next step

The present 0.198 rad result is enough to establish the data and imitation interface. It does not
claim that a frozen SONIC can execute every one of the 21 families. The next implementation step
is a physical executor for the held-out catalogue rows, followed by the `no imitation` and `no
SONIC fine-tune` controls. Unsupported families remain explicit failures rather than being
silently mapped to walking.
