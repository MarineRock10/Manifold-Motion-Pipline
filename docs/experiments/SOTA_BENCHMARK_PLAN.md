# Multi-seed simulation and external comparison protocol

This document freezes the next experimental stage. It separates three claims that are often
mixed together:

1. **Perception-chain validity:** simulated radar + pose/SLAM adapter → 3-D probability map →
   incremental route → `M_e(t)`.
2. **G1 system performance:** all methods run with the same G1 model, SONIC weights, physics
   step, initial state, scene, seed and hard safety gates.
3. **External-paper comparison:** an external method is only called a baseline after its
   algorithm has been reimplemented or its official code has been adapted to the same G1
   simulator. A reported success rate on Digit, Unitree H1, or a different simulator is not
   copied into our table as if it were directly comparable.

## Relevant recent methods

The comparison set is chosen by interface, not by paper title alone:

| Method | Public primary source | What is comparable here | Required adapter |
|---|---|---|---|
| Dynamic subgoal pursuit | [arXiv:2506.02206](https://arxiv.org/abs/2506.02206) | high-level learned subgoals + low-level model-based locomotion | replace Digit/MPC executor with the frozen G1/SONIC executor |
| P3O safe locomotion | [arXiv:2508.07611](https://arxiv.org/abs/2508.07611) | point-cloud-conditioned safe locomotion and safety cost | replay the same radar/voxel observations and use the G1 action interface |
| GuideWalk | [arXiv:2606.10449](https://arxiv.org/abs/2606.10449) | explicit navigation guidance plus terrain-adaptive locomotion | export its guidance policy or reproduce the guidance-only ablation on G1 |
| Polygonal semantic mapping | [arXiv:2411.01919](https://arxiv.org/abs/2411.01919) | online map construction and planner latency | compare map/route metrics separately from whole-body execution |
| Whole-body imitation + centroidal MPC | [arXiv:2508.00362](https://arxiv.org/abs/2508.00362) | motion imitation and dynamic feasibility | use as the motion-reference baseline, with identical obstacle gates |
| Optimization-embedded trajectory generation | [user-provided paper](https://arxiv.org/pdf/2609.09158) | differentiable/embedded feasibility projection | implement the same optimizer budget around the G1 reference trajectory |

The first three are the primary algorithm-level comparison set. The mapping and imitation
papers are component baselines. The user-provided 2609.09158 method is an optimization baseline,
not a navigation dataset, so it is evaluated on the same generated obstacle layouts.

## Datasets and scene sources

No public dataset currently provides the exact tuple “G1 joint reference + synchronized 3-D
radar + SLAM pose + dynamic obstacle + successful whole-body action”. The benchmark therefore
uses a layered protocol:

| Source | Role | How it enters this project |
|---|---|---|
| [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) | robot-compatible motion families | capability screening, Stage 1 paired windows, transition supplements |
| [AMASS](https://amass.is.tue.mpg.de/) | broad human motion prior | optional pretraining/coverage audit; never counted as G1 execution success |
| [BABEL](https://babel.is.tue.mpg.de/) | action labels over AMASS | motion-family stratification and held-out semantic labels |
| [Habitat-Matterport 3D](https://aihabitat.org/datasets/hm3d/) | real indoor geometry | convert selected meshes/occupancy to collision-safe MuJoCo fixtures and generate radar scans |
| [Habitat 3.0](https://aihabitat.org/habitat3/)** | interactive/dynamic indoor scenes | dynamic-obstacle and human-motion layout source; evaluate in a normalized G1 replay |
| [ScanNet](http://www.scan-net.org/) / [Replica](https://github.com/facebookresearch/Replica-Dataset) | point-cloud/map generalization | held-out geometry and SLAM-map stress tests |

`**` Habitat 3.0 is used as a scene/interaction source, not as an unmodified controller
benchmark. The final G1 result always reports the conversion fidelity and collision mesh.

## Frozen multi-seed matrix

The released core pilot has 3 seeds × 5 representative scenario families × 4 methods = 60
paired MuJoCo rows. The full protocol remains 32 primary seeds over 26 scenario variants and
the declared radar/dropout/time-skew/extrinsic conditions. The perception-only matrix already
has 8/8 accepted seeds.

Primary methods:

- `B2`: offline geometry route + offline primitive;
- `Ours-2`: incremental ESDF/D* Lite + online `M_e` + projection;
- `Ours-3`: Ours-2 + shadow gate and semantic hysteresis;
- `Ours-4`: Ours-3 + state/history-conditioned Stage 2 candidate.

Every row stores the complete seed, scene, radar map, corridor, route, executed state, contact
summary, failure taxonomy and planning timings. There is no post-hoc seed removal.

## Proposed metric: MCSA

Traditional navigation metrics (success, collision, path length) miss the central requirement
of this project: the robot must change its action because the measured manifold changed. We add
the **Manifold-Conditioned Safe Adaptation (MCSA)** score, reported beside—not instead of—the
standard metrics:

```text
MCSA = 0.45 safe_success
     + 0.25 condition_action
     + 0.20 clearance_score
     + 0.10 realtime_score
```

`condition_action` is 1 only when the scenario-required primitive family appears (crouch for a
low interval, side gait for a narrow interval, no contraction in open space, or an online
semantic response in a dynamic scene). `clearance_score` maps the exact self/obstacle clearance
from 0.02–0.20 m to [0,1]. `realtime_score = exp(-P95_planning_ms / 200)`. We report the four
components and bootstrap confidence intervals, so MCSA cannot hide a collision or a slow
planner.

An internal MCSA win over B2 is not automatically an external SOTA claim. An external SOTA claim
requires the adapted method to use the same scene-seed pairs, training-data budget and G1/SONIC
interface, plus a statistically significant paired improvement on predeclared primary metrics.

## Commands

```bash
# 8-seed radar/SLAM/P1 acceptance
./scripts/run_simulated_slam_seeds.sh

# 3-seed × 5-scenario × 4-method physical core pilot (60 rows)
./scripts/run_cvpr_multiseed_core.sh

# Aggregate standard metrics and MCSA
./scripts/python.sh -m manifold_motion.evaluation.multiseed_summary \
  --root reports/cvpr/physical_multiseed_core_v1 \
  --out reports/cvpr/physical_multiseed_core_v1/summary.json
```

The current one-seed 26-scenario result is retained at
`docs/experiments/results/cvpr_primary_physical_seed31000_summary.json`. It is a diagnostic
pilot and already shows why a SOTA claim must wait: B2 currently has 17/26 successes while
Ours-4 has 14/26 under that frozen protocol. The core matrix and MCSA report are the next gate;
if Ours does not beat B2, the result is recorded as a negative ablation and the method is not
called SOTA.

The external comparison boundary is executable through
`manifold_motion.evaluation.external_baseline_gate`. Its current report is
`docs/experiments/results/external_baseline_gate.json`; all listed adapters are explicitly
`not_ready` until they pass the identical G1/MuJoCo/SONIC protocol.
