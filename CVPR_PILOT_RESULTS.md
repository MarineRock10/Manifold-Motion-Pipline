# CVPR pilot results (current SONIC pipeline)

This page records the latest deterministic simulation pilot after the alternative-controller
branch was removed from the active experiment. It is an evidence log, not the final multi-seed paper table: the
frozen protocol still requires paired seeds, complete failure rows and the audit in
[`CVPR_EXPERIMENTS.md`](CVPR_EXPERIMENTS.md).

The pre-registered config now contains 26 scenario variants, including the four extended tasks
shown below. With the current paired seed policy this expands the plan to 10,752 rows (7,168
primary and 3,584 diagnostic); this is a planned workload, not a claim that every row has already
been executed.

## Re-run completed on 2026-09-24

Command:

```bash
source /path/to/manifold-motion-venv/bin/activate
export PYTHONPATH=.:/home/xiyuan/Manifold-Motion-Pipline
bash ./run_stage2_manifold_side_on.sh
python -m manifold_motion.incremental_dynamic_benchmark \
  --out reports/manifold_motion/cvpr_pilot_dynamic
```

The side-on suite re-ran the same code path for all four physical fixtures. Each accepted row
uses measured `M_e`, Flow candidates, the SONIC/MuJoCo gate and one continuous state; no segment
index is used as the primitive label.

| fixture | selected behavior | keyframes | obstacle contacts | maximum roll |
|---|---|---:|---:|---:|
| wide | nominal walk | 6/6 | 0 | 9.71° |
| low | walk → crouch | 6/6 | 0 | 15.34° |
| narrow 1.05 m | walk → side-on gait | 6/6 | 0 | 9.71° |
| center block | walk / side / turn / walk | 10/10 | 0 | 11.61° |

The center-block run has route-deviation P95 `0.149 m` and mean joint tracking error `0.116 rad`.
The paired low-versus-wide run changes the measured vertical aperture from `1.338 m` to `0.990 m`
and drops the mean pelvis height by `0.217 m`; the narrow run changes the lateral free semi-axis
from `0.700 m` to `0.345 m` and activates side gait.

## Dynamic replanning pilot

The obstacle sequence is absent → crossing → moved through the route → absent. The latest run
produced 4 route-change events, zero planner failures, and the following timing comparison:

| planner | median update time | reference |
|---|---:|---|
| incremental ESDF + D* Lite | 23.16 ms (post-init) | 2.5 Hz radar budget |
| full voxel A* | 2009.36 ms | same grid and map updates |

This is an `88.11×` median speedup on the current RTX 5060/WSL setup. The result is a planner
pilot; it is not yet the 32-seed paired table.

## Additional accepted primitives

The gallery also includes independently gated jump/landing, transition→crouch without a state
reset, Flow candidate screening, long side gait and long low-clearance sequences. The jump pilot
records `0.253 m` lift, verified landing, no non-foot floor contacts and no obstacle contacts.

## Extended long-sequence pilot

The same pipeline was also run on four new fixtures that are not aliases of the original
wide/low/narrow/center set:

| task | keyframes | routed behavior | turn intervals | obstacle contacts |
|---|---:|---|---:|---:|
| alternating chicane | 17/17 | nominal gait with alternating curvature turns | 9 | 0 |
| low-side-turn compound | 18/18 | low transition, crouch, lateral gait, turn, nominal recovery | 3 | 0 |
| repeated gate cycle | 15/15 | crouch/recovery cycles plus lateral gate | 0 | 0 |
| long slalom | 18/18 | nominal gait plus 3 bilateral side intervals and latched turns | 9 | 0 |

Each run keeps one MuJoCo state and passes the exact G1-surface/self-manifold clearance audit.
These are deterministic qualitative pilot runs; they expand failure-mode coverage but do not
replace the predeclared paired multi-seed benchmark.

## Primary-matrix structural smoke

The 4 primary methods × 26 scenarios × first nominal seed matrix now has an explicit 104-row
adapter preflight. It verifies that every scenario resolves to a tracked fixture, a parameterized
narrow/low generator, or a named dynamic-map event, and that B2/Ours-2/Ours-3/Ours-4 each has an
explicit implementation profile. The preflight passes 104/104 rows.

This is deliberately stored as `structural_preflight_only`: it runs no Flow generation and no
MuJoCo physics, so it is not included in success/collision statistics. The next quantitative gate
is the one-seed physical sweep, followed by the paired eight-seed pilot.
The checked-in summary is [`docs/cvpr_primary_smoke_report.json`](docs/cvpr_primary_smoke_report.json).

```bash
PYTHONPATH=. python3 -m manifold_motion.cvpr_smoke \
  --out reports/cvpr/primary_smoke.jsonl
```

## What remains for the paper table

The next run is the frozen B2/Ours-2/Ours-3/Ours-4 paired sweep. Every scenario/seed must emit
one schema-complete JSON row, including failures; only after the audit passes will bootstrap
confidence intervals and paired effects be reported.
