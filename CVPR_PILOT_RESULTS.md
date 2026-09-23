# CVPR pilot results (current SONIC pipeline)

This page records the latest deterministic simulation pilot after the alternative-controller
branch was removed from the active experiment. It is an evidence log, not the final multi-seed paper table: the
frozen protocol still requires paired seeds, complete failure rows and the audit in
[`CVPR_EXPERIMENTS.md`](CVPR_EXPERIMENTS.md).

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

## What remains for the paper table

The next run is the frozen B2/Ours-2/Ours-3/Ours-4 paired sweep. Every scenario/seed must emit
one schema-complete JSON row, including failures; only after the audit passes will bootstrap
confidence intervals and paired effects be reported.
