# Trained autonomous obstacle-response experiments

These three previews use the same frozen SONIC/MuJoCo executor and the trained v2 environment
composer. The route is rebuilt from each live radar update; the selected family is not indexed by
segment number. A current-state/history shadow rollout, optimization projection and measured
`M_self` surface gate must all pass before a new reference is committed.

| experiment | input change | learned decision | result |
|---|---|---|---|
| static block | fixed 3-D obstacle map | `dodge_lateral`/`walk_curve` family from `M_e + M_self + state/history` | 9/9 keyframes, 0 contacts, accepted |
| dynamic crossing | timestamped obstacle crossing the route | online family re-route with D* Lite and hysteresis | 8/8 keyframes, 0 contacts, accepted |
| projectile overhead | radar-tracked torso/head-height box with finite-difference velocity | reactive hazard head chooses `crouch`, then releases after the object passes | 8/8 keyframes, 0 contacts, accepted |

## Trained models

The composer was trained with a counterfactual environment objective and an environment-only
auxiliary objective:

```bash
./scripts/python.sh -m manifold_motion.stage2.composer train \
  --windows reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows.npz \
  --out reports/manifold_motion/composer_v2_counterfactual \
  --epochs 40 --batch-size 128 --hidden 96 --learning-rate 0.0003 \
  --counterfactual-weight 0.50 --environment-only-weight 0.25 \
  --seed 20260927 --device cpu
```

Its held-out direct classification is Top-1 0.6437 / Top-3 0.7950. Masking geometry reduces
Top-1 to 0.2517 and permuting geometry reduces it to 0.1912, so the action selection is not a
fixed segment/path lookup.

The projectile head is trained from randomized relative pose/velocity observations. Its teacher
rolls out keep, sidestep, crouch, retreat and hop families against the swept ellipsoidal
self-manifold, then scores clearance, feasibility, and effort. The v3 CPU run used 80,000
augmented samples and achieved 90.14% overall accuracy with recall 0.88/0.79/0.92/0.88 for
sidestep/crouch/retreat/hop (keep recall 0.91). After correcting the signed lateral feasibility
term, the final v4 CPU run used 80,000 augmented samples and achieved 90.55% overall accuracy
with recall 0.91/0.81/0.92/0.88 for sidestep/crouch/retreat/hop (keep recall 0.91).
Unsupported low-level families are projected to
the currently available SONIC side/crouch/nominal references and remain subject to the physical
gate; they are not silently claimed as executed jumps.

```bash
./scripts/python.sh -m manifold_motion.stage2.reactive_policy train \
  --out reports/manifold_motion/reactive_hazard_policy_v4 \
  --samples 40000 --epochs 45 --batch-size 512 --hidden 128 --seed 20260927
```

## Acceptance metrics

The exact reports and trajectories are generated under `reports/manifold_motion/autonomous_v2/`
(ignored from git). The compact, reproducible summary is
[`trained_autonomous_acceptance.json`](../demo_gallery/trained_autonomous_acceptance.json).

The centreline high-speed projectile configuration is intentionally a stress test: if the frozen
controller cannot create enough lateral clearance it stops with zero contacts instead of being
counted as a success. The nominal visual uses the feasible overhead projectile schedule.
