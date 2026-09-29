# Online SEED skill composer

> **Default status: geometry-router mode.** The learned 21-family SEED composer described below
> is retained as an archived ablation and is not loaded by the default autonomous suite. The
> active deployment loop uses the measured `M_e/M_self` affordance directly; an optional
> relative-motion hazard head can choose among the already screened walk/side/crouch primitives.

This module still documents the former trained composer interface for reproducibility. It does
not replace SONIC and it does not claim that all 21 families already have independent deployable
controllers. Passing `--composer-checkpoint` explicitly re-enables the learned ablation; omitting
it selects the geometry-only path.

## Runtime contract

At every accepted radar update the loop now performs:

1. simulated radar -> probabilistic 3-D sliding voxel map;
2. incremental ESDF + D* Lite -> local route, SDF and ellipsoid corridor `M_e(t)`;
3. current MuJoCo state -> 69-D state and 12-frame history in policy joint order;
4. exact G1 mesh samples -> measured self-manifold `M_self`;
5. geometry router maps measured aperture/heading to a screened primitive (or, only when an
   explicit checkpoint is supplied, `(state, history, M_e, SDF, M_self, command)` -> 21-family
   probabilities);
6. temporal confidence/dwell hysteresis -> stable family when the learned ablation is enabled;
7. geometry safety shield -> verified legacy Flow token;
8. candidate generation -> optimization-embedded projection -> current-state shadow rollout;
9. frozen SONIC execution with contact and exact self-manifold clearance gates.

The family output remains auditable even when several SEED families currently map to one legacy
SONIC token. For example, `hands_back_walk`, `carry_object` and `door_interaction` can be
distinguished by the composer but still execute through the validated nominal locomotion support.
They become independent executable actions only after family-specific references pass the same
physical gate; the online router never silently treats a classifier label as proof of feasibility.

## Current measured result

Important scope note: the accepted wide/low GIF is the active geometry-routing result, not a
valid LATENT-effect claim. The former `skill_prior_v3` was trained without corridor/SDF/M_self,
and the low-corridor crouch was already available through the deterministic safety router. The
corrected environment-conditioned prior is now trained as `skill_prior_environment_v4`, but its
first generated crouch candidate is physically stable on flat ground while violating the requested
corridor (`max implicit radius 1.63`), so it is deliberately not deployed.

Held-out actor split, replayed clip-by-clip as rolling perception updates:

| Quantity | Result |
|---|---:|
| test windows | 727 |
| continuous clips | 36 |
| raw family top-1 | 54.75% |
| raw family top-3 | 77.72% |
| mean CPU inference latency | 1.62 ms |

Three online MuJoCo/SONIC long routes pass without obstacle contact:

| Scene | Keyframes | Result |
|---|---:|---|
| wide long | 8/8 | accepted |
| right-offset block | 9/9 | accepted |
| left-offset block | 10/10 | accepted |

The long low/side compound fixture remains a recorded negative result under online perception:
both the deterministic M_e baseline and the learned composer reach the same fourth keyframe and
then trigger the exact self-manifold clearance stop. This is a P1/corridor-controller coverage
failure, not a composer regression, and is intentionally not counted as an accepted demo.

## Reproduce

```bash
./scripts/run_stage2_online_composer_long.sh
```

The trained adapter in `targeted_sonic_adapter_v2` is not loaded by this command. SONIC stays
frozen; the adapter remains only a warm-start artifact for the later failure-driven fine-tuning
stage.

## How to read the GIF

`online_composer_wide_vs_low.gif` is the intentionally simple counterfactual demo. The left and
right panels are matched by normalized route progress:

- **WIDE**: `M_e` has a tall vertical semi-axis, so `M_self` stays tall and SONIC receives
  `walk_nominal`.
- **LOW CEILING**: the vertical `M_e` semi-axis contracts below the crouch threshold; the
  measured `M_self` contracts with it, the composer settles on `crouch_walk`, and SONIC receives
  `crouch`.

The cards at the bottom show blue `M_e` height, orange measured `M_self` height, the learned
family/probability, the independent geometry decision, and the primitive actually executed. The
old single-run `online_composer_live_slam.gif` is retained as a raw sensor/planner diagnostic;
it is not intended to prove an action switch by itself.
