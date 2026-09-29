# Geometry-routed Stage-2 mainline

## Decision

The environment-conditioned latent prior and learned high-level composer are no longer in the
default execution path. In the current experiments they changed labels and decoded references,
but the frozen SONIC controller turned those references into a stiff, one-sided gait that was
worse than the independently screened primitive anchors. Their code and reports remain as an
archived ablation; they are not deleted and are not cited as the main result.

## Active pipeline

```text
radar / SLAM pose
      -> rolling 3-D occupancy + incremental ESDF + D* Lite
      -> local environment corridor M_e(t)
      -> measured G1 self-manifold M_self(t)
      -> geometry and safety router with state/history hysteresis
      -> screened SEED/SONIC primitive (walk / lateral / crouch)
      -> normalized gait-phase handoff
      -> optimization projection
      -> frozen SONIC -> MuJoCo
      -> mesh, contact, balance and gait-symmetry acceptance gates
```

Static and moving-obstacle navigation use the geometry router only. Incoming projectiles add a
small relative-motion hazard classifier. That classifier selects an escape family from tracked
position and velocity; it is not a latent pose decoder and cannot synthesize a trajectory. The
selected primitive remains subject to the same `M_e`, `M_self`, projection and physical gates.

## Safety and motion-quality contracts

- Compact actions are forbidden in an unconstrained corridor unless an actual tracked hazard
  requests them.
- Low clearance may request crouch; narrow lateral clearance may request side gait; otherwise
  nominal walking remains the default.
- `M_self` is refitted from the current MuJoCo body geometry and checked against the corridor.
- Primitive boundaries transfer normalized cyclic phase instead of resetting the gait or using
  a global nearest-pose match that can swap the support leg.
- Nominal and lateral intervals fail acceptance when left/right leg velocity energy or joint
  range becomes strongly one-sided.
- Projectile returns stay in a dynamic layer and are not baked into the slowly decaying static
  occupancy volume.

## Reproduction

From WSL:

```bash
./scripts/run_trained_autonomous_suite.sh
```

The command deliberately does not train or load `latent_prior` or `stage2.composer`. It trains
only the bounded projectile hazard classifier, then runs static, dynamic-crossing, grazing and
overhead-projectile cases. Set `RENDER=1` to refresh the four GitHub GIF previews:

```bash
RENDER=1 ./scripts/run_trained_autonomous_suite.sh
```

Reports are written to `reports/manifold_motion/autonomous_v3_motion_quality/`; the checked-in
summary is `docs/demo_gallery/repaired_motion_quality.json`.

## Archived latent ablation

The old prior, checkpoints and visual atlas remain reproducible through
`LATENT_SKILL_PRIOR.md`. They are useful as a negative-result ablation: improved latent-space or
reference-space metrics do not imply improved closed-loop SONIC gait. Re-enabling them requires
an explicit checkpoint argument; no default script depends on them.
