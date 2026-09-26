# SEED capability gate

`data/seed_capability_v1/` is a deliberately bounded slice of the downloaded BONES-SEED archive.
It contains a 30-family semantic catalogue, a 270-row actor-disjoint manifest and a 30-row
one-per-family physical pilot. The metadata label is only a retrieval hint; it is not a learned
capability claim.

Build the catalogue and extract only selected G1 CSV members:

```bash
./scripts/python.sh -m manifold_motion.dataio.seed_capability_catalog build \
  --metadata "/mnt/c/Users/<user>/Downloads/metadata/seed_metadata_v004.csv" \
  --out data/seed_capability_v1 --train 6 --validation 2 --test 2
./scripts/python.sh -m manifold_motion.dataio.seed_capability_catalog extract \
  --archive "/mnt/c/Users/<user>/Downloads/g1.tar.gz" \
  --manifest data/seed_capability_v1/seed_capability_pilot_v1.csv \
  --out data/seed_capability_v1
```

The first physical screen is intentionally one clip per family. Replays are saved under
`reports/manifold_motion/seed_capability_pilot_v1/`; only accepted MuJoCo/SONIC records may enter
the latent-prior training set. `capabilities.json` separates `partial`, `supported` and
`unsupported` families. A family becomes `supported` only after at least two accepted clips from
different actors. Use `confirm` to create a three-clip-per-family confirmation manifest for the
pilot-accepted families, then replay it with the same gate.

The current pilot is expected to contain ordinary locomotion, lateral gait, crouch transitions,
steps, jumps and selected interaction fragments. Crawl, vault, roll-recovery and some dynamic
low-contact actions are intentionally likely to remain unsupported under frozen SONIC; they are
kept as negative evidence and future SONIC-finetuning targets, never silently mixed into positives.

## Current capability result

The September 27 gate tested 30 semantic families. Twenty-one are supported by the current frozen
SONIC controller. The bounded full manifest then accepted 187/194 selected clips (96.4%). Supported
families cover forward/jog/lateral/curved locomotion, turns, crouch motion and transitions, lateral
dodges/lunges/hops, high and box jumps, box steps, kneeling/all-fours, and several object/ladder
interaction fragments. Unsupported families are retained as negative evidence: bend-duck walk,
forward/broad jump, crawl/spider/inchworm, get-up, vault and roll-recovery.

The actor-disjoint split is computed from a stable hash of `actor_uid`; windows from the same actor
never cross train/validation/test.

## State-conditioned latent prior

The accepted clips produce 3,988 windows at 30 Hz (2,610 train, 651 validation, 727 test). The
state-conditioned VAE in `manifold_motion.stage2.latent_prior` learns a separate prior
`p(z | state, history, family, manifold, command)` rather than using a fixed normal latent. The
best current checkpoint obtains test normalized MSE 1.170 versus the zero-normalized baseline
1.629, a 28.2% relative improvement.

```bash
./scripts/python.sh -m manifold_motion.stage2.latent_prior train \
  --windows reports/manifold_motion/seed_capability_windows_v2/seed_stage2_windows.npz \
  --out reports/manifold_motion/skill_prior_v3 --epochs 60 --batch-size 128 \
  --latent-dim 32 --condition-hidden 128 --hidden 256 --beta 0.005 --device cpu
```

## Environment/self-manifold composer

For the high-level router, each accepted execution is reverse-constructed into a time-aligned
environment corridor `M_e(t)=[center xyz, semi xyz, yaw]`, a `(10,10,8)` local SDF, and an executed
self-manifold sequence `[semi_x, semi_y, semi_z]`. The data representation matches online SLAM,
but this pre-training set is explicitly marked `reverse_synthesized_from_R_exec`; it is not claimed
as sensor data.

The composer reads state, 0.4 s history, corridor, SDF, self-manifold and navigation command. It
does **not** read the primitive label as an input. On the held-out actor split its current result is
57.8% Top-1, 79.0% Top-3 and 58.5% macro-F1 over the 21 executable families.

```bash
./scripts/python.sh -m manifold_motion.dataio.seed_windows \
  --replay-root reports/manifold_motion/seed_capability_supported_v1 \
  --manifest data/seed_capability_v1/seed_capability_supported_v1.csv \
  --out reports/manifold_motion/seed_capability_windows_corridor_v1 \
  --label-field family_id --primitive-count 30 --environment-mode reverse_corridor \
  --horizon-seconds 1.2 --history-seconds 0.4 --stride-seconds 0.2

./scripts/python.sh -m manifold_motion.stage2.composer train \
  --windows reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows.npz \
  --out reports/manifold_motion/composer_v1 --epochs 80 --batch-size 128 --hidden 96 --device cpu
```

At runtime, the selected family conditions the latent prior. `constrained_generator.py` then applies
a family-local, zero-initialized body residual, the state-conditioned latent barrier and the
optimization-embedded projection. Frozen SONIC plus MuJoCo remains the final physical gate.

The held-out `hands_back_walk` integration smoke selected the correct family, clipped a proposed
latent Mahalanobis radius from 3.5 to 3.0, reduced the projection objective from 0.0352 to 0.0270,
and passed the final frozen SONIC/MuJoCo selector (`accepted=true`). To reproduce that exact handoff
after training:

```bash
./scripts/python.sh -m manifold_motion.stage2.constrained_generator smoke \
  --prior-checkpoint reports/manifold_motion/skill_prior_v3/skill_prior.pt \
  --prior-windows reports/manifold_motion/seed_capability_windows_v2/seed_stage2_windows.npz \
  --composer-checkpoint reports/manifold_motion/composer_v1/composer.pt \
  --environment-windows reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows.npz \
  --out reports/manifold_motion/constrained_generator_smoke_v1 \
  --true-family hands_back_walk --split 2 --latent-offset 3.5

./scripts/python.sh -m manifold_motion.stage2.select \
  --sample reports/manifold_motion/constrained_generator_smoke_v1/constrained_sample.npz \
  --out reports/manifold_motion/constrained_generator_physical_gate_v1 \
  --source-hz 30 --max-corridor-radius 1.0
```

## Failure-directed supplementation

The first mining pass contains 16 unique failed clips: seven tracking/instability, six jump-phase,
and three non-foot-contact failures. The generated acquisition plan prioritizes short phase-aligned
jump/landing fragments, slow neutral-handoff transitions, and low-clearance clips with explicit
contact/body-envelope labels. Only after those targeted clips pass replay should the local residual
or SONIC adapter be fine-tuned.

The first targeted supplement selected 30 unseen clips (ten affected families), of which 17 passed
the physical gate. The accepted-only action archive contains 4,154 rows with frozen SONIC actions
paired to accepted SEED reference actions. A CPU, rank-16 zero-init adapter warm-start reduced
action MSE from 5.7329 to 5.6831 on the held-out supplement actors; the residual was explicitly
masked to the family-relevant body joints. This is a warm-start diagnostic, not a controller
replacement: a full acceptance matrix and privileged PPO/distillation check are still required.

```bash
./scripts/python.sh -m manifold_motion.dataio.failure_mining cluster \
  --replay-root reports/manifold_motion/seed_capability_pilot_v1 \
                reports/manifold_motion/seed_capability_supported_v1 \
  --manifest data/seed_capability_v1/seed_capability_supported_v1.csv \
  --out reports/manifold_motion/failure_clusters_v1

./scripts/python.sh -m manifold_motion.dataio.failure_mining supplement \
  --metadata /mnt/c/Users/<user>/Downloads/metadata/seed_metadata_v004.csv \
  --exclude-manifest data/seed_capability_v1/seed_capability_manifest_v1.csv \
                    data/seed_capability_v1/seed_capability_supported_v1.csv \
  --out data/seed_capability_v1/seed_failure_supplement_v1.csv --per-family 3

./scripts/python.sh -m manifold_motion.dataio.seed_capability_catalog extract \
  --archive /mnt/c/Users/<user>/Downloads/g1.tar.gz \
  --manifest data/seed_capability_v1/seed_failure_supplement_v1.csv \
  --out data/seed_capability_v1
./scripts/python.sh -m manifold_motion.dataio.seed_replay \
  --manifest data/seed_capability_v1/seed_failure_supplement_v1.csv \
  --data-root data/seed_capability_v1 \
  --out reports/manifold_motion/seed_failure_supplement_v1 --jobs 2
./scripts/python.sh -m manifold_motion.dataio.seed_adapter_dataset \
  --replay-root reports/manifold_motion/seed_failure_supplement_v1 \
  --manifest data/seed_capability_v1/seed_failure_supplement_v1.csv \
  --out reports/manifold_motion/seed_failure_supplement_v1/adapter_dataset_v2.npz
./scripts/python.sh -m manifold_motion.stage2.train_sonic_adapter \
  --dataset reports/manifold_motion/seed_failure_supplement_v1/adapter_dataset_v2.npz \
  --out reports/manifold_motion/targeted_sonic_adapter_v2 --epochs 25 \
  --batch-size 128 --device cpu --rank 16
```
