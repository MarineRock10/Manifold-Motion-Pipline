#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_RUNNER="${PYTHON_RUNNER:-./scripts/python.sh}"
REPLAY_ROOT="${REPLAY_ROOT:-reports/manifold_motion/seed_capability_supported_v1}"
MANIFEST="${MANIFEST:-data/seed_capability_v1/seed_capability_supported_v1.csv}"
SEED_METADATA="${SEED_METADATA:-/mnt/c/Users/Xiyuan Wang/Downloads/metadata/seed_metadata_v004.csv}"
SEED_ARCHIVE="${SEED_ARCHIVE:-/mnt/c/Users/Xiyuan Wang/Downloads/g1.tar.gz}"

"$PYTHON_RUNNER" -m manifold_motion.dataio.seed_windows \
  --replay-root "$REPLAY_ROOT" --manifest "$MANIFEST" \
  --out reports/manifold_motion/seed_capability_windows_v2 \
  --label-field family_id --primitive-count 30 --environment-mode flat \
  --horizon-seconds 1.2 --history-seconds 0.4 --stride-seconds 0.2

"$PYTHON_RUNNER" -m manifold_motion.stage2.latent_prior train \
  --windows reports/manifold_motion/seed_capability_windows_v2/seed_stage2_windows.npz \
  --out reports/manifold_motion/skill_prior_v3 --epochs 60 --batch-size 128 \
  --latent-dim 32 --condition-hidden 128 --hidden 256 --beta 0.005 --device cpu

"$PYTHON_RUNNER" -m manifold_motion.dataio.seed_windows \
  --replay-root "$REPLAY_ROOT" --manifest "$MANIFEST" \
  --out reports/manifold_motion/seed_capability_windows_corridor_v1 \
  --label-field family_id --primitive-count 30 --environment-mode reverse_corridor \
  --horizon-seconds 1.2 --history-seconds 0.4 --stride-seconds 0.2

"$PYTHON_RUNNER" -m manifold_motion.stage2.composer train \
  --windows reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows.npz \
  --out reports/manifold_motion/composer_v1 --epochs 80 --batch-size 128 --hidden 96 --device cpu

"$PYTHON_RUNNER" -m manifold_motion.stage2.constrained_generator smoke \
  --prior-checkpoint reports/manifold_motion/skill_prior_v3/skill_prior.pt \
  --prior-windows reports/manifold_motion/seed_capability_windows_v2/seed_stage2_windows.npz \
  --composer-checkpoint reports/manifold_motion/composer_v1/composer.pt \
  --environment-windows reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows.npz \
  --out reports/manifold_motion/constrained_generator_smoke_v1 \
  --true-family hands_back_walk --split 2 --latent-offset 3.5

"$PYTHON_RUNNER" -m manifold_motion.stage2.select \
  --sample reports/manifold_motion/constrained_generator_smoke_v1/constrained_sample.npz \
  --out reports/manifold_motion/constrained_generator_physical_gate_v1 \
  --source-hz 30 --max-corridor-radius 1.0

"$PYTHON_RUNNER" -m manifold_motion.dataio.failure_mining cluster \
  --replay-root reports/manifold_motion/seed_capability_pilot_v1 "$REPLAY_ROOT" \
  --manifest "$MANIFEST" --out reports/manifold_motion/failure_clusters_v1

if [[ -f "$SEED_METADATA" && -f "$SEED_ARCHIVE" ]]; then
  "$PYTHON_RUNNER" -m manifold_motion.dataio.failure_mining supplement \
    --metadata "$SEED_METADATA" \
    --exclude-manifest data/seed_capability_v1/seed_capability_manifest_v1.csv "$MANIFEST" \
    --out data/seed_capability_v1/seed_failure_supplement_v1.csv --per-family 3
  "$PYTHON_RUNNER" -m manifold_motion.dataio.seed_capability_catalog extract \
    --archive "$SEED_ARCHIVE" --manifest data/seed_capability_v1/seed_failure_supplement_v1.csv \
    --out data/seed_capability_v1
  "$PYTHON_RUNNER" -m manifold_motion.dataio.seed_replay \
    --manifest data/seed_capability_v1/seed_failure_supplement_v1.csv \
    --data-root data/seed_capability_v1 \
    --out reports/manifold_motion/seed_failure_supplement_v1 --jobs 2
  "$PYTHON_RUNNER" -m manifold_motion.dataio.seed_adapter_dataset \
    --replay-root reports/manifold_motion/seed_failure_supplement_v1 \
    --manifest data/seed_capability_v1/seed_failure_supplement_v1.csv \
    --out reports/manifold_motion/seed_failure_supplement_v1/adapter_dataset_v2.npz
  "$PYTHON_RUNNER" -m manifold_motion.stage2.train_sonic_adapter \
    --dataset reports/manifold_motion/seed_failure_supplement_v1/adapter_dataset_v2.npz \
    --out reports/manifold_motion/targeted_sonic_adapter_v2 \
    --epochs 25 --batch-size 128 --device cpu --rank 16
else
  echo "SEED metadata/archive not found; skipped failure-directed extraction and adapter warm-start."
fi

"$PYTHON_RUNNER" -m pytest -q
