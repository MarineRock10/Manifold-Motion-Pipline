#!/usr/bin/env bash
set -euo pipefail

# Resource-bounded Stage-2 adaptation for the deploy P1 condition.
# Safe defaults for a laptop/WSL installation; override with
# STAGE2_EPOCHS=30 STAGE2_BATCH=64 STAGE2_THREADS=8.
repo="$(cd "$(dirname "$0")" && pwd)"
cd "$repo"
export PYTHONPATH="/home/xiyuan/.local/share/sonic-manifold-g1:$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${STAGE2_THREADS:-8}"
export MKL_NUM_THREADS="${STAGE2_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${STAGE2_THREADS:-8}"

windows="reports/manifold_motion/seed_windows_generalization_v2/seed_stage2_windows.npz"
root_out="reports/manifold_motion/stage2_mean_deploy_cpu_v2"
epochs="${STAGE2_EPOCHS:-40}"
batch="${STAGE2_BATCH:-64}"
root_weight="${STAGE2_ROOT_WEIGHT:-2}"

if [[ ! -f "$windows" ]]; then
  echo "missing Stage-2 windows: $windows" >&2
  exit 2
fi

common=(--windows "$windows" --epochs "$epochs" --batch-size "$batch"
        --condition-hidden 128 --hidden 256 --device cpu
        --model-target-field target_ref --root-weight "$root_weight")

echo "[stage2] CPU-safe deploy-condition training"
echo "        windows=$windows epochs=$epochs batch=$batch root_weight=$root_weight threads=$OMP_NUM_THREADS"
echo "        output=$root_out"

# Crouch for low clearance, lateral walk for narrow passages, and nominal walk
# for open space. Existing generalisation checkpoints remain for other families.
python3 -m manifold_motion.stage2_flow train-mean \
  --out "$root_out/primitive2_crouch" --primitive-id 2 "${common[@]}"
python3 -m manifold_motion.stage2_flow train-mean \
  --out "$root_out/primitive4_side" --primitive-id 4 "${common[@]}"
python3 -m manifold_motion.stage2_flow train-mean \
  --out "$root_out/primitive5_walk" --primitive-id 5 "${common[@]}"

cat <<EOF

Deploy-conditioned Stage-2 models are ready:
  $root_out/primitive2_crouch/conditional_mean.pt
  $root_out/primitive4_side/conditional_mean.pt
  $root_out/primitive5_walk/conditional_mean.pt

Run run_deploy_perception_demo.sh afterwards to pass them through the
physical SONIC/MuJoCo gate; a checkpoint is not deployable until that gate
accepts it.
EOF
