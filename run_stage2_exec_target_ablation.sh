#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")" && pwd)"
cd "$repo"
export PYTHONPATH="$repo:/home/xiyuan/.local/share/sonic-manifold-g1"

windows="reports/manifold_motion/seed_windows_generalization_v2/seed_stage2_windows.npz"
root_out="reports/manifold_motion/stage2_mean_exec_generalization_v1"
common=(--windows "$windows" --model-target-field target_exec --epochs 10
        --batch-size 256 --device cpu)

python3 -m manifold_motion.stage2_flow train-mean \
  --out "$root_out/primitive4" --primitive-id 4 "${common[@]}"
python3 -m manifold_motion.stage2_flow train-mean \
  --out "$root_out/primitive5" --primitive-id 5 "${common[@]}"

cat <<EOF
Execution-target conditional means written below:
  $root_out/primitive4/conditional_mean.pt
  $root_out/primitive5/conditional_mean.pt

Pass --mean-model-root $root_out to stage2_manifold_adaptive for the
SONIC-achieved-target ablation. The normalizer is stored in each checkpoint.
EOF
