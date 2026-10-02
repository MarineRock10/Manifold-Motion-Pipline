#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export PYTHONPATH="$repo:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl
out="${CVPR_MULTI_OUT:-reports/cvpr/physical_multiseed_core_v1}"
mkdir -p "$out"
read -r -a seeds <<< "${CVPR_SEEDS:-31000 31001 31002}"
read -r -a scenarios <<< "${CVPR_SCENARIOS:-open-center low-100 narrow-085-left compound-side-low-turn dynamic-crossing-block}"
read -r -a methods <<< "${CVPR_METHODS:-B2 Ours-2 Ours-3 Ours-4}"
for seed in "${seeds[@]}"; do
  args=(--out "$out/seed_${seed}" --seed "$seed" --resume --skip-render
        --row-timeout-s 180 --max-ticks 1400)
  for scenario in "${scenarios[@]}"; do args+=(--only-scenario "$scenario"); done
  for method in "${methods[@]}"; do args+=(--only-method "$method"); done
  "${MANIFOLD_PYTHON:-$repo/scripts/python.sh}" -m manifold_motion.evaluation.cvpr_physical_smoke "${args[@]}"
done
"${MANIFOLD_PYTHON:-$repo/scripts/python.sh}" -m manifold_motion.evaluation.multiseed_summary \
  --root "$out" --out "$out/summary.json" --baseline B2
