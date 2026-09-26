#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
MANIFOLD_PYTHON="${MANIFOLD_PYTHON:-$repo/scripts/python.sh}"
export PYTHONPATH="$repo:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl
"${MANIFOLD_PYTHON}" -m manifold_motion.visualization.long_sequence_gallery \
  --out reports/manifold_motion/stage2_long_sequence_gallery_v1
printf 'PASS: %s\n' "$repo/reports/manifold_motion/stage2_long_sequence_gallery_v1/comparison_report.json"
