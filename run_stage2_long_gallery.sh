#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo"
export PYTHONPATH="$repo:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl
python3 -m manifold_motion.long_sequence_gallery \
  --out reports/manifold_motion/stage2_long_sequence_gallery_v1
printf 'PASS: %s\n' "$repo/reports/manifold_motion/stage2_long_sequence_gallery_v1/comparison_report.json"
