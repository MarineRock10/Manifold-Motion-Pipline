#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export PYTHONPATH="$repo:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl
"${MANIFOLD_PYTHON:-$repo/scripts/python.sh}" -m manifold_motion.evaluation.multiseed_perception \
  --out "${PERCEPTION_SEED_OUT:-reports/cvpr/simulated_slam_multiseed_v1}" \
  --seeds 31000 31001 31002 31003 31004 31005 31006 31007
