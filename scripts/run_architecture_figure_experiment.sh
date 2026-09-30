#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export PYTHONPATH="${repo}:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

"${MANIFOLD_PYTHON:-$repo/scripts/python.sh}" -m manifold_motion.evaluation.architecture_experiment \
  --out "${ARCHITECTURE_OUT:-docs/experiments/architecture_figure_v2}" \
  --fps "${ARCHITECTURE_FPS:-10}"
