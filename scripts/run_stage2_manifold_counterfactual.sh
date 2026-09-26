#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
MANIFOLD_PYTHON="${MANIFOLD_PYTHON:-$repo/scripts/python.sh}"
export PYTHONPATH="${repo}:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl

root="reports/manifold_motion/stage2_manifold_counterfactual_v1"
common=(--side-semi-y-m 0.42 --num-candidates 3)

"${MANIFOLD_PYTHON}" -m manifold_motion.stage2.manifold_adaptive \
  --scene data/g1_flat/scene_manifold_wide_long.xml \
  --title "WIDE CORRIDOR: M_e SELECTS NOMINAL WALK" \
  --out "$root/wide" "${common[@]}"

"${MANIFOLD_PYTHON}" -m manifold_motion.stage2.manifold_adaptive \
  --scene data/g1_flat/scene_manifold_low_long.xml \
  --title "LOW CEILING: M_e SELECTS CROUCH" \
  --out "$root/low" "${common[@]}"

"${MANIFOLD_PYTHON}" -m manifold_motion.stage2.manifold_adaptive \
  --scene data/g1_flat/scene_manifold_narrow_long.xml \
  --title "NARROW PASSAGE: M_e SELECTS SIDE GAIT" \
  --out "$root/narrow" "${common[@]}"

"${MANIFOLD_PYTHON}" -m manifold_motion.stage2.manifold_adaptive \
  --scene data/g1_flat/scene_long_avoidance.xml \
  --title "CENTER BLOCK: M_e(t) SELECTS ROUTE + TURN/SIDE" \
  --out "$root/center" --max-ticks 2200 "${common[@]}"

"${MANIFOLD_PYTHON}" -m manifold_motion.stage2.manifold_counterfactual --root "$root"
printf 'PASS: %s\n' "$repo/$root/comparison_report.json"
