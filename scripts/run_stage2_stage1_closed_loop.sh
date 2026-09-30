#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
MANIFOLD_PYTHON="${MANIFOLD_PYTHON:-$repo/scripts/python.sh}"
export PYTHONPATH="${repo}:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl

root="reports/manifold_motion/stage2_stage1_closed_loop_v1"
router="models/stage1/primitive_router_geometry_v2.pt"
common=(--side-semi-y-m 0.42 --num-candidates 3 --stage1-router "$router")

"${MANIFOLD_PYTHON}" -m manifold_motion.stage2.manifold_adaptive \
  --scene data/g1_flat/scene_manifold_wide_long.xml \
  --title "WIDE CORRIDOR: LEARNED STAGE1 NOMINAL" \
  --out "$root/wide" "${common[@]}"

"${MANIFOLD_PYTHON}" -m manifold_motion.stage2.manifold_adaptive \
  --scene data/g1_flat/scene_manifold_low_long.xml \
  --title "LOW CEILING: LEARNED STAGE1 CROUCH" \
  --out "$root/low" "${common[@]}"

"${MANIFOLD_PYTHON}" -m manifold_motion.stage2.manifold_adaptive \
  --scene data/g1_flat/scene_manifold_narrow_long.xml \
  --title "NARROW PASSAGE: LEARNED STAGE1 SIDE GAIT" \
  --out "$root/narrow" "${common[@]}"

"${MANIFOLD_PYTHON}" -m manifold_motion.stage2.manifold_adaptive \
  --scene data/g1_flat/scene_long_avoidance.xml \
  --title "CENTER BLOCK: LEARNED STAGE1 ROUTE" \
  --out "$root/center" --max-ticks 2200 --side-gait-mode diagonal "${common[@]}"

"${MANIFOLD_PYTHON}" -m manifold_motion.stage2.manifold_counterfactual --root "$root"
printf 'PASS: %s\n' "$repo/$root/comparison_report.json"
