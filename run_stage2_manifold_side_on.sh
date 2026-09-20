#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo"
export PYTHONPATH="${repo}:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl

root="reports/manifold_motion/stage2_manifold_side_on_v3"
common=(--side-semi-y-m 0.42 --num-candidates 3)
counterfactual_planner=(--planner-body-radius-m 0.40 --planner-clearance-m 0.10)

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_wide_long.xml \
  --title "WIDE: NOMINAL FORWARD WALK" \
  --out "$root/wide" "${common[@]}" "${counterfactual_planner[@]}"

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_low_long.xml \
  --title "LOW: FORWARD CROUCH ALIGNED WITH M_e" \
  --out "$root/low" "${common[@]}" "${counterfactual_planner[@]}"

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_narrow_105cm.xml \
  --title "1.05 m NARROW: STRICT SIDE-ON GAIT" \
  --out "$root/narrow" "${common[@]}" "${counterfactual_planner[@]}"

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_long_avoidance.xml \
  --title "CENTER BLOCK: SIDE-ON PASSAGE ACTIONS" \
  --out "$root/center" --max-ticks 2200 --side-gait-mode diagonal "${common[@]}"

python3 -m manifold_motion.stage2_manifold_counterfactual --root "$root"
printf 'PASS: %s\n' "$repo/$root/comparison_report.json"
