#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
PYTHON="${MANIFOLD_PYTHON:-$repo/scripts/python.sh}"
export PYTHONPATH="$repo:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl

checkpoint="reports/manifold_motion/composer_v1/composer.pt"
common=(--online-perception --composer-checkpoint "$checkpoint" --num-candidates 2
        --online-condition-iterations 0 --planner-body-radius-m 0.40
        --planner-side-body-radius-m 0.30 --planner-clearance-m 0.10
        --segment-length-m 0.45 --max-ticks 1800)

"$PYTHON" -m manifold_motion.stage2.manifold_adaptive \
  --scene data/g1_flat/scene_manifold_wide_long.xml \
  --title "WIDE M_e -> WALK" --out reports/manifold_motion/online_composer_wide_long_v1 \
  "${common[@]}"
"$PYTHON" -m manifold_motion.stage2.manifold_adaptive \
  --scene data/g1_flat/scene_manifold_low_long.xml \
  --title "LOW M_e -> CROUCH WALK" --out reports/manifold_motion/online_composer_low_long_v1 \
  "${common[@]}"

"$PYTHON" -m manifold_motion.visualization.online_composer_comparison \
  --wide reports/manifold_motion/online_composer_wide_long_v1 \
  --low reports/manifold_motion/online_composer_low_long_v1 \
  --out docs/demo_gallery/media/online_composer_wide_vs_low.gif \
  --frames 90 --duration-ms 80
