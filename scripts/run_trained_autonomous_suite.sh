#!/usr/bin/env bash
set -euo pipefail

# Bounded CPU reproduction of the three accepted autonomous experiments.  Reports/checkpoints
# are intentionally kept under reports/ (git-ignored); the checked-in GIFs are generated from
# those reports with the renderer documented in docs/demo_gallery/README.md.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="./scripts/python.sh"
WINDOWS="reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows.npz"
COMPOSER="reports/manifold_motion/composer_v2_counterfactual"
REACTIVE="reports/manifold_motion/reactive_hazard_policy_v4"

"$PY" -m manifold_motion.stage2.composer train \
  --windows "$WINDOWS" --out "$COMPOSER" --epochs 40 --batch-size 128 --hidden 96 \
  --learning-rate 0.0003 --counterfactual-weight 0.50 --environment-only-weight 0.25 \
  --seed 20260927 --device cpu
"$PY" -m manifold_motion.stage2.reactive_policy train \
  --out "$REACTIVE" --samples 40000 --epochs 45 --batch-size 512 --hidden 128 --seed 20260927

mkdir -p reports/manifold_motion/autonomous_v2/dynamic_fixture
"$PY" - <<'PY'
from pathlib import Path
from manifold_motion.evaluation.cvpr_physical_smoke import _xml_for_generated_fixture
root = Path('reports/manifold_motion/autonomous_v2')
folders = {'crossing': 'dynamic_fixture', 'projectile_grazing': 'projectile_grazing_fixture',
           'projectile_overhead': 'projectile_overhead_fixture'}
for event, folder in folders.items():
    out = root / folder / 'generated_scene.xml'
    out.parent.mkdir(parents=True, exist_ok=True)
    _xml_for_generated_fixture({'factory': 'dynamic_event_fixture', 'event': event, 'goal_x_m': 3.6}, out)
PY

common=(--composer-checkpoint "$COMPOSER/composer.pt" --composer-action-profile navigation
  --disable-composer-geometry-override --composer-switch-margin 0.12
  --composer-min-dwell-updates 3 --num-candidates 2 --online-condition-iterations 0
  --planner-body-radius-m 0.40 --planner-side-body-radius-m 0.30 --planner-clearance-m 0.10
  --segment-length-m 0.45 --side-gait-mode diagonal --max-ticks 350 --skip-render)

"$PY" -m manifold_motion.stage2.manifold_adaptive \
  --scene data/g1_flat/scene_manifold_block_right.xml --title 'TRAINED COMPOSER STATIC BLOCK' \
  --out reports/manifold_motion/autonomous_v2/static_block_right_nav --online-perception "${common[@]}"
"$PY" -m manifold_motion.stage2.manifold_adaptive \
  --scene reports/manifold_motion/autonomous_v2/dynamic_fixture/generated_scene.xml \
  --title 'TRAINED COMPOSER DYNAMIC CROSSING' --out reports/manifold_motion/autonomous_v2/dynamic_crossing_nav \
  --online-perception --dynamic-obstacle-event crossing "${common[@]}"
"$PY" -m manifold_motion.stage2.manifold_adaptive \
  --scene reports/manifold_motion/autonomous_v2/projectile_overhead_fixture/generated_scene.xml \
  --title 'TRAINED REACTIVE PROJECTILE DUCK V4' --out reports/manifold_motion/autonomous_v2/projectile_overhead_duck_v4 \
  --online-perception --online-perception-scan-ticks 10 --dynamic-obstacle-event projectile_overhead \
  --reactive-policy-checkpoint "$REACTIVE/reactive_policy.pt" --composer-action-profile reactive "${common[@]}"
"$PY" -m manifold_motion.stage2.manifold_adaptive \
  --scene reports/manifold_motion/autonomous_v2/projectile_grazing_fixture/generated_scene.xml \
  --title 'TRAINED REACTIVE PROJECTILE GRAZING V4' --out reports/manifold_motion/autonomous_v2/projectile_grazing_dodge_v4 \
  --online-perception --online-perception-scan-ticks 10 --dynamic-obstacle-event projectile_grazing \
  --reactive-policy-checkpoint "$REACTIVE/reactive_policy.pt" --composer-action-profile reactive "${common[@]}"
