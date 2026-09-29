#!/usr/bin/env bash
set -euo pipefail

# Bounded CPU reproduction of the geometry-routed mainline.  The environment/self manifolds
# select only independently screened SONIC primitives; no latent prior/composer checkpoint is
# trained or loaded by this script.  The small projectile hazard head is not a motion decoder:
# it only chooses evade-left/evade-right/crouch from relative position and velocity.
# Reports/checkpoints are intentionally kept under reports/ (git-ignored); checked-in GIFs are
# generated from those reports with the renderer documented in docs/demo_gallery/README.md.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="./scripts/python.sh"
REACTIVE="reports/manifold_motion/reactive_hazard_policy_v4"
OUT_ROOT="${OUT_ROOT:-reports/manifold_motion/autonomous_v3_motion_quality}"
export OUT_ROOT

"$PY" -m manifold_motion.stage2.reactive_policy train \
  --out "$REACTIVE" --samples 40000 --epochs 45 --batch-size 512 --hidden 128 --seed 20260927

mkdir -p "$OUT_ROOT/dynamic_fixture"
"$PY" - <<'PY'
import os
from pathlib import Path
from manifold_motion.evaluation.cvpr_physical_smoke import _xml_for_generated_fixture
root = Path(os.environ['OUT_ROOT'])
folders = {'crossing': 'dynamic_fixture', 'projectile_grazing': 'projectile_grazing_fixture',
           'projectile_overhead': 'projectile_overhead_fixture'}
for event, folder in folders.items():
    out = root / folder / 'generated_scene.xml'
    out.parent.mkdir(parents=True, exist_ok=True)
    _xml_for_generated_fixture({'factory': 'dynamic_event_fixture', 'event': event, 'goal_x_m': 3.6}, out)
PY

common=(--num-candidates 2 --online-condition-iterations 0
  --planner-body-radius-m 0.40 --planner-side-body-radius-m 0.30 --planner-clearance-m 0.10
  --segment-length-m 0.45 --side-gait-mode diagonal --max-ticks 350 --skip-render)

"$PY" -m manifold_motion.stage2.manifold_adaptive \
  --scene data/g1_flat/scene_manifold_block_right.xml --title 'GEOMETRY ROUTER STATIC BLOCK' \
  --out "$OUT_ROOT/static_block_right_nav" --online-perception "${common[@]}"
"$PY" -m manifold_motion.stage2.manifold_adaptive \
  --scene "$OUT_ROOT/dynamic_fixture/generated_scene.xml" \
  --title 'GEOMETRY ROUTER DYNAMIC CROSSING' --out "$OUT_ROOT/dynamic_crossing_nav" \
  --online-perception --dynamic-obstacle-event crossing "${common[@]}"
"$PY" -m manifold_motion.stage2.manifold_adaptive \
  --scene "$OUT_ROOT/projectile_overhead_fixture/generated_scene.xml" \
  --title 'GEOMETRY ROUTER + HAZARD HEAD: OVERHEAD PROJECTILE' --out "$OUT_ROOT/projectile_overhead_avoidance" \
  --online-perception --online-perception-scan-ticks 10 --dynamic-obstacle-event projectile_overhead \
  --reactive-policy-checkpoint "$REACTIVE/reactive_policy.pt" "${common[@]}"
"$PY" -m manifold_motion.stage2.manifold_adaptive \
  --scene "$OUT_ROOT/projectile_grazing_fixture/generated_scene.xml" \
  --title 'GEOMETRY ROUTER + HAZARD HEAD: GRAZING PROJECTILE' --out "$OUT_ROOT/projectile_grazing_dodge" \
  --online-perception --online-perception-scan-ticks 10 --dynamic-obstacle-event projectile_grazing \
  --reactive-policy-checkpoint "$REACTIVE/reactive_policy.pt" "${common[@]}"

if [[ "${RENDER:-0}" == "1" ]]; then
  declare -A gifs=(
    [static_block_right_nav]=repaired_autonomous_static.gif
    [dynamic_crossing_nav]=repaired_autonomous_dynamic.gif
    [projectile_grazing_dodge]=repaired_autonomous_projectile_grazing.gif
    [projectile_overhead_avoidance]=repaired_autonomous_projectile_overhead.gif
  )
  for run_name in "${!gifs[@]}"; do
    "$PY" -m manifold_motion.visualization.render_stage2_report \
      --run-dir "$OUT_ROOT/$run_name" \
      --out "docs/demo_gallery/media/${gifs[$run_name]}" \
      --fps 12 --scale 0.70
  done
fi
