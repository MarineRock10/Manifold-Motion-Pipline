#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-./scripts/python.sh}"
CHECKPOINT="${LATENT_CHECKPOINT:-reports/manifold_motion/skill_prior_environment_v6_exec/skill_prior.pt}"
WINDOWS="${LATENT_WINDOWS:-reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows.npz}"
RUN_ROOT="${LATENT_COUNTERFACTUAL_ROOT:-reports/manifold_motion/latent_env_counterfactual_v2}"
MEDIA="${LATENT_COUNTERFACTUAL_MEDIA:-docs/demo_gallery/media/latent_environment_counterfactual.gif}"

"$PYTHON" -m scripts.evaluate_latent_counterfactual \
  --prior-checkpoint "$CHECKPOINT" --windows "$WINDOWS" --out "$RUN_ROOT" \
  --base-index 1278 --environment-index 3505 --family-id 6

MUJOCO_GL="${MUJOCO_GL:-egl}" "$PYTHON" -m manifold_motion.visualization.latent_skill_atlas \
  --case "LOW_M_e=$RUN_ROOT/base_environment.npz" \
  --case "WIDE_M_e=$RUN_ROOT/counterfactual_environment.npz" \
  --out "$MEDIA" --report "${MEDIA%.gif}.json" --columns 2 \
  --width 480 --height 320 --frames 48 --fps 12

echo "Latent environment counterfactual: $MEDIA"
