#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="reports/manifold_motion/deploy_perception_demo"
cd "$repo"
export PYTHONPATH="$repo:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL="egl"
python3 -m manifold_motion.deploy_perception \
  --scene data/g1_flat/scene_long_avoidance.xml \
  --goal 3.6 0.0 \
  --out "$out"
stage2_out="reports/manifold_motion/deploy_sonic_stage2_sample"
execution="reports/manifold_motion/deploy_sonic_stage2_validation/executed.npz"
summary="reports/manifold_motion/deploy_sonic_stage2_validation/summary.json"
python3 -m manifold_motion.stage2_flow sample \
  --windows reports/manifold_motion/seed_windows_walk80_v1/seed_stage2_windows.npz \
  --condition-npz "$out/condition.npz" \
  --sampler conditional_mean \
  --mean-model reports/manifold_motion/stage2_mean_walk80_v1/conditional_mean.pt \
  --split 2 --index 0 --out "$stage2_out" --device cpu
python3 -m manifold_motion.render_deploy_sonic_gif \
  --sample "$stage2_out/sample.npz" \
  --condition "$out/condition.npz" \
  --scene data/g1_flat/scene_long_avoidance.xml \
  --executed "$execution" \
  --summary "$summary" \
  --out "$out/deploy_sonic_mujoco_comprehensive.gif" \
  --width 520 --height 390 --fps 20 --distance 2.7
