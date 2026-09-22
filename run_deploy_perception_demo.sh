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
stage2_out="reports/manifold_motion/deploy_perception_continuous"
python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_long_avoidance.xml \
  --out "$stage2_out" \
  --perception-condition "$out/condition.npz" \
  --windows reports/manifold_motion/seed_windows_corridor_stage2_v2/seed_stage2_windows.npz \
  --side-gait-mode diagonal --num-candidates 2 \
  --online-condition-iterations 1 --receding-horizon-ticks 0 \
  --max-ticks 2600 --planner-body-radius-m 0.40 --planner-clearance-m 0.10 \
  --device cpu --fps 20
cp "$stage2_out/manifold_adaptive.gif" "$out/deploy_sonic_mujoco_comprehensive.gif"
