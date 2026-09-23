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
python3 -m manifold_motion.render_synced_deploy_gif \
  --mujoco-gif "$stage2_out/manifold_adaptive.gif" \
  --executed "$stage2_out/executed.npz" \
  --condition "$out/condition.npz" \
  --slam-grid "$out/slam_grid.npz" \
  --radar-returns "$out/radar_returns.npz" \
  --segment-conditions "$stage2_out/segment_conditions.npz" \
  --report "$stage2_out/report.json" \
  --out "$out/deploy_mujoco_slam_synced.gif" \
  --fps 20 --warmup-hold 5
cp "$out/deploy_mujoco_slam_synced.gif" "$out/deploy_sonic_mujoco_comprehensive.gif"
