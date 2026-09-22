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
stage2_out="reports/manifold_motion/deploy_sonic_main_route"
execution="$stage2_out/executed.npz"
summary="$stage2_out/route_summary.json"
set +e
python3 -m manifold_motion.stage2_route \
  --windows reports/manifold_motion/seed_windows_generalization_v2/seed_stage2_windows.npz \
  --router reports/manifold_motion/primitive_router_geometry_v1/router.pt \
  --model 0=reports/manifold_motion/stage2_mean_primitive0_v1/conditional_mean.pt \
  --model 2=reports/manifold_motion/stage2_mean_deploy_cpu_v2/primitive2_crouch/conditional_mean.pt \
  --model 3=reports/manifold_motion/stage2_mean_primitive3_v1/conditional_mean.pt \
  --model 4=reports/manifold_motion/stage2_mean_deploy_cpu_v2/primitive4_side/conditional_mean.pt \
  --model 5=reports/manifold_motion/stage2_mean_deploy_cpu_v2/primitive5_walk/conditional_mean.pt \
  --model 6=reports/manifold_motion/stage2_mean_generalization_v2/primitive6/conditional_mean.pt \
  --condition-npz "$out/condition.npz" \
  --scene data/g1_flat/scene_long_avoidance.xml \
  --split 2 --index 0 --out "$stage2_out" --device cpu
route_status=$?
set -e
if [[ "$route_status" != 0 && "$route_status" != 1 && "$route_status" != 2 ]]; then
  echo "main Stage-2 route failed unexpectedly with exit code $route_status" >&2
  exit "$route_status"
fi
python3 -m manifold_motion.render_deploy_sonic_gif \
  --sample "$stage2_out/sample.npz" \
  --condition "$out/condition.npz" \
  --scene data/g1_flat/scene_long_avoidance.xml \
  --executed "$execution" \
  --summary "$summary" \
  --reuse-executed \
  --out "$out/deploy_sonic_mujoco_comprehensive.gif" \
  --width 520 --height 390 --fps 20 --distance 2.7
