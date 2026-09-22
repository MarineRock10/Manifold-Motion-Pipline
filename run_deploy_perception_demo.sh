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
python3 -m manifold_motion.render_deploy_gif \
  --scene data/g1_flat/scene_long_avoidance.xml \
  --goal 3.6 0.0 \
  --out "$out/deploy_perception_comprehensive.gif"
