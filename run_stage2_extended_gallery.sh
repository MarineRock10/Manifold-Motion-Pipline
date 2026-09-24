#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo"
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=egl

root="reports/manifold_motion/stage2_extended_gallery_v1"
common=(--segment-length-m 0.45 --num-candidates 3 --side-semi-y-m 0.42
        --planner-body-radius-m 0.40 --planner-clearance-m 0.10
        --max-ticks 3200)

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_chicane.xml \
  --title "CHICANE: FOUR ALTERNATING TURNS" \
  --out "$root/chicane_turns" --goal-x 6.0 \
  --side-gait-mode diagonal "${common[@]}"

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_low_side_turn.xml \
  --title "LOW TO SIDE TO TURN: COMPOUND MANIFOLD ROUTE" \
  --out "$root/low_side_turn" --goal-x 5.6 \
  --side-gait-mode diagonal "${common[@]}"

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_gate_cycle.xml \
  --title "GATE CYCLE: CROUCH RECOVERY SIDE CROUCH" \
  --out "$root/gate_cycle" --goal-x 6.4 \
  --side-gait-mode diagonal "${common[@]}"

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_slalom.xml \
  --title "SLALOM: REPEATED HEADING CHANGES" \
  --out "$root/slalom_turns" --goal-x 5.8 \
  --side-gait-mode diagonal "${common[@]}"

python3 - <<'PY'
import json
from pathlib import Path
root = Path("reports/manifold_motion/stage2_extended_gallery_v1")
rows = []
for report in sorted(root.glob("*/report.json")):
    data = json.loads(report.read_text())
    rows.append({
        "id": report.parent.name,
        "scenario": data.get("scenario"),
        "accepted": data.get("accepted"),
        "failed_checks": data.get("failed_checks", []),
        "keyframes": data.get("execution", {}).get("keyframes_reached"),
        "obstacle_contact_ticks": data.get("execution", {}).get("obstacle_contact_ticks"),
        "primitives": [item.get("primitive") for item in data.get("decisions", [])],
        "environment_semi_min_m": data.get("environment_manifold", {}).get("semi_min_m"),
        "self_manifold_clearance_min_m": data.get("robot_self_manifold_safety", {}).get("exact_surface_obstacle_clearance_min_m"),
    })
out = root / "extended_gallery_report.json"
out.write_text(json.dumps({"experiment": "accepted extended environment-manifold sequences", "scenarios": rows}, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY

printf 'PASS: %s\n' "$repo/$root/extended_gallery_report.json"
