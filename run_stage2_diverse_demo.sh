#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")" && pwd)"
cd "$repo"
export PYTHONPATH="$repo:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl

root="reports/manifold_motion/stage2_diverse_demo_v2"
common=(--num-candidates 3 --online-condition-iterations 1
        --planner-body-radius-m 0.40 --planner-clearance-m 0.10)

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_low_short.xml \
  --title "SHORT LOW: CROUCH THEN WALK" \
  --out "$root/low_short" "${common[@]}"

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_block_left.xml \
  --title "LEFT OFFSET BLOCK: SIDE GAIT THEN WALK" \
  --out "$root/block_left" --max-ticks 2200 "${common[@]}"

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_block_right.xml \
  --title "RIGHT OFFSET BLOCK: SIDE GAIT + TURN" \
  --out "$root/block_right" --max-ticks 2200 "${common[@]}"

python3 - <<'PY'
import json
from pathlib import Path

root = Path("reports/manifold_motion/stage2_diverse_demo_v2")
rows = []
for path in sorted(root.glob("*/report.json")):
    report = json.loads(path.read_text())
    execution = report["execution"]
    decisions = report.get("decisions", [])
    rows.append({
        "name": path.parent.name,
        "scenario": report["scenario"],
        "scene": report["scene"],
        "accepted": bool(report["accepted"]),
        "failed_checks": report["failed_checks"],
        "routed_primitives": [row["primitive"] for row in decisions],
        "turn_segments": [row["segment_index"] for row in decisions
                          if row.get("requires_turn")],
        "keyframes_reached": execution["keyframes_reached"],
        "keyframes_total": execution["keyframes_total_excluding_start"],
        "obstacle_contact_ticks": execution["obstacle_contact_ticks"],
        "self_manifold_gate": report["robot_self_manifold_safety"],
        "side_on_ticks": execution["side_on_ticks"],
        "gif": str(path.parent / "manifold_adaptive.gif"),
    })
summary = {
    "experiment": "additional environment-manifold action demos",
    "synthetic_geometry_note": "MuJoCo fixture geometry; no RGB-D/LiDAR claim",
    "accepted": bool(rows) and all(row["accepted"] for row in rows),
    "scenarios": rows,
}
(root / "comparison_report.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if not summary["accepted"]:
    raise SystemExit(2)
PY
