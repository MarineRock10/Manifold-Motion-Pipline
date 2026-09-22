#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")" && pwd)"
cd "$repo"
export PYTHONPATH="$repo:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl

root="reports/manifold_motion/stage2_long_sequence_v1"
common=(--num-candidates 3 --online-condition-iterations 1
        --planner-body-radius-m 0.40 --planner-clearance-m 0.10)

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_long_low_cycle.xml \
  --title "LONG LOW CYCLE: CROUCH -> WALK -> CROUCH" \
  --out "$root/low_cycle" --segment-length-m 0.35 --max-ticks 3000 "${common[@]}"

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_long_combo.xml \
  --title "LONG COMBO: CROUCH -> SIDE GAIT" \
  --out "$root/combo" --segment-length-m 0.45 --max-ticks 2600 "${common[@]}"

python3 - <<'PY'
import json
from pathlib import Path

root = Path("reports/manifold_motion/stage2_long_sequence_v1")
rows = []
for path in (root / "low_cycle" / "report.json", root / "combo" / "report.json"):
    if not path.is_file():
        continue
    report = json.loads(path.read_text())
    execution = report["execution"]
    rows.append({
        "name": path.parent.name,
        "scenario": report["scenario"],
        "scene": report["scene"],
        "accepted": bool(report["accepted"]),
        "failed_checks": report["failed_checks"],
        "routed_primitives": [row["primitive"] for row in report.get("decisions", [])],
        "primitive_switch_count": execution["primitive_switch_count"],
        "keyframes_reached": execution["keyframes_reached"],
        "keyframes_total": execution["keyframes_total_excluding_start"],
        "physics_ticks": execution["physics_ticks"],
        "duration_s": execution["duration_s"],
        "obstacle_contact_ticks": execution["obstacle_contact_ticks"],
        "self_manifold_gate": report.get("robot_self_manifold_safety"),
        "route_deviation_p95_m": execution["route_deviation_p95_m"],
        "max_pre_switch_joint_rms_rad": execution["max_pre_switch_joint_rms_rad"],
        "gif": str(path.parent / "manifold_adaptive.gif"),
    })
summary = {
    "experiment": "long-horizon multi-manifold action sequences",
    "synthetic_geometry_note": "MuJoCo fixture geometry; no RGB-D/LiDAR claim",
    "accepted": bool(rows) and all(row["accepted"] for row in rows),
    "scenarios": rows,
}
(root / "comparison_report.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if not summary["accepted"]:
    raise SystemExit(2)
PY
