#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo"
export PYTHONPATH="${repo}:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl

root="reports/manifold_motion/stage2_no_handcrafted_anchor_v1"
common=(--num-candidates 3 --online-condition-iterations 1 --disable-anchor
        --min-forward-progress-m 0.10 --planner-body-radius-m 0.40
        --planner-clearance-m 0.10 --skip-render)

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_wide_long.xml \
  --title "WIDE: NO HANDCRAFTED ANCHOR" --out "$root/wide" "${common[@]}"
python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_low_long.xml \
  --title "LOW: NO HANDCRAFTED ANCHOR" --out "$root/low" "${common[@]}"
python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_narrow_105cm.xml \
  --title "NARROW: NO HANDCRAFTED ANCHOR" --out "$root/narrow" "${common[@]}"
python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_long_avoidance.xml \
  --title "CENTER: NO HANDCRAFTED ANCHOR" --out "$root/center" \
  --side-gait-mode diagonal --max-ticks 2200 "${common[@]}"

python3 - <<'PY'
import json
from pathlib import Path
root = Path("reports/manifold_motion/stage2_no_handcrafted_anchor_v1")
rows = []
for path in sorted(root.glob("*/report.json")):
    report = json.loads(path.read_text())
    execution = report["execution"]
    rows.append({
        "scenario": report["scenario"], "accepted": bool(report["accepted"]),
        "failed_checks": report["failed_checks"],
        "keyframes_reached": execution["keyframes_reached"],
        "obstacle_contact_ticks": execution["obstacle_contact_ticks"],
        "anchor_policy": report["anchor_policy"],
    })
summary = {"experiment": "no handcrafted anchor; learned conditional mean remains",
           "accepted": bool(rows) and all(row["accepted"] for row in rows), "scenarios": rows}
(root / "comparison_report.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if not summary["accepted"]:
    raise SystemExit(2)
PY
