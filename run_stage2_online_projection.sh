#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo"
export PYTHONPATH="${repo}:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl

root="reports/manifold_motion/stage2_online_projection_v1"
if [[ -f stage2_asset_manifest.sha256 ]]; then
  sha256sum -c stage2_asset_manifest.sha256 >/dev/null
fi
common=(--side-semi-y-m 0.42 --num-candidates 3 --planner-body-radius-m 0.40
        --planner-clearance-m 0.10 --online-condition-iterations 1)

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_wide_long.xml \
  --title "WIDE: ONLINE STATE/HISTORY NOMINAL WALK" \
  --out "$root/wide" "${common[@]}"

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_low_long.xml \
  --title "LOW: ONLINE STATE/HISTORY PROJECTED CROUCH" \
  --out "$root/low" "${common[@]}"

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_manifold_narrow_105cm.xml \
  --title "1.05 m NARROW: ONLINE STRICT SIDE-ON" \
  --out "$root/narrow" "${common[@]}"

python3 -m manifold_motion.stage2_manifold_adaptive \
  --scene data/g1_flat/scene_long_avoidance.xml \
  --title "CENTER BLOCK: ONLINE MANIFOLD ROUTE" \
  --out "$root/center" --side-gait-mode diagonal --max-ticks 2200 "${common[@]}"

python3 - <<'PY'
import json
from pathlib import Path
root = Path("reports/manifold_motion/stage2_online_projection_v1")
rows = []
for report_path in sorted(root.glob("*/report.json")):
    report = json.loads(report_path.read_text())
    execution = report["execution"]
    rows.append({
        "scenario": report["scenario"], "accepted": report["accepted"],
        "failed_checks": report["failed_checks"],
        "online_iterations": report["online_conditioning"]["iterations"],
        "probe_accepted": report["online_conditioning"]["probe_execution"]["accepted"],
        "keyframes_reached": execution["keyframes_reached"],
        "obstacle_contact_ticks": execution["obstacle_contact_ticks"],
        "side_on_ticks": execution["side_on_ticks"],
        "self_manifold_gate": report["robot_self_manifold_safety"],
    })
comparison = {"experiment": "online state/history + optimization-embedded projection",
              "accepted": all(row["accepted"] and row["probe_accepted"] for row in rows),
              "scenarios": rows}
(root / "comparison_report.json").write_text(json.dumps(comparison, indent=2) + "\n")
print(json.dumps(comparison, indent=2))
if not comparison["accepted"]:
    raise SystemExit(2)
PY
printf 'PASS: %s\n' "$repo/$root/comparison_report.json"
