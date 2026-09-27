#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
PYTHON="${MANIFOLD_PYTHON:-$repo/scripts/python.sh}"
export PYTHONPATH="$repo:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl

windows="reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows.npz"
checkpoint="reports/manifold_motion/composer_v1/composer.pt"
root="reports/manifold_motion/online_composer_gallery_v1"
mkdir -p "$root"

# First replay the held-out SEED windows as successive perception frames. This is a fast,
# deterministic check of the learned 21-family online input contract.
"$PYTHON" -m manifold_motion.stage2.online_composer \
  --checkpoint "$checkpoint" --windows "$windows" \
  --out "$root/heldout_rolling_eval"

common=(--online-perception --composer-checkpoint "$checkpoint" --num-candidates 2
        --online-condition-iterations 0 --planner-body-radius-m 0.40
        --planner-side-body-radius-m 0.30 --planner-clearance-m 0.10
        --segment-length-m 0.45 --max-ticks 1800 --skip-render)

"$PYTHON" -m manifold_motion.stage2.manifold_adaptive \
  --scene data/g1_flat/scene_manifold_block_right.xml \
  --title "ONLINE COMPOSER: RIGHT BLOCK" --out "$root/block_right" "${common[@]}"
"$PYTHON" -m manifold_motion.stage2.manifold_adaptive \
  --scene data/g1_flat/scene_manifold_block_left.xml \
  --title "ONLINE COMPOSER: LEFT BLOCK" --out "$root/block_left" "${common[@]}"
"$PYTHON" -m manifold_motion.stage2.manifold_adaptive \
  --scene data/g1_flat/scene_manifold_wide_long.xml \
  --title "ONLINE COMPOSER: WIDE LONG" --out "$root/wide_long" "${common[@]}"

"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("reports/manifold_motion/online_composer_gallery_v1")
rows = []
for report_path in sorted(root.glob("*/report.json")):
    report = json.loads(report_path.read_text())
    if "execution" not in report:
        continue
    execution = report["execution"]
    rows.append({
        "name": report_path.parent.name,
        "accepted": bool(report["accepted"]),
        "failed_checks": report["failed_checks"],
        "keyframes_reached": execution["keyframes_reached"],
        "keyframes_total": execution["keyframes_total_excluding_start"],
        "physics_ticks": execution["physics_ticks"],
        "obstacle_contact_ticks": execution["obstacle_contact_ticks"],
        "primitive_switch_count": execution["primitive_switch_count"],
        "online_updates": report.get("online_composer", {}).get("updates"),
        "composer_mean_latency_ms": report.get("online_composer", {}).get("mean_latency_ms"),
        "composer_families": sorted(set(report.get("online_composer", {}).get("families", []))),
    })
summary = {
    "schema": "manifold-motion.online-composer-gallery.v1",
    "contract": "live radar -> 3-D sliding map -> D* Lite -> M_e -> rolling composer(M_self,state/history) -> frozen SONIC",
    "accepted": bool(rows) and all(row["accepted"] for row in rows),
    "scenarios": rows,
}
(root / "comparison_report.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if not summary["accepted"]:
    raise SystemExit(2)
PY
