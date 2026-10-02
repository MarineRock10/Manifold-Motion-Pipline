#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export PYTHONPATH="$repo:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl
python_runner="${MANIFOLD_PYTHON:-$repo/scripts/python.sh}"
root="${SONIC_ADAPTER_PHYSICAL_OUT:-reports/cvpr/sonic_adapter_physical_v1}"
checkpoint="${SONIC_ADAPTER_CHECKPOINT:-reports/cvpr/sonic_adapter_multiseed_v1/seed_20260923/sonic_condition_adapter.pt}"

run_one() {
  local method="$1" scenario="$2" scene="$3" max_ticks="$4"
  local extra=()
  if [[ "$method" == "adapter" ]]; then
    extra=(--sonic-adapter "$checkpoint")
  fi
  "$python_runner" -m manifold_motion.stage2.manifold_adaptive \
    --scene "data/g1_flat/$scene" --title "${method^^} ${scenario^^}" \
    --out "$root/${method}_${scenario}" --side-semi-y-m 0.42 --num-candidates 3 \
    --planner-body-radius-m 0.40 --planner-clearance-m 0.10 --max-ticks "$max_ticks" \
    --skip-render "${extra[@]}"
}

run_one frozen wide scene_manifold_wide_long.xml 900
run_one adapter wide scene_manifold_wide_long.xml 900
run_one frozen low scene_manifold_low_long.xml 1200
run_one adapter low scene_manifold_low_long.xml 1200
run_one frozen narrow scene_manifold_narrow_105cm.xml 1200
run_one adapter narrow scene_manifold_narrow_105cm.xml 1200

"$python_runner" - "$root" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for report_path in sorted(root.glob("*/report.json")):
    report = json.loads(report_path.read_text())
    execution = report["execution"]
    rows.append({
        "scenario": report_path.parent.name.rsplit("_", 1)[-1],
        "method": report_path.parent.name.split("_", 1)[0],
        "accepted": bool(report["accepted"]),
        "failed_checks": report["failed_checks"],
        "keyframes": execution["keyframes_reached"],
        "physics_ticks": execution["physics_ticks"],
        "terminal_error_m": execution["terminal_error_m"],
        "route_deviation_p95_m": execution["route_deviation_p95_m"],
        "tracking_error_mean_rad": execution["track_err_mean_rad"],
        "obstacle_contact_ticks": execution["obstacle_contact_ticks"],
    })
summary = {"schema": "manifold-motion.sonic-adapter-physical-ab.v1",
           "contract": "same scenes and generated candidates; only bounded residual differs",
           "rows": rows, "all_accepted": all(row["accepted"] for row in rows)}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if not summary["all_accepted"]:
    raise SystemExit(2)
PY
