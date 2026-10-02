#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export PYTHONPATH="$repo:/home/xiyuan/.local/share/sonic-manifold-g1"
export MUJOCO_GL=egl
python_runner="${MANIFOLD_PYTHON:-$repo/scripts/python.sh}"
root="${SONIC_ADAPTER_DYNAMIC_OUT:-reports/cvpr/sonic_adapter_dynamic_v1}"
checkpoint="${SONIC_ADAPTER_CHECKPOINT:-reports/cvpr/sonic_adapter_multiseed_v1/seed_20260923/sonic_condition_adapter.pt}"
fixture_root="${SONIC_DYNAMIC_FIXTURE_ROOT:-reports/manifold_motion/autonomous_v3_motion_quality}"
mkdir -p "$root"

# The residual is trusted at 25% in live dynamic scenes. Static scenes use the full trained
# bound, while radar/SDF distribution shift receives a conservative action trust region.
common=(--online-perception --online-perception-scan-ticks 10 --num-candidates 2
  --online-condition-iterations 0 --planner-body-radius-m 0.40
  --planner-side-body-radius-m 0.30 --planner-clearance-m 0.10
  --segment-length-m 0.45 --max-ticks 350 --skip-render)

run_one() {
  local method="$1" event="$2" scene="$3"
  local extra=()
  if [[ "$method" == "adapter" ]]; then
    extra=(--sonic-adapter "$checkpoint" --sonic-adapter-scale 0.25)
  fi
  "$python_runner" -m manifold_motion.stage2.manifold_adaptive \
    --scene "$scene" --title "${method^^} ${event^^}" \
    --out "$root/${method}_${event}" --dynamic-obstacle-event "$event" \
    --reactive-policy-checkpoint reports/manifold_motion/reactive_hazard_policy_v4/reactive_policy.pt \
    "${common[@]}" "${extra[@]}"
}

scene="$fixture_root/dynamic_fixture/generated_scene.xml"
if [[ ! -f "$scene" ]]; then
  "$python_runner" - <<'PY'
from pathlib import Path
from manifold_motion.evaluation.cvpr_physical_smoke import _xml_for_generated_fixture
out = Path("reports/manifold_motion/autonomous_v3_motion_quality/dynamic_fixture/generated_scene.xml")
out.parent.mkdir(parents=True, exist_ok=True)
_xml_for_generated_fixture({"factory": "dynamic_event_fixture", "event": "crossing", "goal_x_m": 3.6}, out)
PY
fi
run_one frozen crossing "$scene"
run_one adapter crossing "$scene"

"$python_runner" - "$root" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/report.json")):
    report = json.loads(path.read_text())
    execution = report["execution"]
    rows.append({
        "method": path.parent.name.split("_", 1)[0],
        "scenario": "crossing",
        "accepted": bool(report["accepted"]),
        "failed_checks": report["failed_checks"],
        "keyframes": execution["keyframes_reached"],
        "physics_ticks": execution["physics_ticks"],
        "dynamic_wait_ticks": execution["dynamic_wait_ticks"],
        "obstacle_contact_ticks": execution["obstacle_contact_ticks"],
        "adapter": execution.get("sonic_adapter", {}),
    })
summary = {
    "schema": "manifold-motion.sonic-adapter-dynamic-ab.v1",
    "contract": "same online radar/SLAM crossing scene; frozen SONIC versus 0.25 trust-region adapter",
    "rows": rows,
    "all_accepted": bool(rows) and all(row["accepted"] for row in rows),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if not summary["all_accepted"]:
    raise SystemExit(2)
PY
