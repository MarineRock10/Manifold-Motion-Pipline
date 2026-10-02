#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export PYTHONPATH="$repo:/home/xiyuan/.local/share/sonic-manifold-g1"
export OMP_NUM_THREADS="${SONIC_ADAPTER_THREADS:-4}"
dataset="${SONIC_ADAPTER_DATASET:-reports/manifold_motion/seed_failure_supplement_v1/adapter_dataset_v2.npz}"
out_root="${SONIC_ADAPTER_MULTI_OUT:-reports/cvpr/sonic_adapter_multiseed_v1}"
read -r -a seeds <<< "${SONIC_ADAPTER_SEEDS:-20260923 20260924 20260925}"
mkdir -p "$out_root"
for seed in "${seeds[@]}"; do
  "${MANIFOLD_PYTHON:-$repo/scripts/python.sh}" -m manifold_motion.stage2.train_sonic_adapter \
    --dataset "$dataset" --out "$out_root/seed_${seed}" \
    --epochs "${SONIC_ADAPTER_EPOCHS:-10}" --batch-size "${SONIC_ADAPTER_BATCH:-64}" \
    --rank 16 --alpha 1.0 --seed "$seed" --device "${SONIC_ADAPTER_DEVICE:-cpu}"
done
"${MANIFOLD_PYTHON:-$repo/scripts/python.sh}" - "$out_root" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for report in sorted(root.glob("seed_*/report.json")):
    value = json.loads(report.read_text())
    rows.append({
        "seed": int(report.parent.name.split("_")[-1]),
        "train_count": value["train_count"],
        "validation_count": value["validation_count"],
        "test_count": value["test_count"],
        "base_test_mse": value["base_mse"]["test"],
        "adapter_test_mse": value["fitted_mse"]["test"],
        "relative_test_improvement": 1.0 - value["fitted_mse"]["test"] / value["base_mse"]["test"],
        "zero_init_parity": value["zero_init_base_parity_max_abs"],
    })
if not rows:
    raise SystemExit("no adapter reports found")
summary = {
    "schema": "manifold-motion.sonic-adapter-multiseed.v1",
    "dataset": str(root), "seeds": rows,
    "mean_relative_test_improvement": sum(r["relative_test_improvement"] for r in rows) / len(rows),
    "all_zero_init_exact": all(r["zero_init_parity"] == 0.0 for r in rows),
    "deployment_status": "warm-start audit only; full MuJoCo acceptance and PPO/distillation remain required",
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
