#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")" && pwd)"
cd "$repo"
export PYTHONPATH="$repo:/home/xiyuan/.local/share/sonic-manifold-g1${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SONIC_ADAPTER_THREADS:-4}"

transition_root="reports/manifold_motion/seed_transition_stage2_v1"
adapter_root="reports/manifold_motion/sonic_adapter_warmstart_v1"

python3 -m manifold_motion.stage2_transition_dataset \
  --windows reports/manifold_motion/seed_windows_corridor_stage2_v2/seed_stage2_windows.npz \
  --out "$transition_root/transitions.npz"
python3 -m manifold_motion.stage2_transition_gate \
  --transitions "$transition_root/transitions.npz" \
  --out "$transition_root/physical_gate.json" --per-transition 2
python3 -m manifold_motion.stage2_adapter_dataset \
  --transitions "$transition_root/transitions.npz" \
  --out "$adapter_root/action_aligned.npz" --limit "${SONIC_ADAPTER_TRANSITIONS:-24}"
python3 -m manifold_motion.train_sonic_adapter \
  --dataset "$adapter_root/action_aligned.npz" --out "$adapter_root/trained" \
  --epochs "${SONIC_ADAPTER_EPOCHS:-10}" --batch-size "${SONIC_ADAPTER_BATCH:-64}" \
  --rank 16 --alpha 1.0 --device "${SONIC_ADAPTER_DEVICE:-cpu}"

echo "Warm-start checkpoint: $adapter_root/trained/sonic_condition_adapter.pt"
echo "This is not deployable until privileged PPO/distillation and the full MuJoCo matrix pass."
