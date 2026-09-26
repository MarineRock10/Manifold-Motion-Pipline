#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
candidates=()
if [[ -n "${MANIFOLD_PYTHON:-}" ]]; then candidates+=("$MANIFOLD_PYTHON"); fi
candidates+=("$repo/.venv/bin/python" "/home/xiyuan/orcs-review-20260923/.venv/bin/python" "$(command -v python3 || true)")
for candidate in "${candidates[@]}"; do
  [[ -x "$candidate" ]] || continue
  if "$candidate" -c 'import mujoco, torch, onnxruntime' >/dev/null 2>&1; then
    exec "$candidate" "$@"
  fi
done
cat >&2 <<EOF
No dependency-complete Python interpreter found.
Set MANIFOLD_PYTHON to an environment containing MuJoCo, Torch and ONNX Runtime,
or create .venv in the repository.
EOF
exit 127
