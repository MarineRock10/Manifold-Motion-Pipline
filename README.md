# Sonic-Nav

Single-environment research harness for **manifold-conditioned motion learning** on top of a
frozen whole-body controller: a Unitree G1 in MuJoCo driven by NVIDIA GEAR-SONIC ONNX models.

The repository is trimmed to the research-essential subset — `manifold_g1/` plus the SONIC
ONNX assets it loads. Navigation, deployment source, training stacks and docs from the
original Sonic-Nav / GR00T-WholeBodyControl trees were removed.

## Layout

| Path | Contents |
|---|---|
| `manifold_g1/` | MuJoCo environment, frozen SONIC controller, planner wrapper, reference buffers, goal-reaching task, PPO |
| `gear_sonic_deploy/policy/release/` | SONIC encoder/decoder ONNX + observation config (downloaded, git-ignored) |
| `gear_sonic_deploy/planner/target_vel/V2/` | Kinematic planner ONNX (velocity commands → motion reference) |
| `data/` | G1 MuJoCo model, meshes, generated manifold scenes (git-ignored) |
| `reports/` | Training/evaluation logs, checkpoints, videos (git-ignored) |

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 download_from_hf.py          # fetches the SONIC ONNX models from nvidia/GEAR-SONIC
```

The G1 MuJoCo model and meshes live in `data/g1_flat/` (see `manifold_g1/README.md`).

## Run

```bash
make g1-stand     # frozen SONIC stand test
make g1-walk      # planner-driven walk
make g1-train     # PPO on the fixed-manifold goal task
make g1-eval      # greedy evaluation of the trained policy
```

See [`manifold_g1/README.md`](manifold_g1/README.md) for interfaces, curriculum gates and
verified results.
