# Sonic-Nav

Single-environment research harness for **manifold-conditioned motion learning** on top of a
frozen whole-body controller: a Unitree G1 in MuJoCo driven by NVIDIA GEAR-SONIC ONNX models.

The repository is trimmed to the research-essential subset — `manifold_g1/` plus the SONIC
ONNX assets it loads. Navigation, deployment source, training stacks and docs from the
original Sonic-Nav / GR00T-WholeBodyControl trees were removed.

The current stage is the **static manifold → pose policy**: given an ellipsoid (the robot's
standing envelope, squashed in height, pinched in width, shifted sideways), learn the pose
`p(pose | manifold)` that fits the body inside it. The pose has three dimensions — crouch,
lean, waist twist — and every manifold in the family gets its own learned distribution. The
policy is learned in a calibrated geometric model and executed by frozen SONIC in MuJoCo,
which is also the verification path.

## Layout

| Path | Contents |
|---|---|
| `manifold_g1/` | MuJoCo environment, frozen SONIC controller, keyframe reference, static manifold family, PPO |
| `gear_sonic_deploy/policy/release/` | SONIC encoder/decoder ONNX + observation config (downloaded, git-ignored) |
| `data/` | G1 MuJoCo model, meshes, generated manifold scenes (git-ignored) |
| `reports/` | Training logs, checkpoints, verification JSON (git-ignored) |

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 download_from_hf.py          # fetches the SONIC ONNX models from nvidia/GEAR-SONIC
```

The G1 MuJoCo model and meshes live in `data/g1_flat/` (see `manifold_g1/README.md`).

## Run

```bash
make g1-envelope        # calibrate the body model: standing envelope + landmarks over (crouch, lean, twist)
make g1-static-train    # PPO: manifold family -> pose distribution, headless
make g1-static-report   # manifold -> learned pose mean/std, next to the exhaustive optimum
make g1-static-verify   # execute the learned poses on MuJoCo + frozen SONIC
make g1-static-view     # native MuJoCo viewer: u/i height, j/k width, n/m offset, R/P
```

Documents:

| file | contents |
|---|---|
| [`STATIC_MANIFOLD_PIPELINE.md`](STATIC_MANIFOLD_PIPELINE.md) | what is built and verified: body model, manifold family, reward design, results, measured limits |
| [`MINIMAL_GOAL_AND_PATH.md`](MINIMAL_GOAL_AND_PATH.md) | the minimal vertical slice to aim for next, and the path to it |
| [`manifold_g1/README.md`](manifold_g1/README.md) | package-level: modules, commands, viewer keys |
