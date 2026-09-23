# ORCS controller integration

## Decision

Use the official ORCS checkpoints as **native task experts and paper baselines**, but do not label
them as a drop-in replacement for the current ONNX controller. The closest release is
`Orcs-PerLoco-Grail-AdaptSonic`; its public checkpoint has been downloaded and SHA-256 verified.

The official checkpoint contains the frozen SONIC base plus trained LoRA adapters. It is therefore
more useful than training another small imitation adapter from scratch, provided it is evaluated
under the observation and motion-reference contract on which it was trained.

## Measured checkpoint contract

| item | official PerLoco-Grail checkpoint | current deployment |
|---|---:|---:|
| action | 29 G1 joints | 29 G1 joints |
| decoder input | 994 | 994 |
| SONIC token | 64 | 64 |
| adapter rank | 16, seven decoder layers | separate experimental residual |
| adapter condition | 202 | `M_e`, SDF, state/history, command |
| exteroception | 187 downward pelvis rays | 3-D radar probability grid / corridor |
| runtime | PyTorch + pinned `mocke`/`rsl_rl` + MJLab | ONNX Runtime + MuJoCo |

The 202 ORCS augmentation values are `187 height rays + root position + root linear/angular
velocity + root linear/angular command`. The public PerLoco sensor is a 17x11 yaw-aligned downward
grid. It can represent curbs, steps and climb surfaces, but cannot see a side wall or a ceiling.
Consequently:

- use ORCS-Grail directly for curb/terrain generalization;
- do not score it as a narrow-side or low-ceiling method unless a new 3-D observation adapter is
  trained;
- do not copy only the LoRA tensors into the ONNX decoder: its normalizer, frozen base revision,
  tokenizer and reference motion are part of the policy.

## Evaluation tracks

### Track A — official native checkpoint

Run ORCS in its isolated Python 3.11/MJLab environment with the released checkpoint and staged
GRAIL roster. This establishes that the public controller works unchanged and gives a defensible
`ORCS-Grail` baseline for terrain scenes.

### Track B — common MuJoCo scene export

Export the same G1 root/joint/contact traces from ORCS and the current pipeline, then compare:

- task success and illegal contact;
- root/body tracking error;
- joint acceleration, jerk and action saturation;
- minimum measured self-manifold clearance;
- inference latency and GPU memory.

The comparison must use a terrain scenario observable by both policies. It must not use a ceiling
or side wall and then interpret ORCS failure as an algorithmic failure—the released sensor cannot
observe either.

### Track C — future 3-D ORCS adapter

Only if Track A is stable, replace the 187-ray augmentation with an encoder over `M_e(t)`, local
SDF and state/history, preserve the official frozen SONIC base, and train only the adapter using
the ORCS PPO recipe. This is a new policy and must not be called the released ORCS checkpoint.

## Reproducible audit

```bash
python -m manifold_motion.orcs_checkpoint_audit \
  ~/.cache/orcs/releases/v0.1.0/Orcs-PerLoco-Grail-AdaptSonic/checkpoint.pt \
  --out reports/manifold_motion/orcs_grail_audit.json
```

## Native bounded rollout

The repository includes a headless wrapper so a remote WSL session does not depend on an
interactive MuJoCo window. After the isolated ORCS environment and assets are installed, run:

```bash
cd "$ORCS_REPO"
source .venv/bin/activate
export ORCS_RELEASE_ROOT="$ORCS_RELEASE_ROOT"
export MUJOCO_GL=egl
export PYTHONPATH="$MANIFOLD_MOTION_REPO"
python "$MANIFOLD_MOTION_REPO/manifold_motion/orcs_native_rollout.py" \
  --task Orcs-PerLoco-Grail-AdaptSonic --agent release --steps 120 --device cuda:0 \
  --out "$MANIFOLD_MOTION_REPO/artifacts/online_stage2_research/orcs_grail_native/orcs_grail_native.mp4"
```

The checked-in compact clip was generated this way. The wrapper accepts both mjlab's four-return
vector-environment API and Gymnasium's five-return API, and reports the frame count on success.

Sources: [ORCS repository](https://github.com/lok-i/orcs),
[release manifest](https://github.com/lok-i/orcs/blob/main/src/orcs/release.json),
[adapter implementation](https://github.com/lok-i/orcs/blob/main/src/orcs/core/rl.py), and
[public checkpoints](https://huggingface.co/lkrajan/orcs).
