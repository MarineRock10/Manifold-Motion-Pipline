# Manifold-Motion

Research harness for **manifold-conditioned motion generation** on a Unitree G1, executed by a
frozen whole-body controller (NVIDIA GEAR-SONIC ONNX) in MuJoCo.

The idea: an environment constraint is expressed as an **ellipsoid manifold** the body must stay
inside, and the policy learns what to do about it. Today that is `M → pose` (a single held
posture, no time dimension); the target is `M → motion`.

Renamed from `Sonic-Nav`, which described the parent project rather than this one. The repository
is trimmed to the research-essential subset — `manifold_motion/` plus the SONIC ONNX assets it
loads; navigation, deployment source and training stacks from the original Sonic-Nav /
GR00T-WholeBodyControl trees were removed.

## Where to start

| document | what it covers |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | **the whole system**: environment → manifold → primitive → dynamic motion → SONIC, both stages, the capability manifold, the reverse-data loops, the frozen interfaces, and the nine phases with their acceptance criteria |
| [`PIPELINE.md`](PIPELINE.md) | **what is built**: only Phase 1–3 (`M → pose`, no time dimension) — the three stages, their measured numbers, the bugs that were fixed, what is still unsolved |

Start with ARCHITECTURE to see where the project is going and what the interfaces are; PIPELINE
is the honest account of how far it has actually got.

## Layout

| Path | Contents |
|---|---|
| `manifold_motion/` | MuJoCo environment, frozen SONIC controller, manifold family, behaviour cloning, in-the-loop fine-tune, viewers |
| `gear_sonic_deploy/policy/release/` | SONIC encoder/decoder ONNX + observation config (downloaded, git-ignored) |
| `data/` | G1 MuJoCo model, meshes, generated manifold scenes (git-ignored) |
| `reports/` | Clips, demonstrations, checkpoints, verification JSON (git-ignored — **back these up separately**) |

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 download_from_hf.py          # fetches the SONIC ONNX models from nvidia/GEAR-SONIC
```

The G1 model and meshes live in `data/g1_flat/`.

## Run

Three stages, each judged by its own metric:

```bash
# ① data
python3 -m manifold_motion.dataset show                # health: speed, envelope, coverage
python3 -m manifold_motion.demo_spread                 # how much recorded poses vary per manifold

# ② the clone
make g1-bc-train                                   # or: python3 -m manifold_motion.bc train
make g1-bc-eval                                    # 94.9% of training manifolds fit

# ③ the in-the-loop fine-tune
make g1-sonic-rl                                   # ~15 min on CPU
make g1-sonic-eval                                 # BC vs fine-tuned, on the real controller

# verification and the containment ruler
make g1-verify                                     # 6/10 inside, r(sonic) median 0.976
make g1-ruler                                      # the ellipsoid-vs-box check

# see it
python3 -m manifold_motion.view_data replay            # ① the recording
python3 -m manifold_motion.show_bc                     # ② the clone on its training manifolds
python3 -m manifold_motion.show_rl                     # ③ the fine-tune; keys reshape the manifold
```

Viewer keys and what each number on screen means: [`PIPELINE.md`](PIPELINE.md) §7.

## Current numbers

| stage | metric | value |
|---|---|---|
| ① data | SONIC tracking error | 0.054 rad (137 clips, 18705 windows, 10336 pose pairs) |
| ② clone | pose inside its recorded envelope | 9336/9840 = **94.9%** |
| ③ in-loop | achieved pose inside the perturbed envelope | success **1.00**, r **0.82** |
| gate | `verify_sonic` | **6/10** inside, r(sonic) median 0.976 |

## Known blocker

Tilted and very narrow manifolds (`report_specs` t=±10) sit at r 1.12–1.33. Diagnosed: the
manifolds are geometrically solvable (a body pitched 7–8° fits at r 0.97), the policy responds
in the right direction but with about half the needed amplitude, and training on tilted
manifolds makes things **worse** rather than better — the policy answers with waist pitch, which
the frozen controller does not follow. The fix is a pose channel the controller tracks
(`vr_3point_local_target`), not more training pressure. Details: [`PIPELINE.md`](PIPELINE.md) §6.
