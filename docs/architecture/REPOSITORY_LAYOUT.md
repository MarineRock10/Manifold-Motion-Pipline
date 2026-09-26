# Repository layout

The repository is intentionally split by responsibility. New code should be placed in the
narrowest matching package; compatibility aliases are not kept in the package root.

```text
manifold_motion/
├── core/          geometry, kinematics, manifold and path contracts
├── simulation/    G1 MuJoCo, SONIC, reference playback and collection
├── dataio/        SEED metadata, capability manifests, replay and windows
├── perception/    radar/SLAM, voxel map, ESDF and dynamic-scene adapters
├── planning/      safe corridors, D* Lite/incremental planning and routing
├── stage1/        static manifold → pose BC/RL pipeline
├── stage2/        dynamic latent/Flow candidates, projection and execution
├── evaluation/    CVPR matrix, smoke tests and quantitative audits
├── visualization/ renderers, GIFs and GitHub gallery tooling
└── tools/         calibration and one-off geometry utilities

scripts/           WSL/Powershell launchers; python.sh selects a dependency-complete runtime
data/              tracked lightweight model/scene assets and selected SEED slices
reports/           ignored run outputs, checkpoints and physical evidence
docs/              design history, protocols, results and visual gallery
```

## Runtime contract

Run commands from the repository root. `scripts/python.sh` verifies that the selected interpreter
can import MuJoCo, Torch and ONNX Runtime. Set `MANIFOLD_PYTHON` when using a different WSL
virtual environment:

```bash
MANIFOLD_PYTHON=/path/to/python ./scripts/python.sh -m pytest -q
```

The package root is not a dumping ground for new modules. A module that is part of the runnable
Stage-2 chain belongs in `dataio`, `perception`, `planning`, `stage2`, `evaluation` or
`visualization` as appropriate.