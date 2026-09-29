# CVPR data preparation: manifold–action catalogue

The first CVPR experiment artifact is a shared index for both stages. It is built from accepted
SEED replay windows, so every row has the same provenance chain:

```text
SEED G1 clip -> frozen SONIC replay -> MuJoCo physical gate
             -> executed envelope M_self(t)
             -> reverse corridor construction M_e(t)
             -> state/history + reference and executed targets
```

The current corridor archive contains 3,988 windows from 118 actors and 21 action families. The
actor hash split is retained at the window level, so an actor cannot appear in train, validation
and test simultaneously. Each catalogue row records its `M_e` hash, family/action ID, source
clip, split, window origin, corridor aperture statistics and measured `M_self` statistics. The
full tensors remain in the NPZ file; the CSV is the reviewable list used by experiment manifests.

Build it from the already accepted corridor windows:

```bash
./scripts/python.sh -m manifold_motion.dataio.manifold_action_catalog \
  --windows reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows.npz \
  --metadata reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows_metadata.json \
  --out data/manifold_action_catalog_v1
```

The command creates:

- `data/manifold_action_catalog_v1/manifold_action_catalog_v1.csv`: one row per reverse-synthesized
  environment-manifold/action window;
- `data/manifold_action_catalog_v1/manifold_action_catalog_v1.json`: counts, tensor shapes,
  provenance and leakage checks;
- `data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz`: versioned derived training
  tensors (stored with Git LFS; the raw licensed SEED/G1 archives are intentionally not included).

## Paired GIF visual audit

The data-preparation result is now inspectable as **one manifold GIF paired with one action GIF**:
[paired gallery](../demo_gallery/manifold_action_pairs_v1/README.md).
For an offline searchable page, open `docs/demo_gallery/manifold_action_pairs_v1/index.html`.

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ./scripts/python.sh \
  -m manifold_motion.visualization.manifold_action_pairs \
  --out docs/demo_gallery/manifold_action_pairs_v1
```

The default renders one representative for each of the 21 present source families. Each sample creates:

- `me_action_<row>_<family>_manifold.gif`: cyan reverse-synthesized `M_e(t)` and orange recorded `M_self(t)`;
- `me_action_<row>_<family>_action.gif`: the corresponding G1 execution poses;
- `me_action_<row>_<family>_pair.gif`: both views in one GIF, guaranteeing frame synchronization;
- `*_keyframes.jpg`: beginning, middle, and end for static inspection.

Every animation has 36 frames from a 30 Hz, 1.2-second source window. Default playback is 15 fps
(0.5×, 2.4 seconds per loop). GIF delays alternate 60/70 ms, preserving the correct overall rate.
The source joint, root-position, and root-rotation targets are checked before rendering, so a wrong
record or time offset fails instead of silently fabricating a pairing. Source IDs and maximum target
serialization error remain in the accompanying manifest for traceability, but the user-facing entry
point is the GIF gallery, not the numeric files. The two cameras are fixed during the clip and fit
their respective views independently; both use a 0.5 m floor grid.

To inspect arbitrary catalogue rows without rendering all 3,988 overlapping windows:

```bash
MUJOCO_GL=egl ./scripts/python.sh -m manifold_motion.visualization.manifold_action_pairs \
  --rows 150 634 1652 --out reports/manifold_motion/selected_data_pairs
```

Selection is deterministic: the default takes the window with the largest mean joint-frame delta
per source family. This is an illustrative audit, **not random sampling or a success-rate estimate**.
The action is a kinematic replay of recorded `R_exec`, with the original absolute joint angles and
root trajectory; no default-pose offset, trajectory rescaling, or new controller rollout is added.
The finger joints are held at the replay pipeline's neutral setting because the archive records only
the 29 body joints. The corridor is reverse-synthesized around an admitted flat-ground execution;
there are no newly tested real obstacles here. Labels such as `ladder`, `box_jump`, and
`door_interaction` denote SEED source categories, not proof of completed climbing/jumping/door tasks.
These pairs do not establish that a trained model can autonomously infer the action from `M_e`.

## Training artifacts and next experiment

The CSV and JSON are small metadata artifacts and are versioned. The NPZ is reproducible and is
kept local when it is too large for a source checkout. Rows inherit accepted replay admission;
future corridor perturbations are separate candidate data and must pass the same SONIC/MuJoCo
gate before being included as positives.

This catalogue is the input to the next experiment step:

1. Stage 1 imitation trains `M -> action/reference` on the train split;
2. the frozen-SONIC fine-tuning branch trains only on physically admitted failures;
3. Stage 1 ablations remove imitation or controller fine-tuning while keeping the catalogue and
   actor split fixed;
4. Stage 2 adds `M_e(t)`, `M_self(t)`, state and history, using the identical split and source
   clip IDs for paired comparisons.
