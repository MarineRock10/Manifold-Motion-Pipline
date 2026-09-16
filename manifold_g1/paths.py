"""Every path the pipeline reads or writes, in one place.

These were spelled out per module, which is how the artifact directory rename
(`primitive_torch/` -> `bc/`) needed edits in eight files and how `verify_sonic` ended up
defaulting to a policy in one directory while `show_bc` defaulted to another. One definition
each means a layout change is a one-line change.

`reports/` is git-ignored, so these point at files that exist only on disk - see the backup
notes for what must be preserved when moving machines.
"""

from __future__ import annotations

from pathlib import Path

from . import constants as C

REPORTS = C.REPO / "reports" / "manifold_g1"

# --- dataset -----------------------------------------------------------------
CLIPS = REPORTS / "clips"
DATASET = REPORTS / "dataset"
POSE_DEMOS = DATASET / "pose_demos.npz"
DATASET_INDEX = DATASET / "index.json"
CAPABILITY = DATASET / "capability.json"
BODY_ENVELOPE = REPORTS / "body_envelope.json"

# --- policies and their reports ----------------------------------------------
BC_DIR = REPORTS / "bc"
BC_POLICY = BC_DIR / "bc_policy.pt"
BC_LOG = BC_DIR / "train_log.csv"

RL_DIR = REPORTS / "sonic_rl"
RL_POLICY = RL_DIR / "policy_sonicrl.pt"
RL_LOG = RL_DIR / "train_log.csv"

VERIFY_REPORT = BC_DIR / "verify_sonic.json"
RULER_REPORT = REPORTS / "eval_session.json"

# --- scenes ------------------------------------------------------------------
MANIFOLD_SCENE = C.REPO / "data" / "g1_flat" / "scene_manifold.xml"
