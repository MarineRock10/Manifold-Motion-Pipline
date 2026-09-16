"""The pose policy's interface: observation layout and action mapping.

Shared by behaviour cloning, the SONIC-in-the-loop fine-tune and every viewer, so that all four
build the observation and turn an action into a pose in exactly the same way. Getting this
wrong is not loud: an earlier version trained a clone through `tanh` while the environment
reproduced the raw output, and the policy then posed nothing like the demonstration it was fit
to. There is one definition, here.

The observation is 15 numbers:

    pose[:3] / POSE_LIMIT          3   the first joints of the current pose (scaled)
    pose mean, pose std            2   how much the pose is deformed overall
    containment r                  1   geometric distance to the manifold surface
    (centre - pelvis) / 0.5        3   where the manifold sits relative to the body
    semi-axes                      3   how big it is
    axis                           3   which way its spine points

Enough to describe the task and nothing more, so the same policy runs in the batched geometric
environment, in the SONIC-in-the-loop trainer, and in a viewer.
"""

from __future__ import annotations

import numpy as np

POSE_DIM = 29
POSE_LIMIT = 1.4
OBS_DIM = 15


def action_to_pose(action, torch=None):
    """Network output -> pose delta, the single definition used by training and deployment.

    `tanh` rather than a clamp: clamping saturates, so once the raw output drifts past the limit
    it receives no gradient telling it to come back (measured: a cloned policy's output reached
    2.2, was clamped to 1.0, and posed nothing like the demonstration it was fitted to).

    Accepts a tensor or an array and returns the same kind, so callers do not each need their own
    conversion (that is how the action scale drifted apart between trainers before).
    """
    torch = torch if torch is not None else __import__("torch")
    if not isinstance(action, torch.Tensor):
        return np.tanh(np.asarray(action)) * POSE_LIMIT
    return torch.tanh(action) * POSE_LIMIT


def observation(pose: np.ndarray, primitive, radius: float, pelvis: np.ndarray) -> np.ndarray:
    """The 15-dim observation for one pose and one manifold primitive.

    `primitive` is an `EllipsoidManifold` primitive (centre, semi, quat); `radius` is the
    containment of `pose` in that manifold; `pelvis` is the pelvis position in the pose's frame.
    """
    return np.concatenate([
        np.asarray(pose)[:3] / POSE_LIMIT,
        [float(np.mean(pose)), float(np.std(pose))],
        [float(radius)],
        (np.asarray(primitive.center) - np.asarray(pelvis)) / 0.5,
        np.asarray(primitive.semi),
        np.asarray(primitive.axis()),
    ]).astype(np.float32)
