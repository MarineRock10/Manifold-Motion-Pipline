"""Shared drawing helpers for the MuJoCo viewers.

The viewers each render "the pose the policy asked for" as a point cloud next to the robot the
controller actually produced. That point cloud was written three times (`show_bc`, `show_rl`,
and before them `show_pose`/`show_primitive`) with different sampling strides and slightly
different docstrings - which is how a viewer ends up showing a different number of points than
the one next to it despite claiming to show the same thing.

One definition each:

  * `pose_points`  - the body surface of a pose, in the robot's frame
  * `mark_pose`    - draw it into a viewer's user scene
  * `settle`       - run the policy's relaxation loop the way the environment does
"""

from __future__ import annotations

import numpy as np

from .pose_policy import POSE_DIM, POSE_LIMIT


def pose_points(body, pose: np.ndarray, pelvis: np.ndarray, max_vertices: int = 16) -> np.ndarray:
    """Body-surface points of `pose`, translated to the robot's current position.

    `body` is a `body_model.BodyModel`; `pelvis` the displayed robot's pelvis position, so the
    cloud lands on the robot instead of at the origin.
    """
    return body.mesh_points_pose(pose, max_vertices=max_vertices) + np.asarray(pelvis)


def mark_pose(scene, body, pose: np.ndarray, pelvis: np.ndarray, rgb, stride: int = 100,
              size: float = 0.011) -> None:
    """Draw `pose` as spheres in a viewer's user scene (`scene` is `viewer.user_scn`)."""
    import mujoco

    points = pose_points(body, pose, pelvis)
    step = max(1, len(points) // stride)
    for p in points[::step]:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0.0, 0.0]), p,
                            np.eye(3).flatten(), np.array([*rgb, 0.55]))
        scene.ngeom += 1


def settle(ppo, manifold, radius_of, steps: int = 6) -> np.ndarray:
    """The pose a policy proposes for a manifold: the environment's own relaxation loop.

    The environment integrates `pose += (tanh(action) * POSE_LIMIT - pose) * 0.5` for a few steps
    rather than taking a single action, so a viewer that just calls the policy once shows a pose
    the environment would never settle on. Deterministic: the distribution mean is what deploys,
    and sampling it shows exploration noise instead of the policy's answer.

    `radius_of(pose) -> float` measures containment. It is a parameter because the viewers
    legitimately differ: one measures through the GPU kinematics, another through MuJoCo's mesh
    points. Both agree at geom level, and sharing the loop while keeping the measurement explicit
    is better than sharing a loop that silently picks one.
    """
    import torch

    from .pose_policy import action_to_pose, observation

    pose = np.zeros(POSE_DIM)
    with torch.no_grad():
        for _ in range(steps):
            obs = observation(pose, manifold.primitives[0], float(radius_of(pose)), np.zeros(3))
            action, _, _ = ppo.act(obs, deterministic=True)
            target = np.asarray(action_to_pose(action, torch))
            pose = pose + (target - pose) * 0.5
    return pose
