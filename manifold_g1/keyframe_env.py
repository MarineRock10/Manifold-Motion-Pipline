"""Minimal MuJoCo + frozen SONIC runner that holds a static keyframe.

No planner, no task logic: the caller chooses a pose (crouch, lean), SONIC holds it, and the
body's containment against an ellipsoid manifold is measured pelvis-relative - the pose is
judged by its shape, not by where the robot drifted.
"""

from __future__ import annotations

import numpy as np

from . import constants as C
from .env import G1FlatEnv
from .manifold import EllipsoidManifold, build_scene, update_visuals
from .reference import KeyframeReference
from .sonic import SonicController


class KeyframeEnv:
    """One robot on flat ground; SONIC holds a static keyframe until the pose changes.

    The manifold is visual only (translucent ellipsoids, no collision). Passing one at
    construction puts it in the scene; `set_manifold` then reshapes it in place, so a
    viewer session can change the manifold without rebuilding the model.
    """

    def __init__(self, manifold: EllipsoidManifold | None = None, scene_path=None):
        if scene_path is None:
            scene_path = C.FLAT_SCENE if manifold is None else build_scene(manifold)
        self.env = G1FlatEnv(scene_path)
        self.controller = SonicController()
        self.reference: KeyframeReference | None = None

    def set_manifold(self, manifold: EllipsoidManifold, follow: np.ndarray | None = None) -> None:
        """Reshape the visual ellipsoids; `follow` keeps them centred on a pelvis position."""
        update_visuals(self.env.model, manifold, follow=follow)

    def reset(self, x: float = 0.0) -> None:
        self.env.reset(x=x)
        self.controller.reset()
        self.reference = KeyframeReference.standing(self.env.state()["base_quat"])

    def set_joints(self, q_policy: np.ndarray) -> None:
        """Hold an explicit 29-joint pose (policy order) - the primitive model's output space."""
        assert self.reference is not None, "call reset() first"
        self.reference.set_joints(q_policy)

    def set_pose(self, crouch: float, lean: float = 0.0, twist: float = 0.0,
                 arms: float = 0.0) -> None:
        assert self.reference is not None, "call reset() first"
        self.reference.set_pose(crouch, lean, twist, arms)

    def step(self) -> None:
        state = self.env.state()
        self.controller.append_state(state["q_hw"], state["dq_hw"], state["base_quat"],
                                     state["base_ang_vel"])
        _, q_des, _ = self.controller.act(self.reference, state["base_quat"])
        self.env.set_target(q_des)
        self.env.step()
        self.reference.advance()

    def landmarks(self, names: list[str]) -> dict[str, np.ndarray]:
        """World positions of the named bodies (`head` is virtual: torso + 0.30 m)."""
        import mujoco

        model, data = self.env.model, self.env.data
        out = {}
        for name in names:
            if name == "head":
                torso = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")]
                out[name] = torso + np.array([0.0, 0.0, 0.30])
            else:
                out[name] = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()
        return out

    def state(self) -> dict:
        return self.env.state()
