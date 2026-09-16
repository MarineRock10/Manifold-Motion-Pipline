"""Ellipsoid manifolds as *soft* conditioning geometry.

There is no collision hull for the manifold: the robot stands on flat ground and the
manifold exists only as (a) translucent visual ellipsoids and (b) analytic terms in the
observation / reward:

  containment   r = normalized radius; r > 1 means the body point left the manifold and is
                penalized (soft constraint),
  spine axis    the local +z axis of the primitive is the direction the robot's torso should
                align with, which is how a tilted or flattened manifold shapes the body.

Because nothing collides with the ellipsoids, changing the manifold never requires rebuilding
the MuJoCo model - it is pure data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import constants as C

SCENE_DIR = C.REPO / "data" / "g1_flat"
SCENE_PATH = SCENE_DIR / "scene_manifold.xml"

_SCENE_TEMPLATE = """<mujoco model="g1 soft manifold">
  <include file="g1_29dof_with_hand.xml"/>

  <statistic center="0 0 0.8" extent="{extent:.1f}"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20" offwidth="1280" offheight="960"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="manifold_mat" rgba="0.25 0.6 0.95 0.14"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane" group="1"/>
{ellipsoids}
  </worldbody>
</mujoco>
"""

_ELLIPSOID_GEOM = ('    <geom name="ellipsoid_{i}" type="ellipsoid" size="{a:.6g} {b:.6g} {c:.6g}" '
                   'pos="{x:.6g} {y:.6g} {z:.6g}" quat="{qw:.6g} {qx:.6g} {qy:.6g} {qz:.6g}" '
                   'material="manifold_mat" group="1" contype="0" conaffinity="0"/>')


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class Primitive:
    """One ellipsoid: centre, orientation (wxyz) and semi-axes in its local frame."""

    center: np.ndarray
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    semi: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))

    def radius(self, points: np.ndarray) -> np.ndarray:
        """Normalized radius of world points against this ellipsoid (1 = surface)."""
        rel = np.asarray(points, dtype=np.float64) - self.center
        local = rel @ quat_to_matrix(self.quat)  # rows are world vectors in the local frame
        return np.linalg.norm(local / self.semi, axis=-1)

    def axis(self) -> np.ndarray:
        """World direction of the local +z axis (the body/spine direction of this primitive)."""
        return quat_to_matrix(self.quat)[:, 2]


@dataclass
class EllipsoidManifold:
    """An ellipsoid the robot's body should stay inside of."""

    primitives: list[Primitive]

    def radii(self, points: np.ndarray) -> np.ndarray:
        """Normalized radius per point (1 = on the surface, < 1 inside)."""
        return np.min(np.stack([p.radius(points) for p in self.primitives]), axis=0)

    def scene_xml(self) -> str:
        body = []
        for i, primitive in enumerate(self.primitives):
            q = primitive.quat
            body.append(_ELLIPSOID_GEOM.format(
                i=i, a=primitive.semi[0], b=primitive.semi[1], c=primitive.semi[2],
                x=primitive.center[0], y=primitive.center[1], z=primitive.center[2],
                qw=q[0], qx=q[1], qy=q[2], qz=q[3],
            ))
        extent = 2.0 + float(np.max([p.center[0] + p.semi[0] for p in self.primitives]))
        return _SCENE_TEMPLATE.format(extent=extent, ellipsoids="\n".join(body))


def build_scene(manifold: EllipsoidManifold, path: Path = SCENE_PATH) -> Path:
    """Write the flat-ground scene with the (visual-only) ellipsoids."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifold.scene_xml())
    return path


def update_visuals(model: mujoco.MjModel, manifold: EllipsoidManifold,
                   follow: np.ndarray | None = None) -> None:
    """Move the visual ellipsoids in place; they do not collide, so no model rebuild.

    Manifolds are expressed in the **pelvis frame** (that is the frame containment is judged in,
    and why `family.BASE_CENTER` sits at z = -0.13 m relative to the pelvis). To draw one, pass
    the pelvis position as `follow` and the whole offset - including z - is applied; omitting it
    draws the ellipsoid as if the pelvis were at the origin, which puts it under the floor.
    """
    offset = np.zeros(3) if follow is None else np.asarray(follow, dtype=np.float64)[:3]
    for i, primitive in enumerate(manifold.primitives):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"ellipsoid_{i}")
        if gid < 0:
            raise KeyError(f"ellipsoid_{i} not found; the scene was built for a different count")
        model.geom_size[gid] = primitive.semi
        model.geom_pos[gid] = primitive.center + offset
        model.geom_quat[gid] = primitive.quat
        model.geom_rbound[gid] = float(np.linalg.norm(primitive.semi))
