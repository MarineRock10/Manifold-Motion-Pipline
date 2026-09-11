"""Procedural manifold geometry.

An axis-aligned corridor over flat ground (length L along +x, inner width W) whose ceiling
height ramps linearly from `height_start` at the entrance to `height_goal` at the far end.
The free space is visualised as a chain of translucent ellipsoids (the manifold primitives
of the research plan) whose z-semi-axis follows the local ceiling height.

The box walls/ceiling are the collision hull; the ellipsoid chain is the conditioning
geometry used by the policy observation and the compliance metric. All manifold geoms live
in geom group 1 so a renderer can show the manifold alone (group 1) or the full scene.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from . import constants as C

SCENE_DIR = C.REPO / "data" / "g1_flat"
SCENE_PATH = SCENE_DIR / "scene_manifold.xml"

_SCENE_HEADER = """<mujoco model="g1 manifold corridor">
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
    <material name="wall_mat" rgba="0.65 0.65 0.7 1"/>
    <material name="ceiling_mat" rgba="0.75 0.75 0.8 1"/>
    <material name="manifold_mat" rgba="0.25 0.6 0.95 0.16"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane" group="1"/>
"""

_SLAB_GEOMS = """    <geom name="wall_left_{i}" type="box" size="{half_slab:.9g} {half_t:.9g} {half_h:.9g}"
          pos="{cx:.9g} {left_y:.9g} {half_h:.9g}" material="wall_mat" group="1"/>
    <geom name="wall_right_{i}" type="box" size="{half_slab:.9g} {half_t:.9g} {half_h:.9g}"
          pos="{cx:.9g} {right_y:.9g} {half_h:.9g}" material="wall_mat" group="1"/>
    <geom name="ceiling_{i}" type="box" size="{half_slab:.9g} {half_w_outer:.9g} {half_t:.9g}"
          pos="{cx:.9g} 0 {ceiling_z:.9g}" material="ceiling_mat" group="1"/>
    <geom name="ellipsoid_{i}" type="ellipsoid" size="{semi_x:.9g} {half_w:.9g} {half_h:.9g}"
          pos="{cx:.9g} 0 {half_h:.9g}" material="manifold_mat" group="1" contype="0" conaffinity="0"/>"""


@dataclass
class ManifoldSpec:
    """Corridor manifold: high entrance section, then a lower ceiling section."""

    length: float = 6.0
    width: float = 2.0
    height_start: float = 1.5     # ceiling height at the entrance (x = -L/2)
    height_goal: float = 1.5      # ceiling height of the low section (x >= ramp_end_x)
    start_x: float = -2.0
    goal_x: float = 2.0
    wall_thickness: float = 0.1
    slabs: int = 9
    ramp_end_x: float = 0.0       # x where the ceiling reaches height_goal

    @property
    def min_height(self) -> float:
        return float(min(self.height_start, self.height_goal))

    def grid(self) -> tuple[float, float, float, float]:
        return (self.length, self.width, self.height_start, self.height_goal)

    def slab_edges(self) -> np.ndarray:
        return np.linspace(-0.5 * self.length, 0.5 * self.length, self.slabs + 1)

    def slab_centers(self) -> np.ndarray:
        edges = self.slab_edges()
        return 0.5 * (edges[:-1] + edges[1:])

    def height_at(self, x):
        """Local ceiling height at world x: ramp from height_start down to height_goal."""
        span = max(self.ramp_end_x + 0.5 * self.length, 1e-6)
        fraction = np.clip((np.asarray(x, dtype=np.float64) + 0.5 * self.length) / span, 0.0, 1.0)
        return self.height_start + (self.height_goal - self.height_start) * fraction

    def slab_heights(self) -> np.ndarray:
        return np.asarray(self.height_at(self.slab_centers()), dtype=np.float64)

    def ellipsoids(self) -> list[tuple[float, float, float, float]]:
        """Ellipsoid chain as (x, semi_x, semi_y, semi_z); semi_z follows the local ceiling."""
        centers = self.slab_centers()
        heights = self.slab_heights()
        semi_x = 0.7 * self.length / max(self.slabs, 1)
        return [(float(x), semi_x, 0.5 * self.width, 0.5 * float(h))
                for x, h in zip(centers, heights)]

    def ellipse_radius(self, y, z, x):
        """Normalized radius of points against the local free-space ellipse (1 = boundary)."""
        height = np.asarray(self.height_at(x), dtype=np.float64)
        return np.sqrt((np.asarray(y) / (0.5 * self.width)) ** 2
                       + ((np.asarray(z) - 0.5 * height) / (0.5 * np.maximum(height, 0.2))) ** 2)

    def xml(self) -> str:
        half_t = 0.5 * self.wall_thickness
        edges = self.slab_edges()
        centers = self.slab_centers()
        heights = self.slab_heights()
        half_slab = 0.5 * (edges[1] - edges[0])
        semi_x = 0.7 * self.length / max(self.slabs, 1)

        body = [_SCENE_HEADER.format(extent=self.length + 2.0)]
        for index, (cx, height) in enumerate(zip(centers, heights)):
            body.append(_SLAB_GEOMS.format(
                i=index, cx=float(cx), half_slab=half_slab, half_t=half_t,
                half_h=0.5 * float(height), left_y=-(0.5 * self.width + half_t),
                right_y=0.5 * self.width + half_t,
                half_w_outer=0.5 * self.width + self.wall_thickness,
                ceiling_z=float(height) + half_t, semi_x=semi_x,
                half_w=0.5 * self.width,
            ))
        body.append(f'    <geom name="wall_back" type="box" size="{half_t:.3f} '
                    f'{0.5 * self.width + self.wall_thickness:.3f} {0.5 * self.height_start:.3f}" '
                    f'pos="{self.start_x - 0.8:.3f} 0 {0.5 * self.height_start:.3f}" '
                    f'material="wall_mat" group="1"/>')
        body.append("  </worldbody>\n</mujoco>\n")
        return "\n".join(body)


def build_scene(spec: ManifoldSpec, path: Path = SCENE_PATH) -> Path:
    """Write the scene XML next to the robot model so <include> resolves."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.xml())
    return path


def apply_profile(model: mujoco.MjModel, spec: ManifoldSpec, data: mujoco.MjData | None = None) -> None:
    """Update the slab walls, ceilings and ellipsoid chain in place for a new height profile.

    Every manifold geom's z half-size and z position depend only on the local ceiling height,
    so the model can be re-shaped without rebuilding the scene.
    """
    def geom_id(name: str) -> int:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise KeyError(f"geom '{name}' not found in model")
        return gid

    half_t = 0.5 * spec.wall_thickness
    heights = spec.slab_heights()
    edges = spec.slab_edges()
    half_slab = 0.5 * (edges[1] - edges[0])
    for index, height in enumerate(heights):
        half_h = 0.5 * float(height)
        for name in (f"wall_left_{index}", f"wall_right_{index}", f"ellipsoid_{index}"):
            gid = geom_id(name)
            model.geom_size[gid, 2] = half_h
            model.geom_pos[gid, 2] = half_h
        ceiling = geom_id(f"ceiling_{index}")
        model.geom_pos[ceiling, 2] = float(height) + half_t
        model.geom_size[ceiling, 2] = half_t
    back = geom_id("wall_back")
    model.geom_size[back, 2] = 0.5 * spec.height_start
    model.geom_pos[back, 2] = 0.5 * spec.height_start

    # geom_rbound is computed at compile time and mj_setConst does not refresh it, yet the
    # broad phase and ray casts use it; stale bounds make the reshaped scene behave differently
    # from an equivalent scene built directly from XML.
    for index in range(len(heights)):
        for name in (f"wall_left_{index}", f"wall_right_{index}",
                     f"ceiling_{index}", f"ellipsoid_{index}"):
            gid = geom_id(name)
            if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                model.geom_rbound[gid] = float(np.linalg.norm(model.geom_size[gid]))
            elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                model.geom_rbound[gid] = float(np.max(model.geom_size[gid]))
    model.geom_rbound[back] = float(np.linalg.norm(model.geom_size[back]))

    if data is not None:
        # mj_setConst derives solver constants (body_invweight0, dof_invweight0, ...) from the
        # configuration currently held in `data`, so it must see the reference configuration --
        # otherwise the dynamics silently depend on the pose the previous episode ended in.
        mujoco.mj_resetData(model, data)
        mujoco.mj_setConst(model, data)
        mujoco.mj_forward(model, data)
