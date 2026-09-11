"""Procedural manifold geometry.

Current implementation: an axis-aligned corridor over flat ground (length L along +x,
inner width W, ceiling height H) whose free space is visualised as a chain of
translucent ellipsoids (the manifold primitives of the research plan). The box walls
remain the collision hull; the ellipsoid chain is the conditioning geometry that the
policy observes and that the compliance metric is measured against.

All manifold geoms live in geom group 1 so a renderer can show the manifold alone
(hide group 0, i.e. the robot) or the full scene.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import constants as C

SCENE_DIR = C.REPO / "data" / "g1_flat"
SCENE_PATH = SCENE_DIR / "scene_manifold.xml"

_SCENE_TEMPLATE = """<mujoco model="g1 manifold corridor">
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

    <geom name="wall_left" type="box" size="{half_l:.3f} {half_t:.3f} {half_h:.3f}"
          pos="0 {left_y:.3f} {half_h:.3f}" material="wall_mat" group="1"/>
    <geom name="wall_right" type="box" size="{half_l:.3f} {half_t:.3f} {half_h:.3f}"
          pos="0 {right_y:.3f} {half_h:.3f}" material="wall_mat" group="1"/>
    <geom name="ceiling" type="box" size="{half_l:.3f} {half_w_outer:.3f} {half_t:.3f}"
          pos="0 0 {ceiling_z:.3f}" material="ceiling_mat" group="1"/>
    <geom name="wall_back" type="box" size="{half_t:.3f} {half_w_outer:.3f} {half_h:.3f}"
          pos="{back_x:.3f} 0 {half_h:.3f}" material="wall_mat" group="1"/>
{ellipsoid_geoms}
  </worldbody>
</mujoco>
"""

_ELLIPSOID_GEOM = ('    <geom name="ellipsoid_{i}" type="ellipsoid" '
                   'size="{sx:.3f} {sy:.3f} {sz:.3f}" pos="{x:.3f} 0 {z:.3f}" '
                   'material="manifold_mat" group="1" contype="0" conaffinity="0"/>')


@dataclass
class ManifoldSpec:
    """Axis-aligned corridor manifold. (L, W, H) are the parameters the policy observes."""

    length: float = 6.0
    width: float = 2.0
    height: float = 2.0
    start_x: float = -2.0
    goal_x: float = 2.0
    wall_thickness: float = 0.1
    ellipsoid_spacing: float = 0.75

    @property
    def grid(self) -> tuple[float, float, float]:
        return (self.length, self.width, self.height)

    def ellipsoids(self) -> list[tuple[float, float, float, float]]:
        """Ellipsoid chain along the corridor as (x, semi_x, semi_y, semi_z)."""
        count = max(3, int(round(self.length / max(self.ellipsoid_spacing, 1e-3))) + 1)
        xs = np.linspace(-0.5 * self.length, 0.5 * self.length, count)
        semi_x = self.length / (count - 1)
        return [(float(x), semi_x, 0.5 * self.width, 0.5 * self.height) for x in xs]

    def xml(self) -> str:
        half_l = 0.5 * self.length
        half_t = 0.5 * self.wall_thickness
        ellipsoid_geoms = "\n".join(
            _ELLIPSOID_GEOM.format(i=i, sx=sx, sy=sy, sz=sz, x=x, z=0.5 * self.height)
            for i, (x, sx, sy, sz) in enumerate(self.ellipsoids())
        )
        return _SCENE_TEMPLATE.format(
            extent=self.length + 2.0,
            half_l=half_l,
            half_t=half_t,
            half_h=0.5 * self.height,
            half_w_outer=0.5 * self.width + self.wall_thickness,
            left_y=-(0.5 * self.width + half_t),
            right_y=0.5 * self.width + half_t,
            ceiling_z=self.height + half_t,
            back_x=self.start_x - 0.8,
            ellipsoid_geoms=ellipsoid_geoms,
        )

    def ellipse_radius(self, y, z):
        """Normalized radius of a point against the cross-section free-space ellipse."""
        return np.sqrt((y / (0.5 * self.width)) ** 2 + ((z - 0.5 * self.height) / (0.5 * self.height)) ** 2)


def build_scene(spec: ManifoldSpec, path: Path = SCENE_PATH) -> Path:
    """Write the scene XML next to the robot model so <include> resolves."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.xml())
    return path
