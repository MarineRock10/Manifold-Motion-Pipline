"""The manifold family, defined on the extremity envelope (one ruler for model and MuJoCo).

Earlier the family was built from a *mesh-fit* envelope, while poses were judged with
*pelvis-anchored extremity landmarks* (fingertips, head, knees, toes). Those are different
objects - the mesh fit is larger (its semi-axes are 1.1-1.5x the extremity extents and its
centre sits above the pelvis) - so a standing pose scored r = 0.85 under one ruler and 1.63
under the other, and no pose could ever satisfy the ellipsoid in the 29-joint environment.

Everything now uses the extremity envelope:

    base envelope   semi = (0.272, 0.252, 0.756) m      (standing, pelvis-anchored)
    manifold        semi = base * (depth, width, height), centre = (0, offset, 0)
                    with the pelvis as the origin of the body frame

`ManifoldSpec` keeps its parameters (height, width, depth, offset, tilt_deg); only the base it
scales changes. The recorded demos and the capability map already speak this language.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from .manifold import EllipsoidManifold, Primitive

# standing extremity envelope, pelvis-anchored (see `BodyModel.at_pose(np.zeros(29))`)
# Meshed body envelope at the standing pose, in the pelvis frame: the smallest axis-aligned
# ellipsoid that contains the sampled body surface. It is what the 4-D static stage has always
# used, and its meaning is unambiguous: at scale 1.0 the body fits exactly (r ~ 1.0).
# Measured with `body_envelope.body_points`, which samples every group-0 geom on its surface
# (mesh vertices, capsule end caps, sphere shells) - not at bounding-box corners, which inflate
# a capsule by up to sqrt(2) of its radius. Both the numpy and the torch path use this ruler.
BASE_CENTER_WORLD = np.array([0.1262, 0.0035, 0.6962])
BASE_SEMI = np.array([0.2863, 0.4149, 0.9511])
BASE_CENTER = BASE_CENTER_WORLD - np.array([0.0, 0.0, C.DEFAULT_HEIGHT])   # pelvis-relative


@dataclass(frozen=True)
class ManifoldSpec:
    """A manifold in the family: the standing extremity envelope scaled and shifted."""

    height: float = 1.0        # vertical semi-axis scale
    width: float = 1.0         # lateral semi-axis scale
    depth: float = 1.0         # sagittal semi-axis scale
    offset: float = 0.0        # lateral centre shift [m]
    tilt_deg: float = 0.0      # sagittal tilt of the axis (+ leans forward)

    def build(self, base_semi: np.ndarray = BASE_SEMI,
              base_center: np.ndarray = BASE_CENTER) -> EllipsoidManifold:
        scale = np.array([self.depth, self.width, self.height])
        semi = np.asarray(base_semi, dtype=np.float64) * scale
        center = np.asarray(base_center, dtype=np.float64) * scale
        center = center + np.array([0.0, self.offset, 0.0])
        half = np.radians(self.tilt_deg) / 2.0
        quat = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
        return EllipsoidManifold([Primitive(center=center, quat=quat, semi=semi)])

    def label(self) -> str:
        return (f"h{self.height:.2f}_w{self.width:.2f}_d{self.depth:.2f}"
                f"_o{self.offset:+.02f}_t{self.tilt_deg:+.0f}")


def report_specs() -> list[ManifoldSpec]:
    """A curated spread of manifolds for evaluation, from roomy to barely feasible."""
    return [
        ManifoldSpec(1.00, 1.00, 1.00, 0.00, 0.0),    # the standing envelope itself
        ManifoldSpec(0.90, 1.00, 1.00, 0.00, 0.0),    # height: crouch
        ManifoldSpec(0.82, 1.00, 1.00, 0.00, 0.0),    # height: deep crouch
        ManifoldSpec(1.00, 0.85, 1.00, 0.00, 0.00),   # pinch: arms in
        ManifoldSpec(1.00, 1.00, 0.80, 0.00, 0.00),   # shallow: arms forward
        ManifoldSpec(0.92, 0.90, 1.00, 0.00, 0.0),    # crouch + pinch
        ManifoldSpec(0.95, 1.00, 1.00, +0.03, 0.0),   # shifted sideways
        ManifoldSpec(0.95, 1.00, 1.00, 0.00, +10.0),  # axis tilted forward
        ManifoldSpec(0.95, 1.00, 1.00, 0.00, -10.0),  # axis tilted back
        ManifoldSpec(0.88, 0.90, 0.92, +0.02, 0.0),   # everything at once
    ]
