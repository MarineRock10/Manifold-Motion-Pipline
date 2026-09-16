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

# The standing envelope, pelvis-anchored: the smallest axis-aligned *ellipsoid* that contains the
# sampled body surface at the standing pose, so that at scale 1.0 the body fits exactly (r = 1).
#
# It is the SAME object the demonstration set stores, because that is what the policy learned:
# `demos.pose_envelope` builds each demo's manifold as `fit_ellipsoid(mesh_points_pose(pose))`,
# and `report_specs()` must scale that same shape or verification measures a family the policy
# never saw. Both come from the same two functions (`BodyModel.mesh_points_pose` -> `fit_ellipsoid`),
# evaluated at the standing pose, so they cannot drift apart.
#
# Two things about the values, both of which matter:
#   * the centre is NOT the pelvis. It is 0.11 m below it, because most of the body is legs. An
#     earlier version centred the ellipsoid on the pelvis, which forced the semi-axes to grow
#     ~40% in x to reach the feet and changed the shape - the family then disagreed with the
#     demonstrations it was supposed to describe.
#   * the semi-axes are the *fitted* values, not the box's half-extents. Using box half-extents
#     as ellipsoid semi-axes is a different shape: the standing body scored r = 1.145 against its
#     own scale-1.0 manifold, so the `h1.00_w1.00_d1.00` row of `report_specs` was unsatisfiable
#     by standing still. Verified now: r = 1.0000.
BASE_SEMI = np.array([0.2631, 0.3792, 0.8861])
BASE_CENTER = np.array([0.0923, -0.0077, -0.1100])   # pelvis-relative, seen from the pelvis
BASE_CENTER_WORLD = BASE_CENTER + np.array([0.0, 0.0, C.DEFAULT_HEIGHT])

# Envelope -> manifold margin, shared with the demonstration set (`demos.MARGIN`). A manifold is
# the body envelope times this, so a pose that exactly fills the body reads r = 1/1.05 = 0.952 in
# it, and a policy can hold a pose without sitting on the boundary. The demonstrations, the
# batched environment and `report_specs` all use this one value: when `report_specs` omitted it,
# verification measured a family 5% tighter than anything the policy was trained on, and every
# result read ~5% high (predicted 0.98 * 1.05 = 1.029, measured 1.030).
MARGIN = 1.05


@dataclass(frozen=True)
class ManifoldSpec:
    """A manifold in the family: the standing body envelope scaled and shifted."""

    height: float = 1.0        # vertical semi-axis scale
    width: float = 1.0         # lateral semi-axis scale
    depth: float = 1.0         # sagittal semi-axis scale
    offset: float = 0.0        # lateral centre shift [m]
    tilt_deg: float = 0.0      # sagittal tilt of the axis (+ leans forward)

    def build(self, base_semi: np.ndarray = BASE_SEMI,
              base_center: np.ndarray = BASE_CENTER,
              margin: float = MARGIN) -> EllipsoidManifold:
        """The manifold for this spec: semi-axes scaled, centre kept where the body is.

        The scale applies to the semi-axes only. The centre is a *position* - where the body sits
        relative to the pelvis, 0.11 m below it - so scaling it would move the ellipsoid off the
        body rather than resize it; measured, that alone costs 0.005 in r. The earlier version
        scaled the centre by the spec's height/width/depth, which drifted the manifold further
        the more extreme the spec was, and the offset was then added in world terms on top.
        """
        scale = np.array([self.depth, self.width, self.height]) * margin
        semi = np.asarray(base_semi, dtype=np.float64) * scale
        center = np.asarray(base_center, dtype=np.float64) + np.array([0.0, self.offset, 0.0])
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
