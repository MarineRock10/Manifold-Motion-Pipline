"""Robot body envelope and landmark calibration (the static stage's body model).

Samples the robot's surface (mesh vertices) and its landmark bodies while SONIC holds a
keyframe, fits a body ellipsoid around the point cloud, and writes the calibration table used
by the static pipeline:

  * the standing envelope - the simplest manifold the robot trivially fits inside,
  * landmark positions over a (crouch, lean, twist, arms) grid, which is how the geometric
    body model predicts containment.

Landmarks are the body's *extremities*: for each landmark body the calibration stores the
local-frame point of its farthest mesh vertex from the pelvis (the fingertip for a hand, the
top of the head for the torso). A landmark therefore tracks the real surface through any pose,
which is what makes a laterally narrow manifold meaningful.

    python3 -m manifold_g1.body_envelope --crouch 0,0.4,0.8,1.2,1.6,2.0 --lean 0,0.5,1.0 \
        --twist="-1.2,-0.6,0,0.6,1.2" --arms 0,0.5,1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from . import constants as C
from .keyframe_env import KeyframeEnv

LANDMARK_BODIES = ("pelvis", "torso_link",
                   "left_hand_index_1_link", "right_hand_index_1_link",
                   "left_knee_link", "right_knee_link",
                   "left_ankle_roll_link", "right_ankle_roll_link")


# paired joints that should move together (left/right of the same joint), in policy order
_PAIR_SUFFIXES = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
                  "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw")


def _pairs():
    from . import constants as C
    rows = []
    for suffix in _PAIR_SUFFIXES:
        left, right = f"left_{suffix}", f"right_{suffix}"
        if left in C.MOTOR_NAMES and right in C.MOTOR_NAMES:
            rows.append((int(C.ISAACLAB_TO_MUJOCO[C.MOTOR_NAMES.index(left)]),
                         int(C.ISAACLAB_TO_MUJOCO[C.MOTOR_NAMES.index(right)])))
    return np.array(rows)


_PAIR_INDEX = _pairs()


def body_points(model: mujoco.MjModel, data: mujoco.MjData, max_vertices: int = 400) -> np.ndarray:
    """Surface samples of the robot in world coordinates.

    Meshes are sampled on their vertices; the other primitive geoms are sampled *on their
    surface* rather than at their bounding-box corners, because a capsule's box corners sit at
    radius*sqrt(2) from its axis and would inflate the envelope by up to 40% of its radius.
    """
    points = []
    world = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "world")
    for gid in range(model.ngeom):
        # group-0 geoms are the robot's own surfaces; the floor lives on the `world` body and
        # is also group 0, so filter by body too - otherwise the ground plane enters the
        # envelope and pins its lowest point to the pelvis height
        if model.geom_group[gid] != 0 or model.geom_bodyid[gid] == world:
            continue
        pos, mat = data.geom_xpos[gid], data.geom_xmat[gid].reshape(3, 3)
        gtype = model.geom_type[gid]
        size = np.array(model.geom_size[gid])
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            verts = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]
                                    + model.mesh_vertnum[mid]]
            verts = verts.reshape(-1, 3)
            if len(verts) > max_vertices:
                verts = verts[:: max(1, len(verts) // max_vertices)]
            points.append(verts @ mat.T + pos)
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            points.append(_sphere_points(size[0]) @ mat.T + pos)
        elif gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            # axis along local z: two end caps (capsule) or two rims (cylinder), sampled on the
            # surface with the true radius
            points.append(_cylinder_points(size[0], size[1], capsule=(
                gtype == mujoco.mjtGeom.mjGEOM_CAPSULE)) @ mat.T + pos)
        elif gtype in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            points.append(_box_points(size) @ mat.T + pos)
        # planes and other non-surfaces carry no body extent
    return np.concatenate(points)


def _sphere_points(radius: float, n_theta: int = 8, n_phi: int = 8) -> np.ndarray:
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    return radius * np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                              np.cos(th)], axis=-1).reshape(-1, 3)


def _cylinder_points(radius: float, half_length: float, capsule: bool,
                     n_sides: int = 16, n_caps: int = 4) -> np.ndarray:
    phi = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    ring = radius * np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1)
    points = [ring + np.array([0.0, 0.0, half_length]), ring - np.array([0.0, 0.0, half_length])]
    if capsule:                                   # hemispherical caps, sampled on the surface
        for sign in (1.0, -1.0):
            theta = np.linspace(0, np.pi / 2, n_caps)
            for t in theta:
                points.append(ring * np.cos(t) + np.array([0.0, 0.0, sign * (half_length
                                                                             + radius * np.sin(t))]))
    return np.concatenate(points)


def _box_points(size: np.ndarray) -> np.ndarray:
    return np.array([[sx * size[0], sy * size[1], sz * size[2]]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def _mesh_vertices(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    verts = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != body_id or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        local = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        mat = data.geom_xmat[gid].reshape(3, 3)
        verts.append(local.reshape(-1, 3) @ mat.T + data.geom_xpos[gid])
    return np.concatenate(verts) if verts else np.zeros((0, 3))


def landmark_offsets(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, list[float]]:
    """Local-frame extremity point of every landmark body, taken at the current pose."""
    pelvis = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
    offsets: dict[str, list[float]] = {}
    for name in LANDMARK_BODIES:
        if name == "pelvis":
            # the pelvis stays at its own origin: it is the reference the others are measured
            # against, so a surface extremity here would just shift every comparison
            offsets[name] = [0.0, 0.0, 0.0]
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        verts = _mesh_vertices(model, data, bid)
        if len(verts) == 0:
            offsets[name] = [0.0, 0.0, 0.0]
            continue
        extreme = verts[int(np.argmax(np.linalg.norm(verts - pelvis, axis=1)))]
        local = data.xmat[bid].reshape(3, 3).T @ (extreme - data.xpos[bid])
        offsets[name] = [float(v) for v in local]
    return offsets


def read_landmarks(model: mujoco.MjModel, data: mujoco.MjData,
                   offsets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """World positions of the landmark extremity points."""
    out = {}
    for name, offset in offsets.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        out[name] = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ np.asarray(offset)
    return out


def fit_ellipsoid(points: np.ndarray, center: np.ndarray | None = None) -> dict:
    """Axis-aligned ellipsoid containing every sample (inflate the per-axis half-extents)."""
    center = points.mean(axis=0) if center is None else np.asarray(center, dtype=np.float64)
    rel = points - center
    semi = np.max(np.abs(rel), axis=0)
    semi = semi * float(np.max(np.linalg.norm(rel / semi, axis=1)))
    return {"center": [float(v) for v in center], "semi": [float(v) for v in semi],
            "radius_max": float(np.max(np.linalg.norm(rel / semi, axis=1)))}


def measure(env: KeyframeEnv, crouch: float, lean: float, twist: float, arms: float,
            offsets: dict[str, np.ndarray], settle: float = 2.0, protocol: str = "jump") -> dict:
    """Reach the keyframe, then record the body envelope and landmarks.

    `protocol` matters: the frozen controller is path dependent, so a pose reached by slewing
    (what deployment does) can differ from the same pose reached by jumping to it from a fresh
    reset. Calibrate with the protocol you intend to deploy with.
    """
    env.reset()
    target = np.array([float(crouch), float(lean), float(twist), float(arms)])
    if protocol == "slew":
        pose = np.zeros(4)
        for _ in range(int(settle / C.CONTROL_DT)):
            pose += (target - pose) * min(1.0, C.CONTROL_DT / 0.45)
            env.set_pose(*pose)
            env.step()
        for _ in range(int(1.0 / C.CONTROL_DT)):
            env.step()
    else:
        env.set_pose(crouch, lean, twist, arms)
        for _ in range(int(settle / C.CONTROL_DT)):
            env.step()
    state = env.state()
    points = body_points(env.env.model, env.env.data)
    fit = fit_ellipsoid(points)
    landmarks = read_landmarks(env.env.model, env.env.data, offsets)
    fit.update({
        "crouch": float(crouch), "lean": float(lean), "twist": float(twist), "arms": float(arms),
        "pelvis_z": float(state["base_pos"][2]),
        "top_z": float(points[:, 2].max()),
        "landmarks": {k: [float(v) for v in p] for k, p in landmarks.items()},
    })
    return fit


def main() -> int:
    parser = argparse.ArgumentParser(description="Body envelope / landmark calibration")
    parser.add_argument("--crouch", default="0", help="comma-separated crouch amounts")
    parser.add_argument("--lean", default="0", help="comma-separated lean amounts")
    parser.add_argument("--twist", default="0", help="comma-separated twist amounts")
    parser.add_argument("--arms", default="0", help="comma-separated arm-tuck amounts")
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--protocol", choices=("jump", "slew"), default="slew",
                        help="jump: set the keyframe from a fresh reset; slew: approach it the "
                             "way the viewer and verify do (the frozen controller is path dependent)")
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_g1/body_envelope.json"))
    args = parser.parse_args()

    env = KeyframeEnv()
    env.reset()
    for _ in range(int(1.0 / C.CONTROL_DT)):          # settle standing before taking offsets
        env.step()
    offsets = {k: np.array(v) for k, v in landmark_offsets(env.env.model, env.env.data).items()}

    rows = [measure(env, float(c), float(l), float(t), float(a), offsets, args.settle, args.protocol)
            for c in args.crouch.split(",") for l in args.lean.split(",")
            for t in args.twist.split(",") for a in args.arms.split(",")]

    print(f"{'crouch':>7} {'lean':>6} {'twist':>6} {'arms':>6} {'pelvis_z':>9} {'top_z':>7} "
          f"{'semi_x':>7} {'semi_y':>7} {'semi_z':>7} {'|y|max':>7}")
    for r in rows:
        semi = r["semi"]
        ymax = max(abs(p[1]) for p in r["landmarks"].values())
        print(f"{r['crouch']:>7.2f} {r['lean']:>6.2f} {r['twist']:>6.2f} {r['arms']:>6.2f} "
              f"{r['pelvis_z']:>9.3f} {r['top_z']:>7.3f} {semi[0]:>7.3f} {semi[1]:>7.3f} "
              f"{semi[2]:>7.3f} {ymax:>7.3f}")

    standing = next(r for r in rows if r["crouch"] == 0.0 and r["lean"] == 0.0 and r["twist"] == 0.0)
    print(f"\nstanding envelope = the simplest manifold (contains every sampled point, "
          f"r = {standing['radius_max']:.3f}):")
    print(f"  centre_z {standing['center'][2]:.3f}  semi ({standing['semi'][0]:.3f}, "
          f"{standing['semi'][1]:.3f}, {standing['semi'][2]:.3f})  top {standing['top_z']:.3f}")
    print(f"  head authority:  top_z {standing['top_z']:.3f} -> "
          f"{min(r['top_z'] for r in rows):.3f} m")
    y_base = max(abs(p[1]) for p in standing["landmarks"].values())
    y_best = min(max(abs(p[1]) for p in r["landmarks"].values()) for r in rows)
    x_base = max(abs(p[0]) for p in standing["landmarks"].values())
    x_best = min(max(abs(p[0]) for p in r["landmarks"].values()) for r in rows)
    print(f"  landmark |y| {y_base:.3f} -> {y_best:.3f} m, |x| {x_base:.3f} -> {x_best:.3f} m")
    deep = min(r["pelvis_z"] for r in rows)
    print(f"  lowest pelvis over the grid: {deep:.3f} m (standing {standing['pelvis_z']:.3f} m)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"rows": rows, "landmark_offsets": {k: list(v) for k, v in offsets.items()},
         "crouch": args.crouch, "lean": args.lean, "twist": args.twist,
         "arms": args.arms, "protocol": args.protocol}, indent=2))
    print(f"saved {args.out} ({len(rows)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
