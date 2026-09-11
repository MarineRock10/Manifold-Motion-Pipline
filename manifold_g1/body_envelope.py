"""Robot body envelopes from the MJCF: the simplest manifold is the standing envelope.

Samples the robot's surface (mesh vertices + primitive geoms) in a given pose, fits an
axis-aligned ellipsoid around the pelvis, and can sweep the crouch amount to calibrate how
the envelope shrinks. Those numbers define the static manifold task: "given an ellipsoid,
choose a pose that fits inside it".

    python3 -m manifold_g1.body_envelope                      # standing envelope
    python3 -m manifold_g1.body_envelope --crouch 0,0.4,0.8,1.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from . import constants as C
from .env import G1FlatEnv
from .reference import CROUCH_DIRECTION


def body_points(model: mujoco.MjModel, data: mujoco.MjData, max_vertices: int = 400) -> np.ndarray:
    """Surface samples of the robot (group-0 geoms) in world coordinates."""
    points = []
    for gid in range(model.ngeom):
        if model.geom_group[gid] != 0:
            continue
        pos, mat = data.geom_xpos[gid], data.geom_xmat[gid].reshape(3, 3)
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            start = model.mesh_vertadr[mid]
            count = model.mesh_vertnum[mid]
            verts = model.mesh_vert[start:start + count].reshape(-1, 3)
            if len(verts) > max_vertices:
                verts = verts[:: max(1, len(verts) // max_vertices)]
            points.append(verts @ mat.T + pos)
        else:
            size = np.array(model.geom_size[gid])
            gtype = model.geom_type[gid]
            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                half = np.repeat(size[0], 3)
            elif gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                half = np.array([size[0], size[0], size[1]])
            elif gtype in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                half = size
            else:
                half = np.zeros(3)
            corners = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            points.append(corners @ mat.T + pos)
    return np.concatenate(points)


def fit_ellipsoid(points: np.ndarray, center: np.ndarray | None = None) -> dict:
    """Axis-aligned ellipsoid containing every sample point (inflate-until-contained).

    Centred on the point-cloud centroid by default. Starting from the per-axis half-extents
    and scaling them up by the worst normalized radius gives r = 1 on the surface without
    needing a full minimum-volume fit.
    """
    center = points.mean(axis=0) if center is None else np.asarray(center, dtype=np.float64)
    rel = points - center
    semi = np.max(np.abs(rel), axis=0)
    scale = float(np.max(np.linalg.norm(rel / semi, axis=1)))
    semi = semi * scale
    return {
        "center": [float(v) for v in center],
        "semi": [float(v) for v in semi],
        "radius_max": float(np.max(np.linalg.norm(rel / semi, axis=1))),
    }


def fit_floor_ellipsoid(points: np.ndarray) -> dict:
    """Floor-standing ellipsoid (centre at z = semi_z) that contains every sample point.

    This is the convention the manifold task uses: the ellipsoid sits on the ground and the
    robot must stay inside it. `semi_z` is the top of the robot; the horizontal semi-axes are
    chosen so that every sample satisfies r <= 1.
    """
    top = float(points[:, 2].max())
    cz = top
    height_norm = (points[:, 2] - cz) / cz                     # in [-1, 0]
    shrink = np.sqrt(np.maximum(1.0 - height_norm ** 2, 1e-3))  # local half-width factor
    semi_x = float(np.max(np.abs(points[:, 0]) / shrink))
    semi_y = float(np.max(np.abs(points[:, 1]) / shrink))
    rel = points - np.array([0.0, 0.0, cz])
    semi = np.array([semi_x, semi_y, cz])
    return {
        "top_z": top,
        "semi": [float(semi_x), float(semi_y), float(cz)],
        "center": [0.0, 0.0, cz],
        "radius_max": float(np.max(np.linalg.norm(rel / semi, axis=1))),
    }


def envelope_for_crouch(env: G1FlatEnv, crouch: float, settle: float = 2.0) -> dict:
    """Stand in the given crouch pose (closed loop with SONIC) and fit the body ellipsoid."""
    from .reference import CrouchReference
    from .sonic import SonicController

    controller = SonicController()
    controller.reset()
    env.reset(x=0.0)
    reference = CrouchReference.static_stand(env.state()["base_quat"])
    reference.set_amount(crouch)
    for _ in range(int(settle / C.CONTROL_DT)):
        st = env.state()
        controller.append_state(st["q_hw"], st["dq_hw"], st["base_quat"], st["base_ang_vel"])
        _, q_des, _ = controller.act(reference, st["base_quat"])
        env.set_target(q_des)
        env.step()
    st = env.state()
    points = body_points(env.model, env.data)
    fit = fit_ellipsoid(points)
    fit["crouch_amount"] = float(crouch)
    fit["pelvis_z"] = float(st["base_pos"][2])
    fit["top_z"] = float(points[:, 2].max())

    # landmark body model: pelvis, torso(+virtual head), wrists, knees, ankles
    names = ["pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
             "left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link"]
    landmarks = {}
    for name in names:
        bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            landmarks[name] = [float(v) for v in env.data.xpos[bid]]
    if "torso_link" in landmarks:
        landmarks["head_virtual"] = [landmarks["torso_link"][0], landmarks["torso_link"][1],
                                     landmarks["torso_link"][2] + 0.30]
    fit["landmarks"] = landmarks
    return fit


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit the robot's body envelope from the MJCF")
    parser.add_argument("--crouch", default="0", help="comma-separated crouch amounts to sweep")
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_g1/body_envelope.json"))
    args = parser.parse_args()

    env = G1FlatEnv()
    env.pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    results = [envelope_for_crouch(env, float(c), args.settle) for c in args.crouch.split(",")]

    print(f"{'crouch':>7} {'pelvis_z':>9} {'top_z':>7} | body envelope: "
          f"{'center_z':>9} {'semi_x':>7} {'semi_y':>7} {'semi_z':>7}")
    for r in results:
        print(f"{r['crouch_amount']:>7.2f} {r['pelvis_z']:>9.3f} {r['top_z']:>7.3f} | "
              f"{r['center'][2]:>9.3f} {r['semi'][0]:>7.3f} {r['semi'][1]:>7.3f} {r['semi'][2]:>7.3f}")

    standing = results[0]
    print(f"\nstanding envelope = the simplest manifold (contains every sampled body point, "
          f"r_max = {standing['radius_max']:.3f}):")
    print(f"  EllipsoidManifold.single(semi_x={standing['semi'][0]:.3f}, "
          f"semi_y={standing['semi'][1]:.3f}, semi_z={standing['semi'][2]:.3f}, "
          f"center_z={standing['center'][2]:.3f})")
    print("  the robot stands inside it with no pose change needed (r = 1 on the surface)")
    print("  squashing semi_z below the standing value forces a crouch; the sweep above "
          "gives the body envelope for each crouch amount")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"envelopes": results, "crouch_direction": CROUCH_DIRECTION.tolist()},
                                   indent=2))
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
