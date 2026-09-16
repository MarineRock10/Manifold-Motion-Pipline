"""Look at the behaviour-cloned policy where it was actually trained, driven by SONIC.



`show_pose` evaluates over `family.report_specs()` - a hand-picked list that has nothing to do
with the manifolds the clone saw. Since the clone fits its training manifolds (r ~ 0.85 on them
per its own report) and does badly on the hand-picked ones, that comparison says more about the
distribution shift than about the policy.

This viewer draws the manifolds the clone was trained on (the recorded envelopes from
`pose_demos`), together with:

  * the pose the clone produces for that manifold,
  * the recorded pose that belongs to it,
  * both containments, so the copy quality is visible directly.

Press `S` to swap between them, `N`/`M` to step. If the clone's pose and the recorded pose
coincide, the clone works on its own distribution and any failure elsewhere is generalisation.

    python3 -m manifold_g1.show_bc --policy reports/manifold_g1/primitive_torch/bc_policy.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import constants as C
from .manifold import build_scene, update_visuals
from .primitive import POSE_DIM, load_demos
from .primitive_torch import Config, TorchPPO, TorchPrimitiveEnv, action_to_pose
from .static_fit import BodyModel

DEFAULT = C.REPO / "reports" / "manifold_g1" / "primitive_torch" / "bc_policy.pt"
SCENE = C.REPO / "data" / "g1_flat" / "scene_manifold.xml"


def collect(args) -> list[dict]:
    """One entry per training manifold: the clone's pose, the recorded pose, both radii."""
    import torch

    from .kinematics import TorchKinematics
    from .manifold import EllipsoidManifold, Primitive

    kin = TorchKinematics(device=args.device)
    body = BodyModel()
    _, all_demos = load_demos()
    demo_flat = np.concatenate(list(all_demos.values()))
    env = TorchPrimitiveEnv(kin, demo_flat, Config(), n=1, pool=1, seed=0)
    ppo = TorchPPO(env.obs().shape[1], POSE_DIM, device=args.device)
    ppo.load(args.policy)

    keys = list(all_demos)
    rng = np.random.default_rng(args.seed)
    picked = rng.choice(len(keys), min(args.count, len(keys)), replace=False)
    entries = []
    for i in picked:
        key = keys[i]
        values = np.array([float(v) for v in key.split(",")])
        semi, center = values[:3][None, :], values[3:][None, :]
        manifold = EllipsoidManifold([Primitive(center=center[0], semi=semi[0])])

        env.set_manifolds_from_arrays(semi, center, None)
        with torch.no_grad():
            # deterministic: the distribution mean is the pose the policy proposes. Sampling it
            # (what training does) instead shows the exploration noise, whose magnitude here is
            # larger than the difference between poses.
            action, _, _ = ppo.act_tensor(env.obs(), deterministic=True)
            pose = action_to_pose(action, torch)[0].cpu().numpy()
        recorded = all_demos[key][np.argmin(np.abs(all_demos[key]).mean(axis=1))]

        entries.append({
            "manifold": manifold, "semi": semi[0], "center": center[0],
            "policy_pose": pose, "demo_pose": recorded,
            "r_policy": float(env.radius_np(pose[None, :])[0]),
            "r_demo": float(env.radius_np(recorded[None, :])[0]),
            "pose_err": float(np.abs(pose - recorded).mean()),
        })
    return entries


def main() -> int:
    import time

    import mujoco
    import mujoco.viewer

    from .keyframe_env import KeyframeEnv

    parser = argparse.ArgumentParser(description="Behaviour-cloned policy on its training manifolds")
    parser.add_argument("--policy", type=Path, default=DEFAULT)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("computing the clone's pose for each training manifold...")
    entries = collect(args)
    print(f"{'#':>3} {'semi':>20} {'r(clone)':>9} {'r(recorded)':>12} {'pose err':>9}")
    for i, e in enumerate(entries[:12]):
        print(f"{i:>3} {str(np.round(e['semi'], 3)):>20} {e['r_policy']:>9.2f} "
              f"{e['r_demo']:>12.2f} {e['pose_err']:>9.3f}")
    fits = sum(1 for e in entries if e["r_policy"] <= 1.0)
    print(f"\nthe clone fits {fits}/{len(entries)} of its own training manifolds")
    print("(if this is low, the clone never learned its training set; if it is high, any failure")
    print(" elsewhere is generalisation to unseen manifolds)")

    body = BodyModel()
    build_scene(entries[0]["manifold"])
    env = KeyframeEnv(scene_path=SCENE)
    state = {"index": 0, "which": "policy", "pause": False, "hold": 0}

    def key_callback(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        if key == "n":
            state["index"] = (state["index"] + 1) % len(entries)
            state["hold"] = 0
        elif key == "m":
            state["index"] = (state["index"] - 1) % len(entries)
            state["hold"] = 0
        elif key in "123456789":
            state["index"] = min(int(key) - 1, len(entries) - 1)
            state["hold"] = 0
        elif key == "s":
            state["which"] = "recorded" if state["which"] == "policy" else "policy"
            state["hold"] = 0
        elif key == "r":
            state["hold"] = 0
        elif key == "p":
            state["pause"] = not state["pause"]

    viewer = mujoco.viewer.launch_passive(env.env.model, env.env.data, key_callback=key_callback,
                                          show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, 130.0, -10.0
    print("\nkeys: N/M manifold | 1-9 jump | S clone<->recorded | R re-hold | P pause | Ctrl+C quit")
    print("the robot is driven by SONIC: the clone's pose is a keyframe, physics does the rest")

    shown = None
    clock = time.perf_counter()
    try:
        while viewer.is_running():
            entry = entries[state["index"]]
            which = state["which"]
            requested = entry["policy_pose"] if which == "policy" else entry["demo_pose"]

            if shown != (state["index"], which):          # new pose: reset and set the keyframe
                env.reset()
                env.set_joints(requested)
                shown = (state["index"], which)
                state["hold"] = 0

            if not state["pause"]:
                env.step()                                 # SONIC -> PD torques -> physics
                state["hold"] += 1
                clock += C.CONTROL_DT
                delay = clock - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    clock = time.perf_counter()

            # what the robot actually reached, and what it was asked for
            reached = env.state()["q_hw"][C.MUJOCO_TO_ISAACLAB] - C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB]
            r_req = float(entry["manifold"].radii(body.mesh_points_pose(requested)).max())
            r_act = float(entry["manifold"].radii(body.mesh_points_pose(reached)).max())
            err = np.abs(reached - requested)

            pelvis = env.state()["base_pos"]
            update_visuals(env.env.model, entry["manifold"], follow=pelvis)
            scene = viewer.user_scn
            scene.ngeom = 0
            _dots(scene, body, requested, env.env.data)    # requested pose, for comparison
            viewer.set_texts([
                (None, None, f"[{state['index'] + 1}/{len(entries)}] requesting the {which} pose",
                 f"hold {state['hold'] / 50:.1f}s   semi {np.round(entry['semi'], 3)}"),
                (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                 f"r requested {r_req:.2f}   r achieved {r_act:.2f}",
                 f"joint error {err.mean():.3f} rad (arms {err[15:].mean():.3f})   "
                 f"pelvis z {pelvis[2]:.2f}"),
            ])
            viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    return 0


def _dots(scene, body: BodyModel, pose: np.ndarray, data) -> None:
    """Mark the comparison pose with points, in the frame of the displayed robot."""
    import mujoco

    pelvis = data.qpos[0:3].copy()
    points = body.mesh_points_pose(pose, max_vertices=16) + pelvis
    step = max(1, len(points) // 100)
    for p in points[::step]:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.011, 0, 0]), p,
                            np.eye(3).flatten(), np.array([0.2, 0.9, 1.0, 0.5]))
        scene.ngeom += 1


if __name__ == "__main__":
    raise SystemExit(main())
