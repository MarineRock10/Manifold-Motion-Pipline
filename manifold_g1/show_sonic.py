"""Watch the primitive model on the real execution path: SONIC holds the pose, physics runs.

The earlier viewer posed the robot by writing joint angles straight into `qpos` - pure forward
kinematics, no controller, no physics. That shows the *request*, not the outcome: SONIC adopts
some poses and quietly ignores parts of others (the arms most of all), so the picture could
look contained while the robot would never reach it.

Here the loop is the real one (`KeyframeEnv`): the policy's pose goes to the frozen controller
as a keyframe, the controller drives the joints through PD torques in MuJoCo, and what you see
moving is the simulated robot. Two skeletons are drawn at once:

  * **solid** - the robot as it actually is, driven by SONIC;
  * **wireframe points** - the pose that was requested, so the gap is visible directly.

The overlay reports both containments: `r_req` for the request (what the geometric model would
claim) and `r_act` for what the robot reached (what the controller actually delivered).

    python3 -m manifold_g1.show_sonic --policy reports/manifold_g1/primitive_torch/policy.pt

Keys (MuJoCo keeps Space, +/-, arrows, Tab, [ ]):  N/M next/previous manifold, 1-9 jump,
R reset, H hold longer, E envelope, P pause, Ctrl+C quit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import constants as C
from .family import report_specs
from .manifold import build_scene, update_visuals
from .ppo import PPO
from .primitive import load_demos
from .primitive_torch import Config, TorchPrimitiveEnv
from .static_fit import BodyModel

POLICY = C.REPO / "reports" / "manifold_g1" / "primitive_torch" / "policy.pt"
SCENE = C.REPO / "data" / "g1_flat" / "scene_manifold.xml"


def policy_pose(ppo, kin, manifold, steps: int = 8) -> np.ndarray:
    """The pose the policy settles on for one manifold (numpy, for the viewer thread)."""
    import torch

    from .manifold import quat_to_matrix

    _, allv = load_demos()
    env = TorchPrimitiveEnv(kin, np.concatenate(list(allv.values())), Config(), n=1, pool=1,
                            seed=0)
    primitive = manifold.primitives[0]
    env.pool["semi"][0] = torch.as_tensor(primitive.semi, dtype=kin.dtype, device=kin.device)
    env.pool["center"][0] = torch.as_tensor(primitive.center, dtype=kin.dtype, device=kin.device)
    env.pool["rot"][0] = torch.as_tensor(quat_to_matrix(primitive.quat), dtype=kin.dtype,
                                         device=kin.device)
    env.reset()
    pose = np.zeros(29)
    with torch.no_grad():
        for _ in range(steps):
            action, _, _ = ppo.act(env.obs()[0].cpu().numpy(), deterministic=True)
            _, _, done, info = env.step(
                torch.as_tensor(action[None, :], dtype=kin.dtype, device=kin.device))
            pose = info["pose"][0].cpu().numpy()
            if bool(done[0]):
                break
    return pose


def main() -> int:
    import time

    import mujoco
    import mujoco.viewer

    from .keyframe_env import KeyframeEnv
    from .kinematics import TorchKinematics

    parser = argparse.ArgumentParser(description="Primitive poses on the real SONIC path")
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=8, help="policy steps per manifold")
    parser.add_argument("--hold", type=float, default=2.5, help="seconds to hold before moving on")
    parser.add_argument("--auto", action="store_true", help="advance to the next manifold on a timer")
    args = parser.parse_args()

    kin = TorchKinematics(device=args.device)
    body = BodyModel()
    _, allv = load_demos()
    probe = TorchPrimitiveEnv(kin, np.concatenate(list(allv.values())), Config(), n=1, pool=1,
                              seed=0)
    ppo = PPO(probe.obs().shape[1], 29)
    ppo.load(args.policy)

    specs = report_specs()
    print(f"computing {len(specs)} poses with the policy...")
    poses = [policy_pose(ppo, kin, spec.build(), args.steps) for spec in specs]
    spread = np.stack(poses).std(axis=0)
    print(f"  pose variation across manifolds: mean per-joint std {np.degrees(spread.mean()):.1f} deg")
    if np.degrees(spread.mean()) < 1.0:
        print("  -> nearly the same pose for every manifold: the policy is not conditional")

    build_scene(specs[0].build())            # writes the flat scene + the visual ellipsoid
    env = KeyframeEnv(scene_path=C.REPO / "data" / "g1_flat" / "scene_manifold.xml")
    state = {"index": 0, "tick": 0, "pause": False, "auto": args.auto,
             "hold": int(args.hold / C.CONTROL_DT)}

    def key_callback(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        if key == "n":
            state["index"] = (state["index"] + 1) % len(specs)
            state["tick"] = 0
        elif key == "m":
            state["index"] = (state["index"] - 1) % len(specs)
            state["tick"] = 0
        elif key in "123456789":
            state["index"] = min(int(key) - 1, len(specs) - 1)
            state["tick"] = 0
        elif key == "r":
            state["tick"] = 0
            env.reset()
        elif key == "h":
            state["hold"] = state["hold"] + int(1.0 / C.CONTROL_DT)
        elif key == "p":
            state["pause"] = not state["pause"]

    viewer = mujoco.viewer.launch_passive(env.env.model, env.env.data, key_callback=key_callback,
                                          show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.8, 130.0, -12.0
    print("keys: N/M manifold  1-9 jump  R reset  H hold longer  P pause   "
          "(the robot is driven by SONIC, not posed)")

    current = -1
    clock = time.perf_counter()
    try:
        while viewer.is_running():
            index = state["index"]
            spec = specs[index]
            manifold = spec.build()
            if index != current:                      # new manifold: restart and set the keyframe
                env.reset()
                env.set_joints(poses[index])
                if state["index"] != current:
                    state["tick"] = 0
                current = index

            if not state["pause"]:
                env.step()                            # SONIC drives the joints
                state["tick"] += 1
                if state["auto"] and state["tick"] > state["hold"] + 500:
                    state["index"] = (index + 1) % len(specs)

            reached = env.state()["q_hw"][C.MUJOCO_TO_ISAACLAB] - C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB]
            r_req = float(manifold.radii(body.mesh_points_pose(poses[index])).max())
            r_act = float(manifold.radii(body.mesh_points_pose(reached)).max())
            err = np.abs(reached - poses[index])

            pelvis = env.state()["base_pos"]
            update_visuals(env.env.model, manifold, follow=pelvis)

            scene = viewer.user_scn
            scene.ngeom = 0
            # the requested pose, drawn as loose points next to the real robot
            _ghost(scene, body, poses[index], pelvis)
            viewer.set_texts([
                (None, None, f"[{index + 1}/{len(specs)}] {spec.label()}",
                 f"hold {state['tick'] / 50:.1f}s  ({'SONIC driving' if not state['pause'] else 'paused'})"),
                (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                 f"r requested {r_req:.2f}   r achieved {r_act:.2f}",
                 f"joint error {err.mean():.3f} rad  (arms {err[15:].mean():.3f})  "
                 f"pelvis z {pelvis[2]:.2f}"),
            ])
            viewer.sync()
            clock += C.CONTROL_DT
            delay = clock - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                clock = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    return 0


def _ghost(scene, body: BodyModel, pose: np.ndarray, pelvis: np.ndarray) -> None:
    """Points marking the requested pose, in the world frame of the running robot."""
    import mujoco

    points = body.mesh_points_pose(pose, max_vertices=20) + pelvis
    step = max(1, len(points) // 120)
    for p in points[::step]:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.012, 0, 0]), p,
                            np.eye(3).flatten(), np.array([1.0, 0.85, 0.1, 0.45]))
        scene.ngeom += 1


if __name__ == "__main__":
    raise SystemExit(main())
