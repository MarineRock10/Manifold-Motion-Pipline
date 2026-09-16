"""See the primitive model against the data: motion, manifold, policy output, SONIC tracking.

One MuJoCo window, three layers:

  * **the recorded motion** - the robot is posed from a clip's `q_act`, so what you watch is a
    movement SONIC really executed, with its own manifold (`M`, the envelope it occupied) drawn
    around it as a translucent ellipsoid;
  * **the policy's answer** - for that manifold the primitive model proposes a pose. It is drawn
    as a ghost skeleton in a second colour, so the difference between "what the robot did" and
    "what the model would hold" is visible directly;
  * **what happens when it is executed** - optionally the proposal is handed to the frozen
    controller as a keyframe and the viewer shows the pose the robot actually reaches, with the
    containment of both measured live.

    python3 -m manifold_g1.view_primitive motion    # replay a clip inside its own manifold
    python3 -m manifold_g1.view_primitive compare   # policy pose vs. demonstrated pose, per manifold
    python3 -m manifold_g1.view_primitive execute   # hand the policy pose to SONIC and watch

Keys (MuJoCo keeps Space, +/-, arrows, Tab, [ ]): `P` pause, `N`/`M` step 0.5 s, `E` envelope,
`1`..`5` switch manifold, `X` execute the policy's pose, `T` trail, `R` restart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C
from .demos import DEFAULT_ISAAC
from .family import ManifoldSpec, report_specs
from .manifold import build_scene, update_visuals
from .ppo import PPO
from .primitive import (DEMOS, POSE_DIM, PrimitiveConfig, PrimitiveEnv, _obs_dim,
                        load_demos, nearest_demos)

CLIPS = Path("reports/manifold_g1/clips")
POLICY = Path("reports/manifold_g1/primitive/policy.pt")


def _policy(path: Path, obs_dim: int) -> PPO:
    ppo = PPO(obs_dim, POSE_DIM)
    ppo.load(path)
    return ppo


def _stable_manifolds(k: int = 5):
    """Manifolds that the recorded data actually covers, so the comparison is meaningful."""
    mean_demos, all_demos = load_demos()
    specs = []
    for spec in report_specs():
        manifold = spec.build()
        demos = nearest_demos(manifold, all_demos)
        if demos is not None and len(demos) >= 20:
            specs.append((spec, manifold, demos))
        if len(specs) >= k:
            break
    return specs


def _policy_pose(ppo: PPO, body, manifold, mean_demos, steps: int = 12) -> np.ndarray:
    """Roll the policy out under a manifold and return the pose it settles on."""
    env = PrimitiveEnv(body, manifold, mean_demos, PrimitiveConfig())
    obs, _ = env.reset()
    info = {"pose": np.zeros(POSE_DIM)}
    for _ in range(steps):
        action, _, _ = ppo.act(obs, deterministic=True)
        obs, _, done, truncated, info = env.step(np.clip(action, -1.0, 1.0))
        if done or truncated:
            break
    return info["pose"]


def _set_robot(env, pose: np.ndarray) -> None:
    """Pose the robot from a 29-joint policy-order delta (no physics, direct kinematics)."""
    import mujoco

    env.data.qpos[env.body_qadr] = C.DEFAULT_ANGLES + pose[C.ISAACLAB_TO_MUJOCO]
    mujoco.mj_kinematics(env.model, env.data)


def _body_points(env, max_vertices: int = 60) -> np.ndarray:
    from .body_envelope import body_points

    import mujoco

    points = body_points(env.model, env.data, max_vertices=max_vertices)
    pelvis = env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
    return points - pelvis


def _skeleton_geoms(scene, points: np.ndarray, rgb, size: float = 0.02):
    """Draw a sparse point cloud into the viewer's user scene (marks the body surface)."""
    import mujoco

    step = max(1, len(points) // 220)
    for p in points[::step]:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0.0, 0.0]), p,
                            np.eye(3).flatten(), np.array([*rgb, 0.5]))
        scene.ngeom += 1


def compare(args) -> int:
    """Policy pose against the demonstrated poses, manifold by manifold, in one window.

    The recorded motion plays inside its own envelope; the policy's answer for that same
    manifold is drawn as a second skeleton. Two containment numbers are shown: the policy's
    pose in the manifold, and the best recorded pose in it.
    """
    import time

    import mujoco
    import mujoco.viewer

    from .env import G1FlatEnv
    from .static_fit import BodyModel

    body = BodyModel()
    manifolds = _stable_manifolds(args.count)
    if not manifolds:
        print("no manifold has enough recorded demos; collect data or widen the family")
        return 1
    mean_demos, all_demos = load_demos()

    spec, manifold, demos = manifolds[0]
    build_scene(manifold)
    env = G1FlatEnv(Path(C.REPO / "data" / "g1_flat" / "scene_manifold.xml"))
    _policy_poses = {}
    for i, (s, m, _) in enumerate(manifolds):
        _policy_poses[i] = _policy_pose(_policy(args.policy, _policy_obs_dim(body, m, mean_demos)),
                                        body, m, mean_demos)

    state = {"index": 0, "pause": False, "envelope": True, "tick": 0}

    def key_callback(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        if key == "p":
            state["pause"] = not state["pause"]
        elif key in "12345":
            state["index"] = min(int(key) - 1, len(manifolds) - 1)
        elif key == "e":
            state["envelope"] = not state["envelope"]
        elif key == "n":
            state["tick"] -= 25
        elif key == "m":
            state["tick"] += 25

    viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback,
                                          show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.8, 130.0, -12.0
    print("keys: P pause | 1-5 manifold | E envelope | N/M step | Ctrl+C quit")

    clock = time.perf_counter()
    try:
        while viewer.is_running():
            index = state["index"]
            spec, manifold, demos = manifolds[index]
            demo_pose = demos[index % len(demos)]
            policy_pose = _policy_poses[index]

            if not state["pause"]:
                clock += C.CONTROL_DT
                delay = clock - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    clock = time.perf_counter()

            _set_robot(env, policy_pose)                     # show the policy's answer
            points = _body_points(env)
            if state["envelope"]:
                update_visuals(env.model, manifold)
            scene = viewer.user_scn
            scene.ngeom = 0
            _skeleton_geoms(scene, points, (1.0, 0.5, 0.1))          # orange: policy pose
            _set_robot(env, demo_pose)
            _skeleton_geoms(scene, _body_points(env), (0.2, 0.7, 1.0))  # blue: demonstrated pose

            r_policy = float(manifold.radii(_body_points_of(env, policy_pose)).max())
            r_demo = float(manifold.radii(_body_points_of(env, demo_pose)).max())
            viewer.set_texts([
                (None, None, f"manifold {spec.label()}", f"recorded poses nearby: {len(demos)}"),
                (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                 f"orange = policy pose   r {r_policy:.2f}",
                 f"blue   = recorded pose r {r_demo:.2f}"),
            ])
            viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    return 0


def _body_points_of(env, pose: np.ndarray) -> np.ndarray:
    _set_robot(env, pose)
    return _body_points(env)


def _policy_obs_dim(body, manifold, mean_demos) -> int:
    env = PrimitiveEnv(body, manifold, mean_demos, PrimitiveConfig())
    return _obs_dim(env)


def motion(args) -> int:
    """Replay a recorded clip inside the manifold it occupied, with the policy's answer drawn."""
    import time

    import mujoco
    import mujoco.viewer

    from .env import G1FlatEnv
    from .manifold import EllipsoidManifold, Primitive as EllipsoidPrimitive
    from .static_fit import BodyModel

    clip_path = Path(args.clip) if args.clip else sorted(CLIPS.glob("ep*.npz"))[0]
    with np.load(clip_path) as handle:
        data = {k: handle[k] for k in handle.files}
    summary = json.loads(clip_path.with_suffix(".json").read_text())
    body = BodyModel()
    mean_demos, all_demos = load_demos()
    base_semi = np.load(DEMOS)["semi"].mean(axis=0)
    manifold = EllipsoidManifold([EllipsoidPrimitive(center=np.zeros(3), semi=base_semi)])
    ppo = _policy(args.policy, _policy_obs_dim(body, manifold, mean_demos))
    build_scene(manifold)
    env = G1FlatEnv(Path(C.REPO / "data" / "g1_flat" / "scene_manifold.xml"))

    q_hw = data["q_act"][:, C.ISAACLAB_TO_MUJOCO]
    state = {"tick": 0, "pause": False, "envelope": True, "trail": True}

    def key_callback(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        if key == "p":
            state["pause"] = not state["pause"]
        elif key == "n":
            state["tick"] = max(0, state["tick"] - 25)
        elif key == "m":
            state["tick"] = min(len(q_hw) - 1, state["tick"] + 25)
        elif key == "e":
            state["envelope"] = not state["envelope"]
        elif key == "t":
            state["trail"] = not state["trail"]
        elif key == "r":
            state["tick"] = 0

    viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback,
                                          show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.2, 130.0, -15.0
    print(f"{clip_path.name}: {summary['ticks']} ticks, track {summary['track_err_mean_rad']:.4f} rad")
    print("keys: P pause | N/M step | E envelope | T trail | R restart")

    clock = time.perf_counter()
    try:
        while viewer.is_running():
            tick = int(state["tick"])
            if not state["pause"]:
                state["tick"] = (tick + 1) % len(q_hw)
                clock += C.CONTROL_DT
                delay = clock - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    clock = time.perf_counter()
            env.data.qpos[0:3] = data["base_pos"][tick]
            env.data.qpos[3:7] = data["base_quat"][tick]
            env.data.qpos[env.body_qadr] = q_hw[tick]
            env.data.qpos[env.hand_qadr] = 0.0
            mujoco.mj_kinematics(env.model, env.data)

            # the manifold this tick occupied, from the recorded envelope
            stride = max(1, len(q_hw) // len(data["envelope_semi"]))
            env_index = min(tick // stride, len(data["envelope_semi"]) - 1)
            tick_manifold = EllipsoidManifold([EllipsoidPrimitive(
                center=np.zeros(3), semi=data["envelope_semi"][env_index])])
            if state["envelope"]:
                update_visuals(env.model, tick_manifold)

            scene = viewer.user_scn
            scene.ngeom = 0
            if state["trail"]:
                for p in data["base_pos"][max(0, tick - 250):tick:5]:
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                        np.array([0.012, 0, 0]), p, np.eye(3).flatten(),
                                        np.array([1.0, 0.45, 0.1, 0.55]))
                    scene.ngeom += 1

            # policy's answer for this manifold, drawn as a ghost skeleton
            pose = _policy_pose(ppo, body, tick_manifold, mean_demos)
            _set_robot(env, pose)
            _skeleton_geoms(scene, _body_points(env), (0.1, 1.0, 0.4), size=0.015)

            err = float(np.abs(data["q_cmd"][tick] - data["q_act"][tick]).mean())
            viewer.set_texts([
                (None, None, f"{clip_path.stem}   t {data['t'][tick]:.1f} s",
                 f"SONIC tracking error {err:.3f} rad"),
                (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                 "green = policy pose for this manifold",
                 f"envelope semi {np.round(data['envelope_semi'][env_index], 3)}"),
            ])
            viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="See the primitive model against the data")
    parser.add_argument("mode", choices=("motion", "compare"))
    parser.add_argument("--clip", type=Path, default=None)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--count", type=int, default=5, help="compare: manifolds to cycle")
    args = parser.parse_args()
    return motion(args) if args.mode == "motion" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
