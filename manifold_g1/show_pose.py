"""One manifold, one pose: press a key to see the next.

Precomputed before the window opens, so rendering does no inference:

  * a list of manifolds (the evaluation family, or ones the data covers);
  * for each: the pose the primitive model proposes, and a pose the recorded motion held
    nearby (a demonstration), when one exists.

The robot is posed by direct kinematics - no simulation, no physics - and the manifold is drawn
around it. `N`/`M` step through the list, `S` swaps between the model's pose and the recorded
one, so "what the model would do" and "what the robot actually did" can be compared one
manifold at a time.

    python3 -m manifold_g1.show_pose                 # policy poses over the family
    python3 -m manifold_g1.show_pose --with-demos    # and the recorded poses nearby
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import constants as C
from .family import ManifoldSpec, report_specs
from .manifold import build_scene, update_visuals
from .ppo import PPO
from .primitive import (POSE_DIM, PrimitiveConfig, PrimitiveEnv, _obs_dim, load_demos,
                        nearest_demos)
from .static_fit import BodyModel

# the torch-trained policy is the current one; the numpy-trained policy stays as a comparison
POLICY = Path(C.REPO / "reports" / "manifold_g1" / "primitive_torch" / "policy.pt")
SCENE = Path(C.REPO / "data" / "g1_flat" / "scene_manifold.xml")


def collect(args) -> list[dict]:
    """One entry per manifold: its spec, the model's pose, and optionally a recorded pose."""
    body = BodyModel()
    mean_demos, all_demos = load_demos()
    kin = None
    if args.torch:
        from .kinematics import TorchKinematics
        kin = TorchKinematics(device=args.device)
    entries = []
    ppo = None
    for spec in report_specs():
        manifold = spec.build()
        env = PrimitiveEnv(body, manifold, mean_demos, PrimitiveConfig(), kin=kin)
        if ppo is None:
            ppo = PPO(_obs_dim(env), POSE_DIM)
            ppo.load(args.policy)
        obs, _ = env.reset()
        info = {"pose": np.zeros(POSE_DIM), "manifold_radius": np.nan}
        for _ in range(env.cfg.max_episode_steps):
            action, _, _ = ppo.act(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(np.clip(action, -1.0, 1.0))
            if done or truncated:
                break
        demos = nearest_demos(manifold, all_demos) if args.with_demos else None
        entries.append({
            "spec": spec, "manifold": manifold,
            "policy_pose": info["pose"], "policy_radius": float(info["manifold_radius"]),
            "demo_pose": None if demos is None else demos[np.argmin(np.abs(demos).mean(axis=1))],
            "demo_count": 0 if demos is None else len(demos),
        })
        print(f"  {spec.label():>26}  policy r {info['manifold_radius']:.2f}"
              f"  demos nearby {0 if demos is None else len(demos)}", flush=True)
    return entries


def show(args) -> int:
    import time

    import mujoco
    import mujoco.viewer

    from .env import G1FlatEnv

    print(f"computing {len(report_specs())} manifolds before opening the window...")
    entries = collect(args)
    poses = np.stack([e["policy_pose"] for e in entries])
    spread = poses.std(axis=0)
    print(f"\npose variation across the {len(entries)} manifolds: max per-joint std "
          f"{np.degrees(spread.max()):.1f} deg, mean {np.degrees(spread.mean()):.1f} deg")
    if np.degrees(spread.mean()) < 1.0:
        print("  -> the policy outputs nearly the SAME pose for every manifold: it has not learned")
        print("     a conditional map, so what you see is one pose repeated, not ten answers."
              "\n     Check the manifold pool / curriculum before trusting any of it.")
    build_scene(entries[0]["manifold"])
    env = G1FlatEnv(SCENE)
    state = {"index": 0, "which": "policy", "pause": False}

    def key_callback(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        if key in ("n", "m"):
            step = 1 if key == "m" else -1
            state["index"] = (state["index"] + step) % len(entries)
        elif key in "123456789":
            state["index"] = min(int(key) - 1, len(entries) - 1)
        elif key == "s":
            state["which"] = "demo" if state["which"] == "policy" else "policy"
        elif key == "p":
            state["pause"] = not state["pause"]

    viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback,
                                          show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, 130.0, -10.0
    print("keys:  N / M  previous / next manifold     1-9 jump      S swap policy/demo pose"
          "      P freeze camera      Ctrl+C quit")

    previous = None
    try:
        while viewer.is_running():
            entry = entries[state["index"]]
            which = state["which"]
            pose = entry["policy_pose"] if which == "policy" else entry["demo_pose"]
            if pose is None:
                pose, which = entry["policy_pose"], "policy (no demo nearby)"

            if previous != (state["index"], which):        # only touch the model on change
                env.data.qpos[env.body_qadr] = C.DEFAULT_ANGLES + pose[C.ISAACLAB_TO_MUJOCO]
                mujoco.mj_kinematics(env.model, env.data)
                # the manifold lives in the pelvis frame: hand the pelvis to the renderer
                pelvis = env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY,
                                                         "pelvis")]
                update_visuals(env.model, entry["manifold"], follow=pelvis)
                previous = (state["index"], which)

            radius = float(entry["manifold"].radii(_body_points(env)).max())
            # the manifold is expressed in the pelvis frame; show where the floor cuts it,
            # because an ellipsoid that dips below the floor is expected (the standing envelope
            # reaches from the toes to the head) and is not a rendering bug
            primitive = entry["manifold"].primitives[0]
            centre_world = primitive.center + np.array(
                [0.0, 0.0, env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY,
                                                           "pelvis")][2]])
            below = float(centre_world[2] - primitive.semi[2])
            spec = entry["spec"]
            viewer.set_texts([
                (None, None,
                 f"[{state['index'] + 1}/{len(entries)}]  manifold",
                 f"h {spec.height:.2f}   w {spec.width:.2f}   d {spec.depth:.2f}   "
                 f"offset {spec.offset:+.2f}   tilt {spec.tilt_deg:+.0f} deg"),
                (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                 f"pose: {which}",
                 f"r {radius:.2f}  demos {entry['demo_count']}  "
                 f"manifold z {below:+.2f}..{centre_world[2] + primitive.semi[2]:+.2f} m"),
            ])
            viewer.sync()
            time.sleep(0.03 if not state["pause"] else 0.1)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    return 0


def _body_points(env, max_vertices: int = 60) -> np.ndarray:
    import mujoco

    from .body_envelope import body_points

    points = body_points(env.model, env.data, max_vertices=max_vertices)
    pelvis = env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
    return points - pelvis


def main() -> int:
    parser = argparse.ArgumentParser(description="One manifold, one pose; keys to step")
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--torch", action="store_true",
                        help="use the torch kinematics for containment (matches the torch "
                             "training path; without it the MuJoCo reference is used)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--with-demos", action="store_true",
                        help="also collect a recorded pose per manifold (S swaps)")
    args = parser.parse_args()
    return show(args)


if __name__ == "__main__":
    raise SystemExit(main())
