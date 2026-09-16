"""Behaviour cloning, on screen: what the clone produces for each recorded manifold.

The clone's training pair is (recorded envelope, recorded pose), so the honest way to watch it is
against that pair, on the manifolds the data actually covers. This viewer draws, for one recorded
manifold at a time:

  * the **recorded pose** (cyan dots) - the target, and the pose the envelope was built from;
  * the **clone's pose** (green dots) - what the clone answers for the same envelope;
  * the **SONIC result** (the robot) - the clone's pose handed to the frozen controller as a
    keyframe, with physics doing the rest.

Press `S` to step through which pose the robot is asked to hold, `N`/`M` to change manifold. The
headline number is `r achieved` measured on the robot: that is the stage's own metric, pose
fitting under the recorded envelope, judged through the controller rather than through a model.

    python3 -m manifold_g1.show_bc --policy reports/manifold_g1/bc/bc_policy.pt
    python3 -m manifold_g1.show_bc --device cpu --count 200 --max-spread 0.02
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import constants as C
from .demo_spread import spread
from .manifold import build_scene, update_visuals
from .pose_policy import POSE_DIM, action_to_pose, observation
from .ppo import PPO
from .demos import load_demos
from .static_fit import BodyModel

DEFAULT = C.REPO / "reports" / "manifold_g1" / "bc" / "bc_policy.pt"
SCENE = C.REPO / "data" / "g1_flat" / "scene_manifold.xml"
SETTLE_STEPS = 6


def collect(args) -> tuple[list[dict], dict]:
    """One entry per recorded manifold: the clone's pose, the recorded pose, both radii.

    Manifolds are ordered most-consistent-first, because that is the order the clone can be
    judged in: a manifold whose recorded poses disagree with each other has no single right
    answer, so a mismatched clone pose there says nothing about the clone. `--max-spread` can
    drop the ambiguous ones entirely; by default every recorded manifold is scored, so the
    summary below covers the training distribution and not a hand-picked corner of it.
    """
    import torch

    from .kinematics import TorchKinematics
    from .manifold import EllipsoidManifold, Primitive

    kin = TorchKinematics(device=args.device)
    body = BodyModel()
    _, all_demos = load_demos()
    ppo = PPO(args.obs_dim, POSE_DIM, device=args.device)
    ppo.load(args.policy)

    entries = []
    keys = list(all_demos)
    if args.sample and args.sample < len(keys):
        # Scoring all 10k recorded manifolds before the window opens costs minutes, so by default
        # the set is shuffled and truncated. `--sample 0` scores every manifold (the summary then
        # describes the whole training distribution, which is what `bc eval` reports too).
        keys = [keys[i] for i in np.random.default_rng(args.seed).choice(
            len(keys), args.sample, replace=False)]
    for key in keys:
        poses = all_demos[key]
        values = np.array([float(v) for v in key.split(",")])
        semi, center = values[:3], values[3:]
        _, median_dev, near = spread(poses)
        near_frac = near / len(poses)
        if args.max_spread is not None and median_dev > args.max_spread:
            continue
        manifold = EllipsoidManifold([Primitive(center=center, semi=semi)])
        recorded = poses[np.argmin(np.abs(poses).mean(axis=1))]

        # the clone's answer for this envelope, through the same observation layout training used
        pose = np.zeros(POSE_DIM)
        with torch.no_grad():
            for _ in range(SETTLE_STEPS):
                r_now = float(manifold.radii(_points(kin, pose).cpu().numpy()).max())
                obs = observation(pose, manifold.primitives[0], r_now, np.zeros(3))
                action, _, _ = ppo.act(obs, deterministic=True)
                target = np.asarray(action_to_pose(action, torch))
                pose = pose + (target - pose) * 0.5
        r_clone = float(manifold.radii(
            _points(kin, pose).cpu().numpy()).max())
        r_recorded = float(manifold.radii(
            _points(kin, recorded).cpu().numpy()).max())

        entries.append({
            "key": key, "manifold": manifold, "semi": semi, "center": center,
            "clone_pose": pose, "recorded_pose": recorded,
            "r_clone": r_clone, "r_recorded": r_recorded,
            "pose_err": float(np.abs(pose - recorded).mean()),
            "spread": median_dev, "near_frac": near_frac, "count": len(poses),
        })

    entries.sort(key=lambda e: (e["spread"], -e["near_frac"]))
    flags = {
        "total": len(all_demos),
        "scored": len(entries),
        "session_ok": int(sum(e["r_recorded"] <= 1.0 for e in entries)),
        "clone_ok": int(sum(e["r_clone"] <= 1.0 for e in entries)),
        "multi": int(sum(e["count"] >= 2 for e in entries)),
    }
    return entries[:args.count], flags


def main() -> int:
    import time

    import mujoco
    import mujoco.viewer

    from .keyframe_env import KeyframeEnv

    parser = argparse.ArgumentParser(description="The clone on the manifolds it was trained on")
    parser.add_argument("--policy", type=Path, default=DEFAULT)
    parser.add_argument("--count", type=int, default=60, help="manifolds to show")
    parser.add_argument("--sample", type=int, default=400,
                        help="manifolds to score before opening the window; 0 scores all "
                             "(slow: the summary is then over the whole training distribution)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-spread", type=float, default=None,
                        help="skip manifolds whose recorded poses disagree by more than this "
                             "(rad); unset means every recorded manifold is scored")
    parser.add_argument("--obs-dim", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"scoring the clone on {args.policy.name} over the recorded manifolds...")
    entries, flags = collect(args)
    if not entries:
        print("no recorded manifold found; build the demo set first")
        return 1
    scored = flags["scored"]
    print(f"{scored} of {flags['total']} recorded manifolds scored"
          + (f" (a random {args.sample}; use --sample 0 for all)" if args.sample else "")
          + f"; {flags['multi']} have more than one recorded pose. Showing {len(entries)}.")
    print(f"  the recorded pose fits its own envelope on {flags['session_ok']}/{scored}   "
          f"(data quality)")
    print(f"  the clone's pose fits it on {flags['clone_ok']}/{scored}   "
          f"(clone quality, same manifolds)")
    print("  spread = how much the recorded poses disagree within a manifold; the clone can only")
    print("  be judged on small-spread rows, where the manifold determines the pose")
    print(f"\n  {'#':>3} {'semi':>22} {'n':>4} {'spread':>7} {'r rec':>6} {'r clone':>8} "
          f"{'pose err':>9}")
    for i, e in enumerate(entries[:14]):
        print(f"  {i:>3} {str(np.round(e['semi'], 3)):>22} {e['count']:>4} {e['spread']:>7.3f} "
              f"{e['r_recorded']:>6.2f} {e['r_clone']:>8.2f} {e['pose_err']:>9.3f}")

    body = BodyModel()
    build_scene(entries[0]["manifold"])
    env = KeyframeEnv(scene_path=SCENE)
    state = {"index": 0, "which": "clone", "pause": False, "hold": 0}

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
            state["which"] = "recorded" if state["which"] == "clone" else "clone"
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
    print("\nkeys: N/M manifold | 1-9 jump | S clone<->recorded | R re-hold | P pause | Ctrl+C")
    print("dots are poses, the robot is what SONIC reached; r achieved is measured on the robot")

    shown = None
    clock = time.perf_counter()
    try:
        while viewer.is_running():
            entry = entries[state["index"]]
            which = state["which"]
            requested = entry["clone_pose"] if which == "clone" else entry["recorded_pose"]

            if shown != (state["index"], which):        # new request: reset, then set keyframe
                env.reset()
                env.set_joints(requested)
                shown = (state["index"], which)
                state["hold"] = 0

            if not state["pause"]:
                env.step()                              # SONIC -> PD torques -> physics
                state["hold"] += 1
                clock += C.CONTROL_DT
                delay = clock - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    clock = time.perf_counter()

            reached = env.state()["q_hw"][C.MUJOCO_TO_ISAACLAB] - C.DEFAULT_ANGLES[
                C.MUJOCO_TO_ISAACLAB]
            r_req = float(entry["manifold"].radii(body.mesh_points_pose(requested)).max())
            r_act = float(entry["manifold"].radii(body.mesh_points_pose(reached)).max())
            err = np.abs(reached - requested)

            pelvis = env.state()["base_pos"]
            update_visuals(env.env.model, entry["manifold"], follow=pelvis)
            scene = viewer.user_scn
            scene.ngeom = 0
            _dots(scene, body, entry["recorded_pose"], pelvis, (0.2, 0.9, 1.0))   # target
            if which == "clone":
                _dots(scene, body, entry["clone_pose"], pelvis, (0.3, 1.0, 0.4))  # the answer
            viewer.set_texts([
                (None, None,
                 f"[{state['index'] + 1}/{len(entries)}]  {entry['key']}   "
                 f"({entry['count']} recorded poses, spread {entry['spread']:.3f})",
                 f"robot is holding the {which} pose   held {state['hold'] / 50:.1f}s"),
                (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                 f"r recorded {entry['r_recorded']:.2f}   r clone {entry['r_clone']:.2f}   "
                 f"r achieved {r_act:.2f}",
                 f"joint error {err.mean():.3f} rad (arms {err[15:].mean():.3f})   "
                 f"pelvis z {pelvis[2]:.2f}"),
            ])
            viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    return 0


def _points(kin, pose: np.ndarray):
    """Body-surface points of one pose, as a device tensor (the containment ruler)."""
    import torch

    return kin.forward(torch.as_tensor(pose[None, :], dtype=kin.dtype, device=kin.device))[0]


def _dots(scene, body: BodyModel, pose: np.ndarray, pelvis: np.ndarray, rgb) -> None:
    """A pose marked with points, in the frame of the displayed robot."""
    import mujoco

    points = body.mesh_points_pose(pose, max_vertices=16) + pelvis
    step = max(1, len(points) // 100)
    for p in points[::step]:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.011, 0, 0]), p,
                            np.eye(3).flatten(), np.array([*rgb, 0.55]))
        scene.ngeom += 1


if __name__ == "__main__":
    raise SystemExit(main())
