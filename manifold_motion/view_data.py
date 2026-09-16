"""Look at the collected data: MuJoCo replay and a static overview figure.

    python3 -m manifold_motion.view_data replay --clip reports/manifold_motion/clips/ep0000_seed1000.npz
    python3 -m manifold_motion.view_data plots  --out reports/manifold_motion/dataset/overview.png
    python3 -m manifold_motion.view_data episode --clip ... --out reports/manifold_motion/dataset/ep0000.png

`replay` drives the native MuJoCo viewer from the recording: the robot is posed from the
recorded `q_act`, the translucent ellipsoid is the body envelope of that instant (the reverse
extracted `M_t`), and the overlay shows what was commanded next to what happened.

Keys: `P` pause, `N` / `M` step back / forward 0.5 s, `E` toggle the envelope, `T` toggle the
pelvis trail, `R` restart, mouse to orbit/zoom (MuJoCo keeps Space, +/- , arrows, Tab, [ ]).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C
from .family import MARGIN
from .dataset import present_episodes

CLIP_DIR = Path("reports/manifold_motion/clips")


def _load(path: Path) -> tuple[dict, dict]:
    with np.load(path) as handle:
        data = {k: handle[k] for k in handle.files}
    summary = json.loads(path.with_suffix(".json").read_text())
    return data, summary


def _manifold_at(data: dict, index: int):
    """The recorded body envelope at control tick `index`, as an ellipsoid centred on the pelvis."""
    from .manifold import EllipsoidManifold, Primitive

    stride = max(1, len(data["t"]) // len(data["envelope_semi"]))
    env = min(index // stride, len(data["envelope_semi"]) - 1)
    return EllipsoidManifold([Primitive(center=data["base_pos"][index].copy(),
                                        semi=data["envelope_semi"][env] * MARGIN)])


def replay(args) -> int:
    import time

    import mujoco
    import mujoco.viewer

    from .env import G1FlatEnv
    from .manifold import SCENE_PATH, build_scene, update_visuals

    path = Path(args.clip)
    data, summary = _load(path)
    # hand joints are not recorded (they are held at zero), so the scene is built for the
    # envelope of the first tick and the ellipsoid is reshaped in place from then on
    build_scene(_manifold_at(data, 0))      # writes the scene with the visual ellipsoid in it
    env = G1FlatEnv(SCENE_PATH)
    state = {"tick": 0, "pause": False, "envelope": True, "trail": True}

    def key_callback(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        if key == "p":
            state["pause"] = not state["pause"]
        elif key == "n":
            state["tick"] = max(0, state["tick"] - 25)
        elif key == "m":
            state["tick"] = min(len(data["t"]) - 1, state["tick"] + 25)
        elif key == "e":
            state["envelope"] = not state["envelope"]
        elif key == "t":
            state["trail"] = not state["trail"]
        elif key == "r":
            state["tick"] = 0

    viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback,
                                          show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.2, 130.0, -15.0

    print(f"replay {path.name}: {summary['ticks']} ticks, {summary['travelled_m']:.2f} m, "
          f"track {summary['track_err_mean_rad']:.4f} rad")
    print("keys: P pause | N/M step 0.5 s | E envelope | T trail | R restart")

    q_act, base_pos, base_quat = data["q_act"], data["base_pos"], data["base_quat"]
    # recorded joint angles are in policy order; the MuJoCo model wants hardware order
    q_hw = q_act[:, C.ISAACLAB_TO_MUJOCO]
    clock = time.perf_counter()
    wall_limit = args.seconds if args.seconds > 0 else None
    started = clock
    try:
        while viewer.is_running():
            if wall_limit is not None and time.perf_counter() - started > wall_limit:
                break
            tick = int(state["tick"])
            if not state["pause"]:
                state["tick"] = (tick + 1) % len(q_act)
                clock += C.CONTROL_DT
                delay = clock - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    clock = time.perf_counter()
            env.data.qpos[0:3] = base_pos[tick]
            env.data.qpos[3:7] = base_quat[tick]
            env.data.qpos[env.body_qadr] = q_hw[tick]
            env.data.qpos[env.hand_qadr] = 0.0
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)

            if state["envelope"]:
                update_visuals(env.model, _manifold_at(data, tick))
            else:
                for i in range(4):                      # park the visual ellipsoid far away
                    model_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM,
                                                   f"ellipsoid_{i}")
                    if model_geom >= 0:
                        env.model.geom_pos[model_geom] = (0.0, 0.0, -50.0)

            scn = viewer.user_scn
            scn.ngeom = 0
            if state["trail"]:
                for p in base_pos[max(0, tick - 250):tick:5]:
                    geom = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                        np.array([0.012, 0.0, 0.0]), p,
                                        np.eye(3).flatten(), np.array([1.0, 0.45, 0.1, 0.55]))
                    scn.ngeom += 1

            cmd = data["cmd"][tick]
            err = float(np.abs(data["q_cmd"][tick] - q_act[tick]).mean())
            speed = float(np.linalg.norm(data["base_lin_vel"][tick, :2]))
            contact = data["contact"][tick]
            viewer.set_texts([
                (None, None, f"{path.stem}", f"tick {tick}/{len(q_act)}  ({tick / 50:.1f} s)"),
                (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                 f"cmd vel {cmd[0]:+.2f}  dir {np.degrees(np.arctan2(cmd[2], cmd[1])):+4.0f} deg  "
                 f"face {np.degrees(np.arctan2(cmd[4], cmd[3])):+4.0f} deg",
                 f"speed {speed:.2f} m/s  track {err:.3f} rad  contact "
                 f"{'L' if contact[0] else '-'}{'R' if contact[1] else '-'}"),
            ])
            viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    return 0


def plots(args) -> int:
    """Static overview of a whole collection: trajectories, speed, envelope, coverage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = json.loads((args.dataset / "index.json").read_text())
    clips = Path(index["clips_dir"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    speed_hist, semi_all, top_all, angles = [], [], [], []
    for episode in present_episodes(index):
        data, _ = _load(clips / episode["file"])
        axes[0, 0].plot(data["base_pos"][:, 0], data["base_pos"][:, 1], lw=0.7, alpha=0.6)
        speed_hist.append(np.linalg.norm(data["base_lin_vel"][:, :2], axis=1))
        semi_all.append(data["envelope_semi"])
        top_all.append(data["envelope_top"])
        angles.append(np.degrees(np.arctan2(data["cmd"][:, 4], data["cmd"][:, 3])) % 360)
    if not speed_hist:
        print(f"no indexed clip could be loaded from {clips}; rebuild the dataset index")
        return 1

    axes[0, 0].set_title(f"pelvis paths, {len(speed_hist)} episodes")
    axes[0, 0].set_xlabel("x [m]"); axes[0, 0].set_ylabel("y [m]"); axes[0, 0].axis("equal")

    speed = np.concatenate(speed_hist)
    axes[0, 1].hist(speed, bins=40)
    axes[0, 1].set_title(f"speed [m/s]  p50 {np.median(speed):.2f}  p95 {np.percentile(speed,95):.2f}")
    axes[0, 1].set_xlabel("m/s")

    semi = np.concatenate(semi_all)
    top = np.concatenate(top_all)
    axes[1, 0].hist([semi[:, 0], semi[:, 1], semi[:, 2]], bins=40, label=["x", "y", "z"],
                    histtype="step")
    axes[1, 0].set_title(f"body envelope semi-axes [m], top p5-p95 "
                         f"{np.percentile(top,5):.2f}-{np.percentile(top,95):.2f} m")
    axes[1, 0].legend()

    ang = np.concatenate(angles)
    axes[1, 1].hist(ang, bins=16, range=(0, 360))
    axes[1, 1].set_title("commanded facing direction [deg], 16 bins")
    axes[1, 1].set_xlabel("deg")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"windows {index['windows']} / episodes {len(index['episodes'])}  ->  {args.out}")
    return 0


def episode(args) -> int:
    """Time series of one episode: tracking error, speed, envelope, contact."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, summary = _load(Path(args.clip))
    t = data["t"]
    err = np.abs(data["q_cmd"] - data["q_act"])          # [T, 29]
    stride = max(1, len(t) // len(data["envelope_semi"]))
    env_t = data["envelope_t"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, err.mean(axis=1), label="mean")
    axes[0].plot(t, err[:, :12].mean(axis=1), label="legs")
    axes[0].plot(t, err[:, 15:].mean(axis=1), label="arms")
    axes[0].plot(t, err[:, 12:15].mean(axis=1), label="waist")
    axes[0].set_ylabel("track err [rad]"); axes[0].legend(ncol=4, fontsize=8)
    axes[0].set_title(f"{Path(args.clip).stem}: {summary['travelled_m']:.2f} m, "
                      f"track {summary['track_err_mean_rad']:.4f} rad")

    axes[1].plot(t, data["base_lin_vel"][:, 0], label="vx")
    axes[1].plot(t, data["base_lin_vel"][:, 1], label="vy")
    axes[1].plot(t, np.linalg.norm(data["base_lin_vel"][:, :2], axis=1), label="|v|", alpha=0.6)
    axes[1].set_ylabel("velocity [m/s]"); axes[1].legend(ncol=3, fontsize=8)

    axes[2].plot(env_t, data["envelope_semi"][:, 0], label="env x")
    axes[2].plot(env_t, data["envelope_semi"][:, 1], label="env y")
    axes[2].plot(env_t, data["envelope_semi"][:, 2], label="env z")
    axes[2].set_ylabel("envelope semi [m]"); axes[2].legend(ncol=3, fontsize=8)

    axes[3].plot(t, data["contact"][:, 0].astype(int), label="left")
    axes[3].plot(t, data["contact"][:, 1].astype(int) + 1.05, label="right")
    axes[3].plot(t, data["cmd"][:, 0] * 0.5, label="cmd vel x0.5", alpha=0.7)
    axes[3].plot(t, np.degrees(np.arctan2(data["cmd"][:, 4], data["cmd"][:, 3])) % 360 / 180,
                 label="facing/180")
    axes[3].set_ylabel("contact / command"); axes[3].set_xlabel("time [s]")
    axes[3].legend(ncol=4, fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"{args.clip} -> {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize collected clips")
    parser.add_argument("mode", choices=("replay", "plots", "episode"))
    parser.add_argument("--clip", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=Path("reports/manifold_motion/dataset"))
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/dataset/overview.png"))
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="replay: quit after this many seconds (0 = until the window closes)")
    args = parser.parse_args()
    if args.mode == "replay":
        clip = args.clip or sorted(CLIP_DIR.glob("ep*.npz"))[0]
        args.clip = clip
        return replay(args)
    if args.mode == "plots":
        return plots(args)
    args.clip = args.clip or sorted(CLIP_DIR.glob("ep*.npz"))[0]
    if args.out == Path("reports/manifold_motion/dataset/overview.png"):
        args.out = args.out.with_name(args.clip.stem + ".png")
    return episode(args)


if __name__ == "__main__":
    raise SystemExit(main())
