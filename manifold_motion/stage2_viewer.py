"""Interactive MuJoCo viewer for an evaluated Stage-2 sample.

The robot is always the measured SONIC/MuJoCo execution.  The colored trails are the generated
``R_ref`` and the held-out SEED reference in the sample's local root frame.  This avoids the
common misleading demo where a viewer merely teleports a robot through the neural network's
reference and calls it executed.

Keys: P pause/resume, N/M step -/+0.5 s, R restart, C toggle corridor, 1/2/3 toggle trails,
      Esc/Q close.  Mouse drag orbits; wheel zooms.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from . import constants as C
from .corridor import _resample_corridor
from .env import G1FlatEnv


def _rotation6_to_matrix(value: np.ndarray) -> np.ndarray:
    first = np.asarray(value[:3], dtype=np.float64)
    second = np.asarray(value[3:6], dtype=np.float64)
    first /= max(np.linalg.norm(first), 1e-8)
    second -= first * np.dot(first, second)
    second /= max(np.linalg.norm(second), 1e-8)
    return np.column_stack([first, second, np.cross(first, second)])


def _local_to_world(points: np.ndarray, origin_pos: np.ndarray, origin_quat: np.ndarray) -> np.ndarray:
    rotation = C.quat_to_matrix(origin_quat)
    return np.asarray(points, dtype=np.float64) @ rotation.T + origin_pos


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path) as archive:
        sample = {key: np.asarray(archive[key]) for key in archive.files}
    report_path = path.with_suffix(".json")
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    return sample, report


def _add_sphere(scene: mujoco.MjvScene, position: np.ndarray, radius: float,
                color: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0.0, 0.0]), position,
                        np.eye(3).flatten(), np.asarray(color, dtype=np.float32))
    scene.ngeom += 1


def _add_ellipsoid(scene: mujoco.MjvScene, center: np.ndarray, semi: np.ndarray,
                   yaw: float, color: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                         [np.sin(yaw), np.cos(yaw), 0.0],
                         [0.0, 0.0, 1.0]])
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        np.asarray(semi, dtype=np.float64), center, rotation.flatten(),
                        np.asarray(color, dtype=np.float32))
    scene.ngeom += 1


def replay(args: argparse.Namespace) -> int:
    sample, report = _load(args.sample)
    if "generated_ref" not in sample:
        raise ValueError("sample must contain generated_ref")
    if args.executed is None:
        candidate = args.sample.parent.parent / "stage2_mean_walk80_validation_test2" / "executed.npz"
        args.executed = candidate if candidate.is_file() else None
    if args.executed is None or not args.executed.is_file():
        raise ValueError("an evaluated executed.npz is required; run stage2_validate first")
    with np.load(args.executed) as archive:
        execution = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"q_exec", "base_pos", "base_quat"}
    if not required.issubset(execution):
        raise ValueError(f"{args.executed} is missing {sorted(required - set(execution))}")

    generated = sample["generated_ref"]
    expected = sample.get("expected_ref", generated)
    origin_pos, origin_quat = execution["base_pos"][0], execution["base_quat"][0]
    model_path = _local_to_world(generated[:, 29:32], origin_pos, origin_quat)
    expected_path = _local_to_world(expected[:, 29:32], origin_pos, origin_quat)
    executed_path = execution["base_pos"]
    corridor = sample.get("condition_corridor")
    corridor_50 = _resample_corridor(corridor, len(executed_path)) if corridor is not None else None

    env = G1FlatEnv()
    state = {"tick": 0, "pause": False, "close": False, "corridor": True,
             "model": True, "expected": True, "executed": True}

    def key_callback(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        if key == "p": state["pause"] = not state["pause"]
        elif key == "n": state["tick"] = max(0, state["tick"] - 25)
        elif key == "m": state["tick"] = min(len(executed_path) - 1, state["tick"] + 25)
        elif key == "r": state["tick"] = 0
        elif key == "c": state["corridor"] = not state["corridor"]
        elif key == "1": state["model"] = not state["model"]
        elif key == "2": state["expected"] = not state["expected"]
        elif key == "3": state["executed"] = not state["executed"]
        elif key in {"q", "escape"}: state["close"] = True

    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback,
                                          show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.0, 125.0, -12.0

    print(f"Stage-2 GUI: {args.sample}")
    print(f"primitive={report.get('primitive', 'unknown')}  source_index={report.get('source_index', '?')}")
    print("P pause | N/M step | R restart | C corridor | 1 model | 2 SEED | 3 executed | Q close")
    clock = time.perf_counter()
    try:
        while viewer.is_running() and not state["close"]:
            tick = int(state["tick"])
            q_policy = execution["q_exec"][tick]
            env.data.qpos[:3] = execution["base_pos"][tick]
            env.data.qpos[3:7] = execution["base_quat"][tick]
            env.data.qpos[env.body_qadr] = q_policy[C.ISAACLAB_TO_MUJOCO]
            env.data.qpos[env.hand_qadr] = 0.0
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)

            scene = viewer.user_scn
            scene.ngeom = 0
            if state["model"]:
                for point in model_path[::2]:
                    _add_sphere(scene, point, 0.014, (0.2, 0.75, 1.0, 0.60))
            if state["expected"]:
                for point in expected_path[::2]:
                    _add_sphere(scene, point, 0.012, (0.25, 1.0, 0.35, 0.45))
            if state["executed"]:
                for point in executed_path[max(0, tick - 250):tick:4]:
                    _add_sphere(scene, point, 0.016, (1.0, 0.55, 0.10, 0.75))
            if state["corridor"] and corridor_50 is not None:
                element = corridor_50[tick]
                center = _local_to_world(element[None, :3], origin_pos, origin_quat)[0]
                yaw = element[6] + np.arctan2(C.quat_rotate(origin_quat, np.array([1., 0., 0.]))[1],
                                               C.quat_rotate(origin_quat, np.array([1., 0., 0.]))[0])
                _add_ellipsoid(scene, center, element[3:6], yaw, (0.75, 0.25, 1.0, 0.16))

            viewer.set_texts([
                (None, None, "Stage-2 dynamic sample", f"{tick}/{len(executed_path)-1}  {tick/50:.2f} s"),
                (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                 f"orange=executed  cyan=model  green=SEED  purple=corridor",
                 f"P {'resume' if state['pause'] else 'pause'} | C {'on' if state['corridor'] else 'off'}"),
            ])
            viewer.sync()
            if not state["pause"]:
                state["tick"] = (tick + 1) % len(executed_path)
                clock += C.CONTROL_DT
                delay = clock - time.perf_counter()
                if delay > 0: time.sleep(delay)
                else: clock = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="View an evaluated Stage-2 sample in MuJoCo")
    parser.add_argument("--sample", type=Path, required=True,
                        help="sample.npz produced by stage2_flow sample")
    parser.add_argument("--executed", type=Path, default=None,
                        help="executed.npz produced by stage2_validate")
    args = parser.parse_args()
    return replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
