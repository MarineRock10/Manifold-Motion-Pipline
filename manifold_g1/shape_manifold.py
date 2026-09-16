"""Shape the manifold by hand and watch the policy and the controller respond.

The other viewers cycle through a fixed list of manifolds. This one lets the envelope be
changed while it runs - squash it, pinch it, shift it, tilt its axis - and shows the full chain
live:

    keys -> manifold -> policy pose -> SONIC keyframe -> physics -> achieved pose

Two containments are on screen at once, which is the point: `r requested` is what the geometric
model thinks of the policy's pose, and `r achieved` is what SONIC actually left the robot in.
The gap between them is the execution cost, and it is only visible because the controller is in
the loop rather than modelled.

Keys (MuJoCo's viewer already owns Space, +/-, arrows, Tab and [ ]):

    u / i    squash / raise the envelope       (height)
    j / k    pinch / widen it                  (width)
    , / .    flatten / deepen it               (depth)
    n / m    shift it sideways
    t / y    tilt the axis back / forward
    S        swap the requested pose: policy <-> a recorded demonstration
    R        re-hold from the current pose     P pause
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import constants as C
from .static_fit import BodyModel
from .demos import DEFAULT_ISAAC
from .family import BASE_CENTER, BASE_SEMI
from .keyframe_env import KeyframeEnv
from .manifold import EllipsoidManifold, Primitive, build_scene, update_visuals
from .ppo import PPO
from .primitive import POSE_DIM, load_demos

# default to the RL-finetuned policy (the best available); fall back to the clone
POLICY = C.REPO / "reports" / "manifold_g1" / "primitive_torch" / "policy.pt"
FALLBACK = C.REPO / "reports" / "manifold_g1" / "primitive_torch" / "bc_policy.pt"
SCENE = C.REPO / "data" / "g1_flat" / "scene_manifold.xml"
POSE_LIMIT = 1.4


class Shaper:
    """The manifold as five live parameters, so keys can move them independently."""

    def __init__(self):
        self.height = 1.00
        self.width = 1.00
        self.depth = 1.00
        self.offset = 0.00
        self.tilt = 0.0

    def build(self) -> EllipsoidManifold:
        semi = BASE_SEMI * np.array([self.depth, self.width, self.height])
        center = BASE_CENTER * np.array([self.depth, self.width, self.height])
        center = center + np.array([0.0, self.offset, 0.0])
        half = np.radians(self.tilt) / 2.0
        return EllipsoidManifold([Primitive(center=center, semi=semi,
                                            quat=np.array([np.cos(half), 0.0, np.sin(half), 0.0]))])

    def nudge(self, key: str) -> bool:
        step, tilt_step = 0.02, 2.0
        if key == "u":
            self.height = max(0.55, self.height - step)
        elif key == "i":
            self.height = min(1.30, self.height + step)
        elif key == "j":
            self.width = max(0.50, self.width - step)
        elif key == "k":
            self.width = min(1.30, self.width + step)
        elif key == ",":
            self.depth = max(0.45, self.depth - step)
        elif key == ".":
            self.depth = min(1.30, self.depth + step)
        elif key == "n":
            self.offset = float(np.clip(self.offset - 0.01, -0.10, 0.10))
        elif key == "m":
            self.offset = float(np.clip(self.offset + 0.01, -0.10, 0.10))
        elif key == "t":
            self.tilt = float(np.clip(self.tilt - tilt_step, -25.0, 25.0))
        elif key == "y":
            self.tilt = float(np.clip(self.tilt + tilt_step, -25.0, 25.0))
        else:
            return False
        return True

    def label(self) -> str:
        return (f"h {self.height:.2f}  w {self.width:.2f}  d {self.depth:.2f}  "
                f"offset {self.offset:+.2f}  tilt {self.tilt:+.0f}deg")


def solve_policy(ppo, body: BodyModel, manifold: EllipsoidManifold, kin,
                 steps: int = 6) -> np.ndarray:
    """The pose the policy proposes for this manifold (deterministic, no exploration noise)."""
    import torch

    pose = np.zeros(POSE_DIM)
    for _ in range(steps):
        obs = _observation(body, manifold, pose, kin)
        with torch.no_grad():
            action, _, _ = ppo.act(obs, deterministic=True)
        target = np.tanh(action) * POSE_LIMIT
        pose = pose + (target - pose) * 0.5          # a few relaxation steps, as the env does
    return pose


def _observation(body: BodyModel, manifold: EllipsoidManifold, pose: np.ndarray, kin) -> np.ndarray:
    """The 15-dim observation the policy was trained on, built the same way in both trainers."""
    primitive = manifold.primitives[0]
    r = float(manifold.radii(body.mesh_points_pose(pose)).max())
    pelvis = body.at_pose(pose)[body.pelvis_index()]
    return np.concatenate([
        pose[:3] / POSE_LIMIT, [pose.mean(), pose.std()], [r],
        (primitive.center - pelvis) / 0.5, primitive.semi, primitive.axis(),
    ]).astype(np.float32)


class Points:
    """Body-surface points of the requested pose, for a visual comparison against the robot."""

    def __init__(self, max_vertices: int = 20):
        self.body = BodyModel()
        self.max_vertices = max_vertices


def main() -> int:
    import mujoco
    import mujoco.viewer

    from .kinematics import TorchKinematics

    parser = argparse.ArgumentParser(description="Shape the manifold by hand, live")
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--settle", type=float, default=0.6, help="seconds per hold")
    parser.add_argument("--demo", type=int, default=None,
                        help="fix the demonstration index instead of choosing at random")
    args = parser.parse_args()

    path = args.policy if Path(args.policy).exists() else FALLBACK
    ppo = PPO(15, POSE_DIM, device=args.device)
    ppo.load(path)
    print(f"policy: {path.name}")

    kin = TorchKinematics(device=args.device)
    body = BodyModel()
    mean_demos, all_demos = load_demos()
    keys = [k for k in all_demos if len(all_demos[k])]
    rng = np.random.default_rng(0)
    demo_pose = all_demos[keys[rng.integers(len(keys))]][0]

    shaper = Shaper()
    manifold = shaper.build()
    build_scene(manifold)
    env = KeyframeEnv(scene_path=SCENE)
    env.reset()

    state = {"pose": np.zeros(POSE_DIM), "which": "policy", "pause": False, "dirty": True,
             "elapsed": 0.0}

    def key_callback(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        if shaper.nudge(key):
            state["dirty"] = True
            return
        if key == "s":
            state["which"] = "recorded" if state["which"] == "policy" else "policy"
            state["dirty"] = True
        elif key == "r":
            state["dirty"] = True
        elif key == "p":
            state["pause"] = not state["pause"]

    viewer = mujoco.viewer.launch_passive(env.env.model, env.env.data, key_callback=key_callback,
                                          show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, 130.0, -10.0
    print("keys: u/i height  j/k width  ,/. depth  n/m offset  t/y tilt  "
          "S policy<->recorded  R re-hold  P pause")

    clock = time.perf_counter()
    try:
        while viewer.is_running():
            if state["pause"]:
                viewer.sync()
                time.sleep(0.02)
                continue

            if state["dirty"]:
                state["dirty"] = False
                state["elapsed"] = 0.0
                manifold = shaper.build()
                requested = (solve_policy(ppo, body, manifold, kin) if state["which"] == "policy"
                             else demo_pose)
                env.reset()
                env.set_joints(requested)
                state["pose"] = requested
                state["requested"] = requested
                update_visuals(env.env.model, manifold, follow=env.state()["base_pos"])

            if not state["pause"]:
                env.step()                                  # SONIC drives, physics responds
                state["elapsed"] += C.CONTROL_DT
                clock += C.CONTROL_DT
                delay = clock - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    clock = time.perf_counter()

            requested = state["requested"]
            reached = env.state()["q_hw"][C.MUJOCO_TO_ISAACLAB] - DEFAULT_ISAAC
            r_req = float(manifold.radii(body.mesh_points_pose(requested)).max())
            r_act = float(manifold.radii(body.mesh_points_pose(reached)).max())
            err = np.abs(reached - requested)
            pelvis = env.state()["base_pos"]
            update_visuals(env.env.model, manifold, follow=pelvis)

            scene = viewer.user_scn
            scene.ngeom = 0
            _dots(scene, body, requested, pelvis)
            viewer.set_texts([
                (None, None, f"manifold: {shaper.label()}",
                 f"requesting the {state['which']} pose   held {state['elapsed']:.1f}s"),
                (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                 f"r requested {r_req:.2f}   r achieved {r_act:.2f}",
                 f"joint error {err.mean():.3f} (arms {err[15:].mean():.3f})   pelvis z {pelvis[2]:.2f}"),
            ])
            viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    return 0


def _dots(scene, body: BodyModel, pose: np.ndarray, pelvis: np.ndarray) -> None:
    """The requested pose as points, next to the robot the controller actually produced."""
    import mujoco

    points = body.mesh_points_pose(pose, max_vertices=16) + pelvis
    step = max(1, len(points) // 110)
    for p in points[::step]:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.011, 0, 0]), p,
                            np.eye(3).flatten(), np.array([0.2, 0.9, 1.0, 0.5]))
        scene.ngeom += 1


if __name__ == "__main__":
    raise SystemExit(main())
