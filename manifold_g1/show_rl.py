"""The fine-tuned policy, on the distribution it was fine-tuned on - with keys that change it.

The in-the-loop fine-tune trains on *perturbed* recorded envelopes: one recorded pose, its own
manifold reshaped, and the reward measured on the pose the real controller reached. That is the
distribution this viewer samples, so what it shows is the stage's own metric:

  * a recorded envelope is drawn and perturbed by the same rule training uses (each semi-axis
    scaled by up to +-`--perturb`, the centre shifted by up to 2 cm);
  * the policy's answer (green dots) is handed to SONIC as a keyframe and held;
  * `r achieved` is containment of the pose the robot actually reached - the reward the
    fine-tune maximised - next to `r recorded`, the same pose's containment under the
    *unperturbed* envelope, which is the "did the deformation help" comparison.

The manifold keys are the feature: `h`/`w`/`d` scale the drawn envelope, `n`/`m` shift it, `t`/`y`
tilt it. Every key re-draws the perturbation and re-runs the policy, so the response to a changed
manifold is visible directly rather than inferred from a log.

    python3 -m manifold_g1.show_rl
    python3 -m manifold_g1.show_rl --policy reports/manifold_g1/sonic_rl/policy_sonicrl.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from . import constants as C
from .manifold import build_scene, update_visuals
from .pose_policy import POSE_DIM, action_to_pose, observation
from .ppo import PPO
from .demos import load_demos
from .static_fit import BodyModel

RL_POLICY = C.REPO / "reports" / "manifold_g1" / "sonic_rl" / "policy_sonicrl.pt"
BC_POLICY = C.REPO / "reports" / "manifold_g1" / "bc" / "bc_policy.pt"
LOG = C.REPO / "reports" / "manifold_g1" / "sonic_rl" / "train_log.csv"
SCENE = C.REPO / "data" / "g1_flat" / "scene_manifold.xml"
SETTLE_STEPS = 6


class Episode:
    """One sampled episode: a recorded pose, its own envelope, and the perturbation of it.

    `height`/`width`/`depth` re-scale the drawn envelope on top of the training-style jitter,
    which is what the keys move; `offset` shifts it sideways and `tilt` leans its axis, matching
    the family parameters the reported manifolds use.
    """

    def __init__(self, key: str, poses: np.ndarray, rng: np.random.Generator, perturb: float):
        self.key = key
        self.poses = poses
        self.rng = rng
        self.perturb = perturb
        self.base = np.array([float(v) for v in key.split(",")])
        self.demo = poses[np.argmin(np.abs(poses).mean(axis=1))]     # least extreme pose
        self.height = self.width = self.depth = 1.0
        self.offset = 0.0
        self.tilt = 0.0
        self.redraw()

    def redraw(self) -> None:
        """A fresh perturbation of the recorded envelope, as training draws it."""
        semi, center = self.base[:3].copy(), self.base[3:].copy()
        if self.perturb > 0:
            semi = semi * np.clip(1.0 + self.rng.normal(0, self.perturb, 3), 0.85, 1.15)
            center = center + np.array([self.rng.normal(0, 0.02), self.rng.normal(0, 0.02), 0.0])
        self.semi, self.center = semi, center
        self.clean = self.base[:3]   # the recorded envelope, for the "did it help" column

    def scaled(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The perturbed envelope with the key-driven scaling applied: (semi, center, quat)."""
        semi = self.semi * np.array([self.depth, self.width, self.height])
        center = self.center * np.array([self.depth, self.width, self.height])
        center = center + np.array([0.0, self.offset, 0.0])
        half = np.radians(self.tilt) / 2.0
        return semi, center, np.array([np.cos(half), 0.0, np.sin(half), 0.0])

    def nudge(self, key: str) -> bool:
        """Move one shape parameter. Returns False when the key is not a shape key."""
        step, tilt_step = 0.04, 3.0
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
        elif key == "z":
            self.offset = float(np.clip(self.offset - 0.01, -0.10, 0.10))
        elif key == "x":
            self.offset = float(np.clip(self.offset + 0.01, -0.10, 0.10))
        elif key == "t":
            self.tilt = float(np.clip(self.tilt - tilt_step, -25.0, 25.0))
        elif key == "y":
            self.tilt = float(np.clip(self.tilt + tilt_step, -25.0, 25.0))
        else:
            return False
        return True

    def label(self) -> str:
        return (f"perturb {self.perturb:.0%}   h {self.height:.2f}  w {self.width:.2f}  "
                f"d {self.depth:.2f}  offset {self.offset:+.2f}  tilt {self.tilt:+.0f}deg")


def solve(ppo, body: BodyModel, manifold, kin, steps: int = SETTLE_STEPS) -> np.ndarray:
    """The pose the policy proposes: a few relaxation steps, exactly as the environment does."""
    import torch

    pose = np.zeros(POSE_DIM)
    with torch.no_grad():
        for _ in range(steps):
            r_now = float(manifold.radii(body.mesh_points_pose(pose)).max())
            obs = observation(pose, manifold.primitives[0], r_now, np.zeros(3))
            action, _, _ = ppo.act(obs, deterministic=True)
            target = np.asarray(action_to_pose(action, torch))
            pose = pose + (target - pose) * 0.5
    return pose


def training_summary() -> dict | None:
    """The logged fine-tune, verbatim, so the screen number can be checked against the run."""
    if not LOG.exists():
        return None
    rows = list(csv.DictReader(LOG.open()))
    if not rows:
        return None
    tail = rows[-min(10, len(rows)):]
    return {
        "iterations": len(rows),
        "success": float(np.mean([float(r["success"]) for r in tail])),
        "r_achieved": float(np.mean([float(r["r_achieved"]) for r in tail])),
        "gated": int(sum(int(r["gated"]) for r in rows)),
    }


def main() -> int:
    import time

    import mujoco
    import mujoco.viewer

    from .kinematics import TorchKinematics
    from .keyframe_env import KeyframeEnv
    from .manifold import EllipsoidManifold, Primitive

    parser = argparse.ArgumentParser(description="The fine-tuned policy on perturbed manifolds")
    parser.add_argument("--policy", type=Path, default=RL_POLICY)
    parser.add_argument("--fallback", type=Path, default=BC_POLICY,
                        help="used when --policy does not exist")
    parser.add_argument("--perturb", type=float, default=0.08,
                        help="semi-axis jitter, matching the fine-tune's default")
    parser.add_argument("--obs-dim", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = args.policy if Path(args.policy).exists() else args.fallback
    kin = TorchKinematics(device=args.device)
    body = BodyModel()
    ppo = PPO(args.obs_dim, POSE_DIM, device=args.device)
    ppo.load(path)
    rng = np.random.default_rng(args.seed)
    _, all_demos = load_demos()
    keys = [k for k in all_demos if len(all_demos[k]) >= 4]
    if not keys:
        print("no recorded manifold has enough poses; build the demo set first")
        return 1

    index = int(rng.integers(len(keys)))
    episode = Episode(keys[index], all_demos[keys[index]], rng, args.perturb)

    def manifold_of(ep: Episode):
        semi, center, quat = ep.scaled()
        return EllipsoidManifold([Primitive(center=center, semi=semi, quat=quat)])

    build_scene(manifold_of(episode))
    env = KeyframeEnv(scene_path=SCENE)
    manifest = manifold_of(episode)

    summary = training_summary()
    print(f"policy: {path.name}"
          + (f"   (fallback: {args.policy.name} not found)" if path != args.policy else ""))
    if summary:
        print(f"logged fine-tune: {summary['iterations']} iterations, last-10 success "
              f"{summary['success']:.2f}, r achieved {summary['r_achieved']:.2f}, "
              f"gated {summary['gated']}")
    print(f"perturbation: semi +-{args.perturb:.0%}, centre +-2 cm   "
          f"(the distribution the fine-tune trained on)")

    state = {"pose": np.zeros(POSE_DIM), "requested": np.zeros(POSE_DIM), "which": "policy",
             "pause": False, "dirty": True, "elapsed": 0.0, "clean_r": np.nan}

    def request() -> None:
        """Rebuild the manifold from the current sample and re-run the policy on it."""
        nonlocal manifest
        state["dirty"] = False
        state["elapsed"] = 0.0
        manifest = manifold_of(episode)
        requested = (solve(ppo, body, manifest, kin) if state["which"] == "policy"
                     else episode.demo)
        env.reset()
        env.set_joints(requested)
        state["requested"] = requested
        state["pose"] = requested
        state["clean_r"] = float(EllipsoidManifold(
            [Primitive(center=episode.base[3:], semi=episode.clean)]
        ).radii(body.mesh_points_pose(requested)).max())
        update_visuals(env.env.model, manifest, follow=env.state()["base_pos"])

    def key_callback(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        if episode.nudge(key):
            state["dirty"] = True
            return
        if key == "n":
            episode_swap(+1)
        elif key == "m":
            episode_swap(-1)
        elif key == "s":
            state["which"] = "recorded" if state["which"] == "policy" else "policy"
            state["dirty"] = True
        elif key == "r":
            state["dirty"] = True
        elif key == "p":
            state["pause"] = not state["pause"]

    def episode_swap(step: int) -> None:
        """A new recorded envelope, with a fresh perturbation of it."""
        nonlocal episode
        idx = (keys.index(episode.key) + step) % len(keys)
        episode = Episode(keys[idx], all_demos[keys[idx]], rng, args.perturb)
        state["dirty"] = True

    viewer = mujoco.viewer.launch_passive(env.env.model, env.env.data, key_callback=key_callback,
                                          show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, 130.0, -10.0
    print("\nkeys: u/i height  j/k width  ,/. depth  z/x offset  t/y tilt   "
          "(each re-runs the policy on the reshaped manifold)"
          "\n      N/M new recorded envelope + fresh perturbation | S policy<->recorded"
          "\n      R redraw perturbation | P pause | Ctrl+C quit")

    clock = time.perf_counter()
    try:
        while viewer.is_running():
            if state["dirty"]:
                request()

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
            reached = env.state()["q_hw"][C.MUJOCO_TO_ISAACLAB] - C.DEFAULT_ANGLES[
                C.MUJOCO_TO_ISAACLAB]
            r_req = float(manifest.radii(body.mesh_points_pose(requested)).max())
            r_act = float(manifest.radii(body.mesh_points_pose(reached)).max())
            err = np.abs(reached - requested)
            pelvis = env.state()["base_pos"]
            update_visuals(env.env.model, manifest, follow=pelvis)

            scene = viewer.user_scn
            scene.ngeom = 0
            if state["which"] == "recorded":
                _dots(scene, body, episode.demo, pelvis, (0.2, 0.9, 1.0))
            else:
                _dots(scene, body, requested, pelvis, (0.3, 1.0, 0.4))

            viewer.set_texts([
                (None, None,
                 f"{episode.key}   {episode.label()}   holding {state['elapsed']:.1f}s",
                 f"requesting the {state['which']} pose"),
                (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                 f"r requested {r_req:.2f}   r achieved {r_act:.2f}   "
                 f"r on recorded envelope {state['clean_r']:.2f}",
                 f"joint error {err.mean():.3f} (arms {err[15:].mean():.3f})   "
                 f"pelvis z {pelvis[2]:.2f}"),
            ])
            viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    return 0


def _dots(scene, body: BodyModel, pose: np.ndarray, pelvis: np.ndarray, rgb) -> None:
    """The requested pose as points, next to the robot the controller actually produced."""
    import mujoco

    points = body.mesh_points_pose(pose, max_vertices=16) + pelvis
    step = max(1, len(points) // 110)
    for p in points[::step]:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.011, 0, 0]), p,
                            np.eye(3).flatten(), np.array([*rgb, 0.55]))
        scene.ngeom += 1


if __name__ == "__main__":
    raise SystemExit(main())
