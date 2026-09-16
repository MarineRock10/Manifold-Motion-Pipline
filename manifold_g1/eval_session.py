"""The containment ruler, measured on real clips: box envelope vs the ellipsoid manifolds use.

A recorded clip stores, every `envelope_stride` ticks, the body envelope `M_t` of that instant -
computed by `recompute_envelopes` as the sampled body *surface* in the pelvis frame, at the tick
it belongs to. This script reproduces that measurement and then judges a policy against it, with
two radii per row, because the two are different shapes and the difference is 35%:

  * `box`       - the stored envelope is an axis-aligned *box* (`max|rel|` per axis), so the
                  recorded pose is inside it at exactly 1.000 by construction. This column
                  verifies the reconstruction; it is not a policy score.
  * `ellipsoid` - the manifold family is an *ellipsoid*, whose unit level set is inscribed in
                  that box. A box corner sits at r = 1.73 and the real standing body at r = 1.35
                  against its own recorded envelope, so `r > 1` here does NOT mean "outside the
                  recording" - it means "outside the inscribed ellipsoid", which no recorded pose
                  can satisfy.

**What this means for the numbers elsewhere in the pipeline.** The recorded envelope being a box
while the policy's reward is an ellipsoid norm means the containment target is not the shape the
data describes: maximising `w_inside` moves the body toward an inscribed ellipsoid that excludes
~39% of the standing surface. That is worth knowing before reading any r near 1 as "almost
inside". Fixing it means defining the envelope as an ellipsoid at measurement time (inflate the
box semis by the factor that contains the body, 1.35 for the standing pose), not in the viewer.

    python3 -m manifold_g1.eval_session --policy reports/manifold_g1/primitive_torch/bc_policy.pt
    python3 -m manifold_g1.eval_session --clips 8 --device cpu --show 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C

CLIP_DIR = C.REPO / "reports" / "manifold_g1" / "clips"
DEFAULT_POLICY = C.REPO / "reports" / "manifold_g1" / "primitive_torch" / "bc_policy.pt"
OUT = C.REPO / "reports" / "manifold_g1" / "eval_session.json"
SETTLE_STEPS = 6


def _clips(explicit, limit: int) -> list[Path]:
    if explicit:
        return list(explicit)
    return sorted(CLIP_DIR.glob("ep*.npz"))[:limit]


def evaluate(args) -> int:
    import mujoco
    import torch

    from .body_envelope import body_points
    from .env import G1FlatEnv
    from .kinematics import TorchKinematics
    from .manifold import EllipsoidManifold, Primitive
    from .pose_policy import POSE_DIM, action_to_pose, observation
    from .ppo import PPO
    from .primitive import load_demos

    kin = TorchKinematics(device=args.device)
    _, all_demos = load_demos()
    ppo = PPO(args.obs_dim, POSE_DIM, device=args.device)
    ppo.load(args.policy)

    clips = _clips(args.clip, args.clips)
    if not clips:
        print(f"no clips under {CLIP_DIR}; collect data first")
        return 1

    env = G1FlatEnv(C.FLAT_SCENE)
    rows = []
    for path in clips:
        with np.load(path) as handle:
            data = {k: handle[k] for k in handle.files}
        q_hw = data["q_act"][:, C.ISAACLAB_TO_MUJOCO]
        # `envelope_t[i] == t[i * stride]` (recompute_envelopes writes exactly that), so the tick
        # belonging to envelope i is i * stride. Pairing across the stride measures a pose against
        # a stale envelope and pushes the recorded column above 1 for no good reason.
        stride = max(1, len(data["t"]) // len(data["envelope_semi"]))
        ticks = list(range(0, len(data["envelope_semi"]), max(1, len(data["envelope_semi"])
                                                              // args.per_clip)))

        for index in ticks:
            tick = index * stride
            semi = data["envelope_semi"][index]

            env.reset()
            env.data.qpos[0:3] = data["base_pos"][tick]
            env.data.qpos[3:7] = data["base_quat"][tick]
            env.data.qpos[env.body_qadr] = q_hw[tick]
            env.data.qpos[env.hand_qadr] = 0.0
            mujoco.mj_forward(env.model, env.data)

            points = body_points(env.model, env.data, max_vertices=args.vertices)
            pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            pelvis = env.data.xpos[pelvis_id]
            # the stored envelope is in the pelvis frame, so the pelvis yaw has to come out
            rot = env.data.xmat[pelvis_id].reshape(3, 3)
            recorded_pose = data["q_act"][tick] - C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB]

            manifold = EllipsoidManifold([Primitive(center=np.zeros(3), semi=semi)])
            rel_recorded = (points - pelvis) @ rot
            r_recorded = float(manifold.radii(rel_recorded).max())
            box_recorded = float((np.abs(rel_recorded) / semi).max())   # inside the recorded box?

            demos = _closest_demos(all_demos, semi)
            _, med_dev, near = spread_of(demos)

            pose = np.zeros(POSE_DIM)
            with torch.no_grad():
                for _ in range(SETTLE_STEPS):
                    clone_points = _points(kin, pose)
                    r_now = float(manifold.radii(clone_points).max())
                    obs = observation(pose, manifold.primitives[0], r_now, np.zeros(3))
                    action, _, _ = ppo.act(obs, deterministic=True)
                    target = np.asarray(action_to_pose(action, torch))
                    pose = pose + (target - pose) * 0.5
                clone_points = _points(kin, pose)
                r_cloned = float(manifold.radii(clone_points).max())
                box_cloned = float((np.abs(clone_points) / semi).max())

            rows.append({
                "clip": path.stem, "tick": tick, "semi": semi.tolist(),
                "r_recorded": r_recorded, "r_cloned": r_cloned,
                "box_recorded": box_recorded, "box_cloned": box_cloned,
                "spread": med_dev, "near": near, "demos": int(len(demos)),
                "pose_err": float(np.abs(pose - recorded_pose).mean()),
            })

    report(rows, args)
    OUT.write_text(json.dumps({"policy": str(args.policy), "ruler": "surface, pelvis frame",
                               "rows": rows}, indent=2))
    print(f"\n-> {OUT}")
    return 0


def _points(kin, pose: np.ndarray) -> np.ndarray:
    """Body-surface points of one pose, the same ruler the clip's envelope uses.

    `TorchKinematics.forward` is the torch twin of `body_envelope.body_points` (verified
    geom-by-geom against MuJoCo), and it returns pelvis-anchored points already, so it is
    directly comparable to the stored envelope.
    """
    import torch

    return kin.forward(torch.as_tensor(pose[None, :], dtype=kin.dtype,
                                       device=kin.device))[0].cpu().numpy()


def spread_of(poses: np.ndarray):
    from .demo_spread import spread

    return spread(poses)


def _closest_demos(all_demos: dict, semi: np.ndarray) -> np.ndarray:
    keys = np.stack([np.array([float(v) for v in k.split(",")][:3]) for k in all_demos])
    nearest = int(np.argmin(np.abs(keys - semi[None, :]).mean(axis=1)))
    return all_demos[list(all_demos)[nearest]]


def report(rows: list[dict], args) -> None:
    r_rec = np.array([r["r_recorded"] for r in rows])
    r_clo = np.array([r["r_cloned"] for r in rows])
    b_rec = np.array([r["box_recorded"] for r in rows])
    b_clo = np.array([r["box_cloned"] for r in rows])
    print(f"{'clip':>20} {'tick':>5} | {'ellipsoid':>19} | {'box (recorded envelope)':>24}")
    print(f"{'':>20} {'':>5} | {'recorded':>9} {'cloned':>9} | {'recorded':>11} {'cloned':>12}")
    for r in rows[:args.show]:
        print(f"{r['clip']:>20} {r['tick']:>5} | {r['r_recorded']:>9.2f} {r['r_cloned']:>9.2f} | "
              f"{r['box_recorded']:>11.3f} {r['box_cloned']:>12.2f}")

    print(f"\n{len(rows)} sampled ticks (at the envelope's own sampling instants)")
    print(f"  box test      - recorded pose inside its own envelope: "
          f"{int((b_rec <= 1.0 + 1e-6).sum())}/{len(rows)}   (1.000 by construction: the envelope")
    print(f"                  was measured from that pose; a real departure means the "
          f"reconstruction drifted)")
    print(f"                  the clone's pose inside it:           "
          f"{int((b_clo <= 1.0).sum())}/{len(rows)}   ({float((b_clo <= 1.0).mean()):.0%})")
    print(f"  ellipsoid test - recorded: {int((r_rec <= 1.0).sum())}/{len(rows)}   "
          f"cloned: {int((r_clo <= 1.0).sum())}/{len(rows)}")
    print(f"  recorded: box {np.median(b_rec):.3f} median (max {b_rec.max():.4f}), "
          f"ellipsoid {np.median(r_rec):.2f} median")
    print(f"  cloned  : box {np.median(b_clo):.2f} median, ellipsoid {np.median(r_clo):.2f} median")
    print(f"  clone pose error vs the recorded pose: "
          f"{np.mean([r['pose_err'] for r in rows]):.3f} rad")
    print(f"\nThe two radii are different shapes: the stored envelope is a box, the manifold is the")
    print(f"ellipsoid inscribed in it. A box corner sits at r = 1.73 and the real standing body at")
    print(f"r = 1.35 against its own envelope, so 'ellipsoid r > 1' is not 'outside the recording'.")
    print(f"Use the box column for containment in the recorded envelope; the ellipsoid column is")
    print(f"what the policy's reward actually optimises, and the gap between them is worth knowing.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Recorded-vs-cloned containment, one ruler")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--clip", type=Path, action="append", default=None)
    parser.add_argument("--clips", type=int, default=6, help="how many recorded clips to use")
    parser.add_argument("--per-clip", type=int, default=20,
                        help="envelope samples per clip")
    parser.add_argument("--vertices", type=int, default=120,
                        help="max mesh vertices per geom, matching the envelope's own sampling")
    parser.add_argument("--obs-dim", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--show", type=int, default=14)
    args = parser.parse_args()
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
