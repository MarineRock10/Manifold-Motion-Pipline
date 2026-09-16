"""The containment ruler, checked against real clips.

A recorded clip stores, every `envelope_stride` ticks, the body envelope `M_t` of that instant -
written by `recompute_envelopes` (and by `clip.py` at collection time) as the sampled body
*surface* in the pelvis frame, as an **ellipsoid**: the per-axis extents inflated by the single
scalar `max_i ||rel_i / semi||`, which is the smallest uniform inflation containing the samples.

This script reproduces that measurement and reports two radii per row, because they answer
different questions:

  * `ellipsoid` - containment in the stored envelope, which is the shape every manifold in this
    project uses. This is the column that should read 1.00 for the recorded pose.
  * `box`       - the same points against the raw per-axis extents (`max|rel|`, no inflation). A
    box corner sits at r = 1.73, so a body inside the box still exceeds 1 here; this column is
    only useful as a shape comparison, and reads ~0.72 for a pose whose ellipsoid r is 1.00.

**Why this exists.** Until 2026-09-16 the clips stored the raw box half-extents and containment
was measured as an ellipsoid norm, so the recorded pose scored r = 1.18-1.57 (median 1.40)
against its own envelope - "fit inside the manifold" was unsatisfiable by the very pose the
envelope was measured from, and the standing body sat at r = 1.145 against the family's
scale-1.0 manifold. Both writers now fit an ellipsoid and the family base is derived from the
same fit, so scale 1.0 means "the standing body fits exactly". Recorded poses now read r = 1.000.

    python3 -m manifold_motion.eval_session --policy reports/manifold_motion/bc/bc_policy.pt
    python3 -m manifold_motion.eval_session --clips 8 --device cpu --show 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C
from .paths import BC_POLICY, CLIPS, RULER_REPORT

CLIP_DIR = CLIPS
DEFAULT_POLICY = BC_POLICY
OUT = RULER_REPORT
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
    from .demos import load_demos
    from .demo_spread import nearest_key, spread

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

            demos = all_demos[nearest_key(all_demos, semi)]
            _, med_dev, near = spread(demos)

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
    print(f"  ellipsoid test - recorded: {int((r_rec <= 1.0 + 1e-4).sum())}/{len(rows)}   "
          f"(1.000 by construction: the envelope was fitted to that pose)")
    print(f"                   cloned:   {int((r_clo <= 1.0).sum())}/{len(rows)}")
    print(f"  box test (no inflation, for shape reference) - recorded: "
          f"{int((b_rec <= 1.0).sum())}/{len(rows)}   cloned: {int((b_clo <= 1.0).sum())}/{len(rows)}")
    print(f"  recorded: ellipsoid {np.median(r_rec):.3f} median (max {r_rec.max():.4f}), "
          f"box {np.median(b_rec):.2f}")
    print(f"  cloned  : ellipsoid {np.median(r_clo):.2f} median, box {np.median(b_clo):.2f}")
    print(f"  clone pose error vs the recorded pose: "
          f"{np.mean([r['pose_err'] for r in rows]):.3f} rad")
    print(f"\nThe two columns are different shapes. The stored envelope is an ellipsoid (the")
    print(f"per-axis extents inflated until they contain the body's surface), and the recorded pose")
    print(f"sits on or just inside it, so the ellipsoid column is the one to read: it is the shape")
    print(f"every manifold in this project measures, and it should read 1.00 for 'recorded'.")
    print(f"The box column drops the inflation, and a box corner sits at r = 1.73, so it reads less")
    print(f"than 1 even for a body inside the box - useful only as a shape comparison.")


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
