"""Pose demonstrations cut out of recorded motion (Phase 2.1).

The primitive model outputs a pose, and the motion clips already contain poses: every recorded
tick is a body configuration the frozen controller actually held, together with the body
envelope it occupied. Cutting frames out of the clips gives labelled demonstrations without
any new simulation:

    pose demo = (M, s) -> q      # q is the full 29-joint pose, in policy (IsaacLab) order
                                 # as a delta from the default standing pose

The pose label is the measured joint vector itself, not a projection onto a few hand-designed
directions: a walking frame holds most of its configuration in joints (shoulder roll, elbows,
wrists, hip pitch) that no small basis covers, and the style differences between motion modes
live exactly there. Keeping all 29 joints is what leaves room for style.

    python3 -m manifold_g1.demos build      # -> reports/manifold_g1/dataset/pose_demos.npz
    python3 -m manifold_g1.demos show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C
from .family import MARGIN
from .pose_policy import POSE_DIM
from .reference import ARM_DIRECTION, CROUCH_DIRECTION, LEAN_DIRECTION, TWIST_DIRECTION

DEFAULT_ISAAC = C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB]
DEFAULT_STRIDE = 10                             # 0.2 s at 50 Hz
from .paths import CLIPS, DATASET, POSE_DEMOS

DEMOS = POSE_DEMOS
MAX_POSE_DELTA = 1.4                            # rad; joints further than this are doing motion,
                                                # not holding a pose (a knee at 66 deg swings past)


def demo_key(semi: np.ndarray, center: np.ndarray) -> str:
    """The manifold identity used as a key: semi then centre, rounded to 2 decimals.

    The rounding is deliberate - it is what lets a queried envelope match a recorded one - but
    it is also why a manifold can be very tight: the tolerance is coarse enough that nearby
    envelopes collapse to one key.
    """
    values = list(np.asarray(semi).ravel()) + list(np.asarray(center).ravel())
    return ",".join(f"{v:.2f}" for v in values)


def demo_values(key: str) -> np.ndarray:
    """`demo_key` reversed: a key back to `[semi(3), center(3)]`.

    Pairs with `demo_key`, and exists because that parse was open-coded nine times across the
    package (`np.array([float(v) for v in key.split(",")])`, then `[:3]` for the semi and `[3:]`
    for the centre). A change to the key format - an extra field, a different rounding - would
    have had to find all nine, and a missed one silently pairs poses with the wrong ellipsoid.
    """
    return np.array([float(v) for v in key.split(",")], dtype=np.float64)


def load_demos(path: Path = DEMOS) -> tuple[dict, dict]:
    """Demo poses grouped by the manifold they were recorded under.

    Returns (mean pose per manifold, all poses per manifold). Every consumer - the clone's
    training set, both viewers, the in-loop fine-tune - reads the demonstrations through this
    one function, so they cannot disagree about what the data is.
    """
    with np.load(path) as handle:
        semi, center, pose = handle["semi"], handle["center"], handle["pose"]
    table: dict[str, list] = {}
    for i in range(len(pose)):
        table.setdefault(demo_key(semi[i], center[i]), []).append(pose[i])
    return ({k: np.mean(v, axis=0) for k, v in table.items()},
            {k: np.array(v) for k, v in table.items()})


def pose_from_joints(q_isaac: np.ndarray) -> np.ndarray:
    """The pose label: joint angles relative to the default standing pose."""
    return np.asarray(q_isaac, dtype=np.float64) - DEFAULT_ISAAC


def pose_envelope(body_model, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pelvis-frame ellipsoid that *contains* the body surface at this pose.

    Half-extents alone do not contain a curved point set: an axis-aligned box corner is
    sqrt(3) times its own semi-axis away from the centre, so a body whose surface sits near the
    corners scores r ~ 1.45 even when the box was built from that same body (measured, invariant
    to sampling density). `fit_ellipsoid` inflates the half-extents until every sample is inside,
    which is exactly the missing step - the center it uses is the sample mean, not the extents
    midpoint, for the same reason.
    """
    from .body_envelope import fit_ellipsoid

    fit = fit_ellipsoid(body_model.mesh_points_pose(pose))
    return np.array(fit["semi"]), np.array(fit["center"])


def build(args) -> int:
    from .body_model import BodyModel as _BodyModel

    body_model = _BodyModel()
    rows = []
    for path in sorted(args.clips.glob("*.npz")):
        name = path.stem
        if not (name.startswith("mode") or name.startswith("ep")):
            continue
        summary = json.loads(path.with_suffix(".json").read_text())
        if summary["fell"]:
            continue
        with np.load(path) as handle:
            data = {k: handle[k] for k in handle.files}
        if "q_act" not in data:
            continue

        q = data["q_act"]
        semi = data["envelope_semi"]
        ticks = len(q)
        stride_env = max(1, ticks // len(semi))
        speed = np.linalg.norm(data["base_lin_vel"][:, :2], axis=1)

        for tick in range(0, ticks, args.stride):
            pose = pose_from_joints(q[tick])
            if np.abs(pose).max() > args.max_pose_delta:
                continue                                    # a mid-stride frame, not a held pose
            env_index = min(tick // stride_env, len(semi) - 1)
            # The demo's manifold must contain the demo's pose - that is the whole premise of pairing
            # them. The stored envelope comes from the recorded tick and is centred on that
            # tick's midpoint, so scaling only its semi-axes by MARGIN does not guarantee
            # containment when the midpoint moved even slightly (measured r = 1.22 for poses that
            # should score ~0.95). The manifold is therefore built from this pose: its own
            # midpoint, its own extents, with the margin applied where it is needed.
            ext, center = pose_envelope(body_model, pose)  # pelvis frame
            semi_here = ext * MARGIN
            rows.append({
                "clip": name, "tick": tick,
                "semi": semi_here,
                "center": center,
                "quat": data["base_quat"][tick],
                "pose": pose,
                "residual": float(np.abs(pose).max()),
                "speed": float(speed[tick]),
                "q": q[tick], "dq": data["dq_act"][tick],
                "base_lin_vel": data["base_lin_vel"][tick],
                "base_ang_vel": data["base_ang_vel"][tick],
                "contact": data["contact"][tick],
            })

    if not rows:
        print("no demos: run the mode/episode collection first")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "pose_demos.npz",
        clip=np.array([r["clip"] for r in rows]),
        tick=np.array([r["tick"] for r in rows]),
        semi=np.stack([r["semi"] for r in rows]),
        center=np.stack([r["center"] for r in rows]),
        quat=np.stack([r["quat"] for r in rows]),
        pose=np.stack([r["pose"] for r in rows]),
        residual=np.array([r["residual"] for r in rows]),
        speed=np.array([r["speed"] for r in rows]),
        q=np.stack([r["q"] for r in rows]),
        dq=np.stack([r["dq"] for r in rows]),
        base_lin_vel=np.stack([r["base_lin_vel"] for r in rows]),
        base_ang_vel=np.stack([r["base_ang_vel"] for r in rows]),
        contact=np.stack([r["contact"] for r in rows]),
    )

    poses = np.stack([r["pose"] for r in rows])
    semi = np.stack([r["semi"] for r in rows])
    clips = sorted({r["clip"] for r in rows})
    print(f"pose demos: {len(rows)} frames from {len(clips)} clips "
          f"(stride {args.stride} ticks = {args.stride / 50:.1f} s)")
    print(f"  pose is {POSE_DIM}-D (policy joint order, delta from default); "
          f"|delta| mean {np.abs(poses).mean():.3f}  p95 {np.percentile(np.abs(poses), 95):.3f} rad")
    # the joints that carry the most variation are where style lives
    order = np.argsort(-poses.std(axis=0))[:8]
    print("  most variable joints (deg):")
    for i in order:
        name = C.MOTOR_NAMES[int(C.ISAACLAB_TO_MUJOCO[i])]
        print(f"    {name:22s} mean {np.degrees(poses[:,i].mean()):+7.1f}  "
              f"std {np.degrees(poses[:,i].std()):5.1f}  "
              f"range {np.degrees(poses[:,i].min()):+7.1f}..{np.degrees(poses[:,i].max()):+7.1f}")
    for i, name in enumerate("xyz"):
        print(f"  envelope {name}: mean {semi[:,i].mean():.3f}  std {semi[:,i].std():.3f}")

    health = {
        "demos": len(rows), "clips": clips, "stride": args.stride, "pose_dim": POSE_DIM,
        "pose_stats": {"mean_abs": float(np.abs(poses).mean()),
                       "p95_abs": float(np.percentile(np.abs(poses), 95)),
                       "per_joint_std_deg": {
                           C.MOTOR_NAMES[int(C.ISAACLAB_TO_MUJOCO[i])]:
                               float(np.degrees(poses[:, i].std())) for i in range(POSE_DIM)}},
        "envelope_std": {n: float(semi[:,i].std()) for i, n in enumerate("xyz")},
    }
    (args.out / "pose_demos.json").write_text(json.dumps(health, indent=2))
    print(f"saved {args.out / 'pose_demos.npz'} and {args.out / 'pose_demos.json'}")
    return 0


def show(args) -> int:
    with np.load(args.out / "pose_demos.npz") as handle:
        clip, pose, semi, speed = handle["clip"], handle["pose"], handle["semi"], handle["speed"]
    names = [C.MOTOR_NAMES[int(C.ISAACLAB_TO_MUJOCO[i])] for i in range(pose.shape[1])]
    clips = sorted(set(clip.tolist()))
    print(f"{len(pose)} demos from {len(clips)} clips; per-joint std (deg) — "
          f"mean {np.degrees(pose.std(axis=0)).mean():.1f}, "
          f"max {np.degrees(pose.std(axis=0)).max():.1f}")

    # 1) is there style to imitate at all? spread of the mean pose across clips
    means = []
    for name in clips:
        means.append(pose[clip == name].mean(axis=0))
    means = np.stack(means)
    spread = np.degrees(means.std(axis=0))
    order = np.argsort(-spread)[:10]
    print(f"\nstyle separation across {len(clips)} clips (std of per-clip mean pose, deg):")
    for i in order:
        print(f"  {names[i]:22s} {spread[i]:6.1f}")

    # 2) within a manifold bin, are the demonstrated poses different? (imitation signal)
    key = np.round(np.concatenate([semi[:, :2], speed[:, None]], axis=1), 1)
    unique, counts = np.unique(key, axis=0, return_counts=True)
    bins = []
    for cell in unique[counts >= 5]:
        mask = np.all(key == cell, axis=1)
        if mask.sum() >= 5:
            bins.append(np.degrees(pose[mask].std(axis=0)))
    if bins:
        bins = np.stack(bins)
        print(f"\nwithin-manifold-bin pose spread: {len(bins)} bins with >=5 demos; "
              f"mean per-joint std {bins.mean():.1f} deg, max {bins.max():.1f} deg")
        ratio = bins.mean() / max(np.degrees(pose.std(axis=0)).mean(), 1e-9)
        verdict = ("multi-modal: imitation has signal" if ratio > 0.4
                   else "bins are near-unimodal: little to imitate")
        print(f"  within-bin / global std ratio {ratio:.2f} -> {verdict}")
    else:
        print("\nno manifold bin holds >=5 demos: the sampled manifolds are too spread out")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pose demos cut from recorded motion (Phase 2.1)")
    parser.add_argument("mode", choices=("build", "show"))
    parser.add_argument("--clips", type=Path, default=CLIPS)
    parser.add_argument("--out", type=Path, default=DATASET)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--max-pose-delta", type=float, default=MAX_POSE_DELTA,
                        help="drop frames whose largest joint delta exceeds this (rad)")
    args = parser.parse_args()
    return build(args) if args.mode == "build" else show(args)


if __name__ == "__main__":
    raise SystemExit(main())
