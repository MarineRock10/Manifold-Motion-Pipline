"""Primitive model: manifold -> pose, with a 29-joint pose (Phase 2.2/2.3).

The primitive decides *what posture to hold*: `p_φ(q | M, s)` where `q` is the full 29-joint
pose as a delta from the default standing pose, in policy (IsaacLab) order. Stage 2 turns that
posture into a moving trajectory; this stage only chooses the posture.

Two things are combined, the split the project's design calls for:

  * **RL keeps it feasible** - the policy is trained in the geometric model with the same
    containment/manifold reward as the 4-D static stage, so the pose it picks actually fits
    the ellipsoid and is one the frozen controller can hold;
  * **imitation keeps the styles** - the recorded motion (Phase 2.1 demos) supplies, for each
    manifold, several poses that were really executed, and the policy is pulled towards them.

A plain regression on the demos would collapse to the mean pose (a unimodal answer); the RL
term is what keeps a distribution. The `report` mode measures both: feasibility (containment,
SONIC check) and style spread (how far the sampled poses sit from their own mean, against the
demonstrations' own spread).

    python3 -m manifold_g1.primitive train  --iterations 400          # RL + imitation
    python3 -m manifold_g1.primitive train  --iterations 0 --bc-only  # imitation only
    python3 -m manifold_g1.primitive report
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import constants as C
from .demos import DEFAULT_ISAAC, pose_from_joints
from .manifold import EllipsoidManifold, Primitive as EllipsoidPrimitive
from .ppo import PPO, RolloutBuffer
from .family import BASE_SEMI, ManifoldSpec, report_specs
from .static_fit import BodyModel

REPO = C.REPO
DEMOS = REPO / "reports" / "manifold_g1" / "dataset" / "pose_demos.npz"
OUT_DIR = REPO / "reports" / "manifold_g1" / "primitive"
POSE_DIM = 29
# observation layout: obs() builds it; the dimension is read off at runtime by the env
# (3 pose stats + |q| mean + std + containment + 3 centre + 3 semi + 3 axis)
POSE_LIMIT = 1.4                 # rad, the demo cut-off: poses stay inside the recorded range


@dataclass
class PrimitiveConfig:
    dt: float = 0.1
    max_episode_steps: int = 8          # reach a pose and hold it; the clock is the same as
                                        # deployment (0.8 s), and short enough to end inside one
                                        # rollout, so finished episodes are actually observed
    tau: float = 0.45                            # first-order pose response (measured on SONIC)
    w_inside: float = 1.0                        # per second and unit of containment margin
    margin_cap: float = 0.03
    w_outside: float = 25.0
    w_effort: float = 0.4
    w_spine: float = 5.0
    w_imitation: float = 1.0                     # pull towards the demonstrated pose
    # --- kinematic feasibility: the geometric containment says the pose fits the manifold,
    #     these say the robot could actually stand in it
    w_com: float = 3.0                           # COM not over the feet
    w_foot: float = 3.0                          # a foot lifted off the floor
    w_limit: float = 5.0                         # joint-limit overshoot
    success_hold: int = 3
    success_margin: float = 0.98


class PrimitiveEnv:
    """Geometric primitive environment: the action is a 29-joint pose delta."""

    action_dim = POSE_DIM

    def __init__(self, body: BodyModel, manifold: EllipsoidManifold, demos: dict,
                 cfg: PrimitiveConfig | None = None, kin=None):
        self.body = body
        self.manifold = manifold
        self.demos = demos
        self.cfg = cfg or PrimitiveConfig()
        self.kin = kin                      # TorchKinematics for the kinematic penalties
        self.reset()

    # -- observation -------------------------------------------------------
    def obs(self) -> np.ndarray:
        return self._obs(self.manifold.primitives[0], self.pose)

    def _obs(self, primitive: EllipsoidPrimitive, pose: np.ndarray) -> np.ndarray:
        """Pose embedding + containment + manifold, so model and deployment share one layout."""
        points = self.body.mesh_points_pose(pose)
        pelvis = self.body.at_pose(pose)[self.body.pelvis_index()]
        radius = float(self.manifold.radii(points).max())
        return np.concatenate([
            pose[:3] / POSE_LIMIT, [pose.mean(), pose.std()], [radius],
            (primitive.center - pelvis) / 0.5, primitive.semi, primitive.axis(),
        ]).astype(np.float32)

    def radius(self, pose: np.ndarray) -> float:
        """Containment of the body surface in the manifold (mesh envelope, same as the family)."""
        return float(self.manifold.radii(self.body.mesh_points_pose(pose)).max())

    def _kinematic_penalty(self, cfg: PrimitiveConfig) -> float:
        """How far this pose is from one the robot could stand in (COM / feet / joint limits)."""
        if self.kin is None:
            return 0.0
        out = self.kin.stability(np.asarray(self.pose)[None, :])
        com = out["com"][0].cpu().numpy()
        feet = out["feet_z"][0].cpu().numpy()
        limit = float(out["limit"][0])
        return (cfg.w_com * max(0.0, np.linalg.norm(com[:2]) - 0.08)      # COM inside the feet
                + cfg.w_foot * float(np.maximum(0.0, feet - feet.min() - 0.02).sum())
                + cfg.w_limit * limit)

    def demonstrated(self) -> np.ndarray | None:
        """The demo pose for the current manifold, if this manifold has any."""
        return self.demos.get(self._key(), None)

    def _key(self) -> str:
        primitive = self.manifold.primitives[0]
        return ",".join(f"{v:.2f}" for v in list(primitive.semi) + list(primitive.center))

    def reset(self, **_):
        self.pose = np.zeros(POSE_DIM)
        self.steps = 0
        self.inside_ticks = 0
        self.episode_return = 0.0
        return self.obs(), {"radius": self.radius(self.pose)}

    def step(self, action: np.ndarray):
        cfg = self.cfg
        target = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0) * POSE_LIMIT
        demo = self.demonstrated()
        reward = 0.0
        for _ in range(5):
            alpha = min(1.0, C.CONTROL_DT / cfg.tau)
            self.pose += (target - self.pose) * alpha
            radius = self.radius(self.pose)
            margin = min(1.0 - radius, cfg.margin_cap)
            reward += (cfg.w_inside * margin if radius <= cfg.success_margin
                       else -cfg.w_outside * (radius - cfg.success_margin)) * C.CONTROL_DT
            reward -= cfg.w_effort * float(np.abs(self.pose).mean()) / POSE_LIMIT * C.CONTROL_DT
            reward -= self._kinematic_penalty(cfg) * C.CONTROL_DT
            if demo is not None:                     # imitation: stay near a demonstrated pose
                reward -= cfg.w_imitation * float(np.abs(self.pose - demo).mean()) * C.CONTROL_DT
            self.inside_ticks = self.inside_ticks + 1 if radius <= cfg.success_margin else 0
        self.steps += 1
        terminated = truncated = False
        outcome = "running"
        if self.inside_ticks >= cfg.success_hold:
            reward += 5.0
            terminated, outcome = True, "success"
        elif self.steps >= cfg.max_episode_steps:
            truncated, outcome = True, "timeout"
        self.episode_return += reward
        radius = self.radius(self.pose)
        return self.obs(), float(reward), terminated, truncated, {
            "outcome": outcome, "episode_return": self.episode_return, "pose": self.pose.copy(),
            "manifold_radius": radius}


class _ManifoldBatch:
    """Per-stream ellipsoids held as tensors so containment runs on the GPU in one pass.

    Uploads are deduplicated: episodes end every few steps and a finished stream restarts on one
    of a handful of manifolds, so a reset only touches the device when the stream's manifold
    actually changed (the tensors are compared by identity marker, not by value).
    """

    def __init__(self, kin, n: int):
        self.kin = kin
        self.torch = kin.torch
        self.n = n
        self.marker = [None] * n
        dev, dt = kin.device, kin.dtype
        self.semi = self.torch.ones(n, 3, dtype=dt, device=dev)
        self.center = self.torch.zeros(n, 3, dtype=dt, device=dev)
        self.rot = self.torch.eye(3, dtype=dt, device=dev).expand(n, 3, 3).clone()

    def assign(self, index, manifolds: list, markers: list | None = None) -> None:
        """Upload the manifolds of these streams; skip any stream whose marker is unchanged."""
        if markers is not None:
            fresh = [(int(i), m, k) for i, m, k in zip(index, manifolds, markers)
                     if k is None or self.marker[int(i)] is not k]
            if not fresh:
                return
            index = np.array([f[0] for f in fresh], dtype=np.int64)
            manifolds = [f[1] for f in fresh]
            for i, _, k in fresh:
                self.marker[i] = k
        from .manifold import quat_to_matrix

        torch = self.torch
        idx = torch.as_tensor(np.asarray(index, dtype=np.int64), device=self.kin.device)
        self.semi[idx] = torch.as_tensor(
            np.stack([m.primitives[0].semi for m in manifolds]), dtype=self.kin.dtype,
            device=self.kin.device)
        self.center[idx] = torch.as_tensor(
            np.stack([m.primitives[0].center for m in manifolds]), dtype=self.kin.dtype,
            device=self.kin.device)
        self.rot[idx] = torch.as_tensor(
            np.stack([quat_to_matrix(m.primitives[0].quat) for m in manifolds]),
            dtype=self.kin.dtype, device=self.kin.device)

    def radius(self, poses, streams=None) -> "np.ndarray":
        """Containment for the given rows, each against its stream's manifold; numpy [B]."""
        torch = self.torch
        poses = torch.as_tensor(np.asarray(poses, dtype=np.float64), dtype=self.kin.dtype,
                                device=self.kin.device)
        index = (torch.arange(self.n, device=self.kin.device) if streams is None
                 else torch.as_tensor(np.asarray(streams, dtype=np.int64),
                                      device=self.kin.device))
        points, mask = self.kin.forward(poses, return_mask=True)         # [B, K, 3]
        # [B, 3] -> [B, 1, 3] so the subtraction broadcasts across the point axis, not the batch
        center = self.center[index].unsqueeze(1)
        semi = self.semi[index].unsqueeze(1)
        rot = self.rot[index]
        local = torch.einsum("bki,bij->bkj", points - center, rot)
        radius = (local / semi).norm(dim=-1)
        radius = radius.masked_fill(~mask, -1.0).max(dim=-1).values
        return radius.detach().cpu().numpy()


class BatchedPrimitiveEnv:
    """The same environment for many manifolds at once (the pose model is vectorized)."""

    action_dim = POSE_DIM

    def __init__(self, body: BodyModel, cfg: PrimitiveConfig | None = None, kin=None):
        self.body = body
        self.cfg = cfg or PrimitiveConfig()
        self.kin = kin                      # TorchKinematics, or None for the MuJoCo path
        self.batch = None
        self.n = 0

    def set_manifolds(self, manifolds: list[EllipsoidManifold], demos: list[np.ndarray | None]):
        primitives = [m.primitives[0] for m in manifolds]
        self.n = len(manifolds)
        self.manifolds = list(manifolds)
        self.center = np.array([p.center for p in primitives])
        self.semi = np.array([p.semi for p in primitives])
        self.axis = np.array([p.axis() for p in primitives])
        self.rot = np.array([np.eye(3) if np.allclose(p.quat, [1, 0, 0, 0]) else _quat_to_matrix(p.quat)
                             for p in primitives])
        self.demo = np.array([d if d is not None else np.full(POSE_DIM, np.nan) for d in demos])
        self.manifolds = list(manifolds)
        if self.kin is not None:
            self.batch = _ManifoldBatch(self.kin, self.n)
            self.batch.assign(np.arange(self.n), manifolds, list(manifolds))
        self.pose = np.zeros((self.n, POSE_DIM))
        self.steps = np.zeros(self.n, dtype=int)
        self.inside = np.zeros(self.n, dtype=int)
        self.episode_return = np.zeros(self.n)
        return self.obs()

    def set_manifolds_at(self, index: np.ndarray, manifolds: list[EllipsoidManifold],
                         demos: list[np.ndarray | None]) -> None:
        primitives = [m.primitives[0] for m in manifolds]
        for slot, manifold in zip(index, manifolds):
            self.manifolds[int(slot)] = manifold
        if self.batch is not None:
            self.batch.assign(index, manifolds, manifolds)
        self.center[index] = np.array([p.center for p in primitives])
        self.semi[index] = np.array([p.semi for p in primitives])
        self.axis[index] = np.array([p.axis() for p in primitives])
        self.rot[index] = np.array([_quat_to_matrix(p.quat) for p in primitives])
        self.demo[index] = np.array([d if d is not None else np.full(POSE_DIM, np.nan)
                                     for d in demos])
        self.reset_streams(index)

    def reset_streams(self, index: np.ndarray) -> None:
        self.pose[index] = 0.0
        self.steps[index] = 0
        self.inside[index] = 0
        self.episode_return[index] = 0.0

    def radius(self, poses: np.ndarray, streams: np.ndarray | None = None) -> np.ndarray:
        """Containment per pose, each pose scored against its own stream's manifold.

        `streams` selects which manifolds the rows belong to (default: all of them, in order).
        """
        if self.batch is not None:                  # exact FK on the GPU, one pass
            return self.batch.radius(poses, streams)
        out = np.empty(len(poses))
        select = range(len(poses)) if streams is None else streams
        for i, stream in enumerate(select):         # MuJoCo fallback (reference path)
            points = self.body.mesh_points_pose(poses[i])
            out[i] = float(self.manifolds[int(stream)].radii(points).max())
        return out

    def obs(self) -> np.ndarray:
        pelvis = self.body.at_pose_batch(self.pose)[:, self.body.pelvis_index(), :]
        return np.concatenate([
            self.pose[:, :3] / POSE_LIMIT, self.pose.mean(axis=1, keepdims=True),
            self.pose.std(axis=1, keepdims=True), self.radius(self.pose)[:, None],
            (self.center - pelvis) / 0.5, self.semi, self.axis,
        ], axis=1).astype(np.float32)

    def step(self, actions: np.ndarray):
        cfg = self.cfg
        targets = np.clip(np.atleast_2d(actions), -1.0, 1.0) * POSE_LIMIT
        reward = np.zeros(self.n)
        alpha = min(1.0, C.CONTROL_DT / cfg.tau)
        for _ in range(5):
            self.pose += (targets - self.pose) * alpha
            radius = self.radius(self.pose)
            inside = radius <= cfg.success_margin
            margin = np.minimum(1.0 - radius, cfg.margin_cap)
            reward += np.where(inside, cfg.w_inside * margin,
                               -cfg.w_outside * (radius - cfg.success_margin)) * C.CONTROL_DT
            reward -= cfg.w_effort * np.abs(self.pose).mean(axis=1) / POSE_LIMIT * C.CONTROL_DT
            if self.kin is not None:
                out = self.kin.stability(self.pose)
                com = out["com"].detach().cpu().numpy()
                feet = out["feet_z"].detach().cpu().numpy()
                limit = out["limit"].detach().cpu().numpy()
                reach = np.linalg.norm(com[:, :2], axis=1)
                # a foot counts as lifted when it sits well above the lower of the two
                spread = feet.max(axis=1) - feet.min(axis=1)
                reward -= (cfg.w_com * np.maximum(0.0, reach - 0.08)
                           + cfg.w_foot * np.maximum(0.0, spread - 0.02)
                           + cfg.w_limit * limit) * C.CONTROL_DT
            with np.errstate(invalid="ignore"):
                gap = np.abs(self.pose - self.demo).mean(axis=1)
            gap = np.nan_to_num(gap, nan=0.0)
            reward -= cfg.w_imitation * gap * C.CONTROL_DT
            self.inside = np.where(inside, self.inside + 1, 0)
        self.steps += 1
        success = self.inside >= cfg.success_hold
        timeout = ~success & (self.steps >= cfg.max_episode_steps)
        reward = reward + np.where(success, 5.0, 0.0)
        self.episode_return += reward
        return self.obs(), reward, success, timeout, {
            "radius": self.radius(self.pose), "pose": self.pose.copy(),
            "episode_return": self.episode_return,
            "outcome": np.where(success, "success", np.where(timeout, "timeout", "running"))}


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    from .manifold import quat_to_matrix
    return quat_to_matrix(q)


def demo_key(semi: np.ndarray, center: np.ndarray) -> str:
    return ",".join(f"{v:.2f}" for v in list(np.asarray(semi).ravel()) + list(np.asarray(center).ravel()))


def load_demos(path: Path = DEMOS) -> tuple[dict, dict]:
    """Demo poses grouped by the manifold they were recorded under.

    Returns (mean pose per manifold, all poses per manifold). The keys are coarse (2 decimals)
    because the identities are meant to match between the recorded envelope and a queried one.
    """
    with np.load(path) as handle:
        semi, center, pose = handle["semi"], handle["center"], handle["pose"]
    table: dict[str, list] = {}
    for i in range(len(pose)):
        table.setdefault(demo_key(semi[i], center[i]), []).append(pose[i])
    return ({k: np.mean(v, axis=0) for k, v in table.items()},
            {k: np.array(v) for k, v in table.items()})


def nearest_demos(manifold: EllipsoidManifold, all_demos: dict, radius: float = 0.12,
                  min_count: int = 16):
    """Recorded poses that are both *near* this manifold and *fit inside* it.

    Proximity alone is not enough: a recorded pose carries the envelope of the tick it came
    from, so a pose near (but not inside) the queried manifold teaches the policy to leave the
    ellipsoid - the RL containment term and the imitation term then pull in opposite
    directions, and training drifts out and collapses. So the candidates are filtered by an
    actual containment test, and the caller gets None when too few survive, which degrades that
    manifold to plain RL rather than training on a contradiction.
    """
    primitive = manifold.primitives[0]
    target = np.concatenate([primitive.semi, primitive.center])
    scale = np.maximum(primitive.semi, 1e-3)
    picked = []
    for key, poses in all_demos.items():
        values = np.array([float(v) for v in key.split(",")])
        if float(np.linalg.norm((values[:3] - target[:3]) / scale)) > radius:
            continue
        picked.append(poses)
    if not picked:
        return None
    candidates = np.concatenate(picked)
    fits = _fit_mask(manifold, candidates)
    if fits.sum() < min_count:
        return None
    return candidates[fits]


_FIT_CACHE: dict = {}


def _fit_mask(manifold: EllipsoidManifold, poses: np.ndarray) -> np.ndarray:
    """Which of these poses actually sit inside the manifold (cached per manifold)."""
    key = (tuple(np.round(manifold.primitives[0].semi, 4)),
           tuple(np.round(manifold.primitives[0].center, 4)))
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]
    body = BodyModel()
    fits = np.array([float(manifold.radii(body.mesh_points_pose(q)).max()) <= 1.0 for q in poses])
    _FIT_CACHE[key] = fits
    return fits


_BEST: dict = {}


def _progress(iteration: int, total: int, started: float, stats: dict) -> None:
    """One updating line, metrics first.

    The bar is the only thing that shows overall progress; every number after it is a running
    metric of the current iteration, so the reader never has to guess which percentage means
    what. `best` is the best success so far - a monotone column, which is what makes "is this
    still improving" answerable at a glance.
    """
    import sys

    elapsed = time.perf_counter() - started
    done = iteration / total
    eta = elapsed / max(done, 1e-9) - elapsed
    best = _BEST.get("success")
    if stats["success"] == stats["success"]:                      # not nan
        best = stats["success"] if best is None else max(best, stats["success"])
        _BEST["success"] = best
    bar_width = 24
    filled = int(bar_width * done)
    bar = "#" * filled + "." * (bar_width - filled)
    bands = stats.get("bands")
    band_txt = ""
    if bands:
        band_txt = ("  [e/m/h " + "/".join(
            "--" if v != v else f"{v:.2f}" for v in
            (bands.get("easy"), bands.get("mid"), bands.get("hard"))) + "]")
    line = (f"iter {iteration:4d}/{total} [{bar}]  "
            f"succ {stats['success']:.2f} (best {best if best is not None else float('nan'):.2f})"
            f"{band_txt}  "
            f"r {stats['r']:.2f}  imit {stats['imit']:.3f}  ret {stats['return']:6.2f}  "
            f"eps {stats['eps']:4d}   {elapsed / 60:4.1f}m +{eta / 60:4.1f}m")
    if sys.stdout.isatty():
        sys.stdout.write("\r" + line)
        if iteration >= total:
            sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        print(line, flush=True)


def _demo_for(manifold: EllipsoidManifold, mean_demos: dict, all_demos: dict):
    """A demonstrated pose for this manifold, or None when no recording fits it.

    None means "train this manifold on feasibility alone": a pose that does not fit would make
    the imitation term fight the containment reward, which is what previously collapsed runs.
    """
    near = nearest_demos(manifold, all_demos)
    return near.mean(axis=0) if near is not None else None


def _body_model() -> BodyModel:
    return BodyModel()


def _obs_dim(env) -> int:
    """Observation width, read from the environment so it cannot drift out of sync.

    The single-stream env returns a flat vector, the batched one a matrix; both are accepted.
    """
    obs = env.obs()
    return int(obs.shape[-1])


def train(args) -> int:
    body = _body_model()
    rng = np.random.default_rng(args.seed)
    mean_demos, all_demos = load_demos()
    print(f"demos: {len(mean_demos)} manifolds, {sum(len(v) for v in all_demos.values())} poses")

    cfg = PrimitiveConfig(w_imitation=0.0 if args.bc_only else args.w_imitation)
    kin = None
    if args.surrogate:
        from .kinematics import TorchKinematics
        kin = TorchKinematics(device=args.device)
        print(f"surrogate FK on {kin.device}: {kin.total_vertices} surface points")
    env = BatchedPrimitiveEnv(body, cfg, kin=kin)
    obs_dim = int(env.set_manifolds([report_specs()[0].build()],
                                    [None]).shape[1])
    ppo = PPO(obs_dim, POSE_DIM, init_log_std=[-0.3] * POSE_DIM)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.iterations == 0:
        print("iterations=0: nothing to train")
        return 1

    log = open(out_dir / "train_log.csv", "w")
    log.write("iteration,episodes,success,radius_mean,imitation_gap,return_mean\n")
    started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        frac = (iteration - 1) / max(1, args.iterations - 1)
        ppo.entropy_coef = args.entropy_start + (args.entropy_end - args.entropy_start) * frac
        specs = [report_specs()[i % len(report_specs())]
                 for i in range(args.envs)]
        manifolds = [s.build() for s in specs]
        demos = [_demo_for(m, mean_demos, all_demos) for m in manifolds]
        obs = env.set_manifolds(manifolds, demos)
        buffer = RolloutBuffer(args.rollout_steps * args.envs, obs_dim, POSE_DIM)
        returns, outcomes, radii, gaps = [], [], [], []
        for _ in range(args.rollout_steps):
            action, logprob, value = ppo.act_batch(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            buffer.add_batch(obs, action, logprob, reward, value, done | truncated)
            obs = next_obs
            finished = np.flatnonzero(done | truncated)
            if len(finished):
                returns.extend(info["episode_return"][finished].tolist())
                outcomes.extend(info["outcome"][finished].tolist())
                radii.extend(info["radius"][finished].tolist())
                for i in finished:
                    if not np.isnan(env.demo[i]).all():
                        gaps.append(float(np.abs(info["pose"][i] - env.demo[i]).mean()))
                env.set_manifolds_at(finished, [manifolds[i] for i in finished],
                                     [demos[i] for i in finished])
                obs = env.obs()
        ppo.update(buffer, ppo.value_batch(obs))
        success = float(np.mean([o == "success" for o in outcomes])) if outcomes else float("nan")
        log.write(f"{iteration},{len(outcomes)},{success:.3f},{np.mean(radii) if radii else 0:.3f},"
                  f"{np.mean(gaps) if gaps else float('nan'):.4f},"
                  f"{np.mean(returns) if returns else 0:.3f}\n")
        log.flush()
        _progress(iteration, args.iterations, started, {
            "success": success,
            "r": float(np.mean(radii)) if radii else float("nan"),
            "imit": float(np.mean(gaps)) if gaps else float("nan"),
            "return": float(np.mean(returns)) if returns else float("nan"),
            "eps": len(outcomes),
        })
        ppo.save(out_dir / "policy.pt")
    log.close()
    print(f"\n=== training finished: {args.iterations} iterations in "
          f"{time.perf_counter() - started:.0f} s, success {success:.2f}, "
          f"policy -> {out_dir / 'policy.pt'} ===", flush=True)
    return 0


def report(args) -> int:
    """Feasibility and style spread of the trained primitive, against the demonstrations."""
    body = _body_model()
    mean_demos, all_demos = load_demos()
    env = PrimitiveEnv(body, report_specs()[0].build(), mean_demos,
                       PrimitiveConfig())
    ppo = PPO(_obs_dim(env), POSE_DIM)
    ppo.load(Path(args.policy))
    print(f"{'manifold':>26} {'r_pol':>6} {'r_demo':>7} {'gap_pol':>8} {'gap_spread':>11} "
          f"{'demo_spread':>12} {'outcome':>9}")
    rows = []
    for spec in report_specs():
        manifold = spec.build()
        demos = nearest_demos(manifold, all_demos)
        env.manifold = manifold
        obs, _ = env.reset()
        samples = []
        for _ in range(6):                                   # sample the policy's distribution
            obs, _ = env.reset()
            for _ in range(env.cfg.max_episode_steps):
                action, _, _ = ppo.act(obs)
                action = np.clip(action, -1.0, 1.0)
                obs, _, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            samples.append(info["pose"])
        samples = np.stack(samples)
        radius = float(manifold.radii(body.at_pose_batch(samples)).max(axis=1).mean())
        rows.append({"manifold": spec.label(), "radius": radius,
                     "policy_spread": float(np.degrees(samples.std(axis=0)).mean()),
                     "demo_spread": float(np.degrees(demos.std(axis=0)).mean()) if demos is not None else None})
        spread = "  n/a" if demos is None else f"{np.degrees(demos.std(axis=0)).mean():11.1f}"
        r_demo = "-" if demos is None else f"{float(manifold.radii(body.at_pose_batch(demos)).max(axis=1).mean()):.2f}"
        print(f"{spec.label():>26} {radius:>6.2f} {r_demo:>7} {'-':>8} "
              f"{np.degrees(samples.std(axis=0)).mean():>11.1f} {spread} "
              f"{'success' if radius <= 1 else 'outside':>9}")
    (Path(args.out) / "report.json").write_text(json.dumps(rows, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Primitive model: manifold -> 29-joint pose")
    parser.add_argument("mode", choices=("train", "report"))
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1024, help="unused, kept for symmetry")
    parser.add_argument("--w-imitation", type=float, default=1.0)
    parser.add_argument("--bc-only", action="store_true", help="ablation: no RL feasibility term")
    parser.add_argument("--surrogate", action="store_true",
                        help="judge containment with the batched GPU kinematics instead of "
                             "MuJoCo per-pose sampling (large batches only make sense here)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--entropy-start", type=float, default=0.005)
    parser.add_argument("--entropy-end", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--policy", type=Path, default=OUT_DIR / "policy.pt")
    args = parser.parse_args()
    return train(args) if args.mode == "train" else report(args)


if __name__ == "__main__":
    raise SystemExit(main())
