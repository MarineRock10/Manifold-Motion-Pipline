"""Static manifold fitting: given an ellipsoid, choose a pose that fits inside it.

The simplest manifold is the robot's own standing envelope (see `body_envelope.py`); here the
envelope is reshaped - squashed in height, pinched in width, shifted sideways - and the policy
learns the pose `p(pose | manifold)` that keeps the whole body inside. No locomotion.

Three pieces:

  1. `BodyModel` - the robot's landmark body model, calibrated over a (crouch, lean, twist)
     grid by `body_envelope.py`;
  2. `StaticFitEnv` - a pure-geometry environment (no MuJoCo, no SONIC) in which PPO learns
     `p(pose | manifold)`;
  3. `verify` / `view` - the same policy executed by frozen SONIC holding the keyframe
     (`keyframe_env.py`), which reports the containment gap between model and reality.

Which pose dimension each manifold parameter asks for:

    height  (squash in z)   -> crouch (down to sitting on the heels) + lean
    width   (pinch in y)    -> arms tuck in, plus twist when the manifold is shifted sideways
    depth   (pinch in x)    -> arms tuck in (the hands are the furthest-forward part)
    offset  (shift in y)    -> twist towards the offset, which pulls the far hand in

    python3 -m manifold_g1.static_fit train  --iterations 400
    python3 -m manifold_g1.static_fit verify --policy reports/manifold_g1/static_fit/policy.pt
    python3 -m manifold_g1.static_fit view   --policy reports/manifold_g1/static_fit/policy.pt
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import constants as C
from .body_envelope import read_landmarks
from .manifold import EllipsoidManifold, Primitive, quat_to_matrix
from .ppo import PPO, RolloutBuffer

ENVELOPE_JSON = C.REPO / "reports" / "manifold_g1" / "body_envelope.json"
DEFAULT_POLICY = C.REPO / "reports" / "manifold_g1" / "static_fit" / "policy.pt"
OBS_DIM = 14
ACTION_DIM = 4
# The pose box is exactly the calibrated grid: crouch, lean and arms are one-sided, twist is
# signed, and nothing outside it may be commanded (the body model would have to extrapolate).
# The commanded box is deliberately smaller than the calibrated grid: the frozen controller
# only *holds* part of it (measured, see `verify`), so asking for more would be asking for a
# pose it will not adopt.
CROUCH_MAX = 1.3     # grid goes to 2.0, but tracking saturates around here
LEAN_MAX = 0.5
TWIST_MAX = 1.2      # waist yaw is tracked well
ARMS_MAX = 0.4       # arm keyframes are barely tracked in the deep regime


class BodyModel:
    """Landmark positions as a function of (crouch, lean, twist), trilinearly interpolated."""

    def __init__(self, path: Path = ENVELOPE_JSON):
        data = json.loads(Path(path).read_text())
        rows = data["rows"]
        self.offsets = {k: np.array(v) for k, v in data.get("landmark_offsets", {}).items()}
        self.crouch = np.array(sorted({r["crouch"] for r in rows}))
        self.lean = np.array(sorted({r["lean"] for r in rows}))
        self.twist = np.array(sorted({r["twist"] for r in rows}))
        self.arms = np.array(sorted({r["arms"] for r in rows}))
        self.names = list(rows[0]["landmarks"].keys())
        shape = (len(self.crouch), len(self.lean), len(self.twist), len(self.arms),
                 len(self.names), 3)
        self.grid = np.zeros(shape)
        for r in rows:
            i = int(np.argmin(np.abs(self.crouch - r["crouch"])))
            j = int(np.argmin(np.abs(self.lean - r["lean"])))
            k = int(np.argmin(np.abs(self.twist - r["twist"])))
            m = int(np.argmin(np.abs(self.arms - r["arms"])))
            self.grid[i, j, k, m] = [r["landmarks"][n] for n in self.names]
        standing = next(r for r in rows if r["crouch"] == 0.0 and r["lean"] == 0.0
                        and r["twist"] == 0.0 and r["arms"] == 0.0)
        self.standing_semi = np.array(standing["semi"])
        self.standing_center = np.array(standing["center"])

    def at_pose(self, pose: np.ndarray) -> np.ndarray:
        """Landmark positions for a 29-joint pose delta (policy order).

        Exact, by forward kinematics in MuJoCo: the calibrated 4-axis grid cannot represent a
        29-joint pose like a walking frame, so poses are evaluated directly.
        """
        return self.at_pose_batch(np.asarray(pose, dtype=np.float64)[None, :])[0]

    def mesh_points_pose(self, pose: np.ndarray, max_vertices: int = 120) -> np.ndarray:
        """Sampled body surface for a 29-joint pose delta, pelvis-anchored (mesh envelope).

        This is the ruler the manifold family is defined on (`family.BASE_SEMI` is a mesh fit),
        so containment for arbitrary poses must be judged the same way.
        """
        import mujoco

        from .body_envelope import body_points

        if not hasattr(self, "_kin"):
            self.at_pose_batch(np.zeros((1, len(C.DEFAULT_ANGLES))))
        env = self._kin
        env.data.qpos[env.body_qadr] = C.DEFAULT_ANGLES + np.asarray(pose)[C.ISAACLAB_TO_MUJOCO]
        mujoco.mj_kinematics(env.model, env.data)
        points = body_points(env.model, env.data, max_vertices=max_vertices)
        pelvis = env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
        return points - pelvis

    def at_pose_batch(self, poses: np.ndarray) -> np.ndarray:
        """Landmark positions for a batch of 29-joint pose deltas: [N,29] -> [N,K,3]."""
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, len(C.DEFAULT_ANGLES))
        if not hasattr(self, "_kin"):
            import mujoco
            from .env import G1FlatEnv
            self._kin = G1FlatEnv()
            self._kin_bid = [mujoco.mj_name2id(self._kin.model, mujoco.mjtObj.mjOBJ_BODY, name)
                             for name in self.names]
        import mujoco
        env = self._kin
        # landmark bodies are few, so read xpos/xmat directly instead of rebuilding dicts
        bid = np.array(self._kin_bid)
        offsets = np.stack([self.offsets[n] for n in self.names])       # [K, 3]
        out = np.zeros((len(poses), len(self.names), 3))
        for i, pose in enumerate(poses):
            env.data.qpos[env.body_qadr] = C.DEFAULT_ANGLES + pose[C.ISAACLAB_TO_MUJOCO]
            mujoco.mj_kinematics(env.model, env.data)      # poses only need body frames
            mats = env.data.xmat[bid].reshape(-1, 3, 3)
            points = env.data.xpos[bid] + np.einsum("kij,kj->ki", mats, offsets)
            out[i] = points - points[self.pelvis_index()]
        return out

    def pelvis_index(self) -> int:
        return self.names.index("pelvis")

    def read(self, env) -> dict[str, np.ndarray]:
        """The same landmark points, measured on the robot in MuJoCo."""
        from .body_envelope import read_landmarks

        return read_landmarks(env.env.model, env.env.data, self.offsets)

    def at(self, *pose) -> np.ndarray:
        """Landmark world positions for a pose (pelvis at x = y = 0)."""
        return self.at_batch(np.asarray(pose, dtype=np.float64)[None, :])[0]

    def axis_weights(self, values):
        """Per-axis bracketing index and interpolation weight inside the grid."""
        index, weight = [], []
        for axis, v in zip((self.crouch, self.lean, self.twist, self.arms), values):
            v = np.clip(v, axis[0], axis[-1])
            i = np.clip(np.searchsorted(axis, v) - 1, 0, len(axis) - 2)
            index.append(i)
            weight.append((v - axis[i]) / (axis[i + 1] - axis[i]))
        return index, weight

    def pose_grid(self, n_crouch: int = 9, n_lean: int = 7, n_twist: int = 9, n_arms: int = 7):
        """Batch of poses and their landmark positions: (poses [N,4], landmarks [N,K,3])."""
        axes = np.meshgrid(np.linspace(0.0, self.crouch[-1], n_crouch),
                           np.linspace(0.0, self.lean[-1], n_lean),
                           np.linspace(-self.twist[-1], self.twist[-1], n_twist),
                           np.linspace(0.0, self.arms[-1], n_arms), indexing="ij")
        poses = np.stack([a.ravel() for a in axes], axis=1)
        return poses, self.at_batch(poses)

    def at_batch(self, poses: np.ndarray) -> np.ndarray:
        """Landmark positions for a batch of poses: [N,4] -> [N,K,3] (vectorized interpolation)."""
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4)
        index, weight = self.axis_weights(poses.T)
        out = np.zeros((len(poses), len(self.names), 3))
        for a in (0, 1):
            for b in (0, 1):
                for d in (0, 1):
                    for e in (0, 1):
                        w = ((weight[0] if a else 1 - weight[0])
                             * (weight[1] if b else 1 - weight[1])
                             * (weight[2] if d else 1 - weight[2])
                             * (weight[3] if e else 1 - weight[3]))
                        out += w[:, None, None] * self.grid[index[0] + a, index[1] + b,
                                                           index[2] + d, index[3] + e]
        return out


@dataclass(frozen=True)
class ManifoldSpec:
    """A manifold in the family: the standing envelope squashed, pinched and shifted."""

    height: float = 1.0        # vertical semi-axis and centre height
    width: float = 1.0         # lateral semi-axis
    depth: float = 1.0         # sagittal semi-axis
    offset: float = 0.0        # lateral centre shift [m]
    tilt_deg: float = 0.0      # sagittal tilt of the axis: + leans forward

    def build(self, body: BodyModel) -> EllipsoidManifold:
        semi = body.standing_semi.copy()
        semi[0] = body.standing_semi[0] * self.depth
        semi[1] = body.standing_semi[1] * self.width
        semi[2] = body.standing_semi[2] * self.height
        center = np.array([0.0, self.offset, body.standing_center[2] * self.height])
        half = np.radians(self.tilt_deg) / 2.0
        quat = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
        return EllipsoidManifold([Primitive(center=center, quat=quat, semi=semi)])

    def label(self) -> str:
        return (f"h{self.height:.2f}_w{self.width:.2f}_d{self.depth:.2f}"
                f"_o{self.offset:+.02f}_t{self.tilt_deg:+.0f}")


@dataclass
class StaticConfig:
    dt: float = 0.1                        # policy step
    max_episode_steps: int = 60            # 6 s to find and hold the pose
    tau: float = 0.45                      # first-order pose response
    w_inside: float = 1.0                  # per second and unit of containment margin
    margin_cap: float = 0.03               # margin beyond this earns nothing: without the cap
                                           # the policy balls up even when standing would do
    w_outside: float = 25.0                # per second and unit of violation (a sharp barrier:
                                           # a weak one makes "stay outside, don't pose" rational)
    w_effort: float = 0.4                  # per second at full pose (half-normalized, 4 dims)
    # Alignment has to outbid the effort of leaning, or the policy rationally stays upright:
    # lean 1.0 buys ~10 deg of spine tilt (measured), which is worth ~0.1/s at this weight.
    w_spine: float = 5.0                   # per second at full misalignment with the manifold axis
    success_hold: int = 50                 # control ticks (1 s) inside before success
    success_margin: float = 0.98


def _action_bounds(action: np.ndarray) -> np.ndarray:
    """Map the policy action to a pose target inside the calibrated pose box."""
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
    return np.array([max(0.0, a[0]) * CROUCH_MAX, max(0.0, a[1]) * LEAN_MAX, a[2] * TWIST_MAX,
                     max(0.0, a[3]) * ARMS_MAX])


class StaticFitEnv:
    """Three pose dimensions (crouch, lean, twist) against a static ellipsoid manifold."""

    action_dim = ACTION_DIM

    def __init__(self, body: BodyModel, manifold: EllipsoidManifold, cfg: StaticConfig | None = None):
        self.body = body
        self.manifold = manifold
        self.cfg = cfg or StaticConfig()
        self.reset()

    def radius(self, pose: np.ndarray) -> float:
        return float(self.manifold.radii(self.body.at(*pose)).max())

    def spine_alignment(self) -> float:
        """Cosine between the robot's spine (pelvis -> top of the head) and the manifold axis."""
        points = self.body.at(*self.pose)
        spine = points[self.body.names.index("torso_link")] - points[self.body.pelvis_index()]
        return float(np.dot(spine / np.linalg.norm(spine), self.manifold.primitives[0].axis()))

    def obs(self) -> np.ndarray:
        return self._obs(self.manifold.primitives[0], self.pose,
                         self.body.at(*self.pose)[self.body.pelvis_index()],
                         self.radius(self.pose))

    def _obs(self, primitive: Primitive, pose: np.ndarray, pelvis: np.ndarray,
             radius: float) -> np.ndarray:
        return np.concatenate([
            pose / np.array([CROUCH_MAX, LEAN_MAX, TWIST_MAX, ARMS_MAX]),
            [radius],
            (primitive.center - pelvis) / 0.5,
            primitive.semi,
            primitive.axis(),
        ]).astype(np.float32)

    def reset(self, **_) -> tuple[np.ndarray, dict]:
        self.pose = np.zeros(4)               # standing, outside a reshaped manifold
        self.steps = 0
        self.inside_ticks = 0
        self.episode_return = 0.0
        return self.obs(), {"radius": self.radius(self.pose)}

    def step(self, action: np.ndarray):
        cfg = self.cfg
        target = _action_bounds(action)
        reward = 0.0
        for _ in range(5):                    # 50 Hz inner loop, like the SONIC stack
            alpha = min(1.0, C.CONTROL_DT / cfg.tau)
            self.pose += (target - self.pose) * alpha
            radius = self.radius(self.pose)
            if radius <= cfg.success_margin:
                # margin-seeking, capped: approaching the boundary earns reward (that gradient
                # is what makes "fit inside" learnable), but posing deeper than needed does not
                reward += cfg.w_inside * min(1.0 - radius, cfg.margin_cap) * C.CONTROL_DT
                self.inside_ticks += 1
            else:
                reward -= cfg.w_outside * (radius - cfg.success_margin) * C.CONTROL_DT
                self.inside_ticks = 0
            reward -= cfg.w_effort * float(np.sum(
                np.abs(self.pose) / np.array([CROUCH_MAX, LEAN_MAX, TWIST_MAX, ARMS_MAX]))) / 2.0 \
                * C.CONTROL_DT
            reward -= cfg.w_spine * (1.0 - self.spine_alignment()) * C.CONTROL_DT
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
        pelvis = self.body.at(*self.pose)[self.body.pelvis_index()]
        return self.obs(), float(reward), terminated, truncated, {
            "outcome": outcome, "episode_step": self.steps, "episode_return": self.episode_return,
            "pose": self.pose.copy(), "manifold_radius": radius, "base_z": float(pelvis[2]),
        }


class BatchedStaticEnv:
    """Many manifolds stepped in parallel: one batched numpy computation per control tick.

    Training with N streams costs about the same as one: the body model and the containment
    are vectorized over poses, so the wall time is set by the PPO update, not the environment.
    """

    action_dim = ACTION_DIM

    def __init__(self, body: BodyModel, cfg: StaticConfig | None = None):
        self.body = body
        self.cfg = cfg or StaticConfig()
        self.n = 0

    def set_manifolds(self, specs: list[ManifoldSpec]) -> np.ndarray:
        """Start a fresh episode for every stream on its own manifold."""
        self.n = len(specs)
        self.center = np.zeros((self.n, 3))
        self.semi = np.zeros((self.n, 3))
        self.rot = np.zeros((self.n, 3, 3))
        self.axis = np.zeros((self.n, 3))
        self.pose = np.zeros((self.n, 4))
        self.steps = np.zeros(self.n, dtype=int)
        self.inside_ticks = np.zeros(self.n, dtype=int)
        self.episode_return = np.zeros(self.n)
        self.set_manifolds_at(np.arange(self.n), specs)
        return self.obs()

    def set_manifolds_at(self, index: np.ndarray, specs: list[ManifoldSpec]) -> None:
        """Hand new manifolds to the streams that just finished an episode."""
        primitives = [spec.build(self.body).primitives[0] for spec in specs]
        self.center[index] = np.array([p.center for p in primitives])
        self.semi[index] = np.array([p.semi for p in primitives])
        self.rot[index] = np.array([quat_to_matrix(p.quat) for p in primitives])
        self.axis[index] = np.array([p.axis() for p in primitives])
        self.reset_streams(index)

    def reset_streams(self, index: np.ndarray) -> None:
        self.pose[index] = 0.0
        self.steps[index] = 0
        self.inside_ticks[index] = 0
        self.episode_return[index] = 0.0

    def radius(self, poses: np.ndarray) -> np.ndarray:
        points = self.body.at_batch(poses)                              # [N,K,3]
        rel = points - self.center[:, None, :]
        local = np.einsum("nkj,nji->nki", rel, self.rot)                # into the ellipsoid frame
        return np.linalg.norm(local / self.semi[:, None, :], axis=-1).max(axis=1)

    def spine_alignment(self) -> np.ndarray:
        points = self.body.at_batch(self.pose)
        spine = points[:, self.body.names.index("torso_link"), :] - points[:, 0, :]
        spine /= np.linalg.norm(spine, axis=1, keepdims=True)
        return np.sum(spine * self.axis, axis=1)

    def obs(self) -> np.ndarray:
        pelvis = self.body.at_batch(self.pose)[:, self.body.pelvis_index(), :]
        return np.concatenate([
            self.pose / np.array([CROUCH_MAX, LEAN_MAX, TWIST_MAX, ARMS_MAX]),
            self.radius(self.pose)[:, None],
            (self.center - pelvis) / 0.5,
            self.semi,
            self.axis,
        ], axis=1).astype(np.float32)

    def step(self, actions: np.ndarray):
        cfg = self.cfg
        targets = np.array([_action_bounds(a) for a in np.atleast_2d(actions)])
        reward = np.zeros(self.n)
        alpha = min(1.0, C.CONTROL_DT / cfg.tau)
        for _ in range(5):                       # 50 Hz inner loop, like the SONIC stack
            self.pose += (targets - self.pose) * alpha
            radius = self.radius(self.pose)
            inside = radius <= cfg.success_margin
            margin = np.minimum(1.0 - radius, cfg.margin_cap)
            reward += np.where(inside, cfg.w_inside * margin,
                               -cfg.w_outside * (radius - cfg.success_margin)) * C.CONTROL_DT
            reward -= cfg.w_spine * (1.0 - self.spine_alignment()) * C.CONTROL_DT
            self.inside_ticks = np.where(inside, self.inside_ticks + 1, 0)
            reward -= cfg.w_effort * np.abs(
                self.pose / np.array([CROUCH_MAX, LEAN_MAX, TWIST_MAX, ARMS_MAX])).sum(axis=1) \
                / 2.0 * C.CONTROL_DT
        self.steps += 1
        success = self.inside_ticks >= cfg.success_hold
        truncated = ~success & (self.steps >= cfg.max_episode_steps)
        reward = reward + np.where(success, 5.0, 0.0)
        self.episode_return += reward
        radius = self.radius(self.pose)
        return self.obs(), reward, success, truncated, {
            "radius": radius, "pose": self.pose.copy(), "episode_return": self.episode_return,
            "outcome": np.where(success, "success", np.where(truncated, "timeout", "running")),
        }


POSE_MAX = np.array([CROUCH_MAX, LEAN_MAX, TWIST_MAX, ARMS_MAX])


def _curriculum_limit(args, frac: float) -> float:
    """How much posing the sampled manifolds may demand, along the training schedule.

    Difficulty is one scalar: the L1 magnitude of the least pose that fits (`need`). It rises
    from "a small pose suffices" to the deep-squat-plus-tuck end of the family, so the policy
    meets one new regime at a time instead of all four pose channels at once.
    """
    return args.need_start + (args.need_max - args.need_start) * float(np.clip(frac, 0.0, 1.0))


def _sample_specs(body: BodyModel, rng: np.random.Generator, count: int, args,
                  difficulty_max: float = float("inf")) -> list[ManifoldSpec]:
    """Manifolds in the family that some pose can actually fit into (margin above the boundary).

    Containment needs *every* landmark inside, so a pose is scored by its worst landmark and the
    manifold counts as feasible when the best pose keeps the worst landmark inside.
    """
    height_range, width_range = args.height_range, args.width_range
    depth_range, offset_range = args.depth_range, args.offset_range
    poses, points = body.pose_grid()
    specs = []
    for _ in range(count):
        offset = float(rng.uniform(*offset_range))
        if abs(offset) < args.min_offset:
            offset = float(np.sign(offset) or 1) * args.min_offset
        spec = ManifoldSpec(height=float(rng.uniform(*height_range)),
                            width=float(rng.uniform(*width_range)),
                            depth=float(rng.uniform(*depth_range)), offset=offset,
                            tilt_deg=float(rng.uniform(*args.tilt_range)))
        manifold = spec.build(body)
        radii = manifold.radii(points).max(axis=1)
        index = int(np.argmin(radii))
        if radii[index] > args.feasible_margin:
            continue
        if float(np.abs(poses[index] / POSE_MAX).sum()) > difficulty_max:
            continue
        specs.append(spec)
    return specs


def _best_pose(body: BodyModel, manifold: EllipsoidManifold,
               margin: float = StaticConfig.success_margin) -> tuple[np.ndarray, float]:
    """The task's ground-truth optimum: the least posing that still fits inside.

    Minimizing `r` alone would pick the deepest pose, but the task pays for posing - so the
    optimum is the smallest pose whose worst landmark stays inside (`r <= margin`).
    """
    poses, points = body.pose_grid(n_crouch=21, n_lean=15, n_twist=15, n_arms=15)
    radii = manifold.radii(points).max(axis=1)
    effort = np.abs(poses) / np.array([CROUCH_MAX, LEAN_MAX, TWIST_MAX, ARMS_MAX])
    feasible = radii <= margin
    if feasible.any():
        index = int(np.argmin(np.where(feasible, effort.sum(axis=1), np.inf)))
    else:
        index = int(np.argmin(radii))
    return poses[index], float(radii[index])


def train(args) -> int:
    body = BodyModel()
    rng = np.random.default_rng(args.seed)
    specs = _sample_specs(body, rng, args.candidates, args)
    if not specs:
        print("no feasible manifold found - widen the parameter box")
        return 1
    poses = body.pose_grid()[1]
    scores = np.array([spec.build(body).radii(poses).max(axis=1).min() for spec in specs])
    print(f"manifold family: {len(specs)} feasible of {args.candidates} candidates "
          f"(height {args.height_range}, width {args.width_range}, depth {args.depth_range}, "
          f"offset {args.offset_range})")
    print(f"  room left per manifold (best pose's worst landmark): min {scores.min():.2f} "
          f"median {np.median(scores):.2f} max {scores.max():.2f}  (success needs <= "
          f"{StaticConfig.success_margin})")

    cfg = StaticConfig(w_effort=args.w_effort, w_outside=args.w_outside)
    env = BatchedStaticEnv(body, cfg)
    families: dict[float, list[ManifoldSpec]] = {}

    def family(frac: float) -> list[ManifoldSpec]:
        """Feasible manifolds whose required pose is within the curriculum's limit right now."""
        step = round(frac, 1)
        if step not in families:
            drawn = _sample_specs(body, rng, args.candidates, args,
                                  _curriculum_limit(args, step))
            families[step] = drawn or specs
        return families[step]

    draw = lambda count, frac=1.0: [family(frac)[int(rng.integers(len(family(frac))))]
                                    for _ in range(count)]
    obs = env.set_manifolds(draw(args.envs, 0.0))
    obs_dim = int(obs.shape[1])
    # a wide initial std: the crouch channel alone spans ~0.95 in normalized units, so a narrow
    # initial exploration would never sample the deep squats the family asks for
    ppo = PPO(obs_dim, env.action_dim, init_log_std=[0.0] * env.action_dim)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "train_log.csv", "w")
    log.write("iteration,episodes,success,crouch_mean,lean_mean,twist_mean,return_mean\n")
    print(f"training {args.envs} manifolds in parallel, {args.rollout_steps * args.envs} "
          f"transitions per update")
    started = time.perf_counter()

    for iteration in range(1, args.iterations + 1):
        frac = (iteration - 1) / max(1, args.iterations - 1)
        ppo.entropy_coef = args.entropy_start + (args.entropy_end - args.entropy_start) * frac
        buffer = RolloutBuffer(args.rollout_steps * args.envs, obs_dim, env.action_dim)
        returns, outcomes, final_poses = [], [], []
        for _ in range(args.rollout_steps):
            action, logprob, value = ppo.act_batch(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            buffer.add_batch(obs, action, logprob, reward, value, done | truncated)
            obs = next_obs
            finished = np.flatnonzero(done | truncated)
            if len(finished):
                returns.extend(info["episode_return"][finished].tolist())
                outcomes.extend(info["outcome"][finished].tolist())
                final_poses.extend(info["pose"][finished].tolist())
                env.set_manifolds_at(finished, draw(len(finished), frac))
                obs = env.obs()
        ppo.update(buffer, ppo.value_batch(obs))
        success = float(np.mean([o == "success" for o in outcomes])) if outcomes else float("nan")
        mean_pose = np.mean(final_poses, axis=0) if final_poses else np.zeros(3)
        log.write(f"{iteration},{len(outcomes)},{success:.3f},"
                  f"{mean_pose[0]:.3f},{mean_pose[1]:.3f},{mean_pose[2]:.3f},"
                  f"{np.mean(returns) if returns else 0:.3f}\n")
        log.flush()
        if iteration % max(1, args.iterations // 20) == 0 or iteration == 1:
            print(f"[iter {iteration:3d}] episodes={len(outcomes):3d} success={success:.2f} "
                  f"pose=({mean_pose[0]:.2f},{mean_pose[1]:+.2f},{mean_pose[2]:+.2f})", flush=True)
        ppo.save(out_dir / "policy.pt")
        last_success = success

    _report_table(body, ppo, args, out_dir)
    log.close()
    print(f"\n=== training finished: {args.iterations} iterations in "
          f"{time.perf_counter() - started:.0f} s, success {last_success:.2f}, "
          f"policy -> {Path(args.out) / 'policy.pt'} ===", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(
        {"family": {"height": args.height_range, "width": args.width_range,
                    "offset": args.offset_range},
         "feasible_candidates": len(specs),
         "standing_semi": body.standing_semi.tolist(),
         "standing_center": body.standing_center.tolist()}, indent=2))
    return 0


def _report_table(body: BodyModel, ppo: PPO, args, out_dir: Path | None):
    """Manifold -> learned action distribution, next to the exhaustive optimum."""
    env = StaticFitEnv(body, _report_specs(body, args)[0].build(body),
                       StaticConfig(w_effort=args.w_effort, w_outside=args.w_outside))
    print("\nmanifold -> learned action distribution (greedy) vs the true optimum:")
    print(f"{'manifold':>24} {'crouch':>7} {'lean':>6} {'twist':>6} {'arms':>6} {'r_pol':>6} "
          f"{'r_min':>6} {'spine':>6} {'task-optimal pose (c,l,t,a)':>26} {'std':>22}")
    table = []
    for spec in _report_specs(body, args):
        manifold = spec.build(body)
        env.manifold = manifold
        obs, _ = env.reset()
        for _ in range(env.cfg.max_episode_steps):
            action, _, _ = ppo.act(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        pose = info["pose"]
        mean, std = ppo.action_stats(obs)
        best_pose, best_r = _best_pose(body, manifold)
        _, grid_points = body.pose_grid()
        r_min = float(manifold.radii(grid_points).max(axis=1).min())
        pose_points = body.at(*pose)
        spine = pose_points[body.names.index("torso_link")] - pose_points[body.pelvis_index()]
        spine_tilt = float(np.degrees(np.arctan2(spine[0], spine[2])))
        print(f"{spec.label():>24} {pose[0]:>7.2f} {pose[1]:>+6.2f} {pose[2]:>+6.2f} "
              f"{pose[3]:>6.2f} {info['manifold_radius']:>6.2f} {r_min:>6.2f} {spine_tilt:>+6.1f} "
              f"({best_pose[0]:5.2f},{best_pose[1]:+5.2f},{best_pose[2]:+5.2f},{best_pose[3]:4.2f}) "
              f"({std[0]:.2f},{std[1]:.2f},{std[2]:.2f},{std[3]:.2f})")
        table.append({"manifold": {"height": spec.height, "width": spec.width, "depth": spec.depth,
                                   "offset": spec.offset, "tilt_deg": spec.tilt_deg},
                      "outcome": info["outcome"], "pose": [float(v) for v in pose],
                      "radius": float(info["manifold_radius"]),
                      "action_mean": [float(v) for v in mean], "action_std": [float(v) for v in std],
                      "best_pose": [float(v) for v in best_pose], "best_radius": best_r})
    (out_dir / "greedy_table.json").write_text(json.dumps(table, indent=2))
    if out_dir is not None:
        (out_dir / "greedy_table.json").write_text(json.dumps(table, indent=2))


def report(args) -> int:
    """Print (and save) the learned manifold -> pose distribution of a trained policy."""
    body = BodyModel()
    ppo = PPO(OBS_DIM, ACTION_DIM)
    ppo.load(Path(args.policy))
    _report_table(body, ppo, args, Path(args.out))
    return 0


def _report_specs(body: BodyModel, args) -> list[ManifoldSpec]:
    """A curated spread of manifolds, each asking for a different pose."""
    return [
        ManifoldSpec(1.00, 0.98, 1.00, 0.00, 0.0),    # roomy, upright axis: no pose needed
        ManifoldSpec(1.00, 0.98, 1.00, 0.00, +12.0),  # axis leaning forward: lean to align
        ManifoldSpec(1.00, 0.98, 1.00, 0.00, -12.0),  # axis leaning back
        ManifoldSpec(0.85, 0.98, 1.00, 0.00, +10.0),  # crouch + forward tilt
        ManifoldSpec(0.85, 0.98, 1.00, 0.00, -10.0),
        ManifoldSpec(0.80, 0.98, 1.00, 0.00, 0.0),    # crouch
        ManifoldSpec(0.72, 0.98, 1.00, 0.00, 0.0),    # deep crouch
        ManifoldSpec(1.00, 0.90, 1.00, 0.00, 0.0),    # pinch
        ManifoldSpec(0.85, 0.92, 0.90, +0.04, +8.0),  # crouch + pinch + shift + tilt
        ManifoldSpec(0.85, 0.92, 0.90, -0.04, -8.0),  # mirrored
    ]


def _deploy_obs(body: BodyModel, manifold: EllipsoidManifold, pose: np.ndarray,
                radius: float) -> np.ndarray:
    """Observation built exactly as in `StaticFitEnv`, with the measured radius."""
    primitive = manifold.primitives[0]
    pelvis = body.at(*pose)[body.pelvis_index()]
    return np.concatenate([
        pose / np.array([CROUCH_MAX, LEAN_MAX, TWIST_MAX, ARMS_MAX]),
        [radius],
        (primitive.center - pelvis) / 0.5, primitive.semi, primitive.axis(),
    ]).astype(np.float32)


class PoseSlew:
    """The pose the keyframe is slewed to, with the same first-order lag as the body model.

    Applying the action directly would command a pose the robot cannot reach within one tick,
    while the policy's observation says it is already there - a bang-bang limit cycle.
    """

    def __init__(self, tau: float = StaticConfig.tau):
        self.tau = tau
        self.pose = np.zeros(4)

    def reset(self) -> None:
        self.pose = np.zeros(4)

    def target(self, action: np.ndarray) -> np.ndarray:
        goal = _action_bounds(action)
        self.pose += (goal - self.pose) * min(1.0, C.CONTROL_DT / self.tau)
        return self.pose


def _real_points(body: BodyModel, env, pose: np.ndarray) -> np.ndarray:
    """Measured landmark positions in the model's own frame (pelvis at x = y = 0)."""
    pelvis = env.state()["base_pos"]
    model_pelvis = body.at(*pose)[body.pelvis_index()]
    real = body.read(env)
    return np.array([real[n] - pelvis + model_pelvis for n in body.names])


def verify(args) -> int:
    """Execute the static policy with MuJoCo + SONIC holding the learned keyframes.

    Containment is measured on the *real* landmarks (pelvis-relative, so the robot's drift
    does not enter) and compared with the model's prediction at the same pose.
    """
    from .keyframe_env import KeyframeEnv

    body = BodyModel()
    ppo = PPO(OBS_DIM, ACTION_DIM)
    ppo.load(Path(args.policy))
    env = KeyframeEnv()
    slew = PoseSlew()
    print("SONIC verification (keyframe held by SONIC, containment pelvis-relative):")
    print(f"{'manifold':>24} {'crouch':>7} {'lean':>6} {'twist':>6} {'arms':>6} {'pelvis_z':>9} "
          f"{'head_z':>7} {'r_model':>8} {'r_real':>7} {'worst landmark':>20}")
    rows = []
    for spec in _report_specs(body, args):
        manifold = spec.build(body)
        env.reset()
        slew.reset()
        worst_name, radius_real = "-", float("nan")
        for _ in range(args.ticks):
            pose = slew.pose.copy()
            radii = manifold.radii(_real_points(body, env, pose))
            radius_real = float(radii.max())
            worst_name = body.names[int(np.argmax(radii))]
            action, _, _ = ppo.act(_deploy_obs(body, manifold, pose, radius_real),
                                   deterministic=True)
            env.set_pose(*slew.target(action))
            env.step()
        pose = slew.pose.copy()
        model_points = body.at(*pose)
        print(f"{spec.label():>24} {pose[0]:>7.2f} {pose[1]:>+6.2f} {pose[2]:>+6.2f} "
              f"{pose[3]:>6.2f} {env.state()['base_pos'][2]:>9.3f} "
              f"{model_points[body.names.index('torso_link')][2]:>7.3f} "
              f"{float(manifold.radii(model_points).max()):>8.2f} {radius_real:>7.2f} "
              f"{worst_name:>20s}")
        rows.append({"manifold": {"height": spec.height, "width": spec.width, "depth": spec.depth,
                                  "offset": spec.offset, "tilt_deg": spec.tilt_deg},
                     "pose": [float(v) for v in pose],
                     "pelvis_z": float(env.state()["base_pos"][2]),
                     "head_z": float(model_points[body.names.index("torso_link")][2]),
                     "r_model": float(manifold.radii(model_points).max()),
                     "r_real": radius_real, "worst_landmark": worst_name})
    out = C.REPO / "reports" / "manifold_g1" / "static_fit" / "verify_sonic.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"saved {out.relative_to(C.REPO)}")
    return 0


def diagnostics(args) -> int:
    """Measure what the frozen controller actually does with a keyframe.

    Two tables, both written to `reports/manifold_g1/`:

      * channel tracking - command one pose channel at a time, then fit the *achieved* joint
        angles back onto the four pose directions. If a channel comes back near zero the
        controller ignored it, and no policy can use it;
      * ground-contact poses - kneeling and hand support, with a stability verdict.

    These are the measurements the pose box is derived from; re-run them after changing a
    pose direction in `reference.py`.
    """
    import json

    from .keyframe_env import KeyframeEnv
    from .reference import (ARM_DIRECTION, CROUCH_DIRECTION, LEAN_DIRECTION, TWIST_DIRECTION)

    body = BodyModel()
    env = KeyframeEnv()
    basis = np.stack([CROUCH_DIRECTION, LEAN_DIRECTION, TWIST_DIRECTION, ARM_DIRECTION], axis=1)
    default_isaac = C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB]

    def achieved(pose_target, slew_steps: int = 75, hold_steps: int = 75):
        """Slew to a pose the way deployment does, then read the joints and the landmarks."""
        env.reset()
        slew = PoseSlew()
        for _ in range(slew_steps + hold_steps):
            env.set_pose(*slew.target(np.asarray(pose_target) / POSE_MAX))
            env.step()
        q = env.state()["q_hw"][C.MUJOCO_TO_ISAACLAB]
        delta = q - default_isaac
        fitted, *_ = np.linalg.lstsq(basis, delta, rcond=None)
        return fitted, float(np.linalg.norm(delta - basis @ fitted)), env.state()

    tests = [
        ("standing", (0.0, 0.0, 0.0, 0.0)),
        ("crouch 0.8", (0.8, 0.0, 0.0, 0.0)),
        ("crouch 1.2", (1.2, 0.0, 0.0, 0.0)),
        ("crouch 1.6", (1.6, 0.0, 0.0, 0.0)),
        ("crouch 2.0", (2.0, 0.0, 0.0, 0.0)),
        ("lean 0.5", (0.0, 0.5, 0.0, 0.0)),
        ("lean 1.0", (0.0, 1.0, 0.0, 0.0)),
        ("arms 1.0", (0.0, 0.0, 0.0, 1.0)),
        ("twist +1.2", (0.0, 0.0, 1.2, 0.0)),
        ("crouch 1.2 + lean 0.5", (1.2, 0.5, 0.0, 0.0)),
        ("crouch 1.6 + lean 1.0", (1.6, 1.0, 0.0, 0.0)),
        ("crouch 1.6 + arms 1.0", (1.6, 0.0, 0.0, 1.0)),
        ("crouch 1.2 + lean 0.5 + arms 1.0", (1.2, 0.5, 0.0, 1.0)),
        ("lean 1.0 + arms 1.0", (0.0, 1.0, 0.0, 1.0)),
        ("everything saturated", (2.0, 1.0, 1.2, 1.0)),
    ]
    print("what the frozen controller does with a commanded keyframe:")
    print(f"{'commanded':>30} {'achieved (c,l,t,a)':>26} {'residual':>9} {'verdict':>11}")
    channel_rows = []
    for name, target in tests:
        fitted, residual, _ = achieved(target)
        worst = float(np.max(np.abs(fitted - np.asarray(target))))
        verdict = ("ok" if worst < 0.25 and residual < 0.35
                   else "partial" if worst < 0.6 and residual < 1.0 else "not held")
        print(f"{name:>30} ({fitted[0]:+5.2f},{fitted[1]:+5.2f},{fitted[2]:+5.2f},{fitted[3]:+5.2f}) "
              f"{residual:9.2f} {verdict:>11}")
        channel_rows.append({"command": name, "target": list(target),
                             "achieved": [float(v) for v in fitted],
                             "residual_rad": residual, "verdict": verdict})

    contact_poses = {
        "crouch 2.0 (reference)": dict(left_hip_pitch=-1.2, right_hip_pitch=-1.2,
                                      left_knee=2.0, right_knee=2.0,
                                      left_ankle_pitch=-0.8, right_ankle_pitch=-0.8),
        "kneel, both knees": dict(left_hip_pitch=-1.2, right_hip_pitch=-1.2,
                                  left_knee=2.6, right_knee=2.6,
                                  left_ankle_pitch=-0.2, right_ankle_pitch=-0.2),
        "kneel, both knees deep": dict(left_hip_pitch=-1.5, right_hip_pitch=-1.5,
                                       left_knee=2.8, right_knee=2.8,
                                       left_ankle_pitch=-0.1, right_ankle_pitch=-0.1),
        "kneel, one knee": dict(left_hip_pitch=-1.4, left_knee=2.7, left_ankle_pitch=-0.2,
                                right_hip_pitch=-0.6, right_knee=1.0, right_ankle_pitch=-0.5),
        "kneel + hands down": dict(left_hip_pitch=-1.4, right_hip_pitch=-1.4,
                                   left_knee=2.7, right_knee=2.7,
                                   left_ankle_pitch=-0.2, right_ankle_pitch=-0.2,
                                   left_shoulder_pitch=1.2, right_shoulder_pitch=1.2),
    }
    print("\nground-contact poses:")
    print(f"{'pose':>24} {'head_z':>7} {'pelvis_z':>9} {'knee_z':>7} {'hand_z':>7} "
          f"{'roll':>7} {'drift':>7} {'verdict':>10}")
    contact_rows = []
    for label, joints in contact_poses.items():
        env.reset()
        delta = np.zeros(len(C.MOTOR_NAMES))
        for name, value in joints.items():
            delta[C.ISAACLAB_TO_MUJOCO[C.MOTOR_NAMES.index(name)]] = value
        env.reference.joint_pos = env.reference.joint_pos + delta[None, :]
        for _ in range(150):
            env.step()
        state = env.state()
        measured = body.read(env)                       # extremity landmarks, as the model uses
        points = {name: measured[name] for name in body.names}
        pelvis = points["pelvis"]
        head_z = float(points["torso_link"][2])
        knee_z = float(min(points["left_knee_link"][2], points["right_knee_link"][2]))
        hand_z = float(min(points["left_hand_index_1_link"][2], points["right_hand_index_1_link"][2]))
        quat = state["base_quat"]
        roll = float(np.degrees(np.arctan2(2 * (quat[0] * quat[1] + quat[2] * quat[3]),
                                           1 - 2 * (quat[1] ** 2 + quat[2] ** 2))))
        drift = float(np.linalg.norm(state["base_pos"][:2]))
        fell = abs(roll) > 30 or drift > 0.5
        verdict = "fell over" if fell else "stable"
        print(f"{label:>24} {head_z:7.3f} {pelvis[2]:9.3f} {knee_z:7.3f} {hand_z:7.3f} "
              f"{roll:7.1f} {drift:7.3f} {verdict:>10}")
        contact_rows.append({"pose": label, "head_z": head_z, "pelvis_z": float(pelvis[2]),
                             "knee_z": knee_z, "hand_z": hand_z, "roll_deg": roll,
                             "drift_m": drift, "verdict": verdict})

    out = C.REPO / "reports" / "manifold_g1"
    (out / "pose_tracking.json").write_text(json.dumps(channel_rows, indent=2))
    (out / "contact_poses.json").write_text(json.dumps(contact_rows, indent=2))
    print(f"\nsaved {out.relative_to(C.REPO)}/pose_tracking.json and contact_poses.json")
    return 0


def view(args) -> int:
    """Watch the static policy live; keys reshape the manifold and the pose follows."""
    import time

    import mujoco
    import mujoco.viewer

    from .keyframe_env import KeyframeEnv

    body = BodyModel()
    ppo = PPO(OBS_DIM, ACTION_DIM)
    ppo.load(Path(args.policy))
    spec = ManifoldSpec(height=float(args.height), width=float(args.width),
                        depth=float(args.depth), offset=float(args.offset),
                        tilt_deg=float(args.tilt))
    manifold = spec.build(body)
    env = KeyframeEnv(manifold)
    env.reset()
    slew = PoseSlew()
    state = {"reset": False, "pause": False, "delta": np.zeros(4), "tilt": 0.0}

    # MuJoCo's own viewer already owns Space, +/-, the arrows, Tab and [ ] (camera cycling),
    # so the manifold keys are letters it leaves alone.
    def key_callback(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        if key == "r":
            state["reset"] = True
        elif key == "p":
            state["pause"] = not state["pause"]
        elif key == "u":                       # height down
            state["delta"] = np.array([-0.02, 0.0, 0.0, 0.0])
        elif key == "i":                       # height up
            state["delta"] = np.array([+0.02, 0.0, 0.0, 0.0])
        elif key == "j":                       # width down (pinch)
            state["delta"] = np.array([0.0, -0.02, 0.0, 0.0])
        elif key == "k":                       # width up
            state["delta"] = np.array([0.0, +0.02, 0.0, 0.0])
        elif key == ",":                       # shallower
            state["delta"] = np.array([0.0, 0.0, -0.02])
        elif key == ".":                       # deeper
            state["delta"] = np.array([0.0, 0.0, +0.02])
        elif key == "n":                       # shift left
            state["delta"] = np.array([0.0, 0.0, 0.0, -0.01])
        elif key == "m":                       # shift right
            state["delta"] = np.array([0.0, 0.0, 0.0, +0.01])
        elif key == "t":                       # tilt the axis back
            state["tilt"] = -2.0
        elif key == "y":                       # tilt the axis forward
            state["tilt"] = +2.0

    viewer = mujoco.viewer.launch_passive(env.env.model, env.env.data, key_callback=key_callback,
                                          show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, 130.0, -8.0

    history: list[tuple[float, float]] = []
    figure = mujoco.MjvFigure()
    figure.title = "containment r (blue) and twist (orange)"
    figure.flg_legend = 0
    figure.linergb[0] = (0.35, 0.6, 1.0)
    figure.linergb[1] = (1.0, 0.6, 0.2)
    print("static fit viewer keys (MuJoCo keeps Space, +/- , arrows, Tab, [ ]):")
    print("  u / i   squash / raise the height      j / k   pinch / widen")
    print("  , / .   shallower / deeper             n / m   shift sideways")
    print("  t / y   tilt the axis back / forward   R reset   P pause   Ctrl+C quit")

    clock = time.perf_counter()
    try:
        while viewer.is_running():
            if state["pause"]:
                viewer.sync()
                time.sleep(0.02)
                continue
            if np.any(state["delta"]) or state["tilt"]:
                spec = ManifoldSpec(
                    height=float(np.clip(spec.height + state["delta"][0], 0.50, 1.05)),
                    width=float(np.clip(spec.width + state["delta"][1], 0.40, 1.05)),
                    depth=float(np.clip(spec.depth + state["delta"][2], 0.30, 1.05)),
                    offset=float(np.clip(spec.offset + state["delta"][3], -0.08, 0.08)),
                    tilt_deg=float(np.clip(spec.tilt_deg + state["tilt"], -35.0, 35.0)))
                state["tilt"] = 0.0
                state["delta"] = np.zeros(4)
                manifold = spec.build(body)
                state["reset"] = True
            if state["reset"]:
                state["reset"] = False
                env.reset()
                env.set_manifold(manifold)
                slew.reset()
                history.clear()
                continue
            pose = slew.pose.copy()
            pelvis = env.state()["base_pos"]
            radius = float(manifold.radii(_real_points(body, env, pose)).max())
            action, _, _ = ppo.act(_deploy_obs(body, manifold, pose, radius), deterministic=True)
            env.set_pose(*slew.target(action))
            env.step()
            env.set_manifold(manifold, follow=pelvis)     # keep the visual pelvis-relative in x/y
            history.append((radius, pose[2]))
            n = min(len(history), 400)
            figure.linepnt[0] = figure.linepnt[1] = n
            figure.range[0] = (0, max(n - 1, 1))
            figure.range[1] = (0.0, 1.8)
            for i in range(n):
                figure.linedata[0, 2 * i], figure.linedata[0, 2 * i + 1] = i, history[-n + i][0]
                figure.linedata[1, 2 * i], figure.linedata[1, 2 * i + 1] = i, history[-n + i][1]
            viewer.set_figures((mujoco.MjrRect(10, 10, 320, 150), figure))
            viewer.set_texts([
                (None, None,
                 f"manifold  height {spec.height:.2f}  width {spec.width:.2f}  "
                 f"offset {spec.offset:+.2f}",
                 f"crouch {pose[0]:.2f}  lean {pose[1]:+.2f}  twist {pose[2]:+.2f}  "
                 f"r {radius:.2f}  pelvis z {pelvis[2]:.2f}"),
                (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                 "u/i height  j/k width", ",/. depth  n/m offset  t/y tilt"),
            ])
            viewer.sync()
            clock += 5 * C.CONTROL_DT
            delay = clock - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                clock = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Static manifold -> pose policy")
    parser.add_argument("mode", choices=("train", "report", "verify", "view", "diagnostics"))
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--rollout-steps", type=int, default=64,
                        help="control steps per update; transitions = steps x envs")
    parser.add_argument("--envs", type=int, default=16, help="manifolds trained in parallel")
    parser.add_argument("--candidates", type=int, default=3000,
                        help="sampled manifolds to filter for feasibility")
    parser.add_argument("--height-range", type=float, nargs=2, default=[0.70, 1.00])
    parser.add_argument("--width-range", type=float, nargs=2, default=[0.85, 1.00])
    parser.add_argument("--depth-range", type=float, nargs=2, default=[0.80, 1.00])
    parser.add_argument("--offset-range", type=float, nargs=2, default=[-0.05, 0.05])
    parser.add_argument("--tilt-range", type=float, nargs=2, default=[-12.0, 12.0],
                        help="sagittal tilt of the manifold axis, degrees")
    parser.add_argument("--height", type=float, default=0.78,
                        help="view: manifold height (1.0 = the standing envelope)")
    parser.add_argument("--width", type=float, default=0.85, help="view: manifold width")
    parser.add_argument("--depth", type=float, default=0.75, help="view: manifold depth")
    parser.add_argument("--offset", type=float, default=0.03, help="view: lateral offset [m]")
    parser.add_argument("--tilt", type=float, default=0.0, help="view: manifold axis tilt [deg]")
    parser.add_argument("--min-offset", type=float, default=0.015,
                        help="skip near-centred manifolds, where either twist direction fits")
    parser.add_argument("--need-start", type=float, default=0.6,
                        help="curriculum start: L1 pose magnitude the sampling allows")
    parser.add_argument("--need-max", type=float, default=3.0,
                        help="curriculum end: L1 pose magnitude allowed (deep squat + tuck ~2.5)")
    parser.add_argument("--feasible-margin", type=float, default=0.95,
                        help="keep only manifolds whose best pose has r below this")
    parser.add_argument("--w-outside", type=float, default=25.0,
                        help="violation penalty per second and unit of r past the margin")
    parser.add_argument("--w-effort", type=float, default=0.4,
                        help="reward per second at full pose; higher = pose only when needed")
    parser.add_argument("--entropy-start", type=float, default=0.005)
    parser.add_argument("--entropy-end", type=float, default=0.0004)
    parser.add_argument("--ticks", type=int, default=120, help="verify: control ticks per manifold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_g1/static_fit"))
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    if args.mode == "train":
        return train(args)
    if args.mode == "report":
        return report(args)
    if args.mode == "diagnostics":
        return diagnostics(args)
    return verify(args) if args.mode == "verify" else view(args)


if __name__ == "__main__":
    raise SystemExit(main())
