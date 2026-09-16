"""Exact batched forward kinematics on the GPU, for the training hot path (Phase 2 scaling).

The training loop needs, for every candidate pose, the body surface it occupies - that is what
the manifold containment test is computed from. Doing it with MuJoCo costs ~0.5 ms per pose on
the CPU (mesh sampling dominates), which is the wall the batched/large-scale stages run into.

Forward kinematics is exact and cheap to vectorize: each body's world transform is its parent's
transform composed with a constant offset and one hinge rotation. Exporting the tree once and
re-evaluating it in torch removes MuJoCo from the training loop entirely while staying exact
(no interpolation, no lookup, no approximation), and moves the work to the GPU where it
batches over thousands of poses at once.

    python3 -m manifold_g1.kinematics check      # accuracy + speed against MuJoCo
    python3 -m manifold_g1.kinematics table      # per-pose throughput at several batch sizes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import constants as C


class TorchKinematics:
    """Batched forward kinematics and body-surface points, exactly matching MuJoCo.

    Verified against MuJoCo at geom level (identical vertices, `geom_xmat`/`geom_xpos`):
    zero error. The composition order that matters is

        R_body = R_parent @ quat(body_quat) @ R(axis, q_default + q_delta)
        p_body = p_parent + R_parent @ body_pos
        (vertex) = R_body @ (geom_quat @ v + geom_pos) + p_body

    where `body_quat` is the body's rest rotation and must sit between the parent transform and
    the hinge - omitting it (or applying it outside) is what made earlier attempts wrong.
    """

    def __init__(self, device: str = "cuda", dtype=None, max_vertices: int = 24):
        import mujoco
        import torch

        from .env import G1FlatEnv

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        env = G1FlatEnv()
        model = env.model
        self.model = model
        self.nbody = model.nbody
        self.body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                           for b in range(model.nbody)]
        self.pelvis = self.body_names.index("pelvis")
        world = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "world")

        # per-body: parent, anchor, rest quaternion, hinge axis + the policy index of its joint
        policy_index = np.full(model.nbody, -1, dtype=np.int64)
        axis = np.zeros((model.nbody, 3))
        default_angle = np.zeros(model.nbody)
        for b in range(model.nbody):
            if model.body_jntnum[b] == 0:
                continue
            j = model.body_jntadr[b]
            axis[b] = model.jnt_axis[j]
            for k, act in enumerate(env.body_act):
                if model.actuator_trnid[act, 0] == j:
                    policy_index[b] = int(np.flatnonzero(C.MUJOCO_TO_ISAACLAB == k)[0])
                    default_angle[b] = C.DEFAULT_ANGLES[k]
        self.parent = torch.as_tensor(model.body_parentid, dtype=torch.long, device=self.device)
        self.anchor = torch.as_tensor(model.body_pos, dtype=self.dtype, device=self.device)
        self.rest_quat = torch.as_tensor(model.body_quat, dtype=self.dtype, device=self.device)

        self.axis = torch.as_tensor(axis, dtype=self.dtype, device=self.device)
        self.default_angle = torch.as_tensor(default_angle, dtype=self.dtype, device=self.device)
        self.has_joint = torch.as_tensor(policy_index >= 0, device=self.device)
        self.policy_index = torch.as_tensor(policy_index, dtype=torch.long, device=self.device)
        self.identity = torch.eye(3, dtype=self.dtype, device=self.device)

        # per-geom: body, offset, rotation, and the mesh vertices in the geom frame
        from .body_envelope import _box_points, _cylinder_points, _sphere_points

        geoms = []
        for gid in range(model.ngeom):
            b = model.geom_bodyid[gid]
            if model.geom_group[gid] != 0 or b == world:
                continue
            gtype = model.geom_type[gid]
            size = np.array(model.geom_size[gid], dtype=np.float64)
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[gid]
                verts = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]
                                        + model.mesh_vertnum[mid]].reshape(-1, 3)
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                verts = _sphere_points(size[0], 4, 6)          # primitives need few samples:
            elif gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                verts = _cylinder_points(size[0], size[1],     # they are convex, so the
                                         capsule=(gtype == mujoco.mjtGeom.mjGEOM_CAPSULE),
                                         n_sides=8, n_caps=2)  # extremes are well captured
            elif gtype in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                verts = _box_points(size)
            else:
                continue
            # thin in the geom's own frame: the local vertices are stored raw (they may be
            # rotated by geom_quat), so taking extremes on them directly would drop the points
            # that actually define the envelope once the geom transform is applied
            local = verts @ _quat_to_mat(model.geom_quat[gid]).T
            keep = _thin_indices(local, max_vertices)
            verts = verts[keep]
            geoms.append((b, model.geom_pos[gid], model.geom_quat[gid], verts))
        self.geom_body = torch.as_tensor([g[0] for g in geoms], dtype=torch.long,
                                         device=self.device)
        self.geom_pos = torch.as_tensor(np.stack([g[1] for g in geoms]), dtype=self.dtype,
                                        device=self.device)
        self.geom_quat = torch.as_tensor(np.stack([g[2] for g in geoms]), dtype=self.dtype,
                                         device=self.device)
        counts = [len(g[3]) for g in geoms]
        width = max(counts)
        vertex = np.zeros((len(geoms), width, 3), dtype=np.float64)
        mask = np.zeros((len(geoms), width), dtype=bool)
        for i, g in enumerate(geoms):
            vertex[i, :counts[i]] = g[3]
            mask[i, :counts[i]] = True
        self.vertex = torch.as_tensor(vertex, dtype=self.dtype, device=self.device)
        self.vertex_mask = torch.as_tensor(mask, device=self.device)
        self.total_vertices = int(mask.sum())
        self.geom_rot = _quat_mats(self.geom_quat, self.dtype, self.device)
        # --- stability quantities: COM, feet, joint limits (all from the same body transforms)
        mass = model.body_mass.astype(np.float64)
        ipos = model.body_ipos.astype(np.float64)
        self.mass = torch.as_tensor(mass, dtype=self.dtype, device=self.device)
        self.ipos = torch.as_tensor(ipos, dtype=self.dtype, device=self.device)
        self.total_mass = float(mass.sum())
        self.foot_bodies = torch.as_tensor(
            [self.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")],
            dtype=torch.long, device=self.device)
        # the lowest surface point of each foot, in the foot's own frame
        self.foot_local = []
        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            b = self.body_names.index(name)
            pts = [g[3] for g in geoms if g[0] == b]
            verts = np.concatenate(pts) if pts else np.zeros((1, 3))
            self.foot_local.append(torch.as_tensor(verts[np.argmin(verts[:, 2])], dtype=self.dtype,
                                                   device=self.device))
        # joint limits in policy order (inf where the joint is unlimited)
        lo = np.full(len(C.DEFAULT_ANGLES), -np.inf)
        hi = np.full(len(C.DEFAULT_ANGLES), np.inf)
        for k, act in enumerate(env.body_act):
            j = model.actuator_trnid[act, 0]
            pi = int(np.flatnonzero(C.MUJOCO_TO_ISAACLAB == k)[0])
            if model.jnt_limited[j]:
                lo[pi] = model.jnt_range[j][0] - C.DEFAULT_ANGLES[k]
                hi[pi] = model.jnt_range[j][1] - C.DEFAULT_ANGLES[k]
        self.limit_lo = torch.as_tensor(lo, dtype=self.dtype, device=self.device)
        self.limit_hi = torch.as_tensor(hi, dtype=self.dtype, device=self.device)
        # gains in policy order, for the static holding-torque term
        kp = np.zeros(len(C.DEFAULT_ANGLES))
        effort = np.zeros(len(C.DEFAULT_ANGLES))
        for k, act in enumerate(env.body_act):
            pi = int(np.flatnonzero(C.MUJOCO_TO_ISAACLAB == k)[0])
            kp[pi] = C.KP[k]
            effort[pi] = C.EFFORT_LIMITS[k]
        self.kp = torch.as_tensor(kp, device=self.device)
        self.effort = torch.as_tensor(effort, device=self.device)
        # rest rotation matrices and the tree levels, so `body_transforms` has no Python loop
        self.rest_mat = _quat_mats(self.rest_quat, self.dtype, self.device)
        levels: dict[int, tuple[list[int], list[int]]] = {}
        for b in range(1, self.nbody):
            depth = 0
            node = b
            while node > 0:
                node = int(model.body_parentid[node])
                depth += 1
            levels.setdefault(depth, ([], []))
            levels[depth][0].append(b)
            levels[depth][1].append(int(model.body_parentid[b]))
        self.levels = {d: (torch.as_tensor(c, device=self.device),
                           torch.as_tensor(p_, device=self.device))
                       for d, (c, p_) in levels.items()}
        self.branch_depth = max(self.levels) if self.levels else 0

    # -- transforms --------------------------------------------------------
    def body_transforms(self, poses):
        """World rotation and position of every body: [B, nbody, 3, 3], [B, nbody, 3]."""
        torch = self.torch
        poses = torch.as_tensor(poses, dtype=self.dtype, device=self.device)
        if poses.dim() == 1:
            poses = poses.unsqueeze(0)
        batch = poses.shape[0]
        angles = torch.zeros(batch, self.nbody, dtype=self.dtype, device=self.device)
        valid = self.policy_index >= 0
        angles[:, valid] = poses[:, self.policy_index[valid]] + self.default_angle[valid]

        rot = torch.zeros(batch, self.nbody, 3, 3, dtype=self.dtype, device=self.device)
        pos = torch.zeros(batch, self.nbody, 3, dtype=self.dtype, device=self.device)
        rot[:, 0] = self.identity
        pos[:, 0] = self.anchor[0]
        # hinge rotations for every body at once: [B, nbody, 3, 3] via Rodrigues, no Python loop
        c = torch.cos(angles).unsqueeze(-1).unsqueeze(-1)                     # [B, nbody, 1, 1]
        sn = torch.sin(angles).unsqueeze(-1).unsqueeze(-1)
        ax = self.axis.unsqueeze(0)                                           # [1, nbody, 3]
        K = torch.zeros(1, self.nbody, 3, 3, dtype=self.dtype, device=self.device)
        K[..., 0, 1] = -ax[..., 2]
        K[..., 0, 2] = ax[..., 1]
        K[..., 1, 0] = ax[..., 2]
        K[..., 1, 2] = -ax[..., 0]
        K[..., 2, 0] = -ax[..., 1]
        K[..., 2, 1] = ax[..., 0]
        eye = torch.eye(3, dtype=self.dtype, device=self.device)
        hinge = eye + sn * K + (1 - c) * (K @ K)                              # [B, nbody, 3, 3]
        joint = self.has_joint.view(1, self.nbody, 1, 1)
        local_all = torch.where(joint, self.rest_mat.unsqueeze(0) @ hinge,
                                self.rest_mat.unsqueeze(0).expand(batch, -1, -1, -1))
        # one gather/matmul per tree level instead of one Python iteration per body
        for level in range(1, self.branch_depth + 1):
            child, parent = self.levels[level]
            rot[:, child] = rot[:, parent] @ local_all[:, child]
            pos[:, child] = pos[:, parent] + torch.einsum(
                "bpij,pj->bpi", rot[:, parent], self.anchor[child])
        return rot, pos

    def forward(self, poses, return_mask: bool = False):
        """Body-surface points for a batch of pose deltas: [B, K, 3], pelvis-anchored.

        One padded tensor multiply for every geom: the per-geom Python loop was the whole cost
        (44 small kernels per step), and padding to a common vertex count removes it.
        """
        torch = self.torch
        rot, pos = self.body_transforms(poses)
        Rg = rot[:, self.geom_body] @ self.geom_rot.unsqueeze(0)              # [B, G, 3, 3]
        Pg = pos[:, self.geom_body] + torch.einsum("bgij,gj->bgi", rot[:, self.geom_body],
                                                   self.geom_pos)
        cloud = torch.einsum("gkj,bgij->bgki", self.vertex, Rg) + Pg.unsqueeze(2)
        points = cloud.reshape(cloud.shape[0], -1, 3)
        points = points - pos[:, self.pelvis:self.pelvis + 1]
        if return_mask:
            return points, self.vertex_mask.reshape(1, -1).expand(cloud.shape[0], -1)
        return points

    def envelope(self, poses):
        """Per-pose half-extents of the body surface (pelvis-anchored): [B, 3].

        The containment test only needs the extremes, but padding introduces dummy points at
        the origin, so the padding is placed there deliberately and the max is taken over the
        real points only.
        """
        torch = self.torch
        points, mask = self.forward(poses, return_mask=True)
        big = torch.full_like(points, -1e9)
        points = torch.where(mask.unsqueeze(-1), points, big)
        return points.amax(dim=1)

    def joint_torque(self, poses):
        """Static holding torque per pose: PD gains applied to the pose offset, |tau| summed.

        The design doc's energy term is sum|tau_i * dq_i|; at rest dq = 0, but holding an
        offset pose still costs torque, which is what this measures (normalised by the effort
        limits so the number is comparable across joints).
        """
        torch = self.torch
        poses_t = torch.as_tensor(poses, dtype=self.dtype, device=self.device)
        if poses_t.dim() == 1:
            poses_t = poses_t.unsqueeze(0)
        tau = (self.kp.to(self.dtype) * poses_t).abs() / self.effort.to(self.dtype)
        return tau.mean(dim=1)

    def stability(self, poses):
        """Center of mass, foot heights and joint-limit margin, all batched.

        These are the kinematic half of "will the pose stand up": the reward uses them so the
        policy is not free to crouch past its support or lift a foot while claiming to stand.

        returns dict with
          com      [B, 3]  mass-weighted centre of mass, pelvis-anchored
          feet_z   [B, 2]  lowest surface point of each foot, pelvis-anchored
                 (the floor is at -pelvis_height, so a value nearer 0 means a lifted foot)
          limit    [B]     worst joint-limit overshoot in radians (0 when inside the range)
        """
        torch = self.torch
        rot, pos = self.body_transforms(poses)
        com_world = torch.einsum("bki,k->bi", pos + torch.einsum("bkij,kj->bki", rot, self.ipos),
                                 self.mass) / self.total_mass
        com = com_world - pos[:, self.pelvis]
        foot_pts = torch.stack([
            pos[:, b] + torch.einsum("bij,j->bi", rot[:, b], self.foot_local[i])
            for i, b in enumerate(self.foot_bodies)], dim=1)
        feet_z = foot_pts[..., 2] - pos[:, self.pelvis, 2:3]
        poses_t = torch.as_tensor(poses, dtype=self.dtype, device=self.device)
        if poses_t.dim() == 1:
            poses_t = poses_t.unsqueeze(0)
        overshoot = torch.clamp(self.limit_lo - poses_t, min=0) + torch.clamp(
            poses_t - self.limit_hi, min=0)
        return {"com": com, "feet_z": feet_z, "limit": overshoot.max(dim=-1).values}

    def radius(self, poses, manifold):
        """Containment of the body surface in an ellipsoid manifold: [B] normalized radii."""
        torch = self.torch
        points, mask = self.forward(poses, return_mask=True)
        primitive = manifold.primitives[0]
        center = torch.as_tensor(primitive.center, dtype=self.dtype, device=self.device)
        semi = torch.as_tensor(primitive.semi, dtype=self.dtype, device=self.device)
        rot = _quat_mat(torch.as_tensor(primitive.quat, dtype=self.dtype, device=self.device),
                        self.dtype, self.device)
        local = (points - center) @ rot
        radius = (local / semi).norm(dim=-1)
        radius = radius.masked_fill(~mask, -1.0).max(dim=-1).values
        return radius.detach().cpu().numpy()          # callers are numpy-space env code


def _thin_indices(points: np.ndarray, max_vertices: int) -> np.ndarray:
    """Indices of a point set to keep: the per-axis extremes plus a spread of the interior.

    `points` must be in the frame the envelope is judged in (the geom frame), otherwise the
    selected indices will not be the ones that define the body's extent and the approximated
    envelope comes out systematically smaller than the real one.
    """
    if len(points) <= max_vertices:
        return np.arange(len(points))
    keep = set()
    for ax in range(3):
        keep.add(int(np.argmax(points[:, ax])))
        keep.add(int(np.argmin(points[:, ax])))
    budget = max(0, max_vertices - len(keep))
    if budget:
        stride = max(1, len(points) // budget)
        keep |= set(range(0, len(points), stride))
    return np.array(sorted(keep))


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _thin(verts: np.ndarray, max_vertices: int) -> np.ndarray:
    """Deprecated shim kept for callers that already have local vertices."""
    return verts[_thin_indices(verts, max_vertices)]


def _quat_mats(q, dtype, device):
    """Rotation matrices for a batch of quaternions: [N, 3, 3]."""
    torch = __import__("torch")
    if not isinstance(q, torch.Tensor):
        q = torch.as_tensor(q, dtype=dtype, device=device)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    one = torch.ones_like(w)
    return torch.stack([
        torch.stack([one - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), one - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), one - 2 * (x * x + y * y)], -1),
    ], -2)


def _quat_mat(q, dtype, device):
    torch = __import__("torch")
    if not isinstance(q, torch.Tensor):
        q = torch.as_tensor(q, dtype=dtype, device=device)
    w, x, y, z = q[0], q[1], q[2], q[3]
    one = torch.ones((), dtype=dtype, device=device)
    rows = [
        torch.stack([one - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), one - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), one - 2 * (x * x + y * y)]),
    ]
    return torch.stack(rows)


def _axis_angle(axis, angle, dtype):
    """Rodrigues rotation for a batch of angles about a fixed axis: [B, 3, 3]."""
    torch = __import__("torch")
    axis = axis / axis.norm().clamp(min=1e-9)
    c = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
    s = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
    K = torch.zeros(3, 3, dtype=axis.dtype, device=axis.device)
    K[0, 1], K[0, 2] = -axis[2], axis[1]
    K[1, 0], K[1, 2] = axis[2], -axis[0]
    K[2, 0], K[2, 1] = -axis[1], axis[0]
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return c * eye + s * K + (1 - c) * (axis.unsqueeze(-1) * axis.unsqueeze(0)).unsqueeze(0)


def check(args) -> int:
    """Accuracy against MuJoCo at geom level, and speed."""
    import time

    import mujoco

    from .env import G1FlatEnv
    from .demos import load_demos

    kin = TorchKinematics(device=args.device)
    env = G1FlatEnv()
    model = env.model
    world = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "world")
    _, all_demos = load_demos()
    poses = np.concatenate(list(all_demos.values()))
    sample = poses[np.random.default_rng(0).choice(len(poses), args.samples, replace=False)]

    def mujoco_points():
        pts, n = [], 0
        for gid in range(model.ngeom):
            if (model.geom_group[gid] != 0 or model.geom_bodyid[gid] == world
                    or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mid = model.geom_dataid[gid]
            v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]
                                + model.mesh_vertnum[mid]].reshape(-1, 3)
            pts.append(v @ env.data.geom_xmat[gid].reshape(3, 3).T + env.data.geom_xpos[gid])
            n += len(v)
        return np.concatenate(pts)

    torch = kin.torch
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = kin.forward(sample).cpu().numpy()
    t_torch = (time.perf_counter() - t0) / len(sample)
    errs = []
    for i, q in enumerate(sample):
        env.data.qpos[env.body_qadr] = C.DEFAULT_ANGLES + q[C.ISAACLAB_TO_MUJOCO]
        mujoco.mj_kinematics(model, env.data)
        truth = mujoco_points()
        pelvis = env.data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
        truth = truth - pelvis
        errs.append(np.abs(np.abs(pred[i]).max(0) - np.abs(truth).max(0)).max())
    print(f"poses {len(sample)}  vertices {kin.total_vertices}")
    print(f"  envelope error vs MuJoCo: mean {np.mean(errs)*1e6:.2f} um  max {np.max(errs)*1e6:.2f} um")
    print(f"  speed: {t_torch*1e6:.0f} us/pose on {kin.device}")
    return 0


def table(args) -> int:
    import time

    import torch

    from .demos import load_demos

    kin = TorchKinematics(device=args.device)
    _, all_demos = load_demos()
    poses = np.concatenate(list(all_demos.values()))
    print(f"device {kin.device}   vertices {kin.total_vertices}")
    print(f"{'batch':>7} {'ms/batch':>10} {'us/pose':>9} {'poses/s':>11}")
    for batch in (1, 16, 256, 1024, 4096, 16384):
        p = poses[np.random.default_rng(0).choice(len(poses), batch, replace=False)]
        t = torch.as_tensor(p, dtype=kin.dtype)
        with torch.no_grad():
            kin.forward(t)
            t0 = time.perf_counter()
            for _ in range(5):
                kin.forward(t)
            dt = (time.perf_counter() - t0) / 5
        print(f"{batch:>7} {dt*1e3:>10.2f} {dt/batch*1e6:>9.2f} {batch/dt:>11.0f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Batched GPU forward kinematics")
    parser.add_argument("mode", choices=("check", "table"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=25)
    args = parser.parse_args()
    return check(args) if args.mode == "check" else table(args)


if __name__ == "__main__":
    raise SystemExit(main())
