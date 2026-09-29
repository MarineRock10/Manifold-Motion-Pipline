"""Render paired manifold/action GIFs from the accepted SEED replay catalogue.

The catalogue is a numerical training artifact.  This renderer makes the pairing
auditable: for one selected window it loads the exact source replay, recovers the
same 30 Hz future frames, and writes

* ``*_manifold.gif``: the reverse-synthesized environment corridor ``M_e(t)``
  together with the measured robot envelope ``M_self(t)``;
* ``*_action.gif``: the G1 kinematic execution from that very same replay;
* ``*_pair.gif``: the two views side by side, frame for frame.

The corridor is explicitly labelled as reverse-synthesized from accepted flat-ground
execution.  It is not a sensor-derived obstacle map and these GIFs are replay
visualizations, not a new training or physical-validity claim.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from manifold_motion.core import constants as C
from manifold_motion.dataio.seed_windows import _relative_pose, _resample_record
from manifold_motion.simulation.env import G1FlatEnv


@dataclass(frozen=True)
class RenderConfig:
    width: int = 520
    height: int = 420
    fps: int = 15
    max_families: int = 0


FAMILY_ZH = {
    "all_fours": "四肢支撑", "box_jump": "箱跳来源片段", "button_lever": "按压 / 拨杆动作",
    "carry_object": "搬运姿态", "crouch_transition": "蹲起转换", "crouch_walk": "蹲走",
    "dodge_lateral": "侧向闪避", "door_interaction": "开门来源片段", "forward_lunge": "向前弓步",
    "hands_back_walk": "背手行走", "high_jump": "高跳来源片段", "jog_forward": "向前慢跑",
    "kneel": "跪姿转换", "ladder": "攀梯来源片段", "side_hop": "侧向跳步",
    "step_down_box": "下台阶来源片段", "step_up_box": "上台阶来源片段", "turn_in_place": "原地转向",
    "walk_curve": "曲线行走", "walk_forward": "向前行走", "walk_lateral": "侧向行走",
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _local_to_world(points: np.ndarray, origin_pos: np.ndarray, origin_quat: np.ndarray) -> np.ndarray:
    rot = C.quat_to_matrix(origin_quat)
    return np.asarray(points) @ rot.T + origin_pos


def _yaw_rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _add_ellipsoid(scene: mujoco.MjvScene, center: np.ndarray, semi: np.ndarray,
                   rotation: np.ndarray, rgba: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        np.asarray(semi, dtype=np.float64), np.asarray(center, dtype=np.float64),
        np.asarray(rotation, dtype=np.float64).reshape(-1), np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_sphere(scene: mujoco.MjvScene, center: np.ndarray, radius: float,
                rgba: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0.0, 0.0]),
        np.asarray(center, dtype=np.float64), np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_line(scene: mujoco.MjvScene, start: np.ndarray, end: np.ndarray,
              rgba: tuple[float, float, float, float], radius: float = 0.006) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                       np.eye(3).reshape(-1), np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                         np.asarray(start, dtype=float), np.asarray(end, dtype=float))
    scene.ngeom += 1


def _wire_ellipsoid(scene: mujoco.MjvScene, center: np.ndarray, semi: np.ndarray,
                    rotation: np.ndarray, color: tuple[float, float, float, float],
                    radius: float = 0.006) -> None:
    angles = np.linspace(0, 2 * np.pi, 41)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        ring = np.zeros((len(angles), 3))
        ring[:, a] = semi[a] * np.cos(angles)
        ring[:, b] = semi[b] * np.sin(angles)
        ring = ring @ rotation.T + center
        for start, end in zip(ring[:-1], ring[1:]):
            _add_line(scene, start, end, color, radius)


def _camera(bounds: np.ndarray) -> mujoco.MjvCamera:
    bounds = np.asarray(bounds, dtype=np.float64)
    low = bounds.min(axis=0)
    high = bounds.max(axis=0)
    center = 0.5 * (low + high)
    radius = max(0.55, float(np.linalg.norm(high - low) / 2))
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = center
    camera.distance = 1.10 * radius / math.sin(math.radians(22.5))
    camera.azimuth = 135.0
    camera.elevation = -18.0
    return camera


def _panel(image: Image.Image, title: str, subtitle: str, footer: Iterable[str],
           progress: float, color: str) -> Image.Image:
    # Text sits outside the 3D viewport, so labels never cover the head/feet.
    out = Image.new("RGB", (image.width, image.height + 126), "#101f31")
    out.paste(image, (0, 65))
    draw = ImageDraw.Draw(out)
    draw.text((14, 9), title, fill=color, font=_font(19, True))
    draw.text((14, 37), subtitle, fill="#d3e2ed", font=_font(13))
    for row, line in enumerate(footer):
        draw.text((14, image.height + 75 + 19 * row), line, fill="#d3e2ed", font=_font(12))
    draw.rectangle((0, out.height - 4, int(out.width * progress), out.height), fill=color)
    return out


def _set_pose(env: G1FlatEnv, q_policy: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> None:
    env.data.qpos[:3] = np.asarray(pos, dtype=np.float64)
    env.data.qpos[3:7] = np.asarray(quat, dtype=np.float64)
    env.data.qpos[env.body_qadr] = np.asarray(q_policy, dtype=np.float64)[C.ISAACLAB_TO_MUJOCO]
    env.data.qpos[env.hand_qadr] = 0.0
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def _render_scene(env: G1FlatEnv, renderer: mujoco.Renderer, camera: mujoco.MjvCamera,
                  option: mujoco.MjvOption, q_policy: np.ndarray, pos: np.ndarray,
                  quat: np.ndarray, *, hide_robot: bool,
                  draw: Any | None = None) -> Image.Image:
    _set_pose(env, q_policy, pos, quat)
    option.geomgroup[1] = int(not hide_robot)
    renderer.update_scene(env.data, camera=camera, scene_option=option)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    if draw is not None:
        draw(renderer.scene)
    return Image.fromarray(renderer.render())


def _gif_durations(count: int, fps: int) -> list[int]:
    # GIF delays use 10 ms units. Alternating 60/70 ms preserves 15 fps overall,
    # unlike a constant 66 ms which silently becomes 60 ms in many encoders.
    ticks = np.rint(np.arange(count + 1) * 100.0 / fps).astype(int)
    return (10 * np.diff(ticks)).tolist()


def _save_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError(f"no frames for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    rest = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=256) for frame in frames[1:]]
    first.save(path, save_all=True, append_images=rest, duration=_gif_durations(len(frames), fps),
               loop=0, optimize=True, disposal=2)


def _select_rows(arrays: dict[str, np.ndarray], *, max_families: int) -> list[int]:
    names = [str(x) for x in arrays["primitive_names"]]
    groups: dict[str, list[int]] = {}
    for i, primitive in enumerate(arrays["primitive"]):
        groups.setdefault(names[int(primitive)], []).append(i)
    selected: list[tuple[str, int, float]] = []
    for family, rows in groups.items():
        # The gallery is explicitly a visual audit, so select a deterministic high-motion
        # window rather than silently showing the first repeated walk segment.
        scores = []
        for row in rows:
            scores.append((float(np.mean(np.linalg.norm(np.diff(arrays["target_exec"][row, :, :29], axis=0), axis=1))), row))
        score, row = max(scores)
        selected.append((family, row, score))
    selected.sort(key=lambda item: item[0])
    if max_families > 0:
        selected = selected[:max_families]
    return [row for _, row, _ in selected]


def _load_window(row: int, arrays: dict[str, np.ndarray], metadata: dict[str, Any], record_root: Path) -> dict[str, Any]:
    names = [str(x) for x in arrays["primitive_names"]]
    clip = metadata["clips"][int(arrays["clip_index"][row])]
    motion_id = str(clip["motion_id"])
    record_path = record_root / "records" / f"{motion_id}.npz"
    if not record_path.is_file():
        raise FileNotFoundError(f"accepted source replay not found: {record_path}")
    with np.load(record_path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    data = _resample_record(raw, float(metadata["config"]["output_hz"]))
    origin = int(arrays["source_origin"][row])
    horizon = int(arrays["target_exec"].shape[1])
    future = np.arange(origin + 1, origin + 1 + horizon)
    if future[-1] >= len(data["t"]):
        raise ValueError(f"window {row} exceeds source replay {motion_id}")
    q = data["q_exec"][future]
    root_pos = data["base_pos"][future]
    root_quat = data["base_quat"][future]
    target = arrays["target_exec"][row]
    # A pairing bug here is worse than an empty gallery.  Keep the tolerance tight enough to
    # catch wrong clip/origin lookup while allowing the catalogue's float32 serialization.
    if not np.allclose(q, target[:, :29], atol=2e-4):
        raise ValueError(f"catalogue/action mismatch for row {row} ({motion_id})")
    relative_pos, relative_rot6 = _relative_pose(data["base_pos"], data["base_quat"], origin,
                                                slice(origin + 1, origin + 1 + horizon))
    if not np.allclose(np.concatenate([relative_pos, relative_rot6], axis=1), target[:, 29:],
                       atol=2e-4, rtol=0):
        raise ValueError(f"catalogue/root pose mismatch for row {row} ({motion_id})")
    env_centres = _local_to_world(arrays["corridor"][row, :, :3], data["base_pos"][origin], data["base_quat"][origin])
    origin_rot = C.quat_to_matrix(data["base_quat"][origin])
    env_rot = np.asarray([origin_rot @ _yaw_rotation(yaw) for yaw in arrays["corridor"][row, :, 6]])
    self_rot = np.asarray([C.quat_to_matrix(quat) for quat in root_quat])
    return {
        "row": row,
        "catalog_id": str(arrays["catalog_id"][row]) if "catalog_id" in arrays else f"me_action_{row:06d}",
        "family": names[int(arrays["primitive"][row])],
        "motion_id": motion_id,
        "actor_uid": str(clip.get("actor_uid", "")),
        "q": q,
        "root_pos": root_pos,
        "root_quat": root_quat,
        "env_centres": env_centres,
        "env_semi": arrays["corridor"][row, :, 3:6],
        "env_rot": env_rot,
        "self_semi": arrays["self_manifold"][row],
        "self_rot": self_rot,
        "source_time": data["t"][future] - data["t"][origin],
        "source_start_seconds": float(data["t"][origin]),
        "source_origin": origin,
        "source_hz": float(metadata["config"]["output_hz"]),
        "split": ("train", "validation", "test")[int(arrays["split"][row])],
        "max_target_error": float(np.max(np.abs(np.concatenate([q, relative_pos, relative_rot6], axis=1) - target))),
    }


def _render_one(sample: dict[str, Any], out: Path, config: RenderConfig) -> dict[str, str]:
    env = G1FlatEnv()
    # Render each surface once: the source MJCF has duplicate visual/collision meshes.
    collision = (env.model.geom_bodyid != 0) & (env.model.geom_group != 1)
    env.model.geom_group[collision] = 3
    floor = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    env.model.geom_matid[floor] = -1
    env.model.geom_rgba[floor] = (0.74, 0.79, 0.84, 1.0)
    env.model.vis.global_.offwidth = config.width
    env.model.vis.global_.offheight = config.height
    renderer = mujoco.Renderer(env.model, height=config.height, width=config.width)
    extent = np.sqrt(np.einsum("tij,tj->ti", sample["env_rot"] ** 2, sample["env_semi"] ** 2))
    manifold_bounds = np.concatenate([sample["env_centres"] + extent,
                                      sample["env_centres"] - extent], axis=0)
    # The action camera fits the actual body points rather than the (conservative)
    # enclosing ellipsoid. Both cameras remain fixed for the entire window.
    action_bounds = []
    for q, pos, quat in zip(sample["q"], sample["root_pos"], sample["root_quat"]):
        _set_pose(env, q, pos, quat)
        action_bounds.extend([env.data.xpos[1:].min(axis=0) - 0.14,
                              env.data.xpos[1:].max(axis=0) + 0.14])
    manifold_camera = _camera(manifold_bounds)
    action_camera = _camera(np.asarray(action_bounds))
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[:] = 0
    option.geomgroup[:2] = 1
    manifold_frames: list[Image.Image] = []
    action_frames: list[Image.Image] = []
    pair_frames: list[Image.Image] = []
    try:
        for i, (q, pos, quat) in enumerate(zip(sample["q"], sample["root_pos"], sample["root_quat"])):
            t = float(sample["source_time"][i])
            def common_draw(scene: mujoco.MjvScene) -> None:
                center = sample["root_pos"].mean(axis=0)
                for offset in np.arange(-3, 3.1, 0.5):
                    _add_line(scene, center * [1, 0, 0] + [-3, offset, 0.003],
                              center * [1, 0, 0] + [3, offset, 0.003], (0.43, 0.53, 0.62, 1), 0.002)
                    _add_line(scene, center * [1, 0, 0] + [offset, -3, 0.003],
                              center * [1, 0, 0] + [offset, 3, 0.003], (0.43, 0.53, 0.62, 1), 0.002)
                for start, end in zip(sample["root_pos"][:-1], sample["root_pos"][1:]):
                    _add_line(scene, start * [1, 1, 0] + [0, 0, 0.014],
                              end * [1, 1, 0] + [0, 0, 0.014], (0.03, 0.48, 0.60, 1), 0.008)
            def manifold_draw(scene: mujoco.MjvScene, i: int = i) -> None:
                common_draw(scene)
                _add_ellipsoid(scene, sample["env_centres"][i], sample["env_semi"][i],
                               sample["env_rot"][i], (0.05, 0.70, 0.92, 0.10))
                _wire_ellipsoid(scene, sample["env_centres"][i], sample["env_semi"][i],
                                sample["env_rot"][i], (0.00, 0.56, 0.76, 1))
                _wire_ellipsoid(scene, pos, sample["self_semi"][i], sample["self_rot"][i],
                                (0.96, 0.39, 0.06, 1), 0.008)
                _add_sphere(scene, pos, 0.033, (0.90, 0.16, 0.10, 1.0))
                for k, color in enumerate(((0.85, 0.1, 0.1, 1), (0.12, 0.62, 0.12, 1), (0.15, 0.2, 0.85, 1))):
                    _add_line(scene, pos, pos + sample["self_rot"][i, :, k] * 0.35, color, 0.008)
            def action_draw(scene: mujoco.MjvScene, i: int = i) -> None:
                common_draw(scene)
            manifold = _render_scene(env, renderer, manifold_camera, option, q, pos, quat,
                                     hide_robot=True, draw=manifold_draw)
            subtitle = f"{sample['catalog_id']} | frame {i+1:02d}/{len(sample['q'])} | t={t:.2f}s"
            playback = f"{config.fps / sample['source_hz']:.2f}x speed | ground grid: 0.5 m"
            manifold = _panel(manifold, "MANIFOLD | " + sample["family"], subtitle, [
                "CYAN: reverse corridor M_e   ORANGE: body envelope M_self",
                "semi-axes (m): " + " / ".join(f"{v:.2f}" for v in sample["env_semi"][i]) + " | " + playback,
            ], (i+1)/len(sample['q']), "#71e6fc")
            action = _render_scene(env, renderer, action_camera, option, q, pos, quat,
                                   hide_robot=False, draw=action_draw)
            action = _panel(action, "ACTION | " + sample["family"], subtitle, [
                "Recorded SONIC execution | " + sample["split"] + " | " + sample['actor_uid'],
                f"pelvis z={pos[2]:.2f} m | {playback}",
            ], (i+1)/len(sample['q']), "#ffbd72")
            manifold_frames.append(manifold)
            action_frames.append(action)
            joined = Image.new("RGB", (config.width * 2, manifold.height), (15, 25, 35))
            joined.paste(manifold, (0, 0))
            joined.paste(action, (config.width, 0))
            pair_frames.append(joined)
    finally:
        renderer.close()
    stem = f"{sample['catalog_id']}_{sample['family']}".replace("/", "_")
    paths = {
        "manifold": out / f"{stem}_manifold.gif",
        "action": out / f"{stem}_action.gif",
        "pair": out / f"{stem}_pair.gif",
    }
    _save_gif(manifold_frames, paths["manifold"], config.fps)
    _save_gif(action_frames, paths["action"], config.fps)
    _save_gif(pair_frames, paths["pair"], config.fps)
    # Static proof sheet lets code review inspect beginning/middle/end without a GUI.
    proof = Image.new("RGB", (pair_frames[0].width, pair_frames[0].height * 3))
    for j, index in enumerate((0, len(pair_frames) // 2, len(pair_frames) - 1)):
        proof.paste(pair_frames[index], (0, j * pair_frames[0].height))
    proof.save(out / f"{stem}_keyframes.jpg", quality=90)
    return {key: str(value.name) for key, value in paths.items()}


def _write_gallery(out: Path, entries: list[dict[str, Any]], config: RenderConfig) -> None:
    cards = []
    for item in entries:
        title = FAMILY_ZH.get(item['family'], item['family'])
        cards.append(f'''<article class="card" data-search="{html.escape(title + ' ' + item['family'] + ' ' + item['catalog_id'])}">
  <h2>{html.escape(title)} <small>{html.escape(item['family'])}</small></h2>
  <p><code>{html.escape(item['catalog_id'])}</code> · {item['frames']} 帧 · {html.escape(item['split'])}</p>
  <div class="labels"><span>流形 GIF：环境 + 自身包络</span><span>动作 GIF：G1 执行记录</span></div>
  <img class="synced" src="{html.escape(item['files']['pair'])}" loading="lazy" alt="{html.escape(title)}：流形与动作逐帧同步对照">
  <nav><a href="{html.escape(item['files']['manifold'])}">单独打开流形 GIF</a><a href="{html.escape(item['files']['action'])}">单独打开动作 GIF</a><a href="{html.escape(item['files']['pair'])}">同步对照 GIF</a></nav>
  <p class="source">来源 {html.escape(item['motion_id'])} · actor {html.escape(item['actor_uid'])}</p>
</article>''')
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>流形 GIF ↔ 动作 GIF</title>
<style>body{{font-family:"Microsoft YaHei",Arial,sans-serif;background:#091522;color:#eaf5fa;margin:24px auto;padding:0 18px;max-width:1320px}}h1{{margin-bottom:6px}}p{{color:#b7cad2;line-height:1.7}}.note{{padding:12px 20px;border-left:3px solid #eeaf66;background:#192d3b}}.toolbar{{position:sticky;top:0;background:#091522;padding:10px 0;z-index:1}}input{{width:min(340px,80%);padding:10px;border-radius:6px;border:1px solid #507080;background:#112432;color:#fff}}.grid{{display:grid;grid-template-columns:1fr;gap:22px}}.card{{background:#102331;border:1px solid #294d5d;border-radius:10px;padding:16px}}h2{{font-size:21px;margin:4px 0}}small{{font-size:14px;font-weight:400;color:#aecad4}}code{{font-size:14px;color:#84e9ff}}img{{display:block;width:100%;background:#000;border-radius:6px}}a{{color:#72e9ff;text-decoration:none}}a:hover{{text-decoration:underline}}nav{{display:flex;flex-wrap:wrap;gap:24px;padding-top:14px}}.labels{{display:flex;padding:8px 0;color:#bcd0dd}}.labels span{{width:50%;text-align:center}}.source{{font-size:12px;overflow-wrap:anywhere}}button{{margin-left:12px;background:#243f50;color:white;padding:8px;border:1px solid #507080;border-radius:6px;cursor:pointer}}</style></head>
<body><h1>一个流形 GIF，对应一个动作 GIF</h1><p>{len(entries)} 组成对样本 · 每组两个独立 GIF，另附逐帧同步合成 GIF · 原始窗口 1.2 秒，默认半速观看（2.4 秒）。</p>
<div class="note"><p>左图：青色为反向构建的环境走廊 M_e(t)，橙色为随执行姿态变化的自身包络 M_self(t)。右图：对应窗口的 SONIC/MuJoCo 执行记录，非新生成的模型动作。</p>
<p>这是<strong>数据处理结果</strong>，不是“模型看见流形后自主完成动作”的实验。环境流形由既有动作反推，未放置真实障碍；“开门 / 攀梯 / 箱跳”等名称是 SEED 来源标签，不代表已经完成对应物体交互。蓝色地面线为记录的根轨迹，非规划结果。</p>
<p>默认每个来源动作族选取关节变化较大的一个窗口，便于目视审查；不是随机抽样或成功率统计。镜头在片段内固定；两栏独立取景，网格均为 0.5 米。主预览使用合成 GIF，保证左右严格同步。</p></div>
<div class="toolbar"><input id="search" aria-label="筛选样本" placeholder="筛选：蹲走、侧向、walk、样本编号…"><button id="restart">重新播放可见样本</button></div><main class="grid">{''.join(cards)}</main>
<script>const q=document.getElementById('search');q.oninput=()=>document.querySelectorAll('.card').forEach(c=>c.hidden=!c.dataset.search.toLowerCase().includes(q.value.toLowerCase()));document.getElementById('restart').onclick=()=>document.querySelectorAll('.card:not([hidden]) img').forEach(img=>{{if(img.getBoundingClientRect().bottom>0&&img.getBoundingClientRect().top<innerHeight){{const s=img.src;img.src='';requestAnimationFrame(()=>img.src=s);}}}});</script></body></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")
    md = ["# 流形 GIF ↔ 动作 GIF", "", f"{len(entries)} 组成对样本，左右来源、窗口和时间戳一致。", "",
          "推荐打开 [可视化画廊](index.html)，主预览使用左右合成 GIF 保证同步；下方也保留两个独立 GIF。", "",
          "青色：反向合成环境走廊 M_e；橙色：执行姿态的自身包络 M_self；右侧：SONIC 执行记录的运动学回放。", "",
          "这批是数据处理可视化，不是训练效果或新一轮物理验证。来源动作标签不保证对应交互任务完成。", "",
          "默认每族选关节帧间变化较大的一窗。每窗 36 帧 / 1.2 秒，默认 15 fps 半速播放。两栏独立固定取景，网格 0.5 米。", ""]
    for item in entries:
        md += [f"## {FAMILY_ZH.get(item['family'], item['family'])} · `{item['catalog_id']}`", "",
               f"| Manifold `M_e(t)` + `M_self(t)` | Executed G1 action |",
               "|---|---|",
               f"| ![]({item['files']['manifold']}) | ![]({item['files']['action']}) |", "",
               f"[Open synchronized pair GIF]({item['files']['pair']}) · source `{item['motion_id']}`", ""]
    (out / "README.md").write_text("\n".join(md), encoding="utf-8")


def render_gallery(*, windows: Path, metadata: Path, record_root: Path, out: Path,
                   config: RenderConfig, row_indices: list[int] | None = None) -> dict[str, Any]:
    with np.load(windows, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    if meta.get("environment", {}).get("provenance") != "reverse_synthesized_from_R_exec":
        raise ValueError("this gallery expects explicitly reverse-synthesized corridor data")
    rows = row_indices if row_indices is not None else _select_rows(arrays, max_families=config.max_families)
    if not rows or any(row < 0 or row >= len(arrays["target_exec"]) for row in rows) or len(set(rows)) != len(rows):
        raise ValueError("row indices must be non-empty, unique, and inside the catalogue")
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for number, row in enumerate(rows, start=1):
        sample = _load_window(row, arrays, meta, record_root)
        print(f"[{number:02d}/{len(rows):02d}] {sample['catalog_id']} {sample['family']}", flush=True)
        files = _render_one(sample, out, config)
        entries.append({"catalog_id": sample["catalog_id"], "family": sample["family"],
                        "motion_id": sample["motion_id"], "actor_uid": sample["actor_uid"],
                        "source_origin": sample["source_origin"], "frames": len(sample["q"]),
                        "split": sample["split"], "max_target_error": sample["max_target_error"],
                        "files": files})
    _write_gallery(out, entries, config)
    manifest = {"schema": "manifold-motion.paired-gif-gallery.v1", "fps": config.fps,
                "width": config.width, "height": config.height + 126, "entries": entries,
                "source_hz": float(meta["config"]["output_hz"]),
                "selection": "explicit_rows" if row_indices is not None else "maximum_mean_joint_frame_delta_per_family",
                "source_windows": str(windows), "source_metadata": str(metadata),
                "source_record_root": str(record_root),
                "warning": "M_e is reverse-synthesized from accepted R_exec; GIFs are replay visualizations."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows_metadata.json"))
    parser.add_argument("--record-root", type=Path, default=Path("reports/manifold_motion/seed_capability_supported_v1"))
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/manifold_action_pairs_v1"))
    parser.add_argument("--width", type=int, default=RenderConfig.width)
    parser.add_argument("--height", type=int, default=RenderConfig.height)
    parser.add_argument("--fps", type=int, default=RenderConfig.fps)
    parser.add_argument("--max-families", type=int, default=0)
    parser.add_argument("--rows", type=int, nargs="+", help="render exact catalogue rows instead of one representative per family")
    args = parser.parse_args()
    if min(args.width, args.height, args.fps) <= 0 or args.fps > 100 or args.max_families < 0:
        parser.error("dimensions must be positive; 1 <= fps <= 100; max-families >= 0")
    if args.rows is not None and args.max_families:
        parser.error("--rows and --max-families are mutually exclusive")
    manifest = render_gallery(windows=args.windows, metadata=args.metadata, record_root=args.record_root,
                              out=args.out, config=RenderConfig(args.width, args.height, args.fps, args.max_families),
                              row_indices=args.rows)
    print(json.dumps({"entries": len(manifest["entries"]), "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
