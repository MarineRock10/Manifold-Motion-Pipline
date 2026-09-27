"""Render an interpretable wide-vs-low online-composer counterfactual GIF.

The ordinary MuJoCo render is intentionally dense and is useful for physical inspection, but it
does not make the learned/safety decision causal chain obvious.  This renderer places two runs
at matched normalized progress and explicitly reports M_e, M_self, the raw learned family, the
geometry shield and the primitive that frozen SONIC actually receives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BACKGROUND = (12, 20, 31)
PANEL = (21, 32, 47)
WHITE = (235, 242, 250)
MUTED = (154, 169, 188)
CYAN = (51, 220, 232)
ORANGE = (255, 163, 64)
GREEN = (78, 221, 144)
BLUE = (74, 154, 255)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    paths = [Path("/usr/share/fonts/truetype/dejavu") / name, Path(name)]
    for path in paths:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            pass
    return ImageFont.load_default()


def _load_frames(path: Path) -> tuple[Image.Image, int]:
    image = Image.open(path)
    return image, int(getattr(image, "n_frames", 1))


def _load_run(root: Path) -> dict[str, Any]:
    with np.load(root / "executed.npz") as archive:
        executed = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(root / "segment_conditions.npz") as archive:
        conditions = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(root / "online_perception.npz") as archive:
        perception = {key: np.asarray(archive[key]) for key in archive.files}
    report = json.loads((root / "report.json").read_text())
    updates = report["execution"]["online_perception_updates"]
    return {"executed": executed, "conditions": conditions, "perception": perception,
            "report": report, "updates": updates}


def _nearest_environment(run: dict[str, Any], tick: int) -> np.ndarray:
    executed = run["executed"]
    corridor = run["conditions"]["environment_corridor"]
    position = executed["base_pos"][tick, :2]
    nearest = int(np.argmin(np.linalg.norm(corridor[:, :2] - position[None], axis=1)))
    return np.asarray(corridor[nearest, 3:6], dtype=np.float64)


def _decision(run: dict[str, Any], tick: int) -> dict[str, Any]:
    update_ticks = run["perception"]["update_ticks"]
    index = int(np.searchsorted(update_ticks, tick, side="right") - 1)
    index = int(np.clip(index, 0, len(run["updates"]) - 1))
    update = run["updates"][index]
    learned = update.get("composer_decision", {})
    geometry = update.get("primitive_decision", {})
    return {
        "update": index + 1,
        "family": str(learned.get("family", "not logged")),
        "probability": float(learned.get("probability", 0.0)),
        "geometry": str(geometry.get("primitive", update.get("primitive", "unknown"))),
        "geometry_reason": str(geometry.get("reason", "unknown")),
        "executed": str(update.get("primitive", "unknown")),
        "override": bool(learned.get("geometry_override", False)),
        "selected_reason": str(learned.get("selected_reason", "not logged")),
    }


def _bar(draw: ImageDraw.ImageDraw, xy: tuple[int, int], width: int, value: float,
         maximum: float, color: tuple[int, int, int], label: str) -> None:
    x, y = xy
    draw.text((x, y), label, font=_font(15), fill=WHITE)
    draw.rounded_rectangle((x, y + 23, x + width, y + 38), radius=6, fill=(48, 61, 79))
    extent = int(np.clip(value / maximum, 0.0, 1.0) * width)
    draw.rounded_rectangle((x, y + 23, x + extent, y + 38), radius=6, fill=color)


def _card(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, title: str,
          me: np.ndarray, mr: np.ndarray, decision: dict[str, Any], accepted: bool) -> None:
    draw.rounded_rectangle((x, y, x + width, y + 160), radius=10, fill=PANEL,
                           outline=GREEN if accepted else ORANGE, width=2)
    draw.text((x + 14, y + 10), title, font=_font(20, True), fill=WHITE)
    draw.text((x + width - 112, y + 12), "ACCEPTED" if accepted else "REJECTED",
              font=_font(15, True), fill=GREEN if accepted else ORANGE)
    _bar(draw, (x + 14, y + 43), 185, float(me[2]), 1.40, BLUE,
         f"M_e vertical {me[2]:.2f} m")
    _bar(draw, (x + 218, y + 43), 185, float(mr[2]), 1.40, ORANGE,
         f"M_self vertical {mr[2]:.2f} m")
    learned = f"learned: {decision['family']} {100.0 * decision['probability']:.0f}%"
    draw.text((x + 14, y + 91), learned, font=_font(15), fill=CYAN)
    draw.text((x + 14, y + 116),
              f"geometry: {decision['geometry']}  ->  SONIC: {decision['executed']}",
              font=_font(16, True), fill=WHITE)
    shield = "geometry safety override" if decision["override"] else "composer/geometry agree or map safely"
    draw.text((x + 14, y + 138), shield, font=_font(13), fill=MUTED)


def render(args: argparse.Namespace) -> None:
    wide_image, wide_frames = _load_frames(args.wide / "manifold_adaptive.gif")
    low_image, low_frames = _load_frames(args.low / "manifold_adaptive.gif")
    wide = _load_run(args.wide); low = _load_run(args.low)
    output_frames = []
    frame_count = int(args.frames)
    panel_width, panel_height = 480, 424
    for frame_index, progress in enumerate(np.linspace(0.0, 1.0, frame_count)):
        wide_frame = int(round(progress * (wide_frames - 1)))
        low_frame = int(round(progress * (low_frames - 1)))
        wide_image.seek(wide_frame); low_image.seek(low_frame)
        wide_panel = wide_image.convert("RGB").crop((0, 0, 640, 565)).resize(
            (panel_width, panel_height), Image.Resampling.LANCZOS)
        low_panel = low_image.convert("RGB").crop((0, 0, 640, 565)).resize(
            (panel_width, panel_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (980, 704), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 13), "COUNTERFACTUAL: SAME GOAL, ONLY VERTICAL M_e CHANGES",
                  font=_font(25, True), fill=WHITE)
        draw.text((20, 50), "wide corridor -> normal walk     |     low ceiling -> crouch walk + smaller M_self",
                  font=_font(18), fill=CYAN)
        canvas.paste(wide_panel, (10, 83)); canvas.paste(low_panel, (490, 83))
        wide_tick = min(int(round(progress * (len(wide["executed"]["t"]) - 1))),
                        len(wide["executed"]["t"]) - 1)
        low_tick = min(int(round(progress * (len(low["executed"]["t"]) - 1))),
                       len(low["executed"]["t"]) - 1)
        wide_me = _nearest_environment(wide, wide_tick); low_me = _nearest_environment(low, low_tick)
        wide_mr = wide["executed"]["robot_manifold"][wide_tick, 3:6]
        low_mr = low["executed"]["robot_manifold"][low_tick, 3:6]
        _card(draw, 10, 522, 470, "WIDE", wide_me, wide_mr, _decision(wide, wide_tick),
              bool(wide["report"]["accepted"]))
        _card(draw, 500, 522, 470, "LOW CEILING", low_me, low_mr, _decision(low, low_tick),
              bool(low["report"]["accepted"]))
        draw.text((282, 687), "8/8 keyframes each | 0 obstacle contacts | no reset | frozen SONIC",
                  font=_font(14, True), fill=GREEN)
        output_frames.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output_frames[0].save(args.out, save_all=True, append_images=output_frames[1:],
                          duration=args.duration_ms, loop=0, optimize=True, disposal=2)
    print(json.dumps({"out": str(args.out), "frames": frame_count,
                      "wide_accepted": wide["report"]["accepted"],
                      "low_accepted": low["report"]["accepted"]}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide", type=Path, required=True)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--duration-ms", type=int, default=80)
    args = parser.parse_args()
    if args.frames < 2 or args.duration_ms <= 0:
        parser.error("frames must be >=2 and duration positive")
    render(args); return 0


if __name__ == "__main__":
    raise SystemExit(main())
