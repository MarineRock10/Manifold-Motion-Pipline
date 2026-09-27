"""Rolling-window SEED skill composition for online SLAM/MuJoCo execution.

The offline environment composer is trained on reverse-synthesised SEED windows. This module
adapts the same input contract to live execution: each accepted radar/SLAM update is converted
to a 36-frame local ``M_e`` window, the measured robot envelope is used as ``M_self``, and the
state/history are classified again. Hysteresis and a geometry safety shield prevent a noisy
frame from changing the gait. The frozen SONIC controller remains below this layer.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from manifold_motion.core import constants as C
from manifold_motion.planning.corridor import ExecutedEnvelopeEstimator
from manifold_motion.stage2.composer import ComposerNormalizer, EnvironmentSkillComposer, _forward
from manifold_motion.stage2.flow_route_candidates import RouteFlowSampler
from manifold_motion.stage2.flow import _device, _torch_load


# The frozen Flow/SONIC route executor exposes eight verified legacy tokens. Several richer
# SEED families share safe locomotion support until their own targeted references are added.
FAMILY_TO_LEGACY: dict[str, int] = {
    "walk_forward": 5, "jog_forward": 5, "hands_back_walk": 5, "walk_lateral": 4,
    "walk_curve": 5, "turn_in_place": 6, "crouch_walk": 2, "crouch_transition": 3,
    "dodge_lateral": 4, "forward_lunge": 5, "side_hop": 4, "high_jump": 5,
    "box_jump": 5, "step_up_box": 5, "step_down_box": 5, "kneel": 2, "all_fours": 2,
    "door_interaction": 5, "ladder": 2, "button_lever": 5, "carry_object": 5,
}


@dataclass(frozen=True)
class ComposerDecision:
    family_id: int
    family: str
    probability: float
    top3: list[dict[str, Any]]
    legacy_primitive_id: int | None
    selected_legacy_id: int
    switched: bool
    switch_reason: str
    geometry_override: bool
    latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_id": int(self.family_id), "family": self.family,
            "probability": float(self.probability), "top3": self.top3,
            "legacy_primitive_id": (int(self.legacy_primitive_id)
                                     if self.legacy_primitive_id is not None else None),
            "selected_legacy_id": int(self.selected_legacy_id), "switched": bool(self.switched),
            "switch_reason": self.switch_reason, "geometry_override": bool(self.geometry_override),
            "latency_ms": float(self.latency_ms),
        }


def _resample_corridor(corridor: np.ndarray, frames: int = 36) -> np.ndarray:
    value = np.asarray(corridor, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 7 or len(value) < 2:
        raise ValueError(f"corridor must be [T,7] with T>=2, got {value.shape}")
    if len(value) == frames:
        return value.copy()
    source = np.linspace(0.0, 1.0, len(value)); target = np.linspace(0.0, 1.0, frames)
    result = np.stack([np.interp(target, source, value[:, axis])
                       for axis in range(value.shape[1])], axis=1)
    result[:, 6] = np.interp(target, source, np.unwrap(value[:, 6]))
    return result.astype(np.float32)


def _family_legacy(name: str) -> int | None:
    return FAMILY_TO_LEGACY.get(str(name).lower())


class OnlineSkillComposer:
    """Classify each live rolling M_e/M_self window with debounced safe routing."""

    def __init__(self, checkpoint: Path, *, device: str = "cpu", switch_margin: float = 0.08,
                 min_dwell_updates: int = 2, confidence_floor: float = 0.35,
                 allow_family_contraction: bool = True):
        if switch_margin < 0 or min_dwell_updates < 1 or not 0 <= confidence_floor <= 1:
            raise ValueError("online composer hysteresis parameters are invalid")
        self.device = _device(device); self.checkpoint_path = Path(checkpoint)
        checkpoint_value = _torch_load(self.checkpoint_path, self.device)
        self.normalizer = ComposerNormalizer.from_state_dict(checkpoint_value["normalizer"])
        self.names = [str(value) for value in checkpoint_value["primitive_names"]]
        self.supported = np.asarray(checkpoint_value["supported_mask"], dtype=bool)
        self.model = EnvironmentSkillComposer(**checkpoint_value["architecture"]).to(self.device)
        self.model.load_state_dict(checkpoint_value["model_state"]); self.model.eval()
        self.switch_margin = float(switch_margin); self.min_dwell_updates = int(min_dwell_updates)
        self.confidence_floor = float(confidence_floor)
        self.allow_family_contraction = bool(allow_family_contraction)
        self.stable_family_id: int | None = None; self.pending_family_id: int | None = None
        self.pending_updates = 0; self.update_count = 0; self.switch_count = 0
        self.override_count = 0; self.latencies_ms: list[float] = []; self.decisions: list[dict[str, Any]] = []
        self.envelope = ExecutedEnvelopeEstimator()

    def reset(self) -> None:
        self.stable_family_id = None; self.pending_family_id = None; self.pending_updates = 0
        self.update_count = 0; self.switch_count = 0; self.override_count = 0
        self.latencies_ms.clear(); self.decisions.clear()

    def _predict(self, state: np.ndarray, history: np.ndarray, corridor: np.ndarray,
                 sdf: np.ndarray, self_manifold: np.ndarray, manifold: np.ndarray,
                 command: np.ndarray) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        raw = {
            "state": np.asarray(state, dtype=np.float32).reshape(1, -1),
            "history": np.asarray(history, dtype=np.float32).reshape(1, 12, -1),
            "corridor": _resample_corridor(corridor)[None],
            "sdf": np.asarray(sdf, dtype=np.float32).reshape(1, 10, 10, 8),
            "self_manifold": np.asarray(self_manifold, dtype=np.float32).reshape(1, 36, 3),
            "manifold": np.asarray(manifold, dtype=np.float32).reshape(1, 6),
            "command": np.asarray(command, dtype=np.float32).reshape(1, 9),
        }
        normalized = self.normalizer.normalize(raw)
        data = type("_LiveData", (), {"inputs": normalized, "supported_mask": self.supported})()
        with torch.no_grad():
            logits = _forward(self.model, data, np.asarray([0]), self.device)
            supported = torch.as_tensor(self.supported, device=self.device, dtype=torch.bool)
            logits[:, ~supported] = -1e9
            probability = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        return probability, (time.perf_counter() - started) * 1000.0

    def update(self, *, state: np.ndarray, history: np.ndarray, corridor: np.ndarray,
               sdf: np.ndarray, self_manifold: np.ndarray, manifold: np.ndarray | None = None,
               command: np.ndarray | None = None, safety_primitive_id: int = 5,
               tick: int | None = None) -> dict[str, Any]:
        """Consume one fresh SLAM frame; geometry contraction always has priority."""
        safety_primitive_id = int(safety_primitive_id)
        if safety_primitive_id not in (2, 4, 5): safety_primitive_id = 5
        corridor36 = _resample_corridor(corridor)
        probability, latency = self._predict(
            state, history, corridor36, sdf, self_manifold,
            np.full(6, [2.0, 2.0, 1.5, 0.0, 0.0, 0.0], dtype=np.float32)
            if manifold is None else manifold,
            RouteFlowSampler._yaw_command(corridor36[-1, :3], float(corridor36[-1, 6]))
            if command is None else command,
        )
        family_id = int(np.argmax(probability)); family = self.names[family_id]
        confidence = float(probability[family_id]); legacy = _family_legacy(family)
        top_ids = np.argsort(-probability)[:3]
        top3 = [{"family": self.names[int(i)], "family_id": int(i),
                 "probability": float(probability[int(i)])} for i in top_ids]
        if self.stable_family_id is None:
            self.stable_family_id = family_id; self.pending_family_id = None; self.pending_updates = 0
            switch_reason = "initial_online_composer_decision"; switched = True
        elif family_id == self.stable_family_id:
            self.pending_family_id = None; self.pending_updates = 0
            switch_reason = "stable_family"; switched = False
        else:
            stable_probability = float(probability[self.stable_family_id])
            if confidence < self.confidence_floor or confidence < stable_probability + self.switch_margin:
                self.pending_family_id = None; self.pending_updates = 0
                switch_reason = "hysteresis_margin_or_confidence"; switched = False
            else:
                self.pending_family_id = family_id; self.pending_updates += 1
                if self.pending_updates >= self.min_dwell_updates:
                    self.stable_family_id = family_id; self.pending_family_id = None; self.pending_updates = 0
                    self.switch_count += 1; switch_reason = "debounced_family_switch"; switched = True
                else:
                    switch_reason = "pending_family_dwell"; switched = False
        stable_name = self.names[int(self.stable_family_id)]; stable_legacy = _family_legacy(stable_name)
        geometry_override = False; selected = stable_legacy if stable_legacy in (2, 4, 5) else None
        if safety_primitive_id in (2, 4):
            if selected != safety_primitive_id: geometry_override = True; self.override_count += 1
            selected = safety_primitive_id; selected_reason = "geometry_safety_override"
        elif selected is None or not self.allow_family_contraction:
            selected = 5; selected_reason = "legacy_nominal_safe_fallback"
        elif selected in (2, 4) and confidence < self.confidence_floor:
            selected = 5; selected_reason = "low_confidence_nominal_fallback"
        else:
            selected_reason = switch_reason
        self.update_count += 1; self.latencies_ms.append(float(latency))
        result = ComposerDecision(
            family_id=family_id, family=family, probability=confidence, top3=top3,
            legacy_primitive_id=legacy, selected_legacy_id=int(selected), switched=bool(switched),
            switch_reason=f"{selected_reason}:{switch_reason}", geometry_override=geometry_override,
            latency_ms=float(latency),
        ).as_dict()
        result.update({"tick": (int(tick) if tick is not None else None), "stable_family": stable_name,
                       "stable_family_id": int(self.stable_family_id),
                       "pending_family": (self.names[self.pending_family_id]
                                          if self.pending_family_id is not None else None),
                       "pending_updates": int(self.pending_updates), "safety_primitive_id": safety_primitive_id,
                       "selected_reason": selected_reason})
        self.decisions.append(result); return result

    def update_runtime(self, *, state: dict[str, np.ndarray], state_feature: np.ndarray,
                       history: np.ndarray, corridor: np.ndarray, sdf: np.ndarray,
                       safety_primitive_id: int, tick: int) -> dict[str, Any]:
        """Build M_self from the current MuJoCo state and classify a live perception update."""
        q_policy = np.asarray(state["q_hw"], dtype=np.float64)[C.MUJOCO_TO_ISAACLAB]
        self_semi = self.envelope.sequence(
            q_policy[None], np.asarray(state["base_pos"], dtype=np.float64)[None],
            np.asarray(state["base_quat"], dtype=np.float64)[None],
        )[0]
        self_window = np.repeat(self_semi[None], 36, axis=0).astype(np.float32)
        return self.update(state=state_feature, history=history, corridor=corridor, sdf=sdf,
                           self_manifold=self_window, safety_primitive_id=safety_primitive_id, tick=tick)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True, "checkpoint": str(self.checkpoint_path),
            "input_contract": "69-D state + 12-frame history + live M_e corridor/SDF + measured M_self",
            "updates": int(self.update_count), "switches": int(self.switch_count),
            "geometry_overrides": int(self.override_count),
            "mean_latency_ms": (float(np.mean(self.latencies_ms)) if self.latencies_ms else None),
            "p95_latency_ms": (float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else None),
            "selected_legacy_ids": [int(row["selected_legacy_id"]) for row in self.decisions],
            "families": [str(row["family"]) for row in self.decisions],
        }


def evaluate(args: argparse.Namespace) -> int:
    """Replay held-out windows as successive SLAM frames and measure rolling decisions."""
    with np.load(args.windows, allow_pickle=False) as archive:
        raw = {key: np.asarray(archive[key]) for key in archive.files}
    composer = OnlineSkillComposer(args.checkpoint, device=args.device,
                                    switch_margin=args.switch_margin,
                                    min_dwell_updates=args.min_dwell_updates)
    indices = np.flatnonzero(raw["split"] == args.split)
    groups: dict[int, list[int]] = {}
    for index in indices: groups.setdefault(int(raw["clip_index"][index]), []).append(int(index))
    rows = []
    for clip, clip_indices in sorted(groups.items()):
        composer.reset()
        for index in clip_indices:
            result = composer.update(state=raw["state"][index], history=raw["history"][index],
                                     corridor=raw["corridor"][index], sdf=raw["sdf"][index],
                                     self_manifold=raw["self_manifold"][index],
                                     safety_primitive_id=5, tick=len(rows))
            true_name = str(raw["primitive_names"][int(raw["primitive"][index])])
            rows.append({"clip_index": clip, "source_index": int(index), "true_family": true_name, **result})
    report = {
        "schema": "manifold-motion.online-composer-evaluation.v1", "windows": str(args.windows),
        "checkpoint": str(args.checkpoint), "split": int(args.split), "clips": len(groups),
        "updates": len(rows), "raw_top1_accuracy": float(np.mean([r["family"] == r["true_family"] for r in rows])) if rows else 0.0,
        "raw_top3_accuracy": float(np.mean([any(x["family"] == r["true_family"] for x in r["top3"]) for r in rows])) if rows else 0.0,
        "switch_rate": float(np.mean([r["switched"] for r in rows])) if rows else 0.0,
        "mean_latency_ms": float(np.mean([r["latency_ms"] for r in rows])) if rows else None, "families": {},
    }
    for name in sorted(set(r["true_family"] for r in rows)):
        subset = [r for r in rows if r["true_family"] == name]
        report["families"][name] = {
            "windows": len(subset), "raw_top1_accuracy": float(np.mean([r["family"] == name for r in subset])),
            "top3_accuracy": float(np.mean([any(x["family"] == name for x in r["top3"]) for r in subset])),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out / "decisions.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2)); return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/online_composer_eval_v1"))
    parser.add_argument("--split", type=int, choices=(0, 1, 2), default=2); parser.add_argument("--switch-margin", type=float, default=0.08)
    parser.add_argument("--min-dwell-updates", type=int, default=2); parser.add_argument("--device", default="cpu")
    return evaluate(parser.parse_args())


if __name__ == "__main__": raise SystemExit(main())


__all__ = ["FAMILY_TO_LEGACY", "OnlineSkillComposer", "ComposerDecision"]
