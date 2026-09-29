"""Multi-family, actor-disjoint Stage-2 router/Flow benchmark.

The benchmark chooses one high-confidence example for each distinct family predicted by the
Stage-1 router.  It does not use the recorded primitive to choose a route or a candidate; the
true family is retained only as an audit label.  Each Flow draw is evaluated twice: directly,
and after the bounded optimization-embedded projection, with the same SONIC/MuJoCo gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from manifold_motion.dataio.seed_replay import ReplayConfig, SeedReplayRunner
from manifold_motion.stage2.flow import (
    Normalizer, WindowData, _device, _load_ae, _load_flow, _primitive_name,
    _target_from_model,
)
from manifold_motion.stage2.projection import project_reference
from manifold_motion.stage2.select import _objective
from manifold_motion.stage2.stage1_router_bridge import (
    predicted_primitive_probabilities, replace_primitive_condition,
)
from manifold_motion.stage2.validate import validate_trajectory


def _sample(flow, ae, data: WindowData, index: int, count: int, steps: int,
            seed: int, device: torch.device) -> np.ndarray:
    condition = torch.as_tensor(data.condition[index:index + 1], device=device).expand(count, -1)
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        latent = torch.randn((count, flow.latent_dim), device=device, generator=generator)
        dt = 1.0 / float(steps)
        for step in range(steps):
            time = torch.full((count,), step * dt, device=device)
            latent = latent + dt * flow(latent, time, condition)
        normalized = ae.decode(latent, condition).cpu().numpy().reshape(
            (count,) + data.target_shape
        )
    return _target_from_model(
        data.normalizer.inverse_target(normalized), data.joint_lower, data.joint_upper
    ).astype(np.float32)


def _select_cases(data: WindowData, probabilities: np.ndarray, max_cases: int,
                  min_family_count: int) -> list[dict[str, Any]]:
    test = np.flatnonzero(data.split == 2)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    cases: list[dict[str, Any]] = []
    for family_id in sorted(np.unique(predicted[test]).tolist(),
                            key=lambda value: float(confidence[test[predicted[test] == value]].max()),
                            reverse=True):
        ids = test[predicted[test] == family_id]
        if len(ids) < min_family_count:
            continue
        index = int(ids[np.argmax(confidence[ids])])
        cases.append({
            "index": index,
            "predicted_family_id": int(family_id),
            "predicted_family": _primitive_name(data.raw, family_id),
            "source_family_id": int(data.raw["primitive"][index]),
            "source_family": _primitive_name(data.raw, int(data.raw["primitive"][index])),
            "router_confidence": float(confidence[index]),
            "router_entropy": float(-np.sum(probabilities[index] * np.log(np.maximum(probabilities[index], 1e-8)))),
            "router_match": bool(family_id == int(data.raw["primitive"][index])),
        })
        if len(cases) >= max_cases:
            break
    if not cases:
        raise ValueError("the test split has no predicted family with the requested support")
    return cases


def _evaluate(candidates: np.ndarray, corridor: np.ndarray, family: str, source: Path,
              runner: SeedReplayRunner, config: ReplayConfig, project: bool) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    reports: list[dict[str, Any]] = []
    executions: list[dict[str, np.ndarray]] = []
    for candidate_index, candidate in enumerate(candidates):
        evaluated = candidate
        projection_report = None
        if project:
            evaluated, projection_report = project_reference(candidate, corridor)
        executed, summary = validate_trajectory(
            evaluated, source=source, source_hz=30.0, config=config, corridor=corridor,
            stratum=family, runner=runner,
        )
        report: dict[str, Any] = {
            "candidate_index": candidate_index,
            **_objective(summary, evaluated),
            **summary,
        }
        if projection_report is not None:
            report["projection"] = projection_report
        reports.append(report)
        executions.append(executed)
    return reports, executions


def run(args: argparse.Namespace) -> int:
    torch.set_num_threads(args.threads)
    device = _device(args.device)
    ae, ae_checkpoint = _load_ae(args.autoencoder, device)
    flow, flow_checkpoint = _load_flow(args.flow, device)
    normalizer = Normalizer.from_state_dict(ae_checkpoint["normalizer"])
    data = WindowData.load(args.windows, normalizer=normalizer)
    probabilities = predicted_primitive_probabilities(args.windows, args.router_checkpoint, str(device))
    data = replace_primitive_condition(data, probabilities)
    if flow.condition_dim != data.condition.shape[1] or ae.condition_dim != data.condition.shape[1]:
        raise ValueError("router Flow and active window condition dimensions differ")
    if tuple(data.target_shape) != tuple(ae_checkpoint["target_shape"]):
        raise ValueError("router Flow target shape differs from the active archive")
    cases = _select_cases(data, probabilities, args.max_cases, args.min_family_count)
    runner = SeedReplayRunner()
    config = ReplayConfig()
    args.out.mkdir(parents=True, exist_ok=True)
    case_reports: list[dict[str, Any]] = []
    for case_number, case in enumerate(cases):
        index = int(case["index"])
        candidates = _sample(flow, ae, data, index, args.num_candidates, args.steps,
                             args.seed + case_number, device)
        corridor = np.asarray(data.raw["corridor"][index], dtype=np.float32)
        family = str(case["predicted_family"])
        raw_reports, raw_executions = _evaluate(candidates, corridor, family, args.windows, runner, config, False)
        projected_reports, projected_executions = _evaluate(candidates, corridor, family, args.windows, runner, config, True)
        case_dir = args.out / f"case_{case_number:02d}_{family}"
        case_dir.mkdir(parents=True, exist_ok=True)
        projected_viable = [row for row in projected_reports if row["accepted"]]
        selected_pool = projected_viable or projected_reports
        selected = min(selected_pool, key=lambda row: float(row["score"]))
        selected_index = int(selected["candidate_index"])
        selected_projected, _ = project_reference(candidates[selected_index], corridor)
        np.savez_compressed(
            case_dir / "sample.npz",
            generated_ref=selected_projected,
            generated_ref_candidates=candidates,
            condition_corridor=corridor,
            condition_sdf=np.asarray(data.raw["sdf"][index]),
            source_index=np.asarray(index, dtype=np.int64),
            primitive_name=np.asarray(family),
            router_probabilities=probabilities[index],
        )
        np.savez_compressed(case_dir / "selected_executed.npz", **projected_executions[selected_index])
        case_reports.append({
            **case,
            "case_number": case_number,
            "candidates": int(args.num_candidates),
            "raw_accepted": int(sum(row["accepted"] for row in raw_reports)),
            "projected_accepted": int(sum(row["accepted"] for row in projected_reports)),
            "raw": raw_reports,
            "projected": projected_reports,
            "selected_candidate_index": selected_index,
            "selected_accepted": bool(selected["accepted"]),
            "artifacts": str(case_dir),
        })
    total = len(cases) * args.num_candidates
    summary = {
        "schema": "manifold-motion.stage2.router-flow-multifamily.v1",
        "windows": str(args.windows),
        "router_checkpoint": str(args.router_checkpoint),
        "autoencoder": str(args.autoencoder),
        "flow": str(args.flow),
        "cases": case_reports,
        "aggregate": {
            "cases": len(cases), "candidates_per_case": args.num_candidates,
            "raw_candidates": total,
            "raw_accepted": int(sum(case["raw_accepted"] for case in case_reports)),
            "projected_candidates": total,
            "projected_accepted": int(sum(case["projected_accepted"] for case in case_reports)),
            "projected_case_success": int(sum(case["projected_accepted"] > 0 for case in case_reports)),
        },
        "selection_contract": "families chosen from Stage-1 predicted probabilities; true labels are audit-only",
        "oracle_primitive_not_used": True,
    }
    (args.out / "benchmark.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a multi-family Stage-2 router/Flow physical benchmark")
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path, required=True)
    parser.add_argument("--autoencoder", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--min-family-count", type=int, default=5)
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.max_cases, args.min_family_count, args.num_candidates, args.steps, args.threads) < 1:
        parser.error("benchmark sizes and threads must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
