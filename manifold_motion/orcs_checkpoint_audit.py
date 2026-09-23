"""Audit an official ORCS AdaptSonic checkpoint against this repository's controller contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


EXPECTED_GRAIL_SHA256 = "d6e0560696f0807ddb320a81151ad674fe29991f61583ea9ba1cd5722886c546"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(state: dict, key: str) -> list[int] | None:
    value = state.get(key)
    return list(value.shape) if torch.is_tensor(value) else None


def audit(checkpoint: Path, expected_sha256: str | None = None) -> dict:
    actual_sha256 = _sha256(checkpoint)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    archive = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = archive.get("actor_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain actor_state_dict")
    tensor_count = sum(torch.is_tensor(value) for value in state.values())
    parameter_count = sum(value.numel() for value in state.values() if torch.is_tensor(value))
    adapter_keys = sorted(key for key in state if ".adapters." in key)
    adapter_dim = (_shape(state, "adapter_normalizer._mean") or [None, None])[-1]
    action_dim = (_shape(state, "distribution.std_param") or [None])[-1]
    decoder_input = (_shape(state, "decoder.base.0.weight") or [None, None])[-1]
    encoder_input = (_shape(state, "encoder.0.weight") or [None, None])[-1]
    ranks = sorted({
        int(state[key].shape[0]) for key in adapter_keys
        if key.endswith("down.weight") and torch.is_tensor(state[key])
    })
    return {
        "schema": "manifold-motion.orcs-checkpoint-audit.v1",
        "checkpoint": str(checkpoint),
        "sha256": actual_sha256,
        "sha256_verified": expected_sha256 is None or actual_sha256 == expected_sha256,
        "iteration": archive.get("iter"),
        "actor_tensor_count": tensor_count,
        "actor_parameter_count": parameter_count,
        "action_dim": action_dim,
        "decoder_input_dim": decoder_input,
        "encoder_input_dim": encoder_input,
        "adapter_condition_dim": adapter_dim,
        "adapter_ranks": ranks,
        "adapter_layer_count": len([key for key in adapter_keys if key.endswith("down.weight")]),
        "current_onnx_contract": {
            "encoder_input_dim": 1762,
            "decoder_input_dim": 994,
            "token_dim": 64,
            "action_dim": 29,
        },
        "compatibility": {
            "decoder_shape_matches": decoder_input == 994 and action_dim == 29,
            "direct_onnx_drop_in": False,
            "reason": (
                "ORCS requires its pinned SONIC base, tokenizer, 202-D terrain augmentation, "
                "normalizer and rsl_rl actor. The current deployment exposes separate ONNX "
                "encoder/decoder sessions and has no ORCS adapter input."
            ),
            "native_baseline_supported": True,
            "narrow_or_overhead_obstacle_support": False,
            "sensor_limit": (
                "PerLoco-Grail augmentation uses a 17x11 downward pelvis height scan; "
                "it does not observe side walls or ceilings."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-sha256", default=EXPECTED_GRAIL_SHA256)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.checkpoint, args.expected_sha256 or None)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
