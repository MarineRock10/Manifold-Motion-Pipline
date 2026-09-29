"""Bridge a Stage-1 temporal primitive checkpoint into Stage-2 conditions.

The bridge replaces the categorical primitive one-hot in the Stage-2 window condition with the
probability vector predicted from ``M_e(t)``.  It is intentionally a pure adapter: it does not
change the frozen Stage-1 checkpoint and it never reads the recorded primitive while routing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from manifold_motion.stage1.temporal_primitive import TemporalPrimitiveNet


def predicted_primitive_probabilities(windows: Path, checkpoint: Path, device: str = "cpu") -> np.ndarray:
    import torch

    with np.load(windows, allow_pickle=False) as archive:
        corridor = np.asarray(archive["corridor"], dtype=np.float32)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model = TemporalPrimitiveNet.build(torch, 7, int(ck["hidden"]), int(ck["classes"]))
    model.load_state_dict(ck["model"]); model.to(device); model.eval()
    x = (corridor - np.asarray(ck["geom_mean"])[None, None]) / np.asarray(ck["geom_std"])[None, None]
    with torch.no_grad():
        _, logits = model(torch.as_tensor(x, dtype=torch.float32, device=device))
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1.0, atol=1e-4):
        raise ValueError("Stage-1 router produced invalid probabilities")
    return probabilities


def replace_primitive_condition(data, probabilities: np.ndarray):
    """Return a copy of WindowData whose primitive slice is a predicted distribution."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(data.condition):
        raise ValueError(f"probabilities must be [N,K] with N={len(data.condition)}, got {probabilities.shape}")
    if probabilities.shape[1] != data.primitive_count:
        raise ValueError(f"Stage-1 classes {probabilities.shape[1]} do not match Stage-2 primitive_count {data.primitive_count}")
    output = data.condition.copy()
    start = data.raw["state"].shape[1] + int(np.prod(data.raw["history"].shape[1:]))
    output[:, start:start + data.primitive_count] = probabilities
    data.condition = output
    return data
