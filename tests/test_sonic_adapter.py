from __future__ import annotations

import torch

from manifold_motion.sonic_adapter import SonicAdapterConfig, SonicConditionAdapter


def test_zero_init_is_base_exact() -> None:
    model = SonicConditionAdapter(SonicAdapterConfig(condition_dim=32, rank=4))
    base = torch.randn(5, 29)
    condition = torch.randn(5, 32)
    assert model.zero_init_parity(base, condition) == 0.0
    loss = torch.mean((model(base, condition) - (base + 0.01)) ** 2)
    loss.backward()
    assert model.trainable_parameter_count() > 0


if __name__ == "__main__":
    test_zero_init_is_base_exact()
    print("SONIC adapter parity test passed")
