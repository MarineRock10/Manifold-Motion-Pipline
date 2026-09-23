"""ORCS-style frozen-SONIC condition adapter for a later fine-tuning stage.

The deployed controller in this repository is ONNX and cannot accept an rsl_rl
``SonicWithAdapterModel`` checkpoint directly.  This module defines the compatible experiment
seam: a frozen base action plus a zero-initialized low-rank residual driven by the augmentation
stream ``[M_e, SDF, command, state, history]``.  At initialization the output is exactly the
base action; training can therefore be stopped or rolled back without destroying SONIC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


def stage2_augmentation(state: torch.Tensor, history: torch.Tensor,
                        manifold: torch.Tensor, corridor: torch.Tensor,
                        sdf: torch.Tensor, command: torch.Tensor,
                        primitive_one_hot: torch.Tensor) -> torch.Tensor:
    """Flatten the deployable ORCS-style augmentation stream in a fixed order."""
    batch = state.shape[0]
    return torch.cat([
        state.reshape(batch, -1), history.reshape(batch, -1),
        manifold.reshape(batch, -1), corridor.reshape(batch, -1),
        sdf.reshape(batch, -1), command.reshape(batch, -1),
        primitive_one_hot.reshape(batch, -1),
    ], dim=1)


@dataclass(frozen=True)
class SonicAdapterConfig:
    condition_dim: int
    action_dim: int = 29
    hidden_dim: int = 256
    rank: int = 16
    alpha: float = 1.0
    residual_bound: float = 0.12

    def validate(self) -> None:
        if min(self.condition_dim, self.action_dim, self.hidden_dim, self.rank) <= 0:
            raise ValueError("adapter dimensions must be positive")
        if self.alpha <= 0 or self.residual_bound <= 0:
            raise ValueError("adapter scale/bound must be positive")


class ZeroInitLoRA(nn.Module):
    """Low-rank residual whose B factor is zero at construction."""

    def __init__(self, input_dim: int, output_dim: int, rank: int, alpha: float):
        super().__init__()
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.down = nn.Parameter(torch.empty(input_dim, rank))
        self.up = nn.Parameter(torch.zeros(rank, output_dim))
        nn.init.normal_(self.down, std=0.02)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value @ self.down @ self.up) * self.scale


class SonicConditionAdapter(nn.Module):
    """Frozen-base-compatible residual adapter; only this module is trainable."""

    def __init__(self, config: SonicAdapterConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.condition = nn.Sequential(
            nn.Linear(config.condition_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim),
            nn.SiLU(), nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU(),
        )
        self.residual = ZeroInitLoRA(config.hidden_dim + config.action_dim,
                                     config.action_dim, config.rank, config.alpha)

    def forward(self, base_action: torch.Tensor, augmentation: torch.Tensor) -> torch.Tensor:
        if base_action.shape[-1] != self.config.action_dim:
            raise ValueError("base action dimension does not match adapter config")
        if augmentation.shape[-1] != self.config.condition_dim:
            raise ValueError("augmentation dimension does not match adapter config")
        encoded = self.condition(augmentation)
        delta = self.residual(torch.cat([encoded, base_action], dim=-1))
        return base_action + self.config.residual_bound * torch.tanh(delta)

    def zero_init_parity(self, base_action: torch.Tensor, augmentation: torch.Tensor) -> float:
        with torch.no_grad():
            return float(torch.max(torch.abs(self(base_action, augmentation) - base_action)).item())

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def export_state(self) -> dict:
        return {"config": asdict(self.config), "state_dict": self.state_dict()}
