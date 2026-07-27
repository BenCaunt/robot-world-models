from __future__ import annotations

import torch
from torch import Tensor, nn


class StateDynamicsMLP(nn.Module):
    """Small action-conditioned delta-state predictor used as a pipeline proof."""

    def __init__(
        self,
        *,
        state_dimension: int,
        action_dimension: int,
        hidden_dimension: int = 128,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dimension = state_dimension + action_dimension
        for layer_index in range(hidden_layers):
            layer_input_dimension = (
                input_dimension if layer_index == 0 else hidden_dimension
            )
            layers.extend(
                [
                    nn.Linear(layer_input_dimension, hidden_dimension),
                    nn.SiLU(),
                ]
            )
        layers.append(nn.Linear(hidden_dimension, state_dimension))
        self.network = nn.Sequential(*layers)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        delta = self.network(torch.cat([state, action], dim=-1))
        return state + delta
