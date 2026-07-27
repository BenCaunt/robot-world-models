from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class VisualLatentDynamics(nn.Module):
    """Predict pooled DINOv2 patch tokens and robot state from context and action."""

    def __init__(
        self,
        *,
        state_dimension: int,
        action_dimension: int,
        latent_dimension: int,
        context_frames: int,
        patch_grid: int,
        output_size: int,
        hidden_dimension: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        self.context_frames = context_frames
        self.patch_grid = patch_grid
        condition_dimension = state_dimension + action_dimension
        self.condition = nn.Sequential(
            nn.Linear(condition_dimension, latent_dimension),
            nn.SiLU(),
        )
        predictor_layers: list[nn.Module] = [
            nn.Linear((context_frames + 1) * latent_dimension, hidden_dimension),
            nn.SiLU(),
        ]
        for _ in range(hidden_layers - 1):
            predictor_layers.extend([nn.Linear(hidden_dimension, hidden_dimension), nn.SiLU()])
        predictor_layers.append(nn.Linear(hidden_dimension, latent_dimension))
        self.predictor = nn.Sequential(*predictor_layers)
        self.state_head = nn.Sequential(
            nn.Linear(condition_dimension + latent_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, state_dimension),
        )
        if output_size < patch_grid or output_size % patch_grid:
            raise ValueError("output_size must be a power-of-two multiple of patch_grid")
        scale = output_size // patch_grid
        if scale & (scale - 1):
            raise ValueError("output_size must be a power-of-two multiple of patch_grid")
        decoder_layers: list[nn.Module] = []
        input_channels = latent_dimension
        for stage in range(scale.bit_length() - 1):
            output_channels = max(16, 128 // (2**stage))
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1),
                    nn.SiLU(),
                ]
            )
            input_channels = output_channels
        decoder_layers.extend(
            [
                nn.Conv2d(input_channels, 3, 3, padding=1),
                nn.Sigmoid(),
            ]
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(
        self,
        context: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 4 or context.shape[1] != self.context_frames:
            raise ValueError("context must have shape [batch, context_frames, tokens, latent]")
        condition_input = torch.cat((state, action), dim=-1)
        condition = self.condition(condition_input).unsqueeze(1).expand(-1, context.shape[2], -1)
        token_history = context.permute(0, 2, 1, 3).flatten(start_dim=2)
        delta = self.predictor(torch.cat((token_history, condition), dim=-1))
        predicted_features = functional.normalize(context[:, -1] + delta, dim=-1)
        global_features = context[:, -1].mean(dim=1)
        predicted_state = state + self.state_head(
            torch.cat((condition_input, global_features), dim=-1)
        )
        return predicted_features, predicted_state

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        expected_tokens = self.patch_grid**2
        if features.ndim != 3 or features.shape[1] != expected_tokens:
            raise ValueError(f"features must contain {expected_tokens} spatial tokens")
        spatial = features.transpose(1, 2).reshape(
            features.shape[0],
            features.shape[2],
            self.patch_grid,
            self.patch_grid,
        )
        return self.decoder(spatial)
