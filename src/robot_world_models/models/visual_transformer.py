from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from robot_world_models.models.visual_latent import (
    build_spatial_decoder,
    decode_spatial_features,
)


class VisualSpatiotemporalTransformer(nn.Module):
    """Predict spatial visual tokens with attention across time, space, state, and action."""

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
        attention_heads: int,
    ) -> None:
        super().__init__()
        if hidden_dimension % attention_heads:
            raise ValueError("hidden_dimension must be divisible by attention_heads")
        if hidden_layers < 2:
            raise ValueError("transformer requires at least two hidden layers")
        self.context_frames = context_frames
        self.patch_grid = patch_grid
        token_count = patch_grid**2
        encoder_layers = hidden_layers // 2
        decoder_layers = hidden_layers - encoder_layers

        self.visual_projection = nn.Linear(latent_dimension, hidden_dimension)
        self.state_projection = nn.Linear(state_dimension, hidden_dimension)
        self.action_projection = nn.Linear(action_dimension, hidden_dimension)
        self.temporal_position = nn.Parameter(
            torch.empty(1, context_frames, 1, hidden_dimension)
        )
        self.spatial_position = nn.Parameter(
            torch.empty(1, 1, token_count, hidden_dimension)
        )
        self.condition_position = nn.Parameter(torch.empty(1, 2, hidden_dimension))
        self.output_queries = nn.Parameter(torch.empty(1, token_count, hidden_dimension))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension,
            nhead=attention_heads,
            dim_feedforward=hidden_dimension * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(hidden_dimension),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dimension,
            nhead=attention_heads,
            dim_feedforward=hidden_dimension * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(hidden_dimension),
        )
        self.feature_head = nn.Linear(hidden_dimension, latent_dimension)
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, state_dimension),
        )
        self.decoder = build_spatial_decoder(
            latent_dimension=latent_dimension,
            patch_grid=patch_grid,
            output_size=output_size,
        )
        self._initialize_prediction_prior()

    def _initialize_prediction_prior(self) -> None:
        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        nn.init.trunc_normal_(self.spatial_position, std=0.02)
        nn.init.trunc_normal_(self.condition_position, std=0.02)
        nn.init.trunc_normal_(self.output_queries, std=0.02)
        nn.init.zeros_(self.feature_head.weight)
        nn.init.zeros_(self.feature_head.bias)
        nn.init.zeros_(self.state_head[-1].weight)
        nn.init.zeros_(self.state_head[-1].bias)

    def forward(
        self,
        context: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 4 or context.shape[1] != self.context_frames:
            raise ValueError("context must have shape [batch, context_frames, tokens, latent]")
        expected_tokens = self.patch_grid**2
        if context.shape[2] != expected_tokens:
            raise ValueError(f"context must contain {expected_tokens} spatial tokens")

        visual_tokens = (
            self.visual_projection(context)
            + self.temporal_position
            + self.spatial_position
        ).flatten(start_dim=1, end_dim=2)
        condition_tokens = torch.stack(
            (self.state_projection(state), self.action_projection(action)),
            dim=1,
        )
        memory = self.encoder(
            torch.cat(
                (visual_tokens, condition_tokens + self.condition_position),
                dim=1,
            )
        )
        queries = self.output_queries.expand(context.shape[0], -1, -1)
        predicted_tokens = self.predictor(queries, memory)
        delta = self.feature_head(predicted_tokens)
        predicted_features = functional.normalize(context[:, -1] + delta, dim=-1)
        predicted_state = state + self.state_head(memory[:, -2])
        return predicted_features, predicted_state

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return decode_spatial_features(
            features,
            patch_grid=self.patch_grid,
            decoder=self.decoder,
        )
