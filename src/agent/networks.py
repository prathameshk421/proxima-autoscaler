"""Custom SB3 feature extractor for trend-aware autoscaling observations."""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Split core and trend features, encode trend via GRU, and fuse embeddings."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128) -> None:
        """Initialize extractor layers and validate expected observation shape."""
        if not isinstance(observation_space, spaces.Box):
            raise ValueError("CustomFeatureExtractor expects a Box observation space")

        if len(observation_space.shape) != 1:
            raise ValueError("Observation space must be a flat 1D vector")

        obs_dim = int(observation_space.shape[0])
        if obs_dim <= 8:
            raise ValueError("Observation dimension must be > 8 to include core and trend features")

        super().__init__(observation_space, features_dim)
        self.obs_dim = obs_dim
        self.trend_dim = 8
        self.core_dim = self.obs_dim - self.trend_dim

        self.core_net = nn.Sequential(
            nn.Linear(self.core_dim, 64),
            nn.ReLU(),
        )

        self.trend_gru = nn.GRU(
            input_size=1,
            hidden_size=32,
            batch_first=True,
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations into fused latent features."""
        if observations.dim() != 2:
            raise ValueError(f"Expected 2D tensor [batch, obs_dim], got shape={tuple(observations.shape)}")

        if observations.shape[1] != self.obs_dim:
            raise ValueError(
                f"Expected observation dim {self.obs_dim}, got {int(observations.shape[1])}"
            )

        core_features = observations[:, : self.core_dim]
        trend_history = observations[:, self.core_dim :]

        if trend_history.shape[1] != self.trend_dim:
            raise ValueError("Trend history must have exactly 8 elements")

        core_embedding = self.core_net(core_features)

        trend_sequence = trend_history.unsqueeze(-1)
        _, hidden_state = self.trend_gru(trend_sequence)
        trend_embedding = hidden_state[-1]

        fused = torch.cat([core_embedding, trend_embedding], dim=1)
        return self.fusion_net(fused)