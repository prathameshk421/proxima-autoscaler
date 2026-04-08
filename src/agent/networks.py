"""Custom Neural Network architecture for Proxima Auto-Scaler RL agent.

Separates multi-channel temporal trend data from scalar system features
and processes them through a GRU before merging for the policy/value heads.
"""

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Multi-channel GRU + MLP feature extractor for cloud autoscaling.

    Architecture:
        4 trend channels (demand, util, latency, queue) × 8 timesteps
            → GRU(input_size=4, hidden=48) → 48-dim temporal features
        7 scalar features (capacity, c_max, booting, cooldown, queue, λ, μ)
            → MLP(7 → 64) → 64-dim scalar features
        Concat(48 + 64 = 112)
            → LayerNorm → MLP(112 → features_dim) → output
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        total_dim = observation_space.shape[0]
        self.n_channels = 4
        self.trend_seq_len = 8
        self.trend_total = self.n_channels * self.trend_seq_len  # 32
        self.scalar_dim = total_dim - self.trend_total           # 7

        self.gru = nn.GRU(
            input_size=self.n_channels,
            hidden_size=48,
            batch_first=True,
        )

        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.scalar_dim, 64),
            nn.ReLU(),
        )

        gru_out_dim = 48
        merged_dim = gru_out_dim + 64  # 112

        self.merger = nn.Sequential(
            nn.LayerNorm(merged_dim),
            nn.Linear(merged_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        scalar_features = observations[:, :self.scalar_dim]
        trend_flat = observations[:, self.scalar_dim:]

        # Reshape (batch, 32) → (batch, 8, 4): 8 timesteps × 4 channels
        # Channels are laid out as: demand[0..7], util[8..15], lat[16..23], queue[24..31]
        trend_by_channel = trend_flat.view(batch_size, self.n_channels, self.trend_seq_len)
        # Transpose to (batch, seq_len=8, channels=4)
        trend_input = trend_by_channel.transpose(1, 2)

        gru_out, _ = self.gru(trend_input)
        temporal_features = gru_out[:, -1, :]  # last hidden state

        scalar_out = self.scalar_mlp(scalar_features)

        merged = torch.cat((temporal_features, scalar_out), dim=1)
        return self.merger(merged)
