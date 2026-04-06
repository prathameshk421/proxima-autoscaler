# src/agent/networks.py
"""Custom Neural Network architecture for Proxima Auto-Scaler RL agent."""

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor separating the workload trend sequences from the scalar features.
    
    workload_trend -> GRU -> 32-dim features
    scalar features -> MLP -> 64-dim features
    concat + MLP -> 64-dim output
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        total_dim = observation_space.shape[0]
        self.trend_seq_len = 8
        self.remaining_dim = total_dim - self.trend_seq_len
        
        # 1. GRU for the workload trend (last 8 elements)
        # Input shape expected: (batch, seq=8, input=1)
        self.gru = nn.GRU(
            input_size=1, 
            hidden_size=32, 
            batch_first=True
        )
        
        # 2. MLP for the remaining scalar observations
        self.mlp = nn.Sequential(
            nn.Linear(self.remaining_dim, 64),
            nn.ReLU()
        )
        
        # 3. Final Merger (32 from GRU + 64 from MLP = 96 -> 64)
        self.merger = nn.Sequential(
            nn.Linear(96, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch_size, total_dim)
        
        # Split into scalar portion and workload trend
        remaining_features = observations[:, :self.remaining_dim]
        workload_trend = observations[:, -self.trend_seq_len:]
        
        # --- Trend Pipeline ---
        # Add feature dimension: (batch_size, 8) -> (batch_size, 8, 1)
        gru_in = workload_trend.unsqueeze(-1)
        
        # Pass through GRU. Output shape: (batch, seq_len, hidden_size)
        gru_out, _ = self.gru(gru_in)
        
        # We only care about the last output state
        trend_features = gru_out[:, -1, :]
        
        # --- Scalar Pipeline ---
        scalar_features = self.mlp(remaining_features)
        
        # --- Merge ---
        concat_features = torch.cat((trend_features, scalar_features), dim=1)
        return self.merger(concat_features)
