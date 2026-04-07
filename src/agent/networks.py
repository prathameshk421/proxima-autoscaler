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
        
        self.gru = nn.GRU(
            input_size=1, 
            hidden_size=32, 
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(self.remaining_dim, 64),
            nn.ReLU()
        )
        
        self.merger = nn.Sequential(
            nn.Linear(96, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        remaining_features = observations[:, :self.remaining_dim]
        workload_trend = observations[:, -self.trend_seq_len:]
        
        gru_in = workload_trend.unsqueeze(-1)
        
        gru_out, _ = self.gru(gru_in)
        
        trend_features = gru_out[:, -1, :]
        
        scalar_features = self.mlp(remaining_features)
        
        concat_features = torch.cat((trend_features, scalar_features), dim=1)
        return self.merger(concat_features)
