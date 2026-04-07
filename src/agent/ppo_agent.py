"""PPO model factory and loader for cloud autoscaling environment."""

from __future__ import annotations

from stable_baselines3 import PPO

from src.agent.networks import CustomFeatureExtractor
from src.agent.sb3_compat import patch_torch_load_for_zip_stream


def make_ppo_model(env, tensorboard_log: str | None = None) -> PPO:
    """Create a CPU-only PPO model with stable defaults for delayed-control tasks."""
    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
    }

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        learning_rate=3e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        device="cpu",
        verbose=1,
    )
    return model


def load_ppo_model(path: str, env=None) -> PPO:
    """Load a PPO model on CPU, optionally attaching an environment."""
    patch_torch_load_for_zip_stream()
    return PPO.load(path, env=env, device="cpu")