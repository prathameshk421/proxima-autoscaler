"""PPO agent builder and loader for Proxima Auto-Scaler."""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

from src.agent.networks import CustomFeatureExtractor

def build_ppo(env) -> PPO:
    """
    Build and configure the PPO agent according to the design doc hyperparams.
    
    Args:
        env: The CloudAutoScalingEnv (or VecEnv) to train on.
        
    Returns:
        Configured PPO model instance.
    """
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64),
    )
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=2e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.25,
        ent_coef=0.02,
        vf_coef=0.25,
        max_grad_norm=0.3,
        policy_kwargs=policy_kwargs,
        device="cpu",
        tensorboard_log="./logs/tb",
        verbose=1,
    )
    return model

def load_agent(path: str, env) -> PPO:
    """
    Load a trained PPO model.
    
    Args:
        path: Filepath of the stored model checkpoint.
        env: The CloudAutoScalingEnv (or VecEnv) to evaluate on.
        
    Returns:
        Loaded PPO model ready for inference.
    """
    return PPO.load(path, env=env, device="cpu")
