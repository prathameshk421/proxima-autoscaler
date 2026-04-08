"""PPO agent builder and loader for Proxima Auto-Scaler."""

from __future__ import annotations

from stable_baselines3 import PPO

from src.agent.networks import CustomFeatureExtractor


def build_ppo(env) -> PPO:
    """Build and configure the PPO agent with design-doc hyperparameters.

    Parameters
    ----------
    env
        The ``CloudAutoScalingEnv`` (or ``VecEnv``-wrapped variant) to
        train on.

    Returns
    -------
    PPO
        A freshly initialised PPO model ready for ``.learn()``.
    """
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64),
    )

    return PPO(
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


def load_agent(path: str, env) -> PPO:
    """Load a trained PPO model from a checkpoint file.

    Parameters
    ----------
    path : str
        Filepath of the stored model checkpoint (e.g.
        ``checkpoints/ppo_cloud_final.zip``).
    env
        The ``CloudAutoScalingEnv`` (or ``VecEnv``) to attach for
        inference.

    Returns
    -------
    PPO
        Loaded PPO model ready for ``.predict()``.
    """
    return PPO.load(path, env=env, device="cpu")
