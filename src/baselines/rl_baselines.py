"""RL baseline builders for comparison against PPO.

Provides DQN and A2C agents trained on the same environment to
demonstrate that PPO is the best algorithm choice for this domain.
"""

from __future__ import annotations

from stable_baselines3 import DQN, A2C

from src.agent.networks import CustomFeatureExtractor


def build_dqn(env) -> DQN:
    """Build DQN agent for the DiscreteActionWrapper-wrapped environment.

    DQN requires Discrete action space, so the env must be wrapped
    with DiscreteActionWrapper before calling this function.

    Uses the same CustomFeatureExtractor for fair architectural comparison —
    only the RL algorithm differs (value-based vs. policy gradient).
    """
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64),
    )

    return DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=2_000,
        batch_size=128,
        gamma=0.99,
        target_update_interval=1_000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        device="cpu",
        tensorboard_log="./logs/tb_dqn",
        verbose=1,
    )


def build_a2c(env) -> A2C:
    """Build A2C agent for the MultiDiscrete environment.

    A2C natively supports MultiDiscrete action spaces. Uses the same
    CustomFeatureExtractor for fair comparison — only the RL algorithm
    differs (synchronous advantage actor-critic vs. PPO's clipped objective).
    """
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64),
    )

    return A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device="cpu",
        tensorboard_log="./logs/tb_a2c",
        verbose=1,
    )
