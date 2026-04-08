"""Training script for RL baselines (DQN, A2C) using the same curriculum as PPO."""

from __future__ import annotations

import argparse
import os
import random
import shutil

import numpy as np
import torch
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.cloud_env import CloudAutoScalingEnv
from src.env.discrete_action_wrapper import DiscreteActionWrapper
from src.baselines.rl_baselines import build_dqn, build_a2c


CURRICULUM = [
    ("easy",   150_000),
    ("medium", 250_000),
    ("hard",   600_000),
]


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(difficulty: str, seed: int, discrete: bool = False):
    """Factory for Monitor-wrapped env, optionally with DiscreteActionWrapper."""
    def _thunk():
        env = CloudAutoScalingEnv(difficulty=difficulty, seed=seed)
        if discrete:
            env = DiscreteActionWrapper(env)
        return Monitor(env)
    return _thunk


def train_agent(agent_name: str, seed: int, log_dir: str) -> None:
    """Train a single RL baseline through the full curriculum."""
    _set_seeds(seed)

    discrete = (agent_name == "dqn")
    model = None

    for phase_idx, (difficulty, num_timesteps) in enumerate(CURRICULUM):
        print(f"\n{'='*60}")
        print(f" [{agent_name.upper()}] Phase {phase_idx+1}: '{difficulty}' ({num_timesteps} steps)")
        print(f"{'='*60}")

        train_env = DummyVecEnv([make_env(difficulty, seed, discrete=discrete)])
        eval_env  = DummyVecEnv([make_env(difficulty, 123,  discrete=discrete)])

        if phase_idx == 0:
            if agent_name == "dqn":
                model = build_dqn(train_env)
            else:
                model = build_a2c(train_env)
        else:
            model.set_env(train_env)

        ckpt_dir = f"checkpoints/{agent_name}/best_{difficulty}"
        eval_cb  = EvalCallback(
            eval_env,
            best_model_save_path=ckpt_dir,
            log_path=log_dir,
            eval_freq=5_000,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        )
        periodic_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=f"checkpoints/{agent_name}/periodic_{difficulty}",
            name_prefix=f"{agent_name}_ckpt",
        )

        model.learn(
            total_timesteps=num_timesteps,
            callback=[eval_cb, periodic_cb],
            reset_num_timesteps=False,
        )

    best_hard = f"checkpoints/{agent_name}/best_hard/best_model.zip"
    final     = f"checkpoints/{agent_name}_final.zip"
    if os.path.exists(best_hard):
        shutil.copy2(best_hard, final)
        print(f"\nDeployed best hard checkpoint → {final}")
    else:
        model.save(f"checkpoints/{agent_name}_final")
        print(f"\nSaved final model → {final}")


def main():
    parser = argparse.ArgumentParser(description="Train RL Baselines (DQN, A2C)")
    parser.add_argument("--agents", nargs="+", default=["dqn", "a2c"],
                        choices=["dqn", "a2c"], help="Which agents to train")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    for agent_name in args.agents:
        print(f"\n{'#'*60}")
        print(f" Training {agent_name.upper()} baseline")
        print(f"{'#'*60}")
        train_agent(agent_name, args.seed, args.log_dir)

    print("\nAll RL baselines trained.")


if __name__ == "__main__":
    main()
