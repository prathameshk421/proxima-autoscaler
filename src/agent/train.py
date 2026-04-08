"""Curriculum learning training script for the Proxima Auto-Scaler RL agent.

Trains a PPO agent through a 3-phase curriculum (easy → medium → hard),
then runs baseline comparisons across all difficulties.

Usage::

    python -m src.agent.train --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import shutil

import numpy as np
import torch
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agent.ppo_agent import build_ppo
from src.baselines.heuristics import (
    run_baseline,
    step_scaling_policy,
    target_tracking_policy,
    threshold_policy,
)
from src.env.cloud_env import CloudAutoScalingEnv
from src.env.schemas import Action, Observation

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def make_ppo_policy(model, env: CloudAutoScalingEnv):
    """Create a heuristic-API-compatible wrapper around PPO prediction.

    Returns a callable ``(Observation) -> Action`` so that the PPO agent
    can be evaluated using the same ``run_baseline`` function as the
    heuristic baselines.

    Parameters
    ----------
    model : PPO
        Trained PPO model.
    env : CloudAutoScalingEnv
        Active environment instance (used to build the flat observation).
    """

    def policy(obs: Observation) -> Action:
        obs_array = env._build_obs()
        action_array, _ = model.predict(obs_array, deterministic=True)

        step_scaling = int(action_array[0]) - 2
        t_util_idx = int(action_array[1])
        t_utils = [0.4, 0.5, 0.6, 0.7]
        cap_signal = int(action_array[2])

        return Action(
            step_scaling=step_scaling,
            target_utilization=t_utils[t_util_idx],
            capacity_signal=cap_signal,
        )

    return policy


def main() -> None:
    """Run the full training curriculum and baseline comparison."""
    parser = argparse.ArgumentParser(description="Train PPO Agent for Cloud Auto-Scaling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Logging directory")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)

    curriculum = [
        ("easy", 150_000),
        ("medium", 250_000),
        ("hard", 600_000),
    ]

    model = None

    for phase_idx, (difficulty, num_timesteps) in enumerate(curriculum):
        print(f"\n{'=' * 60}")
        print(
            f" Phase {phase_idx + 1}: Training on '{difficulty}' difficulty ({num_timesteps} steps)"
        )
        print(f"{'=' * 60}")

        train_env = DummyVecEnv(
            [lambda d=difficulty, s=args.seed: Monitor(CloudAutoScalingEnv(difficulty=d, seed=s))]
        )
        eval_env = DummyVecEnv(
            [lambda d=difficulty: Monitor(CloudAutoScalingEnv(difficulty=d, seed=123))]
        )

        if phase_idx == 0:
            model = build_ppo(train_env)
        else:
            model.set_env(train_env)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"checkpoints/best_{difficulty}/",
            log_path=args.log_dir,
            eval_freq=5000,
            n_eval_episodes=10,
            deterministic=False,
            render=False,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=50_000,
            save_path=f"checkpoints/periodic_{difficulty}/",
            name_prefix="ppo_ckpt",
        )

        model.learn(
            total_timesteps=num_timesteps,
            callback=[eval_callback, checkpoint_callback],
            reset_num_timesteps=False,
        )

    best_hard_path = "checkpoints/best_hard/best_model.zip"
    final_path = "checkpoints/ppo_cloud_final.zip"
    if os.path.exists(best_hard_path):
        shutil.copy2(best_hard_path, final_path)
        print(f"\nDeployed best hard checkpoint → {final_path}")
    else:
        model.save("checkpoints/ppo_cloud_final")
        print("\nNo best_hard checkpoint found, saved final model instead.")

    print("\nRunning baseline comparisons across all difficulties...")

    policies = {
        "Threshold": threshold_policy,
        "Step Scaling": step_scaling_policy,
        "Target Tracking": lambda obs: target_tracking_policy(obs, desired_target=0.60),
        "PPO (ours)": None,
    }

    difficulties = ["easy", "medium", "hard"]
    results = {pol: {d: {} for d in difficulties} for pol in policies}

    for diff in difficulties:
        eval_env = CloudAutoScalingEnv(difficulty=diff, seed=999)

        for pol_name, pol_fn in policies.items():
            if pol_name == "PPO (ours)":
                policy_to_run = make_ppo_policy(model, eval_env)
            else:
                policy_to_run = pol_fn

            res = run_baseline(eval_env, policy_to_run, n_episodes=5)
            results[pol_name][diff] = res

    header = (
        f"| {'Policy':<15} | {'Easy R':>8} | {'Med R':>8} | {'Hard R':>8} "
        f"| {'SLA Viol':>8} | {'Waste':>8} |"
    )
    print(f"\n{header}")
    print(f"|{'-' * 17}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|")

    for pol_name in policies:
        easy_r = results[pol_name]["easy"]["avg_reward"]
        med_r = results[pol_name]["medium"]["avg_reward"]
        hard_r = results[pol_name]["hard"]["avg_reward"]

        avg_sla = sum(results[pol_name][d]["sla_violation_rate"] for d in difficulties) / 3
        avg_waste = sum(results[pol_name][d]["capacity_waste"] for d in difficulties) / 3

        print(
            f"| {pol_name:<15} | {easy_r:>8.1f} | {med_r:>8.1f} | {hard_r:>8.1f} "
            f"| {avg_sla:>8.3f} | {avg_waste:>8.1f} |"
        )


if __name__ == "__main__":
    main()
