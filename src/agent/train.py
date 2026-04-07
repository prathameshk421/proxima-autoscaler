# src/agent/train.py
"""Curriculum learning training script for Proxima Auto-Scaler RL agent."""

import argparse
import random
import shutil
import numpy as np
import torch
import os

# Global seeds at the very top (before stable-baselines initializes)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.cloud_env import CloudAutoScalingEnv
from src.env.schemas import Action, Observation
from src.agent.ppo_agent import build_ppo
from src.baselines.heuristics import (
    run_baseline,
    threshold_policy,
    step_scaling_policy,
    target_tracking_policy,
)

def make_ppo_policy(model, env: CloudAutoScalingEnv):
    """Adapter to make PPO compatible with run_baseline evaluating."""
    def policy(obs: Observation) -> Action:
        # PPO requires the flat numpy observation, which we can extract
        # directly from the environment that is stepping through the episode.
        obs_array = env._build_obs()
        action_array, _ = model.predict(obs_array, deterministic=True)
        
        step_scaling = int(action_array[0]) - 2
        t_util_idx = int(action_array[1])
        t_utils = [0.4, 0.5, 0.6, 0.7]
        cap_signal = int(action_array[2])
        
        return Action(
            step_scaling=step_scaling,
            target_utilization=t_utils[t_util_idx],
            capacity_signal=cap_signal
        )
    return policy

def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent for Cloud Auto-Scaling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Logging directory")
    args = parser.parse_args()
    
    # Ensure reproducibility with arg seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    # ── Curriculum Phases ───────────────────────────────────────────────────
    # Total: 1M steps — 2× investment on hard to close the generalization gap
    curriculum = [
        ("easy",   150_000),
        ("medium", 250_000),
        ("hard",   600_000),
    ]
    
    model = None
    
    for phase_idx, (difficulty, num_timesteps) in enumerate(curriculum):
        print(f"\n{'='*60}")
        print(f" Phase {phase_idx + 1}: Training on '{difficulty}' difficulty ({num_timesteps} steps)")
        print(f"{'='*60}")
        
        # Training Environment
        train_env = DummyVecEnv([lambda d=difficulty, s=args.seed: Monitor(CloudAutoScalingEnv(difficulty=d, seed=s))])
        
        # Evaluation Environment
        eval_env = DummyVecEnv([lambda d=difficulty: Monitor(CloudAutoScalingEnv(difficulty=d, seed=123))])
        
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
        
        # Periodic checkpoint every 50k steps for safety
        checkpoint_callback = CheckpointCallback(
            save_freq=50_000,
            save_path=f"checkpoints/periodic_{difficulty}/",
            name_prefix="ppo_ckpt",
        )
        
        model.learn(
            total_timesteps=num_timesteps,
            callback=[eval_callback, checkpoint_callback],
            reset_num_timesteps=False
        )
        
    # Deploy the BEST eval checkpoint from hard phase (not the last checkpoint)
    best_hard_path = "checkpoints/best_hard/best_model.zip"
    final_path = "checkpoints/ppo_cloud_final.zip"
    if os.path.exists(best_hard_path):
        shutil.copy2(best_hard_path, final_path)
        print(f"\nDeployed best hard checkpoint → {final_path}")
    else:
        model.save("checkpoints/ppo_cloud_final")
        print("\nNo best_hard checkpoint found, saved final model instead.")
    
    # ── Evaluation ──────────────────────────────────────────────────────────
    print("\nRunning baseline comparisons across all difficulties...")
    
    policies = {
        "Threshold": threshold_policy,
        "Step Scaling": step_scaling_policy,
        "Target Tracking": lambda obs: target_tracking_policy(obs, desired_target=0.60),
        "PPO (ours)": None  # Will be dynamically built per difficulty
    }
    
    difficulties = ["easy", "medium", "hard"]
    results = {pol: {d: {} for d in difficulties} for pol in policies.keys()}
    
    for diff in difficulties:
        eval_env = CloudAutoScalingEnv(difficulty=diff, seed=999)
        
        for pol_name, pol_fn in policies.items():
            if pol_name == "PPO (ours)":
                policy_to_run = make_ppo_policy(model, eval_env)
            else:
                policy_to_run = pol_fn
                
            res = run_baseline(eval_env, policy_to_run, n_episodes=5)
            results[pol_name][diff] = res

    # ── Comparison Table ────────────────────────────────────────────────────
    print(f"\n| {'Policy':<15} | {'Easy R':>8} | {'Med R':>8} | {'Hard R':>8} | {'SLA Viol':>8} | {'Waste':>8} |")
    print(f"|{'-'*17}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|")
    
    for pol_name in policies.keys():
        easy_r = results[pol_name]["easy"]["avg_reward"]
        med_r  = results[pol_name]["medium"]["avg_reward"]
        hard_r = results[pol_name]["hard"]["avg_reward"]
        
        # Averages across the 3 environments
        avg_sla = sum(results[pol_name][d]["sla_violation_rate"] for d in difficulties) / 3
        avg_waste = sum(results[pol_name][d]["capacity_waste"] for d in difficulties) / 3
        
        print(f"| {pol_name:<15} | {easy_r:>8.1f} | {med_r:>8.1f} | {hard_r:>8.1f} | {avg_sla:>8.3f} | {avg_waste:>8.1f} |")

if __name__ == "__main__":
    main()
