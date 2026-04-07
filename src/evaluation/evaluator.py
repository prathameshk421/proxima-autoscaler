"""Comprehensive evaluation script validating multi-baseline metrics."""

import json
from typing import Callable, Any, Dict
import numpy as np

from stable_baselines3 import PPO

from src.env.cloud_env import CloudAutoScalingEnv
from src.env.schemas import Action, Observation
from src.baselines.heuristics import (
    threshold_policy,
    step_scaling_policy,
    target_tracking_policy
)

def evaluate_episode(env: CloudAutoScalingEnv, policy_fn: Callable[[Observation], Action], is_ppo: bool = False) -> Dict[str, float]:
    """Run a single episode and return metrics for the evaluation."""
    obs_arr, _ = env.reset(seed=env._init_seed)
    
    ep_reward = 0.0
    sla_violations = 0
    total_waste = 0.0
    total_steps = 0
    oscillation_count = 0
    last_dir = 0
    
    tp = 0
    fp = 0
    fn = 0
    
    while True:
        if is_ppo:
            action_arr, _ = policy_fn(obs_arr, deterministic=True)
            step_scaling = int(action_arr[0]) - 2
            cap_signal = int(action_arr[2])
        else:
            obs_pyd = env.state()
            action_obj = policy_fn(obs_pyd)
            
            step_scaling = action_obj.step_scaling
            cap_signal = action_obj.capacity_signal
            
            t_util = action_obj.target_utilization
            t_idx = 1
            if t_util <= 0.45: t_idx = 0
            elif t_util <= 0.55: t_idx = 1
            elif t_util <= 0.65: t_idx = 2
            else: t_idx = 3
                
            action_arr = np.array([
                np.clip(step_scaling + 2, 0, 4),
                t_idx,
                np.clip(cap_signal, 0, 1)
            ])
            
        rho = compute_utilization_direct(env._lam, env.mu, env.current_capacity)
        c_max = env.c_max
        curr_c = env.current_capacity
        lam = env._lam
        mu = env.mu
        
        should_signal = (rho >= 0.95 and curr_c == c_max)
        
        if cap_signal == 1:
            if should_signal: tp += 1
            else: fp += 1
        else:
            if should_signal: fn += 1
            
        obs_arr, reward, terminated, truncated, info = env.step(action_arr)
        
        ep_reward += float(reward)
        total_steps += 1
        
        if step_scaling != 0:
            current_dir = 1 if step_scaling > 0 else -1
            if last_dir != 0 and current_dir != last_dir:
                oscillation_count += 1
            last_dir = current_dir
            
        waste = max(0.0, float(env.current_capacity * env.mu - env._lam))
        total_waste += waste
        
        latency = info["latency_p95"]
        if latency > env.T_sla:
            sla_violations += 1
            
        if terminated or truncated:
            break
            
    accuracy_denom = (tp + fp + fn)
    signal_accuracy = (tp / accuracy_denom) if accuracy_denom > 0 else 1.0
            
    return {
        "avg_reward": ep_reward,
        "sla_violation_rate": sla_violations / max(1, total_steps),
        "oscillation_count": oscillation_count,
        "capacity_waste": total_waste / max(1, total_steps),
        "capacity_signal_accuracy": signal_accuracy
    }
    
def compute_utilization_direct(lam: float, mu: float, c: int) -> float:
    """Helper duplicate of queue_model util since we need to peek before steps."""
    return lam / (c * mu) if c > 0 else 1.0

def main():
    print("Loading Trained PPO artifact: checkpoints/ppo_cloud_final.zip")
    try:
        model = PPO.load("checkpoints/ppo_cloud_final.zip", device="cpu")
    except Exception as e:
        print(f"Error loading PPO: {e}. Defaulting to dummy for script test.")
        model = None
        
    policies = {
        "Threshold": (threshold_policy, False),
        "Step Scaling": (step_scaling_policy, False),
        "Target Tracking": (lambda o: target_tracking_policy(o, desired_target=0.60), False),
        "PPO": (model.predict if model else (lambda o, deterministic: (np.array([2,1,0]), None)), True)
    }
    
    if model is not None:
        expected_dim = int(model.observation_space.shape[0])
        print(f"PPO model expects obs dim: {expected_dim}")
        assert expected_dim == 15, f"Model/env obs dim mismatch: model={expected_dim}, env=15"
    
    difficulties = ["easy", "medium", "hard"]
    seeds = [42, 123, 456]
    
    results = []
    max_waste = 0.0
    
    print("\nExecuting 36 Episode Evaluations...")
    for diff in difficulties:
        for seed in seeds:
            for pol_name, (pol_fn, is_ppo) in policies.items():
                env = CloudAutoScalingEnv(difficulty=diff, seed=seed)
                
                metrics = evaluate_episode(env, pol_fn, is_ppo=is_ppo)
                max_waste = max(max_waste, metrics["capacity_waste"])
                
                metrics.update({
                    "policy": pol_name,
                    "difficulty": diff,
                    "seed": seed
                })
                
                results.append(metrics)
                
    if max_waste == 0.0:
        max_waste = 1.0
    
    for r in results:
        cw_norm = r["capacity_waste"] / max_waste
        osc_rate = r["oscillation_count"] / 200.0
        sla_rate = r["sla_violation_rate"]
        
        score = (1 - sla_rate)*0.5 + (1 - cw_norm)*0.3 + (1 - osc_rate)*0.2
        r["internal_score"] = score
        
    with open("evaluation_report.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nResults isolated and formatted to evaluation_report.json.")
        
    print("\n" + "="*85)
    print(f"| {'Difficulty':<10} | {'Policy':<15} | {'Reward':>9} | {'SLA Viol':>8} | {'Waste':>8} | {'Signal Acc':>10} |")
    print("="*85)
    
    target_tracking_hard_score = 0.0
    ppo_hard_score = 0.0
    
    for diff in difficulties:
        for pol_name in policies.keys():
            pol_runs = [r for r in results if r["policy"] == pol_name and r["difficulty"] == diff]
            
            avg_r = sum(x["avg_reward"] for x in pol_runs) / 3
            avg_sla = sum(x["sla_violation_rate"] for x in pol_runs) / 3
            avg_waste = sum(x["capacity_waste"] for x in pol_runs) / 3
            avg_acc = sum(x["capacity_signal_accuracy"] for x in pol_runs) / 3
            avg_intern = sum(x["internal_score"] for x in pol_runs) / 3
            
            if diff == "hard":
                if pol_name == "Target Tracking":
                    target_tracking_hard_score = avg_intern
                elif pol_name == "PPO":
                    ppo_hard_score = avg_intern
            
            print(f"| {diff:<10} | {pol_name:<15} | {avg_r:>9.1f} | {avg_sla:>8.3f} | {avg_waste:>8.1f} | {avg_acc:>10.3f} |")
        print("-" * 85)
        
    ppo_won = ppo_hard_score > target_tracking_hard_score
    diff_text = "YES" if ppo_won else "NO"
    print(f"\nPPO beats Target Tracking on Hard: {diff_text}  (PPO:{ppo_hard_score:.3f} vs Target:{target_tracking_hard_score:.3f})")

if __name__ == "__main__":
    main()
