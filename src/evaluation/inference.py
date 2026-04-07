"""Standalone inference module for grading the final PPO agent."""

import argparse
from typing import Dict, Any
import numpy as np

from stable_baselines3 import PPO

from src.env.cloud_env import CloudAutoScalingEnv

def extract_obs_dict(obs_arr: np.ndarray, env: CloudAutoScalingEnv) -> Dict[str, Any]:
    """Convert numpy flat observation back to a readable dictionary log."""
    pyd_obs = env.state()
    return pyd_obs.model_dump()

def extract_action_dict(action_arr: np.ndarray) -> Dict[str, Any]:
    """Convert numpy action indices back to readable dictionary log."""
    t_utils = [0.4, 0.5, 0.6, 0.7]
    return {
        "step_scaling": int(action_arr[0]) - 2,
        "target_utilization": t_utils[int(action_arr[1])],
        "capacity_signal": int(action_arr[2])
    }

def adapt_obs_for_model(obs_arr: np.ndarray, expected_dim: int) -> np.ndarray:
    """Align env observation shape with the loaded PPO policy input shape."""
    if obs_arr.shape[0] == expected_dim:
        return obs_arr
    if obs_arr.shape[0] > expected_dim:
        return obs_arr[:expected_dim].astype(np.float32)

    pad_value = float(obs_arr[-1]) if obs_arr.shape[0] > 0 else 0.0
    return np.pad(
        obs_arr.astype(np.float32),
        (0, expected_dim - obs_arr.shape[0]),
        mode="constant",
        constant_values=pad_value,
    )

def run_inference(difficulty: str, seed: int, model_path: str) -> float:
    """Run precisely one episodic inference trace for OpenEnv grading."""
    
    env = CloudAutoScalingEnv(difficulty=difficulty, seed=seed)
    
    model = PPO.load(model_path, device="cpu")
    expected_obs_dim = int(model.observation_space.shape[0])
    
    obs_arr, _ = env.reset(seed=seed)
    total_reward = 0.0
    
    for t in range(200):
        model_obs = adapt_obs_for_model(obs_arr, expected_obs_dim)
        action_arr, _ = model.predict(model_obs, deterministic=True)
        
        obs_dict = extract_obs_dict(obs_arr, env)
        action_dict = extract_action_dict(action_arr)
        
        obs_arr, reward, terminated, truncated, info = env.step(action_arr)
        total_reward += float(reward)
        
        print(f"[STEP {t}] obs={obs_dict} action={action_dict} reward={reward:.4f}")
        
        if terminated or truncated:
            break
            
    min_possible = -200 * 1.5
    max_possible = 200 * 0.5
    
    normalized_score = (total_reward - min_possible) / (max_possible - min_possible)
    normalized_score = float(np.clip(normalized_score, 0.0, 1.0))
    
    print(f"SCORE={normalized_score:.4f}")
    
    return normalized_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser("OpenEnv AutoScaler Inference")
    parser.add_argument("--difficulty", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    
    args = parser.parse_args()
    run_inference(args.difficulty, args.seed, args.model_path)
