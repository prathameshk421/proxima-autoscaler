# app.py
from fastapi import FastAPI, Query
import numpy as np
import uvicorn
from stable_baselines3 import PPO

from src.env.cloud_env import CloudAutoScalingEnv
from src.baselines.heuristics import threshold_policy, step_scaling_policy, target_tracking_policy

app = FastAPI(title="Proxima Auto-Scaler API")

# Load PPO model ONCE at startup to optimize latency
try:
    model = PPO.load("checkpoints/ppo_cloud_final.zip", device="cpu")
    model_loaded = True
except Exception as e:
    print(f"Failed to load PPO model: {e}")
    model = None
    model_loaded = False

@app.get("/")
def root():
    return {
        "name": "Proxima Auto-Scaler API",
        "status": "ok",
        "model_loaded": model_loaded
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model_loaded}

def run_policy_evaluation(env, policy_fn, is_ppo=False):
    """Core evaluation loop matching specifications precisely."""
    obs_arr, _ = env.reset(seed=env._init_seed)
    total_reward = 0.0
    sla_violations = 0
    steps_log = []
    
    for t in range(200):
        if is_ppo:
            action_arr, _ = policy_fn(obs_arr, deterministic=True)
            step_scaling = int(action_arr[0]) - 2
            t_utils = [0.4, 0.5, 0.6, 0.7]
            t_util = t_utils[int(action_arr[1])]
            cap_signal = int(action_arr[2])
        else:
            obs_pyd = env.state()
            action_obj = policy_fn(obs_pyd)
            
            step_scaling = action_obj.step_scaling
            t_util = action_obj.target_utilization
            cap_signal = action_obj.capacity_signal
            
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
            
        # Log before step
        obs_dict = env.state().model_dump()
        action_dict = {
            "step_scaling": step_scaling,
            "target_utilization": t_util,
            "capacity_signal": cap_signal
        }
        
        obs_arr, reward, terminated, truncated, info = env.step(action_arr)
        
        steps_log.append({
            "t": t,
            "obs": obs_dict,
            "action": action_dict,
            "reward": float(reward)
        })
        
        total_reward += float(reward)
        if info["latency_p95"] > env.T_sla:
            sla_violations += 1
            
        if terminated or truncated:
            break
            
    # Calculate required SCORE scalar
    min_possible = -200 * 1.5
    max_possible = 200 * 0.5
    score = float(np.clip((total_reward - min_possible) / (max_possible - min_possible), 0.0, 1.0))
    
    return {
        "score": score,
        "sla_violation_rate": sla_violations / max(1, len(steps_log)),
        "avg_reward": total_reward,
        "steps": steps_log
    }

@app.get("/evaluate")
def evaluate(difficulty: str = Query("easy"), seed: int = Query(42)):
    """Evaluate trained PPO model dynamically."""
    if not model_loaded:
        return {"error": "Model failed to load. Check checkpoints/ppo_cloud_final.zip."}
        
    env = CloudAutoScalingEnv(difficulty=difficulty, seed=seed)
    res = run_policy_evaluation(env, model.predict, is_ppo=True)
    
    return {
        "difficulty": difficulty,
        "seed": seed,
        "score": res["score"],
        "sla_violation_rate": res["sla_violation_rate"],
        "avg_reward": res["avg_reward"],
        "steps": res["steps"]
    }

@app.get("/baselines")
def baselines(difficulty: str = Query("easy"), seed: int = Query(42)):
    """Evaluate baseline heuristics identically."""
    env = CloudAutoScalingEnv(difficulty=difficulty, seed=seed)
    
    pol_names = ["Threshold", "Step Scaling", "Target Tracking"]
    pols = [
        threshold_policy,
        step_scaling_policy,
        lambda o: target_tracking_policy(o, desired_target=0.60)
    ]
    
    results_map = {}
    for name, pol_fn in zip(pol_names, pols):
        res = run_policy_evaluation(env, pol_fn, is_ppo=False)
        # Drop redundant step logs for baseline bulk array (usually large output)
        res.pop("steps")
        results_map[name] = res
        
    return {
        "difficulty": difficulty,
        "seed": seed,
        "baselines": results_map
    }
