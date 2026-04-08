"""FastAPI application for Proxima Auto-Scaler (OpenEnv-compliant)."""

from typing import Dict, Optional

from fastapi import FastAPI, Query
import numpy as np
import uvicorn
from stable_baselines3 import PPO

from src.env.cloud_env import CloudAutoScalingEnv
from src.env.schemas import Action, Observation, Reward
from src.baselines.heuristics import threshold_policy, step_scaling_policy, target_tracking_policy

app = FastAPI(
    title="Proxima Auto-Scaler API",
    description="RL-powered cloud auto-scaling – OpenEnv spec compliant",
    version="0.1.0",
)

# ── Model loading ─────────────────────────────────────────────────────────

try:
    model = PPO.load("checkpoints/ppo_cloud_final.zip", device="cpu")
    model_loaded = True
except Exception as e:
    print(f"Failed to load PPO model: {e}")
    model = None
    model_loaded = False

# ── Persistent env sessions (keyed by difficulty) for step API ────────────

_sessions: Dict[str, CloudAutoScalingEnv] = {}


# ── Helper: normalise cumulative reward → [0, 1] ─────────────────────────

def _normalize_score(total_reward: float, n_steps: int = 200) -> float:
    """Normalize cumulative reward to (0, 1) — strictly excludes 0.0 and 1.0."""
    min_possible = -n_steps * 1.5
    max_possible = n_steps * 0.5
    return float(np.clip(
        (total_reward - min_possible) / (max_possible - min_possible),
        0.001, 0.999,
    ))


def _build_reward(raw: float, info: dict, env: CloudAutoScalingEnv,
                   cumulative: float, step: int) -> dict:
    sla_pen = max(0.0, info["latency_p95"] - env.T_sla) * 12.0
    excess = max(0.0, env.current_capacity * env.mu - env._lam)
    waste_pen = 0.008 * (excess ** 2)
    return {
        "value": raw,
        "sla_penalty": round(sla_pen, 4),
        "waste_penalty": round(waste_pen, 4),
        "score": round(_normalize_score(cumulative, max(step, 1)), 4),
    }


# ══════════════════════════════════════════════════════════════════════════
# OpenEnv step-level endpoints
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "name": "Proxima Auto-Scaler API",
        "status": "ok",
        "model_loaded": model_loaded,
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/tasks")
def list_tasks():
    """List available tasks with difficulties (OpenEnv spec)."""
    return {
        "tasks": [
            {
                "name": "cloud_autoscaling_easy",
                "difficulty": "easy",
                "description": "Predictable sinusoidal workload, 1-step boot delay.",
            },
            {
                "name": "cloud_autoscaling_medium",
                "difficulty": "medium",
                "description": "Fluctuating workload with MMPP bursts, 2-step boot delay.",
            },
            {
                "name": "cloud_autoscaling_hard",
                "difficulty": "hard",
                "description": "Highly unpredictable spiky demand, 3-step boot delay.",
            },
        ]
    }


@app.post("/reset")
def reset_env(difficulty: str = Query("easy"), seed: int = Query(42)):
    """Reset environment for a new episode (OpenEnv spec)."""
    env = CloudAutoScalingEnv(difficulty=difficulty, seed=seed)
    env.reset(seed=seed)
    _sessions[difficulty] = env
    obs_pyd = env.state()
    return {
        "observation": obs_pyd.model_dump(),
        "info": {"difficulty": difficulty, "seed": seed},
    }


@app.get("/state")
def get_state(difficulty: str = Query("easy")):
    """Return current environment state (OpenEnv spec)."""
    env = _sessions.get(difficulty)
    if env is None:
        return {"error": f"No active session for '{difficulty}'. Call /reset first."}
    return {"observation": env.state().model_dump()}


@app.post("/step")
def step_env(
    difficulty: str = Query("easy"),
    step_scaling: int = Query(0, ge=-2, le=2),
    target_utilization: float = Query(0.5),
    capacity_signal: int = Query(0, ge=0, le=1),
):
    """Execute one environment step (OpenEnv spec).

    Returns observation, reward (structured), done flag, and info.
    """
    env = _sessions.get(difficulty)
    if env is None:
        return {"error": f"No active session for '{difficulty}'. Call /reset first."}

    t_util = target_utilization
    if t_util <= 0.45:
        t_idx = 0
    elif t_util <= 0.55:
        t_idx = 1
    elif t_util <= 0.65:
        t_idx = 2
    else:
        t_idx = 3

    action_arr = np.array([
        np.clip(step_scaling + 2, 0, 4),
        t_idx,
        np.clip(capacity_signal, 0, 1),
    ])

    obs_arr, reward, terminated, truncated, info = env.step(action_arr)
    done = terminated or truncated

    # Compute running cumulative reward
    cum_reward = getattr(env, "_cum_reward", 0.0) + float(reward)
    env._cum_reward = cum_reward

    obs_pyd = env.state()
    reward_dict = _build_reward(float(reward), info, env, cum_reward, env.current_step)

    return {
        "observation": obs_pyd.model_dump(),
        "reward": reward_dict,
        "done": done,
        "info": {
            "latency_p95": info["latency_p95"],
            "utilization": info["utilization"],
            "current_capacity": info["current_capacity"],
            "demand_rate": info["lam"],
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# Batch evaluation endpoints (kept for backwards-compat & demo)
# ══════════════════════════════════════════════════════════════════════════

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

    score = _normalize_score(total_reward)

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
        res.pop("steps")
        results_map[name] = res

    return {
        "difficulty": difficulty,
        "seed": seed,
        "baselines": results_map
    }
