"""Rule-based heuristic baselines for Proxima Auto-Scaler.

Provides three heuristic policies (Threshold, Step Scaling, Target Tracking)
that serve as non-learning baselines for evaluating the PPO agent.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from src.env.cloud_env import CloudAutoScalingEnv
from src.env.schemas import Action, Observation


def threshold_policy(obs: Observation) -> Action:
    """Simple threshold scaling based on average utilization.

    Adds one server when utilisation exceeds 80 %, removes one when it
    drops below 30 %, and signals insufficient capacity when the queue
    is large and the server pool is maxed out.
    """
    if obs.cooldown_remaining > 0:
        return Action(step_scaling=0, target_utilization=0.5, capacity_signal=0)

    step_scaling = 0
    if obs.avg_utilization > 0.80:
        step_scaling = 1
    elif obs.avg_utilization < 0.30:
        step_scaling = -1

    capacity_signal = (
        1 if obs.queue_length > 50 and obs.current_capacity == obs.max_capacity else 0
    )

    return Action(
        step_scaling=step_scaling,
        target_utilization=0.5,
        capacity_signal=capacity_signal,
    )


def step_scaling_policy(obs: Observation) -> Action:
    """Multi-tiered scaling steps based on utilization thresholds.

    Uses two thresholds per direction:
    - ``util > 0.90`` → scale +2
    - ``util > 0.75`` → scale +1
    - ``util < 0.25`` → scale −2
    - ``util < 0.40`` → scale −1
    """
    if obs.cooldown_remaining > 0:
        return Action(step_scaling=0, target_utilization=0.5, capacity_signal=0)

    util = obs.avg_utilization
    step = 0
    if util > 0.90:
        step = 2
    elif util > 0.75:
        step = 1
    elif util < 0.25:
        step = -2
    elif util < 0.40:
        step = -1

    capacity_signal = 1 if util > 0.95 and obs.current_capacity == obs.max_capacity else 0

    return Action(
        step_scaling=step,
        target_utilization=0.5,
        capacity_signal=capacity_signal,
    )


def target_tracking_policy(obs: Observation, desired_target: float = 0.60) -> Action:
    """Proportional control maintaining a specific utilization target.

    Computes the required server count to achieve *desired_target*
    utilisation, then emits a scaling delta clamped to [-2, +2].
    A ±10 % dead-band prevents oscillation around the setpoint.

    Parameters
    ----------
    obs : Observation
        Current environment state.
    desired_target : float
        Target utilisation ratio (default 0.60).
    """
    if obs.cooldown_remaining > 0:
        return Action(step_scaling=0, target_utilization=desired_target, capacity_signal=0)

    total_cap = obs.current_capacity + sum(obs.booting_servers)

    if desired_target <= 0:
        return Action(step_scaling=0, target_utilization=desired_target, capacity_signal=0)

    ratio = obs.avg_utilization / desired_target

    if 0.9 < ratio < 1.1:
        return Action(step_scaling=0, target_utilization=desired_target, capacity_signal=0)

    desired_cap = math.ceil(total_cap * ratio)
    step_diff = desired_cap - total_cap
    step_diff = max(-2, min(2, step_diff))

    capacity_signal = 1 if desired_cap > obs.max_capacity else 0

    return Action(
        step_scaling=step_diff,
        target_utilization=desired_target,
        capacity_signal=capacity_signal,
    )


def run_baseline(
    env: CloudAutoScalingEnv,
    policy_fn: Callable[[Observation], Action],
    n_episodes: int = 5,
) -> dict[str, float]:
    """Evaluate a heuristic policy over multiple episodes.

    Parameters
    ----------
    env : CloudAutoScalingEnv
        Environment instance (reset with incrementing seeds).
    policy_fn : callable
        A function ``(Observation) -> Action``.
    n_episodes : int
        Number of evaluation episodes.

    Returns
    -------
    dict
        Aggregated metrics: ``avg_reward``, ``sla_violation_rate``,
        ``oscillation_count``, ``capacity_waste``, ``steps``, and
        ``capacity_signal_accuracy``.
    """
    total_reward = 0.0
    sla_violations = 0
    total_steps = 0
    total_oscillation_count = 0
    total_waste = 0.0

    for ep in range(n_episodes):
        env.reset(seed=env._init_seed + ep)

        last_dir = 0
        ep_reward = 0.0

        while True:
            obs = env.state()
            action_obj = policy_fn(obs)

            step_idx = int(np.clip(action_obj.step_scaling + 2, 0, 4))

            t_util = action_obj.target_utilization
            if t_util <= 0.45:
                t_idx = 0
            elif t_util <= 0.55:
                t_idx = 1
            elif t_util <= 0.65:
                t_idx = 2
            else:
                t_idx = 3

            sig_idx = int(np.clip(action_obj.capacity_signal, 0, 1))
            action_array = np.array([step_idx, t_idx, sig_idx])

            _, reward, terminated, truncated, info = env.step(action_array)
            ep_reward += float(reward)
            total_steps += 1

            current_scale = action_obj.step_scaling
            if current_scale != 0:
                current_dir = 1 if current_scale > 0 else -1
                if last_dir != 0 and current_dir != last_dir:
                    total_oscillation_count += 1
                last_dir = current_dir

            lam = info["lam"]
            current_capacity = info["current_capacity"]
            mu = env.mu

            waste = max(0.0, current_capacity * mu - lam)
            total_waste += float(waste)

            latency = info["latency_p95"]
            if latency > env.T_sla:
                sla_violations += 1

            if terminated or truncated:
                break

        total_reward += ep_reward

    return {
        "avg_reward": total_reward / max(1, n_episodes),
        "sla_violation_rate": float(sla_violations) / max(1, total_steps),
        "oscillation_count": int(total_oscillation_count / max(1, n_episodes)),
        "capacity_waste": total_waste / max(1, total_steps),
        "steps": float(total_steps) / max(1, n_episodes),
        "capacity_signal_accuracy": 0.0,
    }


if __name__ == "__main__":
    env = CloudAutoScalingEnv(difficulty="medium", seed=42)

    print("Evaluating Baseline Policies (Medium Difficulty, 5 Episodes)")
    print("-" * 60)

    res1 = run_baseline(env, threshold_policy, 5)
    print("1. Threshold Policy:")
    for k, v in res1.items():
        if isinstance(v, float):
            print(f"   {k:20s}: {v:8.4f}")
        else:
            print(f"   {k:20s}: {v:8d}")

    res2 = run_baseline(env, step_scaling_policy, 5)
    print("\n2. Step Scaling Policy:")
    for k, v in res2.items():
        if isinstance(v, float):
            print(f"   {k:20s}: {v:8.4f}")
        else:
            print(f"   {k:20s}: {v:8d}")

    res3 = run_baseline(env, target_tracking_policy, 5)
    print("\n3. Target Tracking Policy:")
    for k, v in res3.items():
        if isinstance(v, float):
            print(f"   {k:20s}: {v:8.4f}")
        else:
            print(f"   {k:20s}: {v:8d}")
