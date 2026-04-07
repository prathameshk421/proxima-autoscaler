"""Deterministic baseline heuristics for cloud autoscaling."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.env.cloud_env import CloudAutoScalingEnv
from src.env.schemas import Action, Observation


def _queue_is_high(obs: Observation) -> bool:
    """Return whether queue pressure is high enough to suggest saturation."""
    dynamic_threshold = max(10.0, 0.25 * obs.incoming_request_rate)
    return obs.queue_length >= dynamic_threshold


def _at_capacity(obs: Observation) -> bool:
    """Return whether the cluster is currently at max server count."""
    return obs.current_servers >= obs.max_servers


def threshold_policy(obs: Observation) -> Action:
    """Threshold policy with cooldown-aware no-op and saturation signaling."""
    if obs.cooldown_remaining > 0:
        return Action(mode="step", scale_delta=0)

    if _at_capacity(obs) and _queue_is_high(obs):
        return Action(mode="signal", signal_type="INSUFFICIENT_CAPACITY")

    if obs.utilization > 0.80:
        return Action(mode="step", scale_delta=1)
    if obs.utilization < 0.30:
        return Action(mode="step", scale_delta=-1)
    return Action(mode="step", scale_delta=0)


def step_scaling_policy(obs: Observation) -> Action:
    """Aggressive step-scaling baseline with saturation-aware signaling."""
    if obs.cooldown_remaining > 0:
        return Action(mode="step", scale_delta=0)

    if _at_capacity(obs) and (obs.utilization >= 0.98 or _queue_is_high(obs)):
        return Action(mode="signal", signal_type="INSUFFICIENT_CAPACITY")

    if obs.utilization > 0.90:
        return Action(mode="step", scale_delta=2)
    if obs.utilization > 0.75:
        return Action(mode="step", scale_delta=1)
    if obs.utilization < 0.25:
        return Action(mode="step", scale_delta=-2)
    if obs.utilization < 0.40:
        return Action(mode="step", scale_delta=-1)
    return Action(mode="step", scale_delta=0)


def target_tracking_policy(obs: Observation, desired_target: float = 0.60) -> Action:
    """Target-tracking baseline with tolerance band and bounded corrections."""
    desired_target = min(0.7, max(0.4, float(desired_target)))
    allowed_targets = np.array([0.4, 0.5, 0.6, 0.7], dtype=np.float64)
    target_rounded = float(allowed_targets[np.argmin(np.abs(allowed_targets - desired_target))])

    if obs.cooldown_remaining > 0:
        return Action(mode="step", scale_delta=0)

    implied_required_servers = math.ceil(
        max(1e-6, obs.current_servers * max(0.0, obs.utilization)) / max(0.1, target_rounded)
    )
    if implied_required_servers > obs.max_servers:
        return Action(mode="signal", signal_type="INSUFFICIENT_CAPACITY")

    tolerance = 0.05
    deviation = obs.utilization - target_rounded
    if abs(deviation) <= tolerance:
        return Action(mode="step", scale_delta=0)

    return Action(mode="target", target_utilization=target_rounded)


def _required_servers_estimate(obs: Observation, mu_per_server: float) -> int:
    """Estimate required servers from arrival rate and target utilization."""
    required = math.ceil(
        max(0.0, obs.incoming_request_rate)
        / max(1e-6, mu_per_server * max(0.1, obs.target_utilization))
    )
    return max(0, required)


def run_baseline(
    env: CloudAutoScalingEnv,
    policy_fn: Callable[[Observation], Action],
    n_episodes: int = 5,
) -> dict[str, float]:
    """Run deterministic baseline episodes and return aggregate metrics."""
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")

    total_reward_sum = 0.0
    sla_violations = 0
    total_steps = 0
    total_capacity_waste = 0.0
    total_oscillation_count = 0
    total_signal_tp = 0
    total_signal_fp = 0
    total_signal_fn = 0

    base_seed = int(getattr(env, "seed_value", 42))
    sla_threshold = env.queue_model.base_latency_ms * 2.5

    for episode_idx in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + episode_idx)
        done = False
        cumulative_reward = 0.0

        while not done:
            action = policy_fn(obs)
            obs, reward, done, info = env.step(action)
            cumulative_reward += float(reward)
            total_steps += 1

            if float(info.get("estimated_latency", 0.0)) > sla_threshold:
                sla_violations += 1

            required_servers = _required_servers_estimate(obs, env.mu_per_server)
            capped_required = min(env.max_servers, required_servers)
            waste = max(0.0, obs.current_servers - capped_required) / float(max(1, env.max_servers))
            total_capacity_waste += waste

        total_reward_sum += cumulative_reward
        total_oscillation_count += int(env.oscillation_count)
        total_signal_tp += int(env.signal_tp)
        total_signal_fp += int(env.signal_fp)
        total_signal_fn += int(env.signal_fn)

    signal_denom = total_signal_tp + total_signal_fp + total_signal_fn
    capacity_signal_accuracy = 1.0 if signal_denom == 0 else total_signal_tp / float(signal_denom)

    return {
        "avg_reward": float(total_reward_sum / max(1, total_steps)),
        "sla_violation_rate": float(sla_violations / max(1, total_steps)),
        "oscillation_count": float(total_oscillation_count),
        "capacity_waste": float(total_capacity_waste / max(1, total_steps)),
        "capacity_signal_accuracy": float(capacity_signal_accuracy),
    }


if __name__ == "__main__":
    environment = CloudAutoScalingEnv(difficulty="easy", seed=42)

    threshold_metrics = run_baseline(environment, threshold_policy, n_episodes=2)
    step_metrics = run_baseline(environment, step_scaling_policy, n_episodes=2)
    target_metrics = run_baseline(
        environment,
        lambda obs: target_tracking_policy(obs, desired_target=0.60),
        n_episodes=2,
    )

    print("threshold:", threshold_metrics)
    print("step:", step_metrics)
    print("target:", target_metrics)