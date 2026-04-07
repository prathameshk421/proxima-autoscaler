"""Deterministic multi-agent evaluator for cloud autoscaling tasks."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.agent.action_adapter import action_id_to_action
from src.agent.sb3_compat import patch_torch_load_for_zip_stream
from src.baselines.heuristics import (
    run_baseline,
    step_scaling_policy,
    target_tracking_policy,
    threshold_policy,
)
from src.env.cloud_env import CloudAutoScalingEnv
from src.env.schemas import Action, Observation


def _obs_to_vector(env: CloudAutoScalingEnv, obs: Observation) -> list[float]:
    """Convert typed observation into PPO-ready feature vector."""
    trend = env.workload_generator.get_recent_trend(env.timestep, window=8)
    return obs.to_vector() + trend


def _run_policy_episode(
    env: CloudAutoScalingEnv,
    action_selector: Callable[[Observation], Action],
    seed: int,
) -> dict[str, float]:
    """Run one deterministic episode and compute evaluation metrics."""
    obs, _ = env.reset(seed=seed)
    done = False

    total_reward = 0.0
    steps = 0
    sla_violations = 0
    capacity_waste_acc = 0.0
    sla_threshold = env.queue_model.base_latency_ms * 2.5

    while not done:
        action = action_selector(obs)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        if float(info.get("estimated_latency", 0.0)) > sla_threshold:
            sla_violations += 1

        required_servers = math.ceil(
            max(0.0, obs.incoming_request_rate)
            / max(1e-6, env.mu_per_server * max(0.1, obs.target_utilization))
        )
        capacity_waste_acc += max(0.0, obs.current_servers - min(env.max_servers, required_servers)) / float(
            max(1, env.max_servers)
        )

    signal_tp = float(env.signal_tp)
    signal_fp = float(env.signal_fp)
    signal_fn = float(env.signal_fn)
    signal_denom = signal_tp + signal_fp + signal_fn
    signal_accuracy = 1.0 if signal_denom == 0 else signal_tp / signal_denom

    return {
        "avg_reward": total_reward / max(1, steps),
        "sla_violation_rate": sla_violations / max(1, steps),
        "oscillation_count": float(env.oscillation_count),
        "capacity_waste": capacity_waste_acc / max(1, steps),
        "capacity_signal_accuracy": signal_accuracy,
        "steps": float(steps),
    }


def evaluate_ppo_run(difficulty: str, seed: int, model: PPO) -> dict[str, float]:
    """Evaluate one PPO run for a given difficulty and seed."""
    env = CloudAutoScalingEnv(difficulty=difficulty, seed=seed)

    def _ppo_selector(obs: Observation) -> Action:
        obs_vec = _obs_to_vector(env, obs)
        action_id, _ = model.predict(obs_vec, deterministic=True)
        return action_id_to_action(int(action_id))

    return _run_policy_episode(env, _ppo_selector, seed=seed)


def evaluate_baselines_for_seed(difficulty: str, seed: int) -> dict[str, dict[str, float]]:
    """Evaluate threshold, step, and target baselines on one difficulty/seed."""
    threshold_env = CloudAutoScalingEnv(difficulty=difficulty, seed=seed)
    step_env = CloudAutoScalingEnv(difficulty=difficulty, seed=seed)
    target_env = CloudAutoScalingEnv(difficulty=difficulty, seed=seed)

    threshold_metrics = run_baseline(threshold_env, threshold_policy, n_episodes=1)
    step_metrics = run_baseline(step_env, step_scaling_policy, n_episodes=1)
    target_metrics = run_baseline(target_env, lambda obs: target_tracking_policy(obs, 0.60), n_episodes=1)

    return {
        "threshold": threshold_metrics,
        "step": step_metrics,
        "target_tracking": target_metrics,
    }


def _internal_score(metrics: dict[str, float]) -> float:
    """Compute internal composite score used for ranking."""
    steps = max(1.0, float(metrics.get("steps", 1.0)))
    oscillation_rate = float(metrics.get("oscillation_count", 0.0)) / steps
    capacity_waste_normalized = max(0.0, min(1.0, float(metrics.get("capacity_waste", 0.0))))

    score = (
        (1.0 - float(metrics.get("sla_violation_rate", 0.0))) * 0.5
        + (1.0 - capacity_waste_normalized) * 0.3
        + (1.0 - oscillation_rate) * 0.2
    )
    return max(0.0, min(1.0, score))


def evaluate_all(model_path: str = "checkpoints/ppo_cloud_final.zip") -> dict[str, Any]:
    """Evaluate baselines and PPO across all required difficulties and seeds."""
    difficulties = ["easy", "medium", "hard"]
    seeds = [42, 123, 456]

    patch_torch_load_for_zip_stream()
    model = PPO.load(model_path, device="cpu")

    runs: list[dict[str, Any]] = []
    for difficulty in difficulties:
        for seed in seeds:
            baseline_results = evaluate_baselines_for_seed(difficulty=difficulty, seed=seed)
            for agent_name, metric_values in baseline_results.items():
                enriched = dict(metric_values)
                enriched["score"] = _internal_score(enriched)
                runs.append(
                    {
                        "difficulty": difficulty,
                        "seed": seed,
                        "agent": agent_name,
                        **enriched,
                    }
                )

            ppo_metrics = evaluate_ppo_run(difficulty=difficulty, seed=seed, model=model)
            ppo_metrics["score"] = _internal_score(ppo_metrics)
            runs.append(
                {
                    "difficulty": difficulty,
                    "seed": seed,
                    "agent": "ppo",
                    **ppo_metrics,
                }
            )

    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for difficulty in difficulties:
        grouped[difficulty] = {}
        for agent_name in ["threshold", "step", "target_tracking", "ppo"]:
            subset = [r for r in runs if r["difficulty"] == difficulty and r["agent"] == agent_name]
            grouped[difficulty][agent_name] = {
                "avg_reward": float(np.mean([x["avg_reward"] for x in subset])),
                "sla_violation_rate": float(np.mean([x["sla_violation_rate"] for x in subset])),
                "oscillation_count": float(np.mean([x["oscillation_count"] for x in subset])),
                "capacity_waste": float(np.mean([x["capacity_waste"] for x in subset])),
                "capacity_signal_accuracy": float(np.mean([x["capacity_signal_accuracy"] for x in subset])),
                "score": float(np.mean([x["score"] for x in subset])),
            }

    report = {
        "difficulties": difficulties,
        "seeds": seeds,
        "runs": runs,
        "summary": grouped,
    }

    Path("evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for difficulty in difficulties:
        print(f"\nDifficulty: {difficulty}")
        print("agent                avg_reward  sla_violation_rate  oscillation_count  capacity_waste  signal_accuracy  score")
        for agent_name in ["threshold", "step", "target_tracking", "ppo"]:
            values = grouped[difficulty][agent_name]
            print(
                f"{agent_name:20s} {values['avg_reward']:.4f}      {values['sla_violation_rate']:.4f}"
                f"             {values['oscillation_count']:.2f}              {values['capacity_waste']:.4f}"
                f"          {values['capacity_signal_accuracy']:.4f}         {values['score']:.4f}"
            )

    hard_ppo = grouped["hard"]["ppo"]["score"]
    hard_target = grouped["hard"]["target_tracking"]["score"]
    verdict = "YES" if hard_ppo > hard_target else "NO"
    print(f"PPO beats Target Tracking on Hard: {verdict}")

    return report


if __name__ == "__main__":
    evaluate_all(model_path="checkpoints/ppo_cloud_final.zip")
