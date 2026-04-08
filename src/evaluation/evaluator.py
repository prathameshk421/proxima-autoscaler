"""Comprehensive evaluation with confidence intervals and RL baseline support.

Evaluates PPO, DQN, A2C (RL agents) and heuristic baselines across all
difficulties using multiple seeds. Reports mean ± std for all metrics.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO, DQN, A2C

from src.env.cloud_env import CloudAutoScalingEnv
from src.env.discrete_action_wrapper import DiscreteActionWrapper
from src.env.schemas import Action, Observation
from src.baselines.heuristics import (
    threshold_policy,
    step_scaling_policy,
    target_tracking_policy,
)


EVAL_SEEDS = [42, 123, 456, 789, 1024, 2048, 3000, 4096, 5555, 7777]


def evaluate_episode(
    env: CloudAutoScalingEnv,
    policy_fn,
    is_rl: bool = False,
    needs_discrete_wrapper: bool = False,
    seed: int = 42,
) -> Dict[str, float]:
    """Run a single episode and collect detailed metrics."""
    base = env.unwrapped  # access base env attributes through wrappers
    obs_arr, _ = env.reset(seed=seed)

    ep_reward = 0.0
    sla_violations = 0
    total_waste = 0.0
    total_steps = 0
    oscillation_count = 0
    last_dir = 0
    latencies = []

    tp, fp, fn = 0, 0, 0

    while True:
        if is_rl:
            action_pred, _ = policy_fn(obs_arr, deterministic=True)
            if needs_discrete_wrapper:
                flat = int(np.asarray(action_pred).reshape(-1)[0])
                action_to_env = flat

                a2 = flat % 2
                a1 = (flat // 2) % 4
                a0 = flat // 8
                step_scaling = a0 - 2
                cap_signal = a2
            else:
                action_arr = np.asarray(action_pred).reshape(-1)
                step_scaling = int(action_arr[0]) - 2
                cap_signal = int(action_arr[2])
                action_to_env = action_arr
        else:
            obs_pyd = base.state()
            action_obj = policy_fn(obs_pyd)

            step_scaling = action_obj.step_scaling
            cap_signal = action_obj.capacity_signal

            t_util = action_obj.target_utilization
            t_idx = 1
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
                np.clip(cap_signal, 0, 1),
            ])
            action_to_env = action_arr

        rho = base._lam / (base.current_capacity * base.mu) if base.current_capacity > 0 else 1.0
        should_signal = rho >= 0.95 and base.current_capacity == base.c_max

        if cap_signal == 1:
            if should_signal:
                tp += 1
            else:
                fp += 1
        elif should_signal:
            fn += 1

        obs_arr, reward, terminated, truncated, info = env.step(action_to_env)

        ep_reward += float(reward)
        total_steps += 1
        latencies.append(info["latency_p95"])

        if step_scaling != 0:
            current_dir = 1 if step_scaling > 0 else -1
            if last_dir != 0 and current_dir != last_dir:
                oscillation_count += 1
            last_dir = current_dir

        waste = max(0.0, float(base.current_capacity * base.mu - base._lam))
        total_waste += waste

        if info["latency_p95"] > base.T_sla:
            sla_violations += 1

        if terminated or truncated:
            break

    accuracy_denom = tp + fp + fn
    signal_accuracy = (tp / accuracy_denom) if accuracy_denom > 0 else 1.0

    latencies_arr = np.array(latencies)

    return {
        "avg_reward": ep_reward,
        "sla_violation_rate": sla_violations / max(1, total_steps),
        "oscillation_count": oscillation_count,
        "capacity_waste": total_waste / max(1, total_steps),
        "capacity_signal_accuracy": signal_accuracy,
        "latency_mean": float(np.mean(latencies_arr)),
        "latency_p95": float(np.percentile(latencies_arr, 95)),
        "latency_p99": float(np.percentile(latencies_arr, 99)),
    }


def evaluate_policy_multi_seed(
    policy_name: str,
    policy_fn,
    is_rl: bool,
    difficulty: str,
    seeds: List[int],
    needs_discrete_wrapper: bool = False,
) -> Dict[str, float]:
    """Evaluate a policy across multiple seeds and compute mean ± std."""
    metrics_list = []

    for seed in seeds:
        base_env = CloudAutoScalingEnv(difficulty=difficulty, seed=seed)
        if needs_discrete_wrapper:
            env = DiscreteActionWrapper(base_env)
        else:
            env = base_env
        m = evaluate_episode(
            env,
            policy_fn,
            is_rl=is_rl,
            needs_discrete_wrapper=needs_discrete_wrapper,
            seed=seed,
        )
        metrics_list.append(m)

    result = {}
    for key in metrics_list[0]:
        vals = [m[key] for m in metrics_list]
        result[f"{key}_mean"] = float(np.mean(vals))
        result[f"{key}_std"]  = float(np.std(vals))

    return result


def load_rl_model(agent_type: str, path: str):
    """Load an RL model from checkpoint, returning (model, needs_discrete_wrapper)."""
    cls_map = {"ppo": PPO, "dqn": DQN, "a2c": A2C}
    cls = cls_map.get(agent_type)
    if cls is None:
        return None, False

    try:
        model = cls.load(path, device="cpu")
        needs_discrete = agent_type == "dqn"
        return model, needs_discrete
    except Exception as e:
        print(f"  Warning: Could not load {agent_type} from {path}: {e}")
        return None, False


def main():
    print("=" * 90)
    print("  Proxima Auto-Scaler — Comprehensive Evaluation (mean ± std)")
    print("=" * 90)

    # ── Load RL models ───────────────────────────────────────────────────────
    rl_configs = {
        "PPO": ("ppo", "checkpoints/ppo_cloud_final.zip"),
        "DQN": ("dqn", "checkpoints/dqn_final.zip"),
        "A2C": ("a2c", "checkpoints/a2c_final.zip"),
    }

    rl_policies = {}
    for name, (agent_type, path) in rl_configs.items():
        model, needs_discrete = load_rl_model(agent_type, path)
        if model is not None:
            rl_policies[name] = {
                "fn": model.predict,
                "is_rl": True,
                "needs_discrete": needs_discrete,
            }
            print(f"  Loaded {name}: {path}")
        else:
            print(f"  Skipping {name}: checkpoint not found")

    # ── Heuristic baselines ──────────────────────────────────────────────────
    heuristic_policies = {
        "Threshold": {
            "fn": threshold_policy,
            "is_rl": False,
            "needs_discrete": False,
        },
        "Step Scaling": {
            "fn": step_scaling_policy,
            "is_rl": False,
            "needs_discrete": False,
        },
        "Target Tracking": {
            "fn": lambda o: target_tracking_policy(o, desired_target=0.60),
            "is_rl": False,
            "needs_discrete": False,
        },
    }

    all_policies = {**rl_policies, **heuristic_policies}

    difficulties = ["easy", "medium", "hard"]
    seeds = EVAL_SEEDS

    all_results = []
    flat_results = {}

    # ── Run evaluations ──────────────────────────────────────────────────────
    print(f"\n  Evaluating {len(all_policies)} policies × {len(difficulties)} "
          f"difficulties × {len(seeds)} seeds...\n")

    for pol_name, pol_cfg in all_policies.items():
        flat_results[pol_name] = {}
        for diff in difficulties:
            result = evaluate_policy_multi_seed(
                policy_name=pol_name,
                policy_fn=pol_cfg["fn"],
                is_rl=pol_cfg["is_rl"],
                difficulty=diff,
                seeds=seeds,
                needs_discrete_wrapper=pol_cfg["needs_discrete"],
            )
            flat_results[pol_name][diff] = result
            all_results.append({
                "policy": pol_name,
                "difficulty": diff,
                **result,
            })

    # ── Save JSON report ─────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    report_path = "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Report saved → {report_path}")

    # ── Print comparison table ───────────────────────────────────────────────
    header = (
        f"| {'Policy':<15} | {'Diff':<6} | {'Reward':>14} "
        f"| {'SLA Viol%':>14} | {'Waste':>14} | {'Lat p95':>14} |"
    )
    sep = f"|{'-'*17}|{'-'*8}|{'-'*16}|{'-'*16}|{'-'*16}|{'-'*16}|"

    print(f"\n{'='*96}")
    print(header)
    print(sep)

    for pol_name in all_policies:
        for diff in difficulties:
            m = flat_results[pol_name][diff]
            rew = f"{m['avg_reward_mean']:>6.1f}±{m['avg_reward_std']:<5.1f}"
            sla = f"{m['sla_violation_rate_mean']*100:>5.1f}±{m['sla_violation_rate_std']*100:<5.1f}%"
            wst = f"{m['capacity_waste_mean']:>6.1f}±{m['capacity_waste_std']:<5.1f}"
            lat = f"{m['latency_p95_mean']:>5.2f}±{m['latency_p95_std']:<5.2f}"
            print(f"| {pol_name:<15} | {diff:<6} | {rew:>14} | {sla:>14} | {wst:>14} | {lat:>14} |")
        print(sep)

    print(f"{'='*96}")

    # ── PPO vs best heuristic summary ────────────────────────────────────────
    if "PPO" in flat_results:
        for diff in difficulties:
            ppo_sla = flat_results["PPO"][diff]["sla_violation_rate_mean"]
            best_heur_name = min(
                heuristic_policies.keys(),
                key=lambda n: flat_results[n][diff]["sla_violation_rate_mean"],
            )
            heur_sla = flat_results[best_heur_name][diff]["sla_violation_rate_mean"]
            improvement = (heur_sla - ppo_sla) / max(1e-6, heur_sla) * 100
            print(
                f"  [{diff}] PPO SLA: {ppo_sla*100:.1f}% vs "
                f"{best_heur_name}: {heur_sla*100:.1f}% "
                f"({improvement:+.0f}% improvement)"
            )


if __name__ == "__main__":
    main()
