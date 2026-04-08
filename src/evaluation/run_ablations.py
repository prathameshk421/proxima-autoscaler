"""Ablation study runner for Proxima Auto-Scaler.

Trains ablation variants, evaluates each across all difficulties and
multiple seeds, then outputs a comparison table with mean ± std.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from typing import Dict, List, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

import gymnasium as gym
from torch import nn

from src.env.cloud_env import CloudAutoScalingEnv
from src.agent.networks import CustomFeatureExtractor
from src.evaluation.ablation_configs import AblationConfig, ALL_ABLATIONS


# ── Flat MLP extractor (used by no_gru ablation) ────────────────────────────

class FlatFeatureExtractor(BaseFeaturesExtractor):
    """Simple MLP feature extractor with no temporal reasoning.

    Processes the entire observation vector through a 2-layer MLP,
    treating all features (including the trend window) as independent
    scalar inputs. This is the ablation control for the GRU.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        total_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


# ── Modified environment for ablation overrides ──────────────────────────────

class AblationEnv(CloudAutoScalingEnv):
    """CloudAutoScalingEnv with configurable reward weights and boot delay."""

    def __init__(
        self,
        difficulty: str = "easy",
        seed: int = 42,
        sla_penalty: float = 12.0,
        waste_penalty: float = 0.008,
        override_boot_delay: Optional[int] = None,
    ):
        self._sla_penalty = sla_penalty
        self._waste_penalty = waste_penalty
        self._override_boot_delay = override_boot_delay
        super().__init__(difficulty=difficulty, seed=seed)

    def _configure_difficulty(self, difficulty: str) -> None:
        super()._configure_difficulty(difficulty)
        if self._override_boot_delay is not None:
            self.boot_delay = self._override_boot_delay

    def step(self, action: np.ndarray):
        step_idx = int(np.clip(action[0], 0, 4))
        cap_signal = int(np.clip(action[2], 0, 1))
        step_scaling = step_idx - 2

        if self.cooldown_remaining <= 0 and step_scaling != 0:
            if step_scaling > 0:
                add = min(step_scaling, self.c_max - self.current_capacity)
                if add > 0:
                    self._boot_pipeline.append((add, self.boot_delay))
            else:
                remove = min(abs(step_scaling), self.current_capacity - 1)
                self.current_capacity -= remove
            self.cooldown_remaining = self.cooldown_period
        else:
            self.cooldown_remaining = max(0, self.cooldown_remaining - 1)

        activated = 0
        next_boot = []
        for servers, remaining in self._boot_pipeline:
            remaining -= 1
            if remaining <= 0:
                activated += servers
            else:
                next_boot.append((servers, remaining))
        self._boot_pipeline = next_boot
        self.current_capacity = min(self.c_max, self.current_capacity + activated)

        self._lam = float(self._demand_profile(self.current_step))
        service_capacity = self.current_capacity * self.mu
        served = min(self.queue_length + self._lam, service_capacity)
        self.queue_length = max(0.0, self.queue_length + self._lam - served)

        self.latency_p95 = self._estimate_latency(
            self._lam, self.current_capacity, self.queue_length
        )
        util = self._utilization(self._lam, self.current_capacity)

        self.trend_window.pop(0)
        self.trend_window.append(self._lam)

        sla_breach = max(0.0, self.latency_p95 - self.T_sla)
        reward = 0.5 - self._sla_penalty * sla_breach

        excess_capacity = max(0.0, self.current_capacity * self.mu - self._lam)
        reward -= self._waste_penalty * (excess_capacity ** 2)
        reward -= 0.04 * abs(step_scaling)
        reward -= 0.15 * max(0.0, util - 1.0)

        if cap_signal == 1 and util < 0.9:
            reward -= 0.05

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "latency_p95": float(self.latency_p95),
            "lam": float(self._lam),
            "current_capacity": int(self.current_capacity),
            "utilization": float(util),
        }
        return self._build_obs(), float(reward), terminated, truncated, info


# ── Builder ──────────────────────────────────────────────────────────────────

def build_ablation_ppo(env, config: AblationConfig) -> PPO:
    """Build PPO agent with feature extractor chosen by ablation config."""
    if config.use_gru:
        extractor_class = CustomFeatureExtractor
    else:
        extractor_class = FlatFeatureExtractor

    policy_kwargs = dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=dict(features_dim=64),
    )

    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=2e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.25,
        ent_coef=0.02,
        vf_coef=0.25,
        max_grad_norm=0.3,
        policy_kwargs=policy_kwargs,
        device="cpu",
        verbose=1,
    )


def make_ablation_env(config: AblationConfig, difficulty: str, seed: int):
    """Factory for AblationEnv with the right overrides."""
    def _thunk():
        env = AblationEnv(
            difficulty=difficulty,
            seed=seed,
            sla_penalty=config.sla_penalty_weight,
            waste_penalty=config.waste_penalty_weight,
            override_boot_delay=config.override_boot_delay,
        )
        return Monitor(env)
    return _thunk


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_ablation(
    model: PPO,
    config: AblationConfig,
    difficulty: str,
    seeds: List[int],
) -> Dict[str, float]:
    """Evaluate a trained ablation model across multiple seeds."""
    rewards, sla_rates, wastes = [], [], []

    for seed in seeds:
        env = AblationEnv(
            difficulty=difficulty,
            seed=seed,
            sla_penalty=config.sla_penalty_weight,
            waste_penalty=config.waste_penalty_weight,
            override_boot_delay=config.override_boot_delay,
        )
        obs, _ = env.reset(seed=seed)
        ep_reward = 0.0
        sla_viol = 0
        total_waste = 0.0

        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if info["latency_p95"] > env.T_sla:
                sla_viol += 1
            total_waste += max(0.0, env.current_capacity * env.mu - env._lam)
            if terminated or truncated:
                break

        rewards.append(ep_reward)
        sla_rates.append(sla_viol / 200.0)
        wastes.append(total_waste / 200.0)

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std":  float(np.std(rewards)),
        "sla_mean":    float(np.mean(sla_rates)),
        "sla_std":     float(np.std(sla_rates)),
        "waste_mean":  float(np.mean(wastes)),
        "waste_std":   float(np.std(wastes)),
    }


# ── Main runner ──────────────────────────────────────────────────────────────

def run_single_ablation(
    config: AblationConfig,
    seed: int,
    log_dir: str,
    eval_seeds: List[int],
) -> Dict:
    """Train and evaluate a single ablation variant."""
    print(f"\n{'#'*60}")
    print(f"  Ablation: {config.name}")
    print(f"  {config.description}")
    print(f"{'#'*60}")

    model = None

    for phase_idx, (difficulty, num_timesteps) in enumerate(config.curriculum):
        print(f"\n  Phase {phase_idx+1}: '{difficulty}' ({num_timesteps:,} steps)")

        train_env = DummyVecEnv([make_ablation_env(config, difficulty, seed)])
        eval_env  = DummyVecEnv([make_ablation_env(config, difficulty, 123)])

        if phase_idx == 0:
            model = build_ablation_ppo(train_env, config)
        else:
            model.set_env(train_env)

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=f"checkpoints/ablation_{config.tag}/best_{difficulty}",
            log_path=log_dir,
            eval_freq=5_000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )

        model.learn(
            total_timesteps=num_timesteps,
            callback=[eval_cb],
            reset_num_timesteps=False,
        )

    final_path = f"checkpoints/ablation_{config.tag}_final.zip"
    best_hard = f"checkpoints/ablation_{config.tag}/best_hard/best_model.zip"
    last_diff = config.curriculum[-1][0]
    best_last = f"checkpoints/ablation_{config.tag}/best_{last_diff}/best_model.zip"

    if os.path.exists(best_hard):
        shutil.copy2(best_hard, final_path)
    elif os.path.exists(best_last):
        shutil.copy2(best_last, final_path)
    else:
        model.save(f"checkpoints/ablation_{config.tag}_final")

    print(f"\n  Evaluating {config.name} across difficulties...")
    results = {}
    for diff in ["easy", "medium", "hard"]:
        results[diff] = evaluate_ablation(model, config, diff, eval_seeds)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Ablation Studies")
    parser.add_argument("--ablations", nargs="+",
                        default=list(ALL_ABLATIONS.keys()),
                        choices=list(ALL_ABLATIONS.keys()),
                        help="Which ablations to run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="./logs/ablations")
    parser.add_argument("--n-eval-seeds", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    eval_seeds = [42, 123, 456, 789, 1024, 2048, 3000, 4096, 5555, 7777]
    eval_seeds = eval_seeds[: args.n_eval_seeds]

    all_results = {}

    for abl_name in args.ablations:
        config = ALL_ABLATIONS[abl_name]
        results = run_single_ablation(config, args.seed, args.log_dir, eval_seeds)
        all_results[abl_name] = {
            "description": config.description,
            "results": results,
        }

    report_path = os.path.join(args.log_dir, "ablation_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nReport saved → {report_path}")

    # ── Print comparison table ───────────────────────────────────────────────
    print(f"\n{'='*95}")
    print(f"| {'Variant':<20} | {'Diff':<6} | {'Reward':>12} | {'SLA Viol%':>12} | {'Waste':>12} |")
    print(f"|{'-'*22}|{'-'*8}|{'-'*14}|{'-'*14}|{'-'*14}|")

    for abl_name in args.ablations:
        data = all_results[abl_name]["results"]
        for diff in ["easy", "medium", "hard"]:
            m = data[diff]
            print(
                f"| {ALL_ABLATIONS[abl_name].name:<20} "
                f"| {diff:<6} "
                f"| {m['reward_mean']:>6.1f}±{m['reward_std']:<5.1f} "
                f"| {m['sla_mean']*100:>5.1f}±{m['sla_std']*100:<5.1f}% "
                f"| {m['waste_mean']:>6.1f}±{m['waste_std']:<5.1f} |"
            )
        print(f"|{'-'*22}|{'-'*8}|{'-'*14}|{'-'*14}|{'-'*14}|")

    print(f"{'='*95}")


if __name__ == "__main__":
    main()
