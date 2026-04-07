"""Deterministic PPO inference script for one full cloud autoscaling episode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.agent.action_adapter import action_id_to_action
from src.agent.sb3_compat import patch_torch_load_for_zip_stream
from src.env.cloud_env import CloudAutoScalingEnv
from src.env.schemas import Observation


def _obs_to_vector(env: CloudAutoScalingEnv, obs: Observation) -> list[float]:
    """Convert typed observation to PPO-ready feature vector."""
    trend = env.workload_generator.get_recent_trend(env.timestep, window=8)
    return obs.to_vector() + trend


def run_inference(difficulty: str, seed: int, model_path: str) -> float:
    """Run exactly one full episode and return normalized score in [0,1]."""
    env = CloudAutoScalingEnv(difficulty=difficulty, seed=seed)
    patch_torch_load_for_zip_stream()
    model = PPO.load(model_path, device="cpu")

    obs, _ = env.reset(seed=seed, options={"difficulty": difficulty})
    done = False
    t = 0
    total_reward = 0.0

    while not done:
        obs_vec = _obs_to_vector(env, obs)
        action_id, _ = model.predict(obs_vec, deterministic=True)
        action = action_id_to_action(int(action_id))

        obs, reward, done, _ = env.step(action)
        total_reward += float(reward)

        print(f"[STEP {t}] obs={obs.model_dump()} action={action.to_dict_compact()} reward={reward:.4f}")
        t += 1

    min_possible = -200 * 1.5
    max_possible = 200 * 0.5
    normalized_score = (total_reward - min_possible) / (max_possible - min_possible)
    normalized_score = max(0.0, min(1.0, normalized_score))

    print(f"SCORE={normalized_score:.4f}")
    return normalized_score


def main() -> None:
    """CLI entrypoint for deterministic single-episode PPO inference."""
    parser = argparse.ArgumentParser(description="Run PPO inference for one episode")
    parser.add_argument("--difficulty", type=str, default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()

    run_inference(difficulty=args.difficulty, seed=args.seed, model_path=args.model_path)


if __name__ == "__main__":
    main()
