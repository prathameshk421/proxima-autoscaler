"""Curriculum PPO training script for cloud autoscaling environment."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

if __package__ is None or __package__ == "":
	project_root = Path(__file__).resolve().parents[2]
	if str(project_root) not in sys.path:
		sys.path.insert(0, str(project_root))

from src.agent.action_adapter import action_id_to_action, num_actions
from src.agent.ppo_agent import load_ppo_model, make_ppo_model
from src.baselines.heuristics import run_baseline, threshold_policy
from src.env.cloud_env import CloudAutoScalingEnv
from src.env.schemas import Observation


class SB3CloudAutoScalingEnv(gym.Env[np.ndarray, int]):
	"""Gymnasium wrapper that exposes vector observations for SB3."""

	metadata = {"render_modes": []}

	def __init__(self, difficulty: str, seed: int, episode_length: int = 200) -> None:
		super().__init__()
		self.base_env = CloudAutoScalingEnv(
			difficulty=difficulty,
			seed=seed,
			episode_length=episode_length,
		)
		self.action_space = spaces.Discrete(num_actions())
		self.observation_space = spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(22,),
			dtype=np.float32,
		)

	def _vectorize_observation(self, obs: Observation) -> np.ndarray:
		"""Convert typed observation into flat vector with trend suffix."""
		trend_history = self.base_env.workload_generator.get_recent_trend(
			self.base_env.timestep,
			window=8,
		)
		vector = np.array(obs.to_vector() + trend_history, dtype=np.float32)
		return vector

	def reset(
		self,
		*,
		seed: int | None = None,
		options: dict[str, Any] | None = None,
	) -> tuple[np.ndarray, dict[str, Any]]:
		obs, info = self.base_env.reset(seed=seed, options=options)
		return self._vectorize_observation(obs), info

	def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
		structured_action = action_id_to_action(int(action))
		obs, reward, done, info = self.base_env.step(structured_action)
		return self._vectorize_observation(obs), float(reward), bool(done), False, info


def set_global_determinism(seed: int) -> None:
	"""Set deterministic seeds for Python, NumPy, and PyTorch."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.set_num_threads(1)
	try:
		torch.use_deterministic_algorithms(True)
	except Exception:
		pass


def _make_train_env(difficulty: str, seed: int) -> Monitor:
	"""Create monitored SB3 training environment."""
	return Monitor(SB3CloudAutoScalingEnv(difficulty=difficulty, seed=seed))


def _obs_to_vector_for_model(env: CloudAutoScalingEnv, obs: Observation) -> np.ndarray:
	"""Vectorize observation for PPO inference outside SB3 wrapper."""
	trend_history = env.workload_generator.get_recent_trend(env.timestep, window=8)
	return np.array(obs.to_vector() + trend_history, dtype=np.float32)


def _evaluate_ppo_avg_reward(model_path: str, difficulty: str, seeds: list[int]) -> float:
	"""Evaluate trained PPO model mean episode reward over deterministic seeds."""
	rewards: list[float] = []
	for seed in seeds:
		env = CloudAutoScalingEnv(difficulty=difficulty, seed=seed)
		model = load_ppo_model(model_path)
		obs, _ = env.reset(seed=seed)

		total_reward = 0.0
		done = False
		while not done:
			obs_vec = _obs_to_vector_for_model(env, obs)
			action_id, _ = model.predict(obs_vec, deterministic=True)
			obs, reward, done, _ = env.step(action_id_to_action(int(action_id)))
			total_reward += float(reward)

		rewards.append(total_reward)

	return float(np.mean(rewards))


def train_curriculum(seed: int = 42) -> None:
	"""Train PPO in easy->medium->hard curriculum and save checkpoints."""
	set_global_determinism(seed)

	os.makedirs("checkpoints", exist_ok=True)
	os.makedirs("logs", exist_ok=True)

	phases: list[tuple[str, int, str]] = [
		("easy", 50_000, "checkpoints/ppo_easy.zip"),
		("medium", 100_000, "checkpoints/ppo_medium.zip"),
		("hard", 150_000, "checkpoints/ppo_hard.zip"),
	]

	model = None
	previous_checkpoint: str | None = None

	for phase_index, (difficulty, timesteps, checkpoint_path) in enumerate(phases):
		print(f"[TRAIN] phase={difficulty} timesteps={timesteps}")
		train_env = _make_train_env(difficulty=difficulty, seed=seed)
		eval_env = _make_train_env(difficulty=difficulty, seed=seed + 1000)

		if model is None:
			model = make_ppo_model(train_env, tensorboard_log="logs")
		else:
			# Prefer in-memory continuation to avoid serialization edge cases across phases.
			model.set_env(train_env)

			# Optional fallback to disk load when resuming fresh process state.
			if previous_checkpoint is not None and not os.path.exists(previous_checkpoint):
				raise RuntimeError("Missing previous checkpoint for curriculum continuation")

		eval_callback = EvalCallback(
			eval_env,
			best_model_save_path=f"logs/best_{difficulty}",
			log_path=f"logs/eval_{difficulty}",
			eval_freq=max(5_000, timesteps // 10),
			n_eval_episodes=3,
			deterministic=True,
			render=False,
		)

		model.learn(
			total_timesteps=timesteps,
			reset_num_timesteps=(phase_index == 0),
			callback=eval_callback,
			progress_bar=False,
		)
		model.save(checkpoint_path)
		previous_checkpoint = checkpoint_path

		train_env.close()
		eval_env.close()

	final_checkpoint = "checkpoints/ppo_cloud_final.zip"
	if model is None:
		raise RuntimeError("Training failed to initialize PPO model")
	model.save(final_checkpoint)
	print(f"[TRAIN] saved final model to {final_checkpoint}")

	seeds = [42, 123, 456]
	ppo_reward = _evaluate_ppo_avg_reward(final_checkpoint, difficulty="hard", seeds=seeds)

	threshold_rewards: list[float] = []
	for eval_seed in seeds:
		baseline_env = CloudAutoScalingEnv(difficulty="hard", seed=eval_seed)
		metrics = run_baseline(baseline_env, threshold_policy, n_episodes=1)
		threshold_rewards.append(float(metrics["avg_reward"]))

	threshold_reward = float(np.mean(threshold_rewards))
	print(
		"[COMPARE] hard difficulty mean avg_reward: "
		f"PPO={ppo_reward:.4f}, Threshold={threshold_reward:.4f}"
	)


def main() -> None:
	"""CLI entrypoint for curriculum PPO training."""
	parser = argparse.ArgumentParser(description="Train PPO with easy->medium->hard curriculum")
	parser.add_argument("--seed", type=int, default=42, help="Global deterministic seed")
	args = parser.parse_args()

	train_curriculum(seed=args.seed)


if __name__ == "__main__":
	main()
