"""OpenAI-compatible baseline inference runner for cloud autoscaling tasks.

This script evaluates one agent across easy/medium/hard tasks and fixed seeds,
emitting required logs in this format:

[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ is None or __package__ == "":
	project_root = Path(__file__).resolve().parent
	if str(project_root) not in sys.path:
		sys.path.insert(0, str(project_root))

from src.baselines.heuristics import step_scaling_policy, target_tracking_policy, threshold_policy
from src.env.cloud_env import CloudAutoScalingEnv
from src.env.grader import grade_episode
from src.env.schemas import Action, Observation


def _to_bool_text(value: bool) -> str:
	"""Convert bool to lower-case text expected by benchmark logs."""
	return "true" if value else "false"


def log_start(task: str, env: str, model: str) -> None:
	"""Print standardized start-of-episode log line."""
	print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
	"""Print standardized per-step log line."""
	print(
		f"[STEP] step={step} action={action} reward={reward:.2f} "
		f"done={_to_bool_text(done)} error={error or 'null'}",
		flush=True,
	)


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
	"""Print standardized end-of-episode log line."""
	rewards_text = ",".join(f"{x:.2f}" for x in rewards)
	print(
		f"[END] success={_to_bool_text(success)} steps={steps} score={score:.3f} rewards={rewards_text}",
		flush=True,
	)


def _safe_action_text(action: Action) -> str:
	"""Return a compact JSON string for action logging."""
	return json.dumps(action.to_dict_compact(), separators=(",", ":"), sort_keys=True)


def _build_llm_selector(model_name: str, api_base_url: str, api_key: str) -> Callable[[Observation], tuple[Action, str | None]]:
	"""Create an LLM-backed action selector using OpenAI-compatible API."""
	try:
		from openai import OpenAI
	except Exception as exc:  # pragma: no cover - import depends on runtime env
		raise RuntimeError("Missing dependency 'openai'. Install with: pip install openai") from exc

	client = OpenAI(base_url=api_base_url, api_key=api_key)

	system_prompt = (
		"You control a cloud autoscaling environment. Return ONLY valid JSON with keys for an action. "
		"Allowed actions: "
		"{\"mode\":\"step\",\"scale_delta\":-2|-1|0|1|2}, "
		"{\"mode\":\"target\",\"target_utilization\":0.4|0.5|0.6|0.7}, "
		"{\"mode\":\"signal\",\"signal_type\":\"INSUFFICIENT_CAPACITY\"}."
	)

	def _selector(observation: Observation) -> tuple[Action, str | None]:
		payload = observation.model_dump()
		user_prompt = (
			"Observation: "
			+ json.dumps(payload, separators=(",", ":"), sort_keys=True)
			+ "\nChoose one action JSON only."
		)

		try:
			response = client.chat.completions.create(
				model=model_name,
				temperature=0,
				messages=[
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": user_prompt},
				],
				response_format={"type": "json_object"},
			)
			content = response.choices[0].message.content or "{}"
			parsed = json.loads(content)
			return Action(**parsed), None
		except Exception as exc:
			# Keep episodes progressing deterministically even if LLM output is invalid.
			return Action(mode="step", scale_delta=0), str(exc)

	return _selector


def _build_rule_selector(agent: str) -> Callable[[Observation], tuple[Action, str | None]]:
	"""Create deterministic baseline selector."""
	if agent == "threshold":
		return lambda obs: (threshold_policy(obs), None)
	if agent == "step":
		return lambda obs: (step_scaling_policy(obs), None)
	if agent == "target_tracking":
		return lambda obs: (target_tracking_policy(obs, desired_target=0.6), None)
	raise ValueError(f"Unsupported deterministic agent: {agent}")


def _score_episode(
	env: CloudAutoScalingEnv,
	rewards: list[float],
	latency_history: list[float],
	server_history: list[float],
	saturation_events: int,
	difficulty: str,
) -> float:
	"""Compute deterministic normalized episode score in [0,1]."""
	steps = max(1, len(rewards))
	metrics = {
		"steps": float(steps),
		"avg_reward": float(sum(rewards) / steps),
		"avg_latency": float(sum(latency_history) / max(1, len(latency_history))),
		"sla_threshold": float(env.queue_model.base_latency_ms * 2.5),
		"avg_servers": float(sum(server_history) / max(1, len(server_history))),
		"min_servers": float(env.min_servers),
		"max_servers": float(env.max_servers),
		"oscillation_count": float(env.oscillation_count),
		"signal_tp": float(env.signal_tp),
		"signal_fp": float(env.signal_fp),
		"signal_fn": float(env.signal_fn),
		"saturation_events": float(saturation_events),
	}
	return float(grade_episode(metrics, difficulty=difficulty))


def run_episode(
	difficulty: str,
	seed: int,
	benchmark_name: str,
	model_name: str,
	selector: Callable[[Observation], tuple[Action, str | None]],
	success_threshold: float,
) -> dict[str, Any]:
	"""Run one deterministic episode and emit required benchmark logs."""
	env = CloudAutoScalingEnv(difficulty=difficulty, seed=seed)
	observation, _ = env.reset(seed=seed, options={"difficulty": difficulty})

	log_start(task=difficulty, env=benchmark_name, model=model_name)

	done = False
	step_index = 0
	rewards: list[float] = []
	latency_history: list[float] = []
	server_history: list[float] = []
	saturation_events = 0

	while not done:
		step_index += 1
		action, error_text = selector(observation)

		observation, reward, done, info = env.step(action)
		reward_value = float(reward)
		rewards.append(reward_value)
		latency_history.append(float(info.get("estimated_latency", 0.0)))
		server_history.append(float(info.get("current_servers", env.current_servers)))
		if bool(info.get("saturated", False)):
			saturation_events += 1

		log_step(
			step=step_index,
			action=_safe_action_text(action),
			reward=reward_value,
			done=bool(done),
			error=error_text,
		)

	score = _score_episode(
		env=env,
		rewards=rewards,
		latency_history=latency_history,
		server_history=server_history,
		saturation_events=saturation_events,
		difficulty=difficulty,
	)
	success = score >= success_threshold
	log_end(success=success, steps=step_index, score=score, rewards=rewards)

	return {
		"difficulty": difficulty,
		"seed": seed,
		"steps": step_index,
		"score": score,
		"success": success,
		"avg_reward": float(sum(rewards) / max(1, len(rewards))),
	}


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for baseline inference runs."""
	parser = argparse.ArgumentParser(description="OpenAI-compatible baseline inference runner")
	parser.add_argument("--agent", choices=["llm", "threshold", "step", "target_tracking"], default="llm")
	parser.add_argument("--tasks", nargs="+", default=["easy", "medium", "hard"])
	parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
	parser.add_argument("--benchmark", type=str, default=os.getenv("MY_ENV_V4_BENCHMARK", "openenv-cloud-autoscaling"))
	parser.add_argument("--api-base-url", type=str, default=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"))
	parser.add_argument("--model-name", type=str, default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"))
	parser.add_argument("--success-threshold", type=float, default=0.60)
	parser.add_argument("--output", type=str, default="baseline_inference_report.json")
	return parser.parse_args()


def main() -> None:
	"""CLI entrypoint for deterministic baseline inference across tasks and seeds."""
	args = parse_args()

	if args.agent == "llm":
		api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
		if not api_key:
			raise RuntimeError("Set HF_TOKEN or OPENAI_API_KEY to run --agent llm")
		selector = _build_llm_selector(
			model_name=args.model_name,
			api_base_url=args.api_base_url,
			api_key=api_key,
		)
		effective_model_name = args.model_name
	else:
		selector = _build_rule_selector(args.agent)
		effective_model_name = args.agent

	runs: list[dict[str, Any]] = []
	for task in args.tasks:
		if task not in {"easy", "medium", "hard"}:
			raise ValueError(f"Invalid task '{task}'. Expected easy/medium/hard")
		for seed in args.seeds:
			run_result = run_episode(
				difficulty=task,
				seed=int(seed),
				benchmark_name=args.benchmark,
				model_name=effective_model_name,
				selector=selector,
				success_threshold=float(args.success_threshold),
			)
			runs.append(run_result)

	mean_score = float(sum(item["score"] for item in runs) / max(1, len(runs)))
	output_payload = {
		"agent": args.agent,
		"model_name": effective_model_name,
		"benchmark": args.benchmark,
		"tasks": args.tasks,
		"seeds": args.seeds,
		"mean_score": mean_score,
		"runs": runs,
	}
	Path(args.output).write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
	print(f"Saved report to {args.output}", flush=True)


if __name__ == "__main__":
	main()
