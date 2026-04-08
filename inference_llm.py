"""OpenEnv-compliant LLM inference script for Proxima Auto-Scaler.

Uses the OpenAI-compatible API client to have an LLM make cloud scaling
decisions through the environment's step()/reset()/state() API.

Required environment variables:
    API_BASE_URL   – Base URL for the LLM API (default: HF router)
    MODEL_NAME     – Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       – Hugging Face token (or API_KEY / OPENAI_API_KEY)
"""

import json
import os
import re
import sys
from typing import List

import numpy as np
from openai import OpenAI

from src.env.cloud_env import CloudAutoScalingEnv
from src.env.schemas import Action, Observation, Reward
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or ""
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
MAX_STEPS = 200
TEMPERATURE = 0.2
MAX_TOKENS = 256

SYSTEM_PROMPT = """\
You are a cloud auto-scaling agent. Your job is to decide how many servers to
add/remove, what target utilization to set, and whether to signal insufficient
capacity.

At each step you receive an observation of the current system state and must
respond with a JSON action object with exactly these fields:
{
  "step_scaling": <int from -2 to 2>,
  "target_utilization": <float: 0.4, 0.5, 0.6, or 0.7>,
  "capacity_signal": <int: 0 or 1>
}

Guidelines:
- Positive step_scaling adds servers; negative removes them.
- Scale up proactively when utilization trends upward or queue grows.
- Scale down carefully when utilization is low to save cost.
- Set capacity_signal=1 ONLY when at max capacity AND utilization ≥ 0.95.
- Minimise SLA breaches (p95 latency > 2.0s) while avoiding waste.
- Respond ONLY with the JSON object, no extra text.
"""


def _fmt_obs(obs: Observation) -> str:
    """Format observation as a concise prompt string."""
    return (
        f"Servers: {obs.current_capacity}/{obs.max_capacity} | "
        f"Util: {obs.avg_utilization:.2f} | "
        f"Queue: {obs.queue_length:.1f} | "
        f"Demand: {obs.demand_rate:.1f} | "
        f"ServiceRate: {obs.service_rate:.1f} | "
        f"P95 Latency: {obs.p95_latency:.3f}s | "
        f"Cooldown: {obs.cooldown_remaining} | "
        f"Booting: {obs.booting_servers} | "
        f"Trend(last 8): {[round(x, 1) for x in obs.trend]}"
    )


def _parse_action(text: str) -> Action:
    """Extract JSON action from LLM response, with fallback defaults."""
    # Try to find JSON in the response
    json_match = re.search(r'\{[^}]+\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return Action(
                step_scaling=max(-2, min(2, int(data.get("step_scaling", 0)))),
                target_utilization=float(data.get("target_utilization", 0.5)),
                capacity_signal=max(0, min(1, int(data.get("capacity_signal", 0)))),
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    # Fallback: no-op
    return Action(step_scaling=0, target_utilization=0.5, capacity_signal=0)


def _action_to_array(action: Action) -> np.ndarray:
    """Convert Action pydantic model to gymnasium MultiDiscrete array."""
    t_util = action.target_utilization
    if t_util <= 0.45:
        t_idx = 0
    elif t_util <= 0.55:
        t_idx = 1
    elif t_util <= 0.65:
        t_idx = 2
    else:
        t_idx = 3
    return np.array([
        np.clip(action.step_scaling + 2, 0, 4),
        t_idx,
        np.clip(action.capacity_signal, 0, 1),
    ])


def _normalize_score(total_reward: float) -> float:
    """Normalize cumulative reward to [0.0, 1.0] per the grader spec."""
    min_possible = -200 * 1.5
    max_possible = 200 * 0.5
    return float(np.clip(
        (total_reward - min_possible) / (max_possible - min_possible),
        0.0, 1.0,
    ))


# ── Logging helpers matching the required inference script format ──────────

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str | None):
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END]   success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Main inference loop ───────────────────────────────────────────────────

def run_task(difficulty: str, seed: int = 42) -> float:
    """Run one episode on the given difficulty, return normalised score."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CloudAutoScalingEnv(difficulty=difficulty, seed=seed)

    log_start(
        task=f"cloud_autoscaling_{difficulty}",
        env="proxima-autoscaler",
        model=MODEL_NAME,
    )

    obs_arr, _ = env.reset(seed=seed)
    rewards: List[float] = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step in range(1, MAX_STEPS + 1):
        obs_pyd = env.state()
        user_msg = _fmt_obs(obs_pyd)
        messages.append({"role": "user", "content": user_msg})

        error = None
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            reply = completion.choices[0].message.content or ""
            action_pyd = _parse_action(reply)
        except Exception as exc:
            error = str(exc)
            reply = ""
            action_pyd = Action(step_scaling=0, target_utilization=0.5, capacity_signal=0)

        messages.append({"role": "assistant", "content": reply})
        # Keep context window manageable
        if len(messages) > 20:
            messages = messages[:1] + messages[-18:]

        action_arr = _action_to_array(action_pyd)
        obs_arr, reward, terminated, truncated, info = env.step(action_arr)
        rewards.append(float(reward))

        done = terminated or truncated
        action_str = action_pyd.model_dump_json()
        log_step(step, action_str, float(reward), done, error)

        if done:
            break

    score = _normalize_score(sum(rewards))
    success = score >= 0.5
    log_end(success, len(rewards), score, rewards)
    return score


def main():
    """Run inference across all three task difficulties."""
    difficulties = ["easy", "medium", "hard"]
    seed = int(os.getenv("SEED", "42"))

    scores = {}
    for diff in difficulties:
        print(f"\n{'='*60}")
        print(f"  Running task: {diff} (seed={seed})")
        print(f"{'='*60}\n")
        scores[diff] = run_task(diff, seed=seed)

    print(f"\n{'='*60}")
    print("  FINAL SCORES")
    print(f"{'='*60}")
    for diff, s in scores.items():
        print(f"  {diff:>8s}: {s:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'average':>8s}: {avg:.4f}")


if __name__ == "__main__":
    main()
