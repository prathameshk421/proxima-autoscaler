"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from dotenv import load_dotenv

# Auto-load values from .env for local runs.
load_dotenv()

# Kept to satisfy sample requirements for docker-image based runners.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
# Prefer injected API_KEY (validator proxy), fall back to HF_TOKEN for local use.
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

# MANDATORY: keep these as explicit getenv defaults per submission template.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = (os.getenv("ENV_URL") or "http://localhost:7860").rstrip("/")
# Default to llm so submission runs always exercise the provided LLM proxy.
POLICY_MODE = (os.getenv("POLICY_MODE") or "llm").strip().lower()
BENCHMARK = os.getenv("BENCHMARK") or "cloud-autoscaling"
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD") or "0.1")

TEMPERATURE = float(os.getenv("TEMPERATURE") or "0.2")
MAX_TOKENS = int(os.getenv("MAX_TOKENS") or "220")
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC") or "45")

REWARD_MIN = -5.0
REWARD_MAX = 2.0
TASK_MAX_STEPS = {
    "easy": 50,
    "medium": 75,
    "hard": 100,
}


def _parse_tasks() -> List[str]:
    raw = os.getenv("TASKS") or os.getenv("TASK_NAME") or "easy,medium,hard"
    tasks = [task.strip() for task in raw.split(",") if task.strip()]
    return tasks or ["easy", "medium", "hard"]


TASKS = _parse_tasks()

SYSTEM_PROMPT = textwrap.dedent(
    """
        You are a cloud autoscaling controller.
        Decide one action for the next control step.

        Action schema:
        {
            "mode": "step" | "target" | "signal",
            "scale_delta": integer,                    // used when mode="step", preferred range [-2, 2]
            "target_utilization": float,               // used when mode="target", range [0.1, 0.95]
            "signal_type": "INSUFFICIENT_CAPACITY" | null  // used when mode="signal"
        }

        Return ONLY valid JSON (single object), no markdown and no extra text.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    if error:
        error_val = str(error).replace("\n", "\\n").replace("\r", "\\r")
    else:
        error_val = "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(
    task: str,
    step: int,
    observation: Dict[str, Any],
    last_reward: float,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Task: {task}
        Step: {step}
        Current observation:
        {json.dumps(observation, indent=2, sort_keys=True)}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Choose the best next autoscaling action.
        Return strict JSON action only.
        """
    ).strip()


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _compact_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=True)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def _heuristic_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    util = float(observation.get("utilization", 0.0) or 0.0)
    queue = int(observation.get("queue_length", 0) or 0)
    max_servers = int(observation.get("max_servers", 1) or 1)
    active = int(observation.get("active_servers", 1) or 1)
    cooldown = int(observation.get("cooldown_remaining", 0) or 0)
    queue_ratio = float(observation.get("queue_ratio", 0.0) or 0.0)

    if cooldown > 0:
        return {"mode": "signal", "signal_type": None}

    if active >= max_servers and (util > 0.95 or queue > 0):
        return {"mode": "signal", "signal_type": "INSUFFICIENT_CAPACITY"}

    if util > 0.90 or queue_ratio > 0.35:
        return {"mode": "step", "scale_delta": 2}
    if util > 0.78 or queue_ratio > 0.15:
        return {"mode": "step", "scale_delta": 1}

    if util < 0.25 and queue == 0 and active > 2:
        return {"mode": "step", "scale_delta": -2}
    if util < 0.45 and queue == 0 and active > 1:
        return {"mode": "step", "scale_delta": -1}

    return {"mode": "step", "scale_delta": 0}


def _sanitize_action(raw_action: Dict[str, Any], fallback_obs: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(raw_action.get("mode", "step")).lower()

    if mode == "target":
        target = raw_action.get("target_utilization", 0.6)
        try:
            target_value = float(target)
        except (TypeError, ValueError):
            target_value = 0.6
        return {
            "mode": "target",
            "target_utilization": round(_clamp(target_value, 0.1, 0.95), 4),
        }

    if mode == "signal":
        signal = raw_action.get("signal_type")
        if signal != "INSUFFICIENT_CAPACITY":
            signal = None
        return {"mode": "signal", "signal_type": signal}

    if mode != "step":
        return _heuristic_action(fallback_obs)

    delta = raw_action.get("scale_delta", 0)
    try:
        delta_int = int(delta)
    except (TypeError, ValueError):
        delta_int = 0
    delta_int = max(-2, min(2, delta_int))
    return {"mode": "step", "scale_delta": delta_int}


def get_model_message(
    client: OpenAI,
    task: str,
    step: int,
    observation: Dict[str, Any],
    last_reward: float,
    history: List[str],
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(task, step, observation, last_reward, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        parsed = _extract_json_object(text)
        if parsed is None:
            return _heuristic_action(observation)
        return _sanitize_action(parsed, observation)
    except Exception as exc:
        # Keep stdout strictly on required lines; diagnostics go to stderr.
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return _heuristic_action(observation)


def _compute_score(task: str, rewards: List[float], episode_completed: bool) -> float:
    if not rewards or not episode_completed:
        return 0.0

    task_steps = TASK_MAX_STEPS.get(task, len(rewards))
    # Environment marks done at max_steps-1 (step counter starts at 0).
    effective_steps = max(1, task_steps - 1)
    total = sum(rewards)
    worst_possible = REWARD_MIN * effective_steps
    best_possible = REWARD_MAX * effective_steps
    score = (total - worst_possible) / (best_possible - worst_possible)
    return _clamp(score, 0.0, 1.0)


async def _reset_episode(http: httpx.AsyncClient, task: str) -> Dict[str, Any]:
    response = await http.post(f"{ENV_URL}/reset", params={"task": task})
    response.raise_for_status()
    payload = response.json()
    if "session_id" not in payload or "observation" not in payload:
        raise ValueError(f"Malformed /reset response: {payload}")
    return payload


async def _step_episode(
    http: httpx.AsyncClient,
    session_id: str,
    action: Dict[str, Any],
) -> Dict[str, Any]:
    response = await http.post(
        f"{ENV_URL}/step",
        params={"session_id": session_id},
        json=action,
    )
    response.raise_for_status()
    payload = response.json()
    if "observation" not in payload:
        raise ValueError(f"Malformed /step response: {payload}")
    return payload


async def run_task(task: str, client: Optional[OpenAI], http: httpx.AsyncClient) -> None:
    using_llm = POLICY_MODE == "llm" and client is not None
    model_label = MODEL_NAME if using_llm else "heuristic-rule"
    log_start(task=task, env=BENCHMARK, model=model_label)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    episode_completed = False

    try:
        reset_payload = await _reset_episode(http, task)
        session_id = str(reset_payload["session_id"])
        observation = reset_payload["observation"]
        last_reward = 0.0

        max_steps = TASK_MAX_STEPS.get(task, 100)

        for step in range(1, max_steps + 1):
            if bool(observation.get("done", False)):
                episode_completed = True
                break

            if using_llm and client is not None:
                action = get_model_message(
                    client=client,
                    task=task,
                    step=step,
                    observation=observation,
                    last_reward=last_reward,
                    history=history,
                )
            else:
                action = _heuristic_action(observation)

            step_payload = await _step_episode(http, session_id, action)
            step_obs = step_payload.get("observation", {})
            reward = float(step_payload.get("reward", 0.0) or 0.0)
            done = bool(step_payload.get("done", False))
            info = step_payload.get("info") or {}
            error = info.get("last_action_error") if isinstance(info, dict) else None

            rewards.append(reward)
            steps_taken = step
            action_str = _compact_json(action)

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            history.append(f"step={step} action={action_str} reward={reward:.2f}")
            last_reward = reward
            observation = step_obs

            if done:
                episode_completed = True
                break

        score = _compute_score(task, rewards, episode_completed=episode_completed)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] task={task} error={exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client: Optional[OpenAI] = None
    if POLICY_MODE == "llm":
        if API_KEY:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        else:
            raise RuntimeError(
                "POLICY_MODE=llm requires API_KEY (or HF_TOKEN) to be set."
            )

    timeout = httpx.Timeout(HTTP_TIMEOUT_SEC)
    async with httpx.AsyncClient(timeout=timeout) as http:
        for task in TASKS:
            await run_task(task, client, http)


if __name__ == "__main__":
    asyncio.run(main())
