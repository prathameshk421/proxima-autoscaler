#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Virtual Environment Setup (recommended for reproducible local runs)
#
# 1) Create a virtual environment in the project root:
#    python3 -m venv .venv
#
# 2) Activate it:
#    macOS/Linux: source .venv/bin/activate
#    Windows (PowerShell): .venv\Scripts\Activate.ps1
#    Windows (cmd.exe): .venv\Scripts\activate.bat
#
# 3) Install dependencies:
#    pip install -r requirements.txt
#
# 4) Run this script:
#    python run_inference.py --local --task easy
#
# 5) Deactivate when done:
#    deactivate
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console

# Load .env values for local development convenience.
load_dotenv()

DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
LOCAL_SERVER_URL = "http://127.0.0.1:7860"
DEFAULT_TIMEOUT_SEC = 25.0
DEFAULT_RETRIES = 3
REWARD_MIN = -5.0
REWARD_MAX = 2.0
SUCCESS_THRESHOLD = 0.1

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a cloud autoscaling controller.
    Return exactly one JSON object for the next action.

    Action schema:
    {
      "mode": "step" | "target" | "signal",
      "scale_delta": integer,
      "target_utilization": float,
      "signal_type": "INSUFFICIENT_CAPACITY" | null
    }

    Constraints:
    - Use step mode for direct up/down scaling.
    - Keep scale_delta in [-2, 2].
    - Use signal mode only when maxed out and overloaded.
    - Return JSON only (no markdown, no extra text).
    """
).strip()


def load_openenv_metadata(config_path: Path) -> Tuple[str, List[str]]:
    """Read environment name and task names from openenv.yaml with safe fallbacks."""
    default_env_name = "cloud-autoscaling"
    default_tasks = ["easy", "medium", "hard"]

    if not config_path.exists():
        return default_env_name, default_tasks

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        env_name = str(payload.get("name") or default_env_name)

        tasks: List[str] = []
        for item in payload.get("tasks", []):
            if isinstance(item, dict) and item.get("name"):
                tasks.append(str(item["name"]))

        if not tasks:
            tasks = default_tasks

        return env_name, tasks
    except Exception:
        return default_env_name, default_tasks


def parse_args(task_choices: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one OpenEnv inference episode against HF Space or local server."
    )
    parser.add_argument(
        "--url",
        "--space-url",
        dest="url",
        type=str,
        default=None,
        help="Hugging Face Space URL. Required unless --local is used.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=task_choices,
        default=task_choices[0],
        help="Task name to execute.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name for LLM policy calls.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="Maximum steps to execute before stopping.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=f"Run against local server ({LOCAL_SERVER_URL}).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=DEFAULT_TIMEOUT_SEC,
        help="HTTP and model timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Retry count for transient failures.",
    )
    args = parser.parse_args()

    if not args.local and not args.url:
        parser.error("--url/--space-url is required unless --local is set.")

    if args.max_steps <= 0:
        parser.error("--max-steps must be > 0.")

    if args.timeout_sec <= 0:
        parser.error("--timeout-sec must be > 0.")

    if args.retries < 0:
        parser.error("--retries must be >= 0.")

    return args


def bool_lower(value: bool) -> str:
    return "true" if value else "false"


def compact_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=True)


def sanitize_error(error_value: Optional[Any]) -> str:
    if error_value in (None, "", "null"):
        return "null"
    return str(error_value).replace("\n", "\\n").replace("\r", "\\r")


def emit_start(log_lines: List[str], task: str, env_name: str, model: str) -> None:
    line = f"[START] task={task} env={env_name} model={model}"
    print(line, flush=True)
    log_lines.append(line)


def emit_step(
    log_lines: List[str],
    step: int,
    action: Dict[str, Any],
    reward: float,
    done: bool,
    error: Optional[Any],
) -> None:
    action_text = compact_json(action)
    line = (
        "[STEP] "
        f"step={step} "
        f"action={action_text} "
        f"reward={reward:.2f} "
        f"done={bool_lower(done)} "
        f"error={sanitize_error(error)}"
    )
    print(line, flush=True)
    log_lines.append(line)


def emit_end(log_lines: List[str], success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
    line = (
        "[END]   "
        f"success={bool_lower(success)} "
        f"steps={steps} "
        f"score={score:.3f} "
        f"rewards={rewards_text}"
    )
    print(line, flush=True)
    log_lines.append(line)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    raw_text = raw_text.strip()
    if not raw_text:
        return None

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(raw_text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None

    return None


def heuristic_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    util = float(observation.get("utilization", 0.0) or 0.0)
    queue = int(observation.get("queue_length", 0) or 0)
    queue_ratio = float(observation.get("queue_ratio", 0.0) or 0.0)
    active = int(observation.get("active_servers", 1) or 1)
    max_servers = int(observation.get("max_servers", 1) or 1)
    cooldown = int(observation.get("cooldown_remaining", 0) or 0)

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


def sanitize_action(raw_action: Dict[str, Any], fallback_obs: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(raw_action.get("mode", "step")).lower()

    if mode == "target":
        target = raw_action.get("target_utilization", 0.6)
        try:
            target_value = float(target)
        except (TypeError, ValueError):
            target_value = 0.6
        return {
            "mode": "target",
            "target_utilization": round(clamp(target_value, 0.1, 0.95), 4),
        }

    if mode == "signal":
        signal = raw_action.get("signal_type")
        if signal != "INSUFFICIENT_CAPACITY":
            signal = None
        return {"mode": "signal", "signal_type": signal}

    if mode != "step":
        return heuristic_action(fallback_obs)

    delta = raw_action.get("scale_delta", 0)
    try:
        delta_value = int(delta)
    except (TypeError, ValueError):
        delta_value = 0
    delta_value = int(clamp(delta_value, -2, 2))
    return {"mode": "step", "scale_delta": delta_value}


def llm_action(
    llm_client: OpenAI,
    model_name: str,
    task: str,
    step: int,
    observation: Dict[str, Any],
    history: List[str],
    timeout_sec: float,
    retries: int,
    console: Console,
) -> Dict[str, Any]:
    user_prompt = textwrap.dedent(
        f"""
        Task: {task}
        Step: {step}
        Observation:
        {json.dumps(observation, indent=2, sort_keys=True)}

        Recent history:
        {chr(10).join(history[-4:]) if history else "none"}

        Choose the next autoscaling action.
        Return strict JSON only.
        """
    ).strip()

    for attempt in range(1, retries + 2):
        try:
            completion = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=220,
                stream=False,
                timeout=timeout_sec,
            )

            content = (completion.choices[0].message.content or "").strip()
            parsed = extract_json_object(content)
            if parsed is None:
                raise ValueError("model did not return valid JSON action")
            return sanitize_action(parsed, observation)
        except Exception as exc:
            if attempt > retries:
                console.print(
                    "[yellow]LLM call failed after retries; switching to heuristic action.[/]"
                )
                console.print(f"[dim]{type(exc).__name__}: {exc}[/dim]")
                return heuristic_action(observation)

            backoff_sec = min(8.0, 0.75 * (2 ** (attempt - 1)))
            console.print(
                f"[yellow]LLM call failed ({type(exc).__name__}), retry {attempt}/{retries} in "
                f"{backoff_sec:.1f}s...[/]"
            )
            time.sleep(backoff_sec)

    return heuristic_action(observation)


def request_json_with_retries(
    http_client: httpx.Client,
    method: str,
    endpoint: str,
    timeout_sec: float,
    retries: int,
    console: Console,
    **kwargs: Any,
) -> Dict[str, Any]:
    for attempt in range(1, retries + 2):
        try:
            response = http_client.request(method, endpoint, timeout=timeout_sec, **kwargs)
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError, json.JSONDecodeError) as exc:
            should_retry = attempt <= retries

            if isinstance(exc, httpx.HTTPStatusError):
                status_code = exc.response.status_code
                should_retry = should_retry and status_code in {429, 500, 502, 503, 504}

            if not should_retry:
                raise

            backoff_sec = min(8.0, 0.75 * (2 ** (attempt - 1)))
            console.print(
                f"[yellow]Request failed ({type(exc).__name__}), retry {attempt}/{retries} in "
                f"{backoff_sec:.1f}s...[/]"
            )
            time.sleep(backoff_sec)

    raise RuntimeError("request retry loop exited unexpectedly")


def compute_score(rewards: List[float]) -> float:
    if not rewards:
        return 0.0

    horizon = len(rewards)
    worst_possible = REWARD_MIN * horizon
    best_possible = REWARD_MAX * horizon
    score = (sum(rewards) - worst_possible) / (best_possible - worst_possible)
    return clamp(score, 0.0, 1.0)


def short_model_name(model_name: str) -> str:
    raw = model_name.split("/")[-1]
    compact = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")
    if not compact:
        compact = "model"
    return compact[:40]


def make_results_dir(task: str, model_name: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = Path("results") / f"{ts}_{task}_{short_model_name(model_name)}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def write_artifacts(
    run_dir: Path,
    raw_logs: List[str],
    reward_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    raw_logs_path = run_dir / "raw_logs.txt"
    summary_path = run_dir / "summary.json"
    rewards_csv_path = run_dir / "rewards.csv"
    metadata_path = run_dir / "metadata.json"

    raw_logs_path.write_text("\n".join(raw_logs) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with rewards_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["step", "reward", "cumulative_reward", "done", "error", "action"],
        )
        writer.writeheader()
        for row in reward_rows:
            writer.writerow(row)


def main() -> int:
    project_root = Path(__file__).resolve().parent
    env_name, task_choices = load_openenv_metadata(project_root / "openenv.yaml")

    args = parse_args(task_choices)
    base_url = LOCAL_SERVER_URL if args.local else str(args.url).rstrip("/")

    console = Console(stderr=True)
    run_started = datetime.now()

    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    use_llm = bool(api_key)

    if use_llm:
        console.print(f"[bold green]Using LLM policy:[/] {args.model}")
    else:
        console.print("[bold yellow]HF_TOKEN/API_KEY not found. Using heuristic fallback policy.[/]")

    llm_client = OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
        api_key=api_key or "placeholder",
    )

    run_dir = make_results_dir(task=args.task, model_name=args.model)

    raw_stdout_lines: List[str] = []
    reward_rows: List[Dict[str, Any]] = []
    rewards: List[float] = []
    history: List[str] = []

    steps_taken = 0
    success = False
    done = False
    fatal_error: Optional[str] = None
    last_env_error: Optional[str] = None

    emit_start(raw_stdout_lines, task=args.task, env_name=env_name, model=args.model)

    http_timeout = httpx.Timeout(args.timeout_sec, connect=args.timeout_sec)
    with httpx.Client(base_url=base_url, timeout=http_timeout) as http_client:
        try:
            reset_payload = request_json_with_retries(
                http_client=http_client,
                method="POST",
                endpoint="/reset",
                params={"task": args.task},
                timeout_sec=args.timeout_sec,
                retries=args.retries,
                console=console,
            )
            session_id = str(reset_payload["session_id"])
            observation = dict(reset_payload["observation"])

            for step in range(1, args.max_steps + 1):
                if done:
                    break

                if use_llm:
                    action = llm_action(
                        llm_client=llm_client,
                        model_name=args.model,
                        task=args.task,
                        step=step,
                        observation=observation,
                        history=history,
                        timeout_sec=args.timeout_sec,
                        retries=args.retries,
                        console=console,
                    )
                else:
                    action = heuristic_action(observation)

                step_payload = request_json_with_retries(
                    http_client=http_client,
                    method="POST",
                    endpoint="/step",
                    params={"session_id": session_id},
                    json=action,
                    timeout_sec=args.timeout_sec,
                    retries=args.retries,
                    console=console,
                )

                observation = dict(step_payload.get("observation", {}))
                reward = float(step_payload.get("reward", 0.0) or 0.0)
                done = bool(step_payload.get("done", False))
                info = step_payload.get("info")
                last_env_error = info.get("last_action_error") if isinstance(info, dict) else None

                rewards.append(reward)
                steps_taken = step

                emit_step(
                    log_lines=raw_stdout_lines,
                    step=step,
                    action=action,
                    reward=reward,
                    done=done,
                    error=last_env_error,
                )

                history.append(f"step={step} action={compact_json(action)} reward={reward:.2f}")

                reward_rows.append(
                    {
                        "step": step,
                        "reward": f"{reward:.6f}",
                        "cumulative_reward": f"{sum(rewards):.6f}",
                        "done": bool_lower(done),
                        "error": sanitize_error(last_env_error),
                        "action": compact_json(action),
                    }
                )

            score = compute_score(rewards)
            success = done and score >= SUCCESS_THRESHOLD and fatal_error is None

        except Exception as exc:
            fatal_error = f"{type(exc).__name__}: {exc}"
            console.print(f"[bold red]Run failed:[/] {fatal_error}")
            score = compute_score(rewards)
            success = False

    emit_end(
        log_lines=raw_stdout_lines,
        success=success,
        steps=steps_taken,
        score=score,
        rewards=rewards,
    )

    run_finished = datetime.now()
    summary = {
        "success": success,
        "steps": steps_taken,
        "score": round(score, 6),
        "total_reward": round(sum(rewards), 6),
        "avg_reward": round((sum(rewards) / len(rewards)) if rewards else 0.0, 6),
        "done": done,
        "fatal_error": fatal_error,
        "last_env_error": sanitize_error(last_env_error),
    }
    metadata = {
        "timestamp_start": run_started.isoformat(),
        "timestamp_end": run_finished.isoformat(),
        "duration_seconds": round((run_finished - run_started).total_seconds(), 3),
        "environment": env_name,
        "task": args.task,
        "model": args.model,
        "policy": "llm" if use_llm else "heuristic",
        "base_url": base_url,
        "local": args.local,
        "max_steps": args.max_steps,
        "timeout_sec": args.timeout_sec,
        "retries": args.retries,
        "python": sys.version,
        "results_dir": str(run_dir),
    }

    write_artifacts(
        run_dir=run_dir,
        raw_logs=raw_stdout_lines,
        reward_rows=reward_rows,
        summary=summary,
        metadata=metadata,
    )

    console.print(f"[bold cyan]Artifacts saved:[/] {run_dir}")
    console.print(f"[cyan]Summary:[/] success={bool_lower(success)} score={score:.3f} steps={steps_taken}")

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
