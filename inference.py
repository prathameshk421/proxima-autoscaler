import json
import os

from dotenv import load_dotenv
from openai import OpenAI


SYSTEM_PROMPT = """
You are a cloud infrastructure scaling agent. Output ONLY a JSON action, no explanation.

Action schema:
{
  "mode": "step" | "target" | "signal",
  "scale_delta": <int -2 to +2>,          // only when mode=step
  "target_utilization": <float 0.4-0.7>,  // only when mode=target
  "signal_type": "INSUFFICIENT_CAPACITY" | null
}

Decision rules (apply in order):
1. If cooldown_remaining > 0: you CANNOT scale. Use mode=signal with signal_type=null,
   OR mode=target (it will queue the intent). DO NOT use mode=step with nonzero delta.
2. If active_servers == max_servers AND utilization > 0.95:
   MUST use mode=signal, signal_type=INSUFFICIENT_CAPACITY.
3. If utilization > 0.85 AND active_servers < max_servers AND cooldown_remaining == 0:
   mode=step, scale_delta=2.
4. If utilization < 0.4 AND active_servers > 1 AND cooldown_remaining == 0:
   mode=step, scale_delta=-1.
5. Otherwise: mode=step, scale_delta=0.

Output ONLY the JSON object. No markdown, no explanation.
"""


# Load local .env when present so baseline runs without manual exports.
load_dotenv()


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
POLICY_MODE = os.getenv("POLICY_MODE", "auto").strip().lower()


def _format_error(error_value):
    if not error_value:
        return "null"
    return str(error_value)


def _heuristic_action(obs: dict) -> dict:
    cooldown = int(obs.get("cooldown_remaining", 0) or 0)
    active = int(obs.get("active_servers", 1) or 1)
    max_servers = int(obs.get("max_servers", 1) or 1)
    utilization = float(obs.get("utilization", 0.0) or 0.0)

    if cooldown > 0:
        return {"mode": "signal", "signal_type": None}
    if active >= max_servers and utilization > 0.95:
        return {"mode": "signal", "signal_type": "INSUFFICIENT_CAPACITY"}
    if utilization > 0.85 and active < max_servers:
        return {"mode": "step", "scale_delta": 2}
    if utilization < 0.4 and active > 1:
        return {"mode": "step", "scale_delta": -1}
    return {"mode": "step", "scale_delta": 0}


def call_env(endpoint, method="POST", body=None):
    import urllib.request

    url = f"{ENV_URL}/{endpoint}"
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read())


def run_task(task_name: str):
    rewards = []
    score = 0.0
    success = False
    obs = None
    steps_taken = 0

    use_llm = POLICY_MODE == "llm" or (POLICY_MODE == "auto" and bool(API_KEY))
    policy_name = MODEL_NAME if use_llm else "heuristic-rule"

    print(f"[START] task={task_name} env=cloud-autoscaling model={policy_name}", flush=True)

    try:
        if POLICY_MODE not in {"auto", "llm", "heuristic"}:
            raise RuntimeError("POLICY_MODE must be one of: auto, llm, heuristic")
        if POLICY_MODE == "llm" and not API_KEY:
            raise RuntimeError("Missing API key. Set HF_TOKEN or API_KEY.")

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if use_llm else None

        reset_resp = call_env(f"reset?task={task_name}")
        session_id = reset_resp["session_id"]
        obs = reset_resp["observation"]

        for step in range(1, 101):
            if use_llm:
                user_msg = f"Current state:\n{json.dumps(obs, indent=2)}\n\nOutput your action JSON:"
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=100,
                    temperature=0.2,
                )
                raw = (response.choices[0].message.content or "").strip()
                action = json.loads(raw)
                if not isinstance(action, dict):
                    raise ValueError(f"Model returned non-object JSON action: {raw}")
            else:
                action = _heuristic_action(obs)

            result = call_env(f"step?session_id={session_id}", body=action)
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            obs = result.get("observation", obs)
            info = result.get("info", {})
            error = None
            if isinstance(info, dict):
                error = info.get("last_action_error")

            rewards.append(reward)
            steps_taken = step

            print(
                f"[STEP] step={step} action={json.dumps(action, separators=(',', ':'))} "
                f"reward={reward:.2f} done={str(done).lower()} error={_format_error(error)}",
                flush=True,
            )
            if done:
                break

        score = compute_score(rewards, task_name)
        success = score >= 0.1
    except Exception:
        success = False
        score = 0.0
    finally:
        r_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps_taken} "
            f"score={score:.2f} rewards={r_str}",
            flush=True,
        )

    return score


def compute_score(rewards, task_name):
    bounds = {
        "easy": {"min": -35.0, "max": 50.0},
        "medium": {"min": -52.5, "max": 75.0},
        "hard": {"min": -70.0, "max": 100.0},
    }
    b = bounds.get(task_name, {"min": -50.0, "max": 100.0})
    raw = sum(rewards)
    return max(0.0, min(1.0, (raw - b["min"]) / (b["max"] - b["min"])))


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)
