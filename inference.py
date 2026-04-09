import json
import os

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


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")


def _format_error(error_value):
    if not error_value:
        return "null"
    return str(error_value).replace("\n", "\\n").replace("\r", "\\r")


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

    print(f"[START] task={task_name} env=cloud-autoscaling model={MODEL_NAME}", flush=True)

    try:
        if not API_KEY:
            raise RuntimeError("Missing API key. Set HF_TOKEN or OPENAI_API_KEY.")

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        reset_resp = call_env(f"reset?task={task_name}")
        session_id = reset_resp["session_id"]
        obs = reset_resp["observation"]

        for step in range(1, 101):
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
                f"[STEP] step={step} action={json.dumps(action, separators=(",", ":"))} "
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
            f"score={score:.3f} rewards={r_str}",
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
