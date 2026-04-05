# Round 1 — Problem Statement

## THE TASK

Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard `step() / reset() / state()` API.

---

## KEY REQUIREMENTS AT A GLANCE

* Must simulate a real-world task (not games or toys)
* Implement full OpenEnv spec:

  * typed models
  * `step()` / `reset()` / `state()`
  * `openenv.yaml`
* Minimum 3 tasks with agent graders (easy → medium → hard, scores/reward 0.0–1.0)
* Meaningful reward function with partial progress signals
* Baseline inference script with reproducible scores
* Deploy to Hugging Face Spaces + working Dockerfile
* README with:

  * environment description
  * action/observation spaces
  * setup instructions

---

## FUNCTIONAL REQUIREMENTS

### Real-world task simulation

The environment must simulate a task humans actually do.
Not games, not toys.

Examples:

* Email triage
* Code review
* Data cleaning
* Scheduling
* Customer support
* Content moderation

---

### OpenEnv spec compliance

Implement the full OpenEnv interface:

* Typed `Observation`, `Action`, and `Reward` (Pydantic models)
* `step(action)` → returns observation, reward, done, info
* `reset()` → returns initial observation
* `state()` → returns current state
* `openenv.yaml` with metadata
* Must pass `openenv validate`

---

### Minimum 3 tasks with agent graders

* Each task defines a concrete objective
* Programmatic grader scoring (0.0–1.0)
* Tasks must range: **easy → medium → hard**
* Graders must be:

  * deterministic
  * reproducible

---

### Meaningful reward function

* Provides signal across the entire trajectory
* Rewards partial progress
* Penalizes undesirable behavior:

  * infinite loops
  * destructive actions

---

### Baseline inference script

* Uses OpenAI API client
* Reads credentials from environment variables (`OPENAI_API_KEY`)
* Produces reproducible baseline score across all tasks

---

## NON-FUNCTIONAL REQUIREMENTS

### Deploys to a Hugging Face Space

* Must run as a containerized HF Space
* Must be tagged with `openenv`

---

### Containerized execution

Must include a working Dockerfile:

```bash
docker build
docker run
```

---

### Documentation

README must include:

* Environment description and motivation
* Action & observation space definitions
* Task descriptions with difficulty
* Setup and usage instructions
* Baseline scores

---

## SCORING BREAKDOWN

| Parameter                      | Weight | Description                       |
| ------------------------------ | ------ | --------------------------------- |
| Real-world utility             | 30%    | Genuine usefulness of environment |
| Task & grader quality          | 25%    | Task clarity + grading accuracy   |
| Environment design             | 20%    | State design, reward shaping      |
| Code quality & spec compliance | 15%    | OpenEnv compliance + structure    |
| Creativity & novelty           | 10%    | Innovation                        |

---

## EVALUATION CRITERIA

### Phase 1: Automated Validation

* HF Space deploys
* OpenEnv spec compliance
* Dockerfile builds
* Baseline reproduces
* 3+ tasks

### Phase 2: Agentic Evaluation

* Baseline agent re-run
* Standard Open LLM agent evaluation
* Score variance check

### Phase 3: Human Review

* Reviewed by Meta and Hugging Face engineers

---

## DISQUALIFICATION CRITERIA

* Environment does not deploy or respond
* Plagiarized work
* Graders always return same score
* No inference script

---

## REQUIRED ENV VARIABLES

```bash
API_BASE_URL
MODEL_NAME
HF_TOKEN
```

---

## INFERENCE SCRIPT RULES

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

---

# SAMPLE INFERENCE SCRIPT

```python
# FULL SCRIPT (unchanged)

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from my_env_v4 import MyEnvV4Action, MyEnvV4Env
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1

_MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = """
You are interacting with a simple echo environment.
Each turn you must send a message. The environment will echo it back.
Reward is proportional to message length.
"""

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)

    rewards = []
    steps_taken = 0

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    result = await env.reset()

    for step in range(1, MAX_STEPS + 1):
        message = "hello"
        result = await env.step(MyEnvV4Action(message=message))

        reward = result.reward or 0.0
        rewards.append(reward)

        log_step(step, message, reward, result.done, None)

        if result.done:
            break

    score = sum(rewards) / MAX_TOTAL_REWARD
    success = score >= SUCCESS_SCORE_THRESHOLD

    await env.close()
    log_end(success, step, score, rewards)

if __name__ == "__main__":
    asyncio.run(main())
```

---

# PRE-VALIDATION SCRIPT

```bash
#!/usr/bin/env bash

set -uo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <ping_url> [repo_dir]"
  exit 1
fi

echo "Pinging $PING_URL/reset ..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$PING_URL/reset")

if [ "$HTTP_CODE" != "200" ]; then
  echo "FAILED: HF Space not responding"
  exit 1
fi

echo "Docker build..."
docker build "$REPO_DIR" || exit 1

echo "Running openenv validate..."
openenv validate || exit 1

echo "All checks passed!"
```

---

