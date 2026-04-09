# Cloud Auto-Scaling OpenEnv — Implementation Plan v2.0

> **Hackathon:** Scaler School of Technology × Meta-PyTorch
> **Repo state:** Empty branch — start from scratch
> **Stack:** Python · FastAPI · NumPy · Docker · Hugging Face Spaces · OpenAI-compatible client
> **Structure:** Phase-based (each phase = separate chat)

---

## 0. TL;DR

A fully compliant **OpenEnv environment** simulating cloud auto-scaling. Agent observes system metrics, picks a scaling action, receives shaped reward balancing cost vs. latency vs. stability. Three deterministic tasks, UUID-based session management, baseline LLM inference script, deployed to HF Spaces as a containerized FastAPI app.

---

## Repository Layout

```
cloud-autoscaling-openenv/
├── env/
│   ├── __init__.py
│   ├── models.py          # Pydantic: Observation, Action, Reward, Info
│   ├── simulation.py      # Core sim (workload, servers, latency, queue)
│   ├── reward.py          # Reward function
│   ├── tasks.py           # Task configs: easy / medium / hard
│   └── environment.py     # CloudAutoScalingEnv — step/reset/state
├── server/
│   ├── main.py            # FastAPI app
│   └── routers.py         # Route handlers
├── baseline/
│   └── inference.py       # OpenAI-client baseline script
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Phase 0 — Repo Bootstrap

**Goal:** Clean working environment, nothing broken before a line of logic is written.

```bash
# 1. Create and enter project folder
mkdir cloud-autoscaling-openenv && cd cloud-autoscaling-openenv

# 2. Init git
git init
git checkout -b main

# 3. Create virtualenv
python3 -m venv .venv
source .venv/bin/activate          # Mac/Linux
# .venv\Scripts\activate           # Windows

# 4. Install deps
pip install fastapi "uvicorn[standard]" pydantic numpy openai
pip install pytest httpx            # for testing
pip freeze > requirements.txt

# 5. .gitignore
cat > .gitignore << 'EOF'
.venv/
__pycache__/
*.pyc
.env
*.egg-info/
dist/
.pytest_cache/
EOF

# 6. Create folder skeleton
mkdir -p env server baseline
touch env/__init__.py server/__init__.py baseline/__init__.py

# 7. Initial commit
git add .
git commit -m "chore: project skeleton"
```

**Exit condition:** `python -c "import fastapi, pydantic, numpy, openai; print('ok')"` prints `ok`.

---

## Phase 1 — Data Models (`env/models.py`)

**Goal:** All Pydantic models defined, importable, no logic yet.

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class CloudObservation(BaseModel):
    active_servers: int
    max_servers: int
    request_rate: float               # req/s arriving
    utilization: float                # avg across active servers [0,1]
    queue_length: int                 # unserved requests waiting
    latency_ms: float                 # estimated p99 latency
    target_utilization: float         # current target (mode 2)
    cooldown_remaining: int           # steps until next scale allowed
    step_number: int
    done: bool
    # Normalized fields (future-proof for RL)
    queue_ratio: float = 0.0          # queue_length / MAX_QUEUE
    capacity_ratio: float = 0.0       # active_servers / max_servers

class CloudAction(BaseModel):
    mode: Literal["step", "target", "signal"]
    scale_delta: int = 0
    target_utilization: float = 0.6
    signal_type: Optional[Literal["INSUFFICIENT_CAPACITY"]] = None

class CloudReward(BaseModel):
    value: float
    breakdown: dict

class StepResult(BaseModel):
    observation: CloudObservation
    reward: float
    done: bool
    info: dict
```

**Exit condition:** `python -c "from env.models import CloudObservation, CloudAction, StepResult; print('ok')"` prints `ok`.

---

## Phase 2 — Simulation Core (`env/simulation.py`)

**Goal:** Deterministic workload generator and server pool. No environment logic yet.

### Constants

```python
SERVER_CAPACITY = 100   # req/s per server
STARTUP_DELAY   = 2     # steps before new server is live
COOLDOWN_STEPS  = 3     # steps locked after any scaling action
MAX_QUEUE       = 500   # drop requests beyond this
LATENCY_BASE_MS = 10.0
LATENCY_SCALE   = 5.0
```

### WorkloadGenerator

Three profiles, seeded NumPy RNG, **pre-generate entire trajectory in `__init__`** so step order doesn't matter:

| Profile | Description |
|---------|-------------|
| `smooth` | `A·sin(2π·t/T) + A + ε` |
| `fluctuating` | smooth + moderate noise |
| `spiky` | fluctuating + burst events at spike_probability |

```python
class WorkloadGenerator:
    def __init__(self, profile: str, seed: int, max_steps: int,
                 spike_probability: float = 0.0):
        rng = np.random.default_rng(seed)
        # pre-generate and cache full trajectory
        self._workload = self._generate(rng, profile, max_steps, spike_probability)

    def get(self, step: int) -> float:
        return float(self._workload[step])
```

### ServerPool

```python
class ServerPool:
    def __init__(self, initial: int, min_s: int, max_s: int): ...

    def apply_action(self, delta: int, current_step: int):
        # Respect cooldown. Clamp to [min_s, max_s].
        # New servers go to pending with arrival_step = current_step + STARTUP_DELAY

    def tick(self, current_step: int):
        # Graduate pending → active if arrival_step <= current_step

    @property
    def effective_capacity(self) -> float:
        return self.active * SERVER_CAPACITY
```

### Latency Model (M/D/1 queuing approximation)

```python
def compute_latency(request_rate, capacity, queue_length) -> float:
    utilization = min(request_rate / max(capacity, 1), 1.0)
    if utilization < 0.8:
        return LATENCY_BASE_MS + utilization * 20
    overflow = (utilization - 0.8) / 0.2
    return LATENCY_BASE_MS + 20 + overflow**2 * 200 + queue_length * 0.5
```

**Exit condition:** Print 50-step workload trace for all 3 profiles. Run twice — values must be identical.

---

## Phase 3 — Reward Function (`env/reward.py`)

**Goal:** Multi-term reward, all terms bounded, predictable total range.

> **Note:** The reward function is NOT trained. It is a static formula computed every step.
> What gets optimized (in future RL training) is the agent/policy — not this function.

```python
def compute_reward(obs, action, prev_obs, oscillation_penalty,
                   correctly_signaled) -> CloudReward:

    # Latency penalty
    latency_penalty = 0 if obs.latency_ms < 50 else \
        -min((obs.latency_ms - 50) / 50, 1.0) * 2.0

    # Throughput reward
    served_ratio = min(obs.request_rate /
                       max(obs.active_servers * SERVER_CAPACITY, 1), 1.0)
    throughput_reward = served_ratio * 1.0

    # Cost penalty (over-provisioning)
    over_provision = max(obs.active_servers * SERVER_CAPACITY - obs.request_rate, 0)
    cost_penalty = -min(over_provision / SERVER_CAPACITY, 1.0) * 0.5

    # Queue penalty
    queue_penalty = -min(obs.queue_length / 100, 1.0) * 1.0

    # Invalid action penalty (tried to scale during cooldown)
    invalid_penalty = -0.5 if (obs.cooldown_remaining > 0
                                and action.mode != "signal") else 0.0

    # Capacity signal bonus
    signal_bonus = 1.0 if correctly_signaled else 0.0

    total = (throughput_reward + latency_penalty + cost_penalty
             + queue_penalty + oscillation_penalty + invalid_penalty + signal_bonus)

    return CloudReward(value=round(total, 4), breakdown={
        "throughput": throughput_reward,
        "latency": latency_penalty,
        "cost": cost_penalty,
        "queue": queue_penalty,
        "oscillation": oscillation_penalty,
        "invalid": invalid_penalty,
        "signal": signal_bonus,
    })
```

**Exit condition:** Call with dummy observations, verify breakdown values sum to `value`, verify no term exceeds its bound.

---

## Phase 4 — Environment Class (`env/environment.py`)

**Goal:** Full `reset/step/state` loop. Most complex phase.

### Critical fixes applied here

**Queue drain logic (fixed — requests no longer vanish):**
```python
# In step():
total_demand = obs.request_rate + self._queue
fulfilled = min(total_demand, self._server_pool.effective_capacity)
self._queue = max(0, int(total_demand - fulfilled))
self._queue = min(self._queue, MAX_QUEUE)
```

**Target utilization math (fixed — dimensionality correct):**
```python
# mode="target"
required_servers = math.ceil(
    obs.request_rate / (action.target_utilization * SERVER_CAPACITY)
)
delta = required_servers - self._server_pool.active
self._server_pool.apply_action(delta, self._step)
```

**Oscillation tracking (zero-aware):**
```python
from collections import deque
self._scale_history: deque = deque(maxlen=4)

def _oscillation_penalty(self) -> float:
    # Filter zeros before checking sign changes
    nonzero = [x for x in self._scale_history if x != 0]
    if len(nonzero) < 2:
        return 0.0
    sign_changes = sum(
        1 for a, b in zip(nonzero, nonzero[1:]) if (a > 0) != (b > 0)
    )
    return -0.5 if sign_changes >= 3 else 0.0
```

### Full class outline

```python
class CloudAutoScalingEnv:
    def __init__(self, task_name: str = "easy"):
        self._task = TASKS[task_name]
        self._reset_internals()

    def reset(self) -> CloudObservation:
        self._reset_internals()
        return self._build_observation()

    def step(self, action: CloudAction) -> StepResult:
        # 1. Apply action (mode dispatch)
        # 2. Advance workload
        # 3. Tick server pool
        # 4. Drain queue: total_demand = request_rate + queue
        #    fulfilled = min(total_demand, effective_capacity)
        #    queue = min(max(0, total_demand - fulfilled), MAX_QUEUE)
        # 5. Compute utilization, latency
        # 6. Compute reward
        # 7. Check done
        # 8. Return StepResult

    def state(self) -> CloudObservation: ...

    def _build_observation(self) -> CloudObservation:
        # Includes queue_ratio and capacity_ratio normalized fields
        ...
```

**Exit condition:**
```python
env = CloudAutoScalingEnv("easy")
obs = env.reset()
for _ in range(20):
    result = env.step(CloudAction(mode="step", scale_delta=1))
    print(result.reward, result.observation.latency_ms, result.observation.queue_length)
```
Rewards should vary, queue should drain when capacity is available, not vanish.

---

## Phase 5 — FastAPI Server (`server/main.py`)

**Goal:** HTTP wrapper with UUID session management.

### Session-safe design (critical — fixes concurrent grader clobber)

```python
from fastapi import FastAPI, HTTPException
from uuid import uuid4
from env.environment import CloudAutoScalingEnv
from env.models import CloudAction, StepResult, CloudObservation

app = FastAPI(title="CloudAutoScaling OpenEnv")

# Keyed by session_id UUID — NOT task name
sessions: dict[str, CloudAutoScalingEnv] = {}

@app.post("/reset")
def reset(task: str = "easy") -> dict:
    session_id = str(uuid4())
    sessions[session_id] = CloudAutoScalingEnv(task_name=task)
    obs = sessions[session_id].reset()
    return {"session_id": session_id, "observation": obs.model_dump()}

@app.post("/step")
def step(action: CloudAction, session_id: str) -> StepResult:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id].step(action)

@app.get("/state")
def state(session_id: str) -> CloudObservation:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id].state()

@app.get("/health")
def health():
    return {"status": "ok"}
```

**Exit condition:**
```bash
uvicorn server.main:app --reload
curl -X POST "localhost:8000/reset?task=easy"
# Copy session_id from response, use in next call
curl -X POST "localhost:8000/step?session_id=<uuid>" \
  -H "Content-Type: application/json" \
  -d '{"mode": "step", "scale_delta": 1}'
```

---

## Phase 6 — Baseline Inference Script (`inference.py`)

**Goal:** OpenAI-client script, tight system prompt, correct log format.

### System prompt (handles cooldown + prevents hallucination)

```python
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
```

### Script structure

```python
import os, json
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

def call_env(endpoint, method="POST", body=None):
    import urllib.request
    url = f"{ENV_URL}/{endpoint}"
    data = json.dumps(body).encode() if body else b""
    req = urllib.request.Request(url, data=data,
          headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=10) as r:   # timeout=10 always
        return json.loads(r.read())

def run_task(task_name: str):
    if not API_KEY:
        raise RuntimeError("Missing API key. Set HF_TOKEN or OPENAI_API_KEY.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Reset — capture session_id
    reset_resp = call_env(f"reset?task={task_name}")
    session_id = reset_resp["session_id"]
    obs = reset_resp["observation"]

    rewards = []
    print(f"[START] task={task_name} env=cloud-autoscaling model={MODEL_NAME}", flush=True)

    for step in range(1, 101):
        user_msg = f"Current state:\n{json.dumps(obs, indent=2)}\n\nOutput your action JSON:"
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=100,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        action = json.loads(raw)
        if not isinstance(action, dict):
            raise ValueError(f"Model returned non-object JSON action: {raw}")

        result = call_env(f"step?session_id={session_id}", body=action)
        reward = result.get("reward", 0.0)
        done   = result.get("done", False)
        obs    = result.get("observation", obs)
        rewards.append(reward)

        print(f"[STEP] step={step} action={json.dumps(action)} "
              f"reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
        if done:
            break

    score = compute_score(rewards, task_name)
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(score >= 0.1).lower()} steps={len(rewards)} "
          f"score={score:.3f} rewards={r_str}", flush=True)
    return score

def compute_score(rewards, task_name):
    BOUNDS = {
        "easy":   {"min": -35.0, "max": 50.0},
        "medium": {"min": -52.5, "max": 75.0},
        "hard":   {"min": -70.0, "max": 100.0},
    }
    b = BOUNDS.get(task_name, {"min": -50, "max": 100})
    raw = sum(rewards)
    return max(0.0, min(1.0, (raw - b["min"]) / (b["max"] - b["min"])))

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)
```

**Exit condition:** `python inference.py` runs all 3 tasks, produces valid `[END]` lines, scores between 0.0 and 1.0.

---

## Phase 7 — OpenEnv Compliance + Docker + HF Deploy

**Goal:** Ship it.

### `openenv.yaml`

```yaml
name: cloud-autoscaling
version: "1.0.0"
description: >
  Intelligent cloud auto-scaling environment where an RL/LLM agent
  learns to provision servers under dynamic workloads.
tasks:
  - name: easy
    description: Predictable sinusoidal workload
    difficulty: easy
  - name: medium
    description: Fluctuating noisy workload
    difficulty: medium
  - name: hard
    description: Spiky workload with capacity saturation events
    difficulty: hard
observation_space:
  type: object
  fields:
    - active_servers: int
    - max_servers: int
    - request_rate: float
    - utilization: float
    - queue_length: int
    - latency_ms: float
    - target_utilization: float
    - cooldown_remaining: int
    - step_number: int
    - done: bool
action_space:
  type: object
  fields:
    - mode: "step | target | signal"
    - scale_delta: int
    - target_utilization: float
    - signal_type: "INSUFFICIENT_CAPACITY | null"
reward_range: [-3.5, 2.0]
tags:
  - openenv
  - cloud
  - autoscaling
  - reinforcement-learning
```

### `Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Deploy checklist

- [ ] `docker build -t cloud-env . && docker run -p 7860:7860 cloud-env`
- [ ] Hit `localhost:7860/health` — returns `{"status": "ok"}`
- [ ] Run baseline against `localhost:7860`, confirm scores match dev
- [ ] Push to HF Spaces with `openenv` tag in Space metadata
- [ ] Run baseline against live HF URL, confirm scores match local

---

## Pitfalls Reference

| Pitfall | Fix |
|---------|-----|
| Two grader threads clobber same env | Session UUID key — Phase 5 |
| Requests vanish from queue | `total_demand = rate + queue` drain — Phase 4 |
| Target util dimensionality bug | `ceil(rate / (target_util * SERVER_CAPACITY))` — Phase 4 |
| Oscillation check skips zeros | Filter zeros before sign-change scan — Phase 4 |
| Unnormalized obs for RL | `queue_ratio`, `capacity_ratio` in model — Phase 1 |
| LLM ignores cooldown | Explicit rule #1 in system prompt — Phase 6 |
| HF port mismatch | Port 7860 hardcoded in Dockerfile |
| Missing `openenv` tag | Add in HF Space metadata YAML |
| Baseline urllib no timeout | `timeout=10` on all calls — Phase 6 |

---

## Phase Summary

| Phase | Deliverable | Key files |
|-------|-------------|-----------|
| 0 | Repo + venv + skeleton | `.gitignore`, `requirements.txt` |
| 1 | Pydantic models | `env/models.py` |
| 2 | Simulation core | `env/simulation.py` |
| 3 | Reward function | `env/reward.py` |
| 4 | Environment loop | `env/environment.py`, `env/tasks.py` |
| 5 | HTTP server | `server/main.py`, `server/routers.py` |
| 6 | Baseline script | `inference.py` |
| 7 | Ship | `openenv.yaml`, `Dockerfile`, `README.md` |