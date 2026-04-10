# Cloud Auto-Scaling OpenEnv — Implementation Plan v3.0

> **Hackathon:** Scaler School of Technology × Meta-PyTorch
> **Repo state:** Fully implemented v3.0
> **Stack:** Python · FastAPI · NumPy · Docker · Hugging Face Spaces · OpenAI-compatible client
> **Structure:** Phase-based (each phase = separate chat)

---

## 0. TL;DR

A fully compliant **OpenEnv environment** simulating cloud auto-scaling. Agent observes system metrics, picks a scaling action, receives shaped reward balancing cost vs. latency vs. stability. Three deterministic tasks, UUID-based session management, baseline LLM inference script, deployed to HF Spaces as a containerized FastAPI app.

**v3.0 key upgrades:** Poisson+lognormal+Pareto workloads, M/M/c Erlang C latency model, exponential SLO penalty, L2 stability penalty, Lindley's queue recursion, HPA-style ratio scaling with deadband.

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

**Goal:** Deterministic workload generator, server pool, and latency model. No environment logic yet.

### Constants

```python
SERVER_CAPACITY   = 100     # req/s per server
STARTUP_DELAY     = 2       # steps before new server is live
COOLDOWN_STEPS    = 3       # steps locked after any scaling action
MAX_QUEUE         = 500     # drop requests beyond this
LATENCY_BASE_MS   = 10.0    # minimum baseline latency
STEP_DURATION_SEC = 15.0    # seconds per step (matches Kubernetes HPA scrape interval)
                            # CRITICAL: all rate→count conversions must multiply by this
```

> **v3.0 change:** Added `STEP_DURATION_SEC = 15.0`. Removed unused `LATENCY_SCALE`.

### WorkloadGenerator (v3.0 — Poisson + Lognormal + Pareto)

**Why the change:** v2.0 used Gaussian noise (`randn()`), which produces symmetric variance and allows negative rates. Real cloud traffic is discrete arrivals drawn from a Poisson process. Lognormal rate variance produces right-skewed spikes matching Alibaba/Google cluster traces.

**Critical design rule:** ALL numpy RNG calls must happen inside `__init__` using vectorized array operations. Never call `rng` inside `get()`. The seeded `rng` must be consumed in exactly the same order every time `reset()` re-creates the generator — this guarantees determinism.

Three profiles, seeded NumPy RNG, **pre-generate entire trajectory in `__init__`** so step order doesn't matter:

| Profile | Generation Method |
|---------|-------------------|
| `smooth` | Non-Stationary Poisson Process: `rng.poisson(λ(t))` where `λ(t) = A·sin(2π·t/T) + A` |
| `fluctuating` | Poisson with lognormal multiplier: `rng.poisson(λ(t) · lognormal(0, 0.25))` |
| `spiky` | fluctuating + Pareto bursts: 2% probability, α=2.0, capped at 10× base amplitude |

```python
class WorkloadGenerator:
    VALID_PROFILES = ("smooth", "fluctuating", "spiky")

    def __init__(self, profile: str, seed: int, max_steps: int,
                 base_amp: float = 300.0, period: int = 50,
                 spike_probability: float = 0.0):  # spike_probability kept for API compat, ignored
        rng = np.random.default_rng(seed)
        t = np.arange(max_steps)

        # Base diurnal cycle — sinusoidal, always
        lambda_t = base_amp * np.sin(2 * np.pi * t / period) + base_amp
        lambda_t = np.maximum(10.0, lambda_t)   # minimum traffic floor

        if profile == "smooth":
            # Pure Non-Stationary Poisson Process
            self._workload = rng.poisson(lambda_t).astype(float)

        elif profile == "fluctuating":
            # Lognormal multiplier: right-skewed, never negative
            lognormal_multipliers = rng.lognormal(mean=0.0, sigma=0.25, size=max_steps)
            lambda_noisy = lambda_t * lognormal_multipliers
            self._workload = rng.poisson(lambda_noisy).astype(float)

        elif profile == "spiky":
            lognormal_multipliers = rng.lognormal(mean=0.0, sigma=0.25, size=max_steps)
            lambda_noisy = lambda_t * lognormal_multipliers
            base = rng.poisson(lambda_noisy).astype(float)
            # Pareto spikes: 2% probability, alpha=2.0, capped at 10x base_amp
            spike_mask = rng.random(size=max_steps) < 0.02
            spike_multipliers = np.minimum(rng.pareto(a=2.0, size=max_steps), 10.0)
            spikes = (spike_mask * base_amp * spike_multipliers).astype(float)
            self._workload = base + spikes

        else:
            # Fallback — treat as smooth
            self._workload = rng.poisson(lambda_t).astype(float)

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

### Latency Model (v3.0 — M/M/c Erlang C)

**Why the change:** v2.0 used an M/D/1 model with a piecewise quadratic approximation. This assumed a single server with deterministic service times — wrong for a parallel pod/VM cluster. M/M/c (Erlang C formula) is the industry standard for homogeneous server pools. It naturally produces asymptotic latency explosion near 100% utilization — no fake quadratic penalty needed.

```python
import math

def compute_mmc_latency(request_rate: float, active_servers: int,
                        latency_base_ms: float = LATENCY_BASE_MS) -> float:
    c = max(1, min(active_servers, 15))   # cap at 15 — prevents factorial overflow
    mu = float(SERVER_CAPACITY)            # service rate per server (req/s)
    rho = request_rate / (c * mu)         # system utilization factor

    # Overload state: theoretical latency → ∞, return timeout cap
    if rho >= 0.99:
        return latency_base_ms + 15000.0

    # Erlang C formula: probability that a request must wait
    crho = c * rho
    numerator = (crho ** c) / (math.factorial(c) * (1 - rho))
    sum_k = sum((crho ** k) / math.factorial(k) for k in range(c))
    p_wait = numerator / (sum_k + numerator)

    # Expected wait time in queue: seconds → milliseconds
    wait_time_ms = (p_wait / (c * mu * (1 - rho))) * 1000.0

    return latency_base_ms + wait_time_ms
```

**Behavior:**
- Low load (50 req/s, 5 servers) → ~10ms
- Near capacity (490 req/s, 5 servers) → ~105ms
- Overload (500+ req/s, 5 servers) → 15010ms (timeout cap)

**Exit condition:** Print 50-step workload trace for all 3 profiles. Run twice — values must be identical.

---

## Phase 3 — Reward Function (`env/reward.py`)

**Goal:** Multi-term reward with exponential SLO penalty and L2 stability penalty. All terms bounded, total clipped to `[-5.0, 2.0]`.

> **Note:** The reward function is NOT trained. It is a static formula computed every step.
> What gets optimized (in future RL training) is the agent/policy — not this function.

### Constants

```python
SLO_TARGET_MS = 100.0   # SLO latency threshold (ms)
SLO_K         = 3.0     # steepness of exponential penalty (2.0–4.0 range)
MAX_ACTION    = 2       # max |scale_delta| used for L2 normalization
REWARD_RANGE  = (-5.0, 2.0)
```

### Reward Terms (v3.0)

| # | Term | Range | Formula | Weight |
|---|------|-------|---------|--------|
| 1 | **Throughput reward** | [0, 1.0] | `min(request_rate / max(active_servers × SERVER_CAPACITY, 1), 1.0)` | ×1.0 |
| 2 | **Exponential SLO penalty** | [0, −15.0] | `if latency > SLO_TARGET_MS: -(exp(SLO_K × (latency - SLO_TARGET) / SLO_TARGET) - 1)`, capped at −10.0 | ×1.5 |
| 3 | **Cost penalty** | [−0.5, 0] | `−min(over_provision / SERVER_CAPACITY, 1.0) × 0.5` | ×1.0 |
| 4 | **Queue penalty** | [−1.0, 0] | `−min(queue_length / 100, 1.0) × 1.0` | ×1.0 |
| 5 | **L2 stability penalty** | [−0.2, 0] | `−(scale_delta / MAX_ACTION)²` | ×0.2 |
| 6 | **Invalid action penalty** | {−0.5, 0} | −0.5 if scaling during cooldown | ×1.0 |
| 7 | **Signal bonus** | {0, 1.0} | +1.0 if correctly signaled INSUFFICIENT_CAPACITY | ×1.0 |

**Total = w_eff×throughput − w_slo×slo_penalty + cost + queue − w_stab×stability + invalid + signal**

Clipped to `[-5.0, 2.0]` for value-network stability during RL training.

### v3.0 Design Rationale

**Why exponential SLO penalty (replaces linear latency penalty):**
The old penalty treated a 10ms increase at 20ms the same as at 490ms. In production, latency past the SLO threshold causes cascading failures. An exponential penalty correctly models this — near-zero under normal conditions, explodes past the threshold.

**Why L2 action delta norm (replaces sign-change oscillation deque):**
The old discrete cliff (`if sign_changes >= 3: -0.5`) creates a discontinuous reward landscape that destabilizes RL training. L2 norm is a smooth continuous gradient — micro-adjustments get negligible penalty, large swings get geometric penalty. `scale_delta=1 → 0.05 penalty`, `scale_delta=2 → 0.2 penalty`.

### Signature

```python
def compute_reward(obs, action, prev_delta: int, correctly_signaled: bool) -> CloudReward:
```

> **v3.0 change:** Removed `prev_obs` and `oscillation_penalty` parameters. Added `prev_delta`. The L2 computation happens inside `reward.py` now — no external state needed.

### Full Implementation

```python
import math

def compute_reward(obs, action, prev_delta: int, correctly_signaled: bool):
    # 1. Throughput reward — fraction of demand being served [0, +1.0]
    served_ratio = min(
        obs.request_rate / max(obs.active_servers * SERVER_CAPACITY, 1), 1.0
    )
    throughput_reward = served_ratio * 1.0

    # 2. Exponential SLO latency penalty
    if obs.latency_ms > SLO_TARGET_MS:
        slo_penalty = math.exp(SLO_K * ((obs.latency_ms - SLO_TARGET_MS)
                                         / SLO_TARGET_MS)) - 1.0
        slo_penalty = min(slo_penalty, 10.0)   # cap — prevents exploding gradients
    else:
        slo_penalty = 0.0

    # 3. Cost penalty — over-provisioning [-0.5, 0]
    over_provision = max(obs.active_servers * SERVER_CAPACITY - obs.request_rate, 0)
    cost_penalty = -min(over_provision / SERVER_CAPACITY, 1.0) * 0.5

    # 4. Queue penalty [-1.0, 0]
    queue_penalty = -min(obs.queue_length / 100, 1.0) * 1.0

    # 5. L2 action delta stability penalty
    stability_penalty = (action.scale_delta / max(1, MAX_ACTION)) ** 2

    # 6. Invalid action penalty
    invalid_penalty = -0.5 if (
        obs.cooldown_remaining > 0
        and action.mode == "step"
        and action.scale_delta != 0
    ) else 0.0

    # 7. Capacity signal bonus
    signal_bonus = 1.0 if correctly_signaled else 0.0

    # Weights
    w_eff  = 1.0
    w_slo  = 1.5
    w_stab = 0.2

    total = (
        w_eff  * throughput_reward
        - w_slo  * slo_penalty
        + cost_penalty
        + queue_penalty
        - w_stab * stability_penalty
        + invalid_penalty
        + signal_bonus
    )

    # Clip reward
    total = max(-5.0, min(total, 2.0))

    return CloudReward(value=round(total, 4), breakdown={
        "throughput":  round(w_eff * throughput_reward, 4),
        "slo_penalty": round(-w_slo * slo_penalty, 4),
        "cost":        round(cost_penalty, 4),
        "queue":       round(queue_penalty, 4),
        "stability":   round(-w_stab * stability_penalty, 4),
        "invalid":     round(invalid_penalty, 4),
        "signal":      round(signal_bonus, 4),
    })
```

**Exit condition:** Call with dummy observations, verify breakdown values sum to `value` (after clipping), verify no term exceeds its bound.

---

## Phase 4 — Environment Class (`env/environment.py`)

**Goal:** Full `reset/step/state` loop. Most complex phase.

### v3.0 Critical Design Notes

1. **Queue drain** uses Lindley's recurrence with `STEP_DURATION_SEC` for dimensional correctness
2. **Target-utilization** uses HPA-style ratio scaling with ±10% deadband
3. **Stability** is handled by L2 penalty in `reward.py` — no oscillation deque needed

### Queue Drain Logic (v3.0 — Lindley's Recurrence)

**Why the change:** v2.0 added `request_rate` (req/s) directly to `queue_length` (a count). This is dimensionally invalid — it only works if each step is exactly 1 second. With `STEP_DURATION_SEC = 15`, we need to convert the rate to absolute arrivals per step first.

```python
# In step():
# request_rate is req/s, must be converted to arrivals per step
arrivals       = int(self._request_rate * STEP_DURATION_SEC)
capacity_count = int(self._server_pool.active * SERVER_CAPACITY * STEP_DURATION_SEC)
total_demand   = self._queue + arrivals
fulfilled      = min(total_demand, capacity_count)
new_queue      = max(0, total_demand - fulfilled)
dropped        = max(0, new_queue - MAX_QUEUE)       # requests lost to buffer overflow
self._queue    = min(new_queue, MAX_QUEUE)
utilization    = fulfilled / capacity_count if capacity_count > 0 else 1.0
```

**Dimensional analysis:**
- `arrivals = rate(req/s) × STEP_DURATION(s) = count(req)`
- `capacity_count = servers × capacity(req/s) × STEP_DURATION(s) = count(req)`
- Both are now in the same units (request counts per step)

### Target-Utilization Mode (v3.0 — HPA Ratio + Deadband)

**Why the change:** v2.0's `ceil(rate / (target × capacity))` recomputes an absolute server count from the instantaneous request rate. On noisy workloads this causes micro-scaling every single step. Kubernetes HPA instead computes a ratio and skips action if within ±10% of target.

```python
# mode == "target"
# HPA-style ratio scaling with 10% tolerance deadband
target_util = max(min(action.target_utilization, 0.95), 0.1)
self._target_utilization = target_util
ratio = self._utilization / target_util   # use current utilization
if 0.9 <= ratio <= 1.1:
    delta = 0   # within deadband — no action
else:
    desired = math.ceil(self._server_pool.active * ratio)
    delta = desired - self._server_pool.active
self._server_pool.apply_action(delta, self._step)
```

### Latency Computation (v3.0)

```python
# Old (v2.0):
latency = compute_latency(self._request_rate, self._server_pool.effective_capacity, self._queue)

# New (v3.0):
latency = compute_mmc_latency(self._request_rate, self._server_pool.active)
```

### Reward Computation (v3.0)

```python
# Old (v2.0):
osc_pen = self._oscillation_penalty()
reward = compute_reward(obs, action, prev_obs, osc_pen, correctly_signaled)

# New (v3.0):
reward = compute_reward(obs, action, action.scale_delta, correctly_signaled)
```

### Removed Components (v3.0)

- `from collections import deque` — no longer needed
- `self._scale_history = deque(maxlen=4)` — removed from `_reset_internals()`
- `_oscillation_penalty()` method — deleted entirely
- All references to `self._oscillation_penalty()` in `step()` — removed

### Imports (v3.0)

```python
from env.simulation import (
    MAX_QUEUE,
    SERVER_CAPACITY,
    STEP_DURATION_SEC,       # NEW
    ServerPool,
    WorkloadGenerator,
    compute_mmc_latency,     # RENAMED from compute_latency
)
```

### Full Step Loop Order

```python
class CloudAutoScalingEnv:
    def step(self, action: CloudAction) -> StepResult:
        # 0. Snapshot previous observation
        # 1. Dispatch action (mode: step / target / signal)
        # 2. Advance step counter
        # 3. Advance workload (get next request rate)
        # 4. Tick server pool (graduate pending → active)
        # 5. Drain queue (Lindley's recurrence with STEP_DURATION_SEC)
        # 6. Compute utilization & latency (M/M/c Erlang C)
        # 7. Check signal correctness
        # 8. Compute reward (exponential SLO + L2 stability)
        # 9. Check termination
        # 10. Return StepResult
```

**Exit condition:**
```python
env = CloudAutoScalingEnv("easy")
obs = env.reset()
for _ in range(20):
    result = env.step(CloudAction(mode="step", scale_delta=1))
    print(result.reward, result.observation.latency_ms, result.observation.queue_length)
```
Rewards should vary, queue should drain when capacity is available, latency should drop as servers scale up.

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
reward_range: [-5.0, 2.0]
tags:
  - openenv
  - cloud
  - autoscaling
  - reinforcement-learning
```

> **v3.0 change:** `reward_range` updated from `[-3.5, 2.0]` to `[-5.0, 2.0]`.

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

## v3.0 Formula Quick Reference

### Latency (Erlang C — M/M/c)

```
c = active_servers (capped at 15)
μ = SERVER_CAPACITY (100 req/s per server)
ρ = request_rate / (c × μ)

If ρ ≥ 0.99 → latency = LATENCY_BASE_MS + 15000.0 (timeout)

Erlang C probability of waiting:
  P_wait = [(cρ)^c / (c! × (1-ρ))] / [Σ_{k=0}^{c-1} (cρ)^k/k! + (cρ)^c/(c!×(1-ρ))]

Expected wait time:
  W_q = P_wait / (c × μ × (1-ρ))

Latency = LATENCY_BASE_MS + W_q × 1000 (ms)
```

### Queue Drain (Lindley's Recurrence)

```
arrivals       = ⌊request_rate × STEP_DURATION_SEC⌋
capacity_count = ⌊active_servers × SERVER_CAPACITY × STEP_DURATION_SEC⌋
total_demand   = queue + arrivals
fulfilled      = min(total_demand, capacity_count)
new_queue      = max(0, total_demand - fulfilled)
dropped        = max(0, new_queue - MAX_QUEUE)
queue          = min(new_queue, MAX_QUEUE)
utilization    = fulfilled / capacity_count
```

### Reward

```
total = (1.0 × throughput)
      - (1.5 × slo_penalty)
      + cost_penalty
      + queue_penalty
      - (0.2 × stability_penalty)
      + invalid_penalty
      + signal_bonus

clipped to [-5.0, 2.0]

where:
  throughput     = min(rate / max(servers × 100, 1), 1.0)
  slo_penalty    = min(exp(3.0 × (latency - 100) / 100) - 1, 10.0)  if latency > 100ms, else 0
  cost_penalty   = -min(overcapacity / 100, 1.0) × 0.5
  queue_penalty  = -min(queue / 100, 1.0) × 1.0
  stability      = (scale_delta / 2)²
  invalid        = -0.5 if scaling during cooldown, else 0
  signal         = +1.0 if correctly signaled INSUFFICIENT_CAPACITY
```

### HPA Ratio Scaling (Target Mode)

```
ratio = utilization / target_utilization
if 0.9 ≤ ratio ≤ 1.1 → delta = 0           (deadband)
else → desired = ⌈active_servers × ratio⌉
       delta = desired - active_servers
```

---

## Pitfalls Reference (v3.0 Updated)

| Pitfall | Fix |
|---------|-----|
| Two grader threads clobber same env | Session UUID key — Phase 5 |
| Requests vanish from queue | Lindley's recurrence with `STEP_DURATION_SEC` — Phase 4 |
| Dimensionally invalid queue math | `rate × STEP_DURATION_SEC` converts req/s → count — Phase 4 |
| Target util causes micro-scaling | HPA ratio + ±10% deadband — Phase 4 |
| Latency model wrong for parallel servers | Erlang C M/M/c replaces M/D/1 — Phase 2 |
| Linear latency penalty insensitive to SLO violations | Exponential SLO penalty — Phase 3 |
| Oscillation deque creates discontinuous reward | L2 stability penalty (smooth gradient) — Phase 3 |
| Factorial overflow for large server counts | `c` capped at 15 in Erlang C — Phase 2 |
| Breakdown sum ≠ value after clipping | `_verify_bounds` clips sum before comparing — Phase 3 |
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
| 2 | Simulation core (Poisson workloads, Erlang C latency) | `env/simulation.py` |
| 3 | Reward function (exp SLO, L2 stability) | `env/reward.py` |
| 4 | Environment loop (Lindley queue, HPA deadband) | `env/environment.py`, `env/tasks.py` |
| 5 | HTTP server | `server/main.py`, `server/app.py` |
| 6 | Baseline script | `inference.py` |
| 7 | Ship | `openenv.yaml`, `Dockerfile`, `README.md` |

---

## v2.0 → v3.0 Changelog Summary

| Component | v2.0 | v3.0 |
|-----------|------|------|
| Workload generation | Gaussian noise (`randn()`) | Poisson + lognormal + Pareto |
| Latency model | M/D/1 piecewise quadratic | M/M/c Erlang C |
| Latency penalty | Linear (−2.0 cap) | Exponential SLO (−15.0 cap, ×1.5 weight) |
| Oscillation penalty | Sign-change deque (discrete cliff) | L2 action delta norm (smooth gradient) |
| Queue drain | `rate + queue` (dimensionally wrong) | Lindley's recurrence × `STEP_DURATION_SEC` |
| Target scaling | `ceil(rate / (target × cap))` | HPA ratio + ±10% deadband |
| Reward range | `[-3.5, 2.0]` | `[-5.0, 2.0]` (clipped) |
| `compute_reward` signature | `(obs, action, prev_obs, oscillation_penalty, correctly_signaled)` | `(obs, action, prev_delta, correctly_signaled)` |
| `compute_latency` signature | `(request_rate, capacity, queue_length)` | `compute_mmc_latency(request_rate, active_servers)` |
| Time constant | implicit 1s steps | explicit `STEP_DURATION_SEC = 15.0` |