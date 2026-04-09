# 🖥️ Cloud Auto-Scaling OpenEnv

An [OpenEnv](https://huggingface.co/openenv)-compliant reinforcement learning environment that simulates intelligent cloud infrastructure auto-scaling. An RL or LLM agent observes real-time system metrics and decides how to provision servers — balancing latency, cost, and stability under dynamic workloads.

---

## What is this?

Modern cloud systems need to scale server capacity up and down in response to changing traffic. Scale too slowly and users experience high latency. Scale too aggressively and you waste money on idle servers. This environment formalizes that problem as an RL task.

An agent receives system observations every step (request rate, utilization, queue depth, latency, cooldown) and outputs a scaling action. It is rewarded for keeping latency low, avoiding over-provisioning, draining the queue, and staying stable — penalized for thrashing, ignoring cooldowns, and letting queues overflow.

---

## Environment Description

- **Observation:** System metrics snapshot at each timestep
- **Action:** Scale up/down by step, target a utilization level, or signal capacity exhaustion
- **Reward:** Multi-term shaped reward (latency + throughput + cost + queue + stability)
- **Episodes:** 50–100 steps depending on task difficulty
- **Determinism:** All tasks are seeded — same seed always produces the same workload

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `active_servers` | int | Currently provisioned servers |
| `max_servers` | int | Hard capacity ceiling |
| `request_rate` | float | Incoming requests per second |
| `utilization` | float | Average utilization across active servers [0, 1] |
| `queue_length` | int | Unserved requests waiting |
| `latency_ms` | float | Estimated p99 latency in milliseconds |
| `target_utilization` | float | Current target utilization (used in target mode) |
| `cooldown_remaining` | int | Steps until next scaling action is allowed |
| `step_number` | int | Current step in the episode |
| `done` | bool | Whether the episode has ended |
| `queue_ratio` | float | `queue_length / MAX_QUEUE` — normalized [0, 1] |
| `capacity_ratio` | float | `active_servers / max_servers` — normalized [0, 1] |

---

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `mode` | `"step" \| "target" \| "signal"` | Action type |
| `scale_delta` | int | Servers to add/remove (used when `mode=step`) |
| `target_utilization` | float | Desired utilization 0.4–0.7 (used when `mode=target`) |
| `signal_type` | `"INSUFFICIENT_CAPACITY" \| null` | Capacity signal (used when `mode=signal`) |

### Mode details

- **`step`** — Directly add or remove servers (`scale_delta` in range -2 to +2). Subject to cooldown.
- **`target`** — Specify a desired utilization level; the environment computes the required server count automatically.
- **`signal`** — Report that capacity is exhausted (`INSUFFICIENT_CAPACITY`) when at max servers and still overloaded. Earns a bonus if correctly used.

---

## Reward Function

Total reward per step is the sum of the following terms:

| Term | Range | Description |
|------|-------|-------------|
| Throughput reward | `[0, +1.0]` | Fraction of demand being served |
| Latency penalty | `[-2.0, 0]` | Penalty when p99 latency exceeds 50ms |
| Cost penalty | `[-0.5, 0]` | Penalty for idle over-provisioned capacity |
| Queue penalty | `[-1.0, 0]` | Penalty proportional to queue depth |
| Oscillation penalty | `[-0.5, 0]` | Penalty for rapidly flipping scale direction |
| Invalid action penalty | `[-0.5, 0]` | Penalty for scaling during cooldown |
| Signal bonus | `[0, +1.0]` | Bonus for correctly signaling capacity exhaustion |

**Total reward range:** `[-3.5, +2.0]` per step

**Episode score (0.0–1.0):**
```
score = (sum_rewards - worst_possible) / (best_possible - worst_possible)
```

---

## Tasks

### Easy
- **Workload:** Smooth sinusoidal pattern
- **Steps:** 50
- **Max servers:** 10
- **Goal:** Learn basic up/down provisioning on a predictable curve

### Medium
- **Workload:** Fluctuating with moderate noise
- **Steps:** 75
- **Max servers:** 12
- **Goal:** Filter noise vs. real demand shifts; avoid overreacting to spikes

### Hard
- **Workload:** Spiky with sudden burst events (15% spike probability)
- **Steps:** 100
- **Max servers:** 15
- **Goal:** Handle sudden bursts that may exceed max capacity; use capacity signal correctly

---

## Baseline Scores

Scores from the LLM baseline agent (Qwen2.5-72B-Instruct via HuggingFace router):

| Task | Score |
|------|-------|
| easy | TBD |
| medium | TBD |
| hard | TBD |

---

## Setup

### Local (development)

```bash
git clone <your-repo-url>
cd cloud-autoscaling-openenv

python3 -m venv .venv
source .venv/bin/activate       # Mac/Linux
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
uvicorn server.main:app --reload --port 7860
```

### Docker

```bash
docker build -t cloud-autoscaling-openenv .
docker run -p 7860:7860 cloud-autoscaling-openenv
```

### Hugging Face Space

The environment is deployed at: `https://huggingface.co/spaces/<your-space>`

Set the following environment variables in your Space settings:
- `API_BASE_URL` — LLM inference endpoint (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME` — Model to use (default: `Qwen/Qwen2.5-72B-Instruct`)
- `HF_TOKEN` — Your HuggingFace token

---

## API Usage

All endpoints are available at `http://localhost:7860` (or your HF Space URL).

### `POST /reset?task=<easy|medium|hard>`

Start a new episode. Returns a `session_id` — pass this to all subsequent calls.

```bash
curl -X POST "http://localhost:7860/reset?task=easy"
```

```json
{
  "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "observation": {
    "active_servers": 3,
    "request_rate": 142.5,
    "utilization": 0.47,
    ...
  }
}
```

### `POST /step?session_id=<uuid>`

Take one action. Returns the next observation, reward, and done flag.

```bash
curl -X POST "http://localhost:7860/step?session_id=f47ac10b-..." \
  -H "Content-Type: application/json" \
  -d '{"mode": "step", "scale_delta": 1}'
```

```json
{
  "observation": { ... },
  "reward": 0.62,
  "done": false,
  "info": {}
}
```

### `GET /state?session_id=<uuid>`

Get the current observation without advancing the step.

```bash
curl "http://localhost:7860/state?session_id=f47ac10b-..."
```

### `GET /health`

```bash
curl "http://localhost:7860/health"
# {"status": "ok"}
```

---

## Running the Baseline

```bash
export HF_TOKEN=your_token_here
export ENV_URL=http://localhost:7860

python inference.py
```

Expected output format:

```
[START] task=easy env=cloud-autoscaling model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"mode": "step", "scale_delta": 1} reward=0.72 done=false error=null
[STEP] step=2 action={"mode": "step", "scale_delta": 0} reward=0.65 done=false error=null
...
[END] success=true steps=50 score=0.731 rewards=0.72,0.65,...
```

---

## Simulation Details

- **Server startup delay:** 2 steps (new servers aren't instantly available)
- **Cooldown:** 3 steps after any scaling action before the next is allowed
- **Queue capacity:** 500 requests max — overflow is dropped
- **Latency model:** M/D/1 queuing approximation with non-linear blow-up above 80% utilization
- **Server capacity:** 100 req/s per server

---

## References

1. arXiv:2001.09957 — Survey on RL for cloud resource management (state design)
2. arXiv:2005.14410 — Serverless RL auto-scaling (reward design: latency + queue + over-provisioning)
3. IBM CLOUD 2024 — Latency/cost trade-off framing
4. Nature Scientific Reports 2025 — Multi-objective DQN reward structure
5. OpenEnv documentation — `step/reset/state` contract, typed models, yaml spec