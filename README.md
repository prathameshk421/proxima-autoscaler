---
title: Cloud Autoscaling OpenEnv
sdk: docker
app_port: 7860
tags:
  - openenv
  - cloud
  - autoscaling
  - reinforcement-learning
---

# Cloud Autoscaling OpenEnv

## Abstract
This repository implements an OpenEnv-compatible benchmark for cloud autoscaling control. The environment models non-stationary request demand, delayed server provisioning, cooldown-constrained control, finite queue capacity, and queueing-theoretic latency behavior. It is intended for controlled evaluation of reinforcement learning and LLM-based policies under deterministic task seeds.

## Research Objective
The benchmark studies a standard control trade-off in cloud systems:

- Under-provisioning increases queue buildup and latency violations.
- Over-provisioning reduces delay but increases idle-capacity cost.
- Aggressive scaling can destabilize control decisions.

The policy objective is to maximize cumulative reward while preserving service quality and cost efficiency across an episode.

## Formal Environment Definition
At each timestep, the environment returns an observation vector and accepts one action. The transition includes workload evolution, server activation delays, queue updates, and latency estimation. The step API returns:

- next observation
- scalar reward
- termination flag
- reward-component metadata in `info`

### Observation Space

| Field | Type | Description |
|---|---|---|
| `active_servers` | int | Active server count |
| `max_servers` | int | Capacity ceiling for the task |
| `request_rate` | float | Incoming requests per second |
| `utilization` | float | Approximate utilization in [0, 1] |
| `queue_length` | int | Requests waiting in queue |
| `latency_ms` | float | Estimated latency (ms) |
| `target_utilization` | float | Current target setpoint for target mode |
| `cooldown_remaining` | int | Steps before another scale action is allowed |
| `step_number` | int | Timestep index |
| `done` | bool | Episode termination indicator |
| `queue_ratio` | float | Queue occupancy normalized by max queue |
| `capacity_ratio` | float | Active servers normalized by task max |

### Action Space

| Field | Type | Description |
|---|---|---|
| `mode` | `"step" | "target" | "signal"` | Control mode |
| `scale_delta` | int | Server delta in step mode (clamped to [-2, 2]) |
| `target_utilization` | float | Utilization target in target mode |
| `signal_type` | `"INSUFFICIENT_CAPACITY" | null` | Capacity-exhaustion signal in signal mode |

Mode semantics:

- `step`: direct horizontal scaling by integer server delta.
- `target`: HPA-style target utilization control with deadband.
- `signal`: report capacity saturation without changing server count.

### Reward Components
The total reward is composed from throughput, SLO penalty, cost, queue pressure, action stability, invalid-action penalty, and optional signal bonus.

Component ranges in implementation:

- throughput term: [0.0, 1.0]
- SLO term (weighted negative): [-15.0, 0.0]
- cost term: [-0.5, 0.0]
- queue term: [-1.0, 0.0]
- stability term (weighted negative): [-0.2, 0.0]
- invalid-action term: {-0.5, 0.0}
- signal bonus: {0.0, 1.0}

Final per-step reward is clipped to [-5.0, 2.0].

### Episode Score Normalization
Evaluation score is normalized to [0, 1] from cumulative reward:

```text
score = (sum_rewards - worst_possible) / (best_possible - worst_possible)
```

where `worst_possible = -5.0 * max_steps` and `best_possible = 2.0 * max_steps`.

## System Dynamics
The simulation currently uses the following assumptions:

- server capacity: 100 requests/second/server
- startup delay: 2 steps for newly added servers
- cooldown: 3 steps after each scaling action
- queue cap: 500 requests (overflow is dropped)
- step duration: 15 seconds
- latency model: M/M/c (Erlang C)

Queue update logic is based on demand-service balance per step:

- arrivals = request_rate * step_duration
- service_capacity = active_servers * server_capacity * step_duration
- queue_next = min(max_queue, max(0, queue + arrivals - fulfilled))

## Task Configuration

| Task | Workload Profile | Steps | Initial Servers | Max Servers | Seed |
|---|---|---:|---:|---:|---:|
| easy | smooth sinusoidal demand | 50 | 3 | 10 | 42 |
| medium | fluctuating demand with noise | 75 | 3 | 12 | 42 |
| hard | spiky demand with burst events | 100 | 3 | 15 | 42 |

## Repository Structure

- `env/`: environment, simulation, reward, models, and task configuration
- `server/main.py`: FastAPI service implementing reset/step/state endpoints
- `openenv.yaml`: OpenEnv metadata (name, tasks, spaces)
- `inference.py`: submission-format inference script
- `run_inference.py`: local/remote analysis runner
- `tests/test_environment.py`: environment behavior tests
- `pre_validation_script.sh`: end-to-end validation helper

## Setup

### Local Development

```bash
git clone <repo-url>
cd proxima-autoscaler

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn server.main:app --reload --port 7860
```

### Docker

```bash
docker build -t cloud-env .
docker run -p 7860:7860 cloud-env
```

### Health Check

```bash
curl http://localhost:7860/health
```

## API Contract
All endpoints are served from `http://localhost:7860` or your deployed URL.

### Reset

```bash
curl -X POST "http://localhost:7860/reset?task=easy"
```

Returns `session_id` and initial observation.

### Step

```bash
curl -X POST "http://localhost:7860/step?session_id=<uuid>" \
  -H "Content-Type: application/json" \
  -d '{"mode":"step","scale_delta":1}'
```

Returns observation, reward, done flag, and info.

### State

```bash
curl "http://localhost:7860/state?session_id=<uuid>"
```

Returns current observation without advancing the episode.

## Inference and Evaluation

### Submission Script (`inference.py`)
The script emits strict line-oriented logs expected by the evaluation pipeline:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

Required environment variables for LLM mode:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` (or `API_KEY`)

Heuristic run:

```bash
POLICY_MODE=heuristic ENV_URL=http://localhost:7860 python inference.py
```

LLM run:

```bash
HF_TOKEN=<token> POLICY_MODE=llm ENV_URL=http://localhost:7860 python inference.py
```

### Analysis Script (`run_inference.py`)
Example local run:

```bash
python run_inference.py --local --task easy --max-steps 12
```

## Validation

Run unit tests:

```bash
pytest -q
```

Run OpenEnv validation:

```bash
openenv validate
```

Run deployment validation helper:

```bash
bash pre_validation_script.sh <hf-space-url> .
```

## Reproducibility Notes

- Task workloads are deterministic for fixed seeds.
- Session IDs are UUID-based to support concurrent evaluators.
- Use fixed model, temperature, and timeout settings for comparable experiments.
- Compare local and deployed runs with identical task and policy mode.

## References

1. Cloud autoscaling control literature on latency-cost trade-offs in RL.
2. Queueing theory references for M/M/c latency approximation (Erlang C).
3. OpenEnv API specification and validation workflow.
