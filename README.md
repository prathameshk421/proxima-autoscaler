---
title: Proxima Autoscalar
sdk: docker
app_port: 7860
tags:
  - openenv
  - cloud
  - autoscaling
  - reinforcement-learning
---

# Proxima Autoscalar

Proxima Autoscalar is our hackathon project: an OpenEnv-compatible cloud autoscaling environment where an RL or LLM policy controls server scaling under changing traffic.

## What We Built

- A deterministic autoscaling simulation with realistic constraints.
- A FastAPI environment server compatible with OpenEnv workflow.
- Three difficulty levels (`easy`, `medium`, `hard`) for policy evaluation.
- Inference scripts for both heuristic and LLM policies.

## Core Environment

At each step, the agent reads system state and sends one scaling action.

Observation fields:

- `active_servers`
- `max_servers`
- `request_rate`
- `utilization`
- `queue_length`
- `latency_ms`
- `target_utilization`
- `cooldown_remaining`
- `step_number`
- `done`
- `queue_ratio`
- `capacity_ratio`

Action schema:

```json
{
  "mode": "step | target | signal",
  "scale_delta": -2,
  "target_utilization": 0.6,
  "signal_type": "INSUFFICIENT_CAPACITY"
}
```

Control modes:

- `step`: scale by integer delta (`-2` to `+2`).
- `target`: provide utilization target and let environment compute scaling.
- `signal`: emit `INSUFFICIENT_CAPACITY` when saturated.

Reward combines throughput, SLO pressure, cost, queue pressure, stability, invalid-action penalty, and signal bonus.

## Task Levels

| Task | Profile | Steps | Max Servers |
|---|---|---:|---:|
| easy | smooth | 50 | 10 |
| medium | fluctuating | 75 | 12 |
| hard | spiky | 100 | 15 |

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.main:app --reload --port 7860
```

Health check:

```bash
curl http://localhost:7860/health
```

## API

- `POST /reset?task=easy|medium|hard`
- `POST /step?session_id=<uuid>`
- `GET /state?session_id=<uuid>`
- `GET /health`

Example:

```bash
curl -X POST "http://localhost:7860/reset?task=easy"
curl -X POST "http://localhost:7860/step?session_id=<uuid>" \
  -H "Content-Type: application/json" \
  -d '{"mode":"step","scale_delta":1}'
```

## Run Inference

Heuristic baseline:

```bash
POLICY_MODE=heuristic ENV_URL=http://localhost:7860 python inference.py
```

LLM policy:

```bash
HF_TOKEN=<token> POLICY_MODE=llm ENV_URL=http://localhost:7860 python inference.py
```

Analysis runner:

```bash
python run_inference.py --local --task easy --max-steps 12
```

## Validation

```bash
pytest -q
openenv validate
bash pre_validation_script.sh <hf-space-url> .
```

## Project Structure

- `env/`: simulation, reward, task config, models
- `server/main.py`: OpenEnv API server
- `openenv.yaml`: OpenEnv metadata
- `inference.py`: submission-format inference script
- `run_inference.py`: run and analyze episodes
- `tests/test_environment.py`: behavior tests
