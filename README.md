---
title: Cloud AutoScaling RL (OpenEnv)
emoji: ☁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - cloud-autoscaling
---

# Proxima Auto-Scaler ☁️

**Reinforcement Learning for Cloud Auto-Scaling** — Meta-PyTorch Hackathon Entry

An RL agent that learns to dynamically scale cloud server capacity using Proximal Policy Optimization (PPO). The environment simulates realistic M/M/c queueing dynamics with multi-phase workloads (sinusoidal, noisy, MMPP bursty) and delayed server boot times, forcing the agent to plan scaling decisions ahead of demand.

---

## Architecture

| Component | Description |
|-----------|-------------|
| **Environment** | Gymnasium-based cloud simulator with Erlang-C latency model, configurable difficulty tiers, and D_startup boot delay |
| **Agent** | PPO with custom GRU+MLP feature extractor (SB3), MultiDiscrete action space for step scaling, target utilization, and capacity signaling |
| **Baselines** | Threshold, Step Scaling, and Target Tracking (HPA-style) heuristic policies |
| **Training** | Curriculum learning across easy → medium → hard difficulties with observation normalization |
| **Inference** | LLM-based agent using OpenAI API client for OpenEnv-compliant evaluation |

---

## Observation Space

The environment exposes a 15-dimensional continuous observation vector:

| Index | Feature | Range | Description |
|:-----:|---------|:-----:|-------------|
| 0 | `current_capacity` | [1, c_max] | Active servers currently running |
| 1 | `max_capacity` | [6–20] | Maximum servers available (difficulty-dependent) |
| 2 | `total_booting` | [0, c_max] | Servers in the boot pipeline |
| 3 | `cooldown_remaining` | [0, 3] | Steps until next scaling action is allowed |
| 4 | `queue_length` | [0, ∞) | Backlog of unserved requests |
| 5 | `demand_rate` (λ) | [0, ∞) | Current incoming request rate |
| 6 | `service_rate` (μ) | 4.5–5.0 | Per-server service capacity |
| 7–14 | `trend_window` | [0, ∞) | Last 8 demand rate observations (temporal context) |

### Pydantic Schema (`src.env.schemas.Observation`)

```python
class Observation(BaseModel):
    current_capacity: int     # [1, c_max]
    max_capacity: int         # difficulty-dependent max
    queue_length: float       # ≥ 0
    avg_utilization: float    # λ / (c × μ)
    cooldown_remaining: int   # [0, cooldown_period]
    booting_servers: List[int]
    demand_rate: float        # current λ
    service_rate: float       # per-server μ
    p95_latency: float        # estimated p95 latency (seconds)
    trend: List[float]        # last 8 demand observations
```

---

## Action Space

**MultiDiscrete([5, 4, 2])** — three simultaneous control channels:

| Channel | Name | Values | Description |
|:-------:|------|:------:|-------------|
| 0 | Step Scaling | {0,1,2,3,4} → {-2,-1,0,+1,+2} | Servers to add/remove |
| 1 | Target Utilization | {0,1,2,3} → {0.4, 0.5, 0.6, 0.7} | Utilization setpoint |
| 2 | Capacity Signal | {0, 1} | Signal insufficient capacity |

### Pydantic Schema (`src.env.schemas.Action`)

```python
class Action(BaseModel):
    step_scaling: int          # [-2, +2]
    target_utilization: float  # {0.4, 0.5, 0.6, 0.7}
    capacity_signal: int       # {0, 1}
```

---

## Reward Function

The reward function balances four competing objectives per step:

```
reward = 0.5                           # base survival reward
       - 12.0 × max(0, p95 - T_sla)   # SLA breach penalty
       - 0.008 × excess_capacity²      # quadratic waste penalty
       - 0.04 × |step_scaling|         # action magnitude penalty
       - 0.15 × max(0, util - 1.0)    # congestion penalty
       - 0.05 × false_cap_signal       # false capacity signal penalty
```

### Pydantic Schema (`src.env.schemas.Reward`)

```python
class Reward(BaseModel):
    value: float       # raw step reward
    sla_penalty: float # SLA breach component
    waste_penalty: float # over-provisioning cost
    score: float       # normalised task score ∈ [0.0, 1.0]
```

Score normalisation: `score = clip((Σreward - min_possible) / (max_possible - min_possible), 0, 1)` where `min_possible = -200 × 1.5` and `max_possible = 200 × 0.5`.

---

## Tasks

Three difficulty tiers with deterministic, reproducible grading:

| Task | Difficulty | λ_base | λ_amp | Noise σ | Boot Delay | c_max | Description |
|------|:----------:|:------:|:-----:|:-------:|:----------:|:-----:|-------------|
| `cloud_autoscaling_easy` | Easy | 16.0 | 6.0 | 1.0 | 1 step | 12 | Smooth sinusoidal workload, minimal noise |
| `cloud_autoscaling_medium` | Medium | 20.0 | 9.0 | 1.6 | 2 steps | 16 | Fluctuating demand + MMPP burst patterns |
| `cloud_autoscaling_hard` | Hard | 24.0 | 12.0 | 2.2 | 3 steps | 20 | Aggressive spikes, high noise, long boot |

Each task runs for 200 timesteps with a fixed random seed for reproducibility.

---

## Baseline Scores

Scores normalised to [0.0, 1.0] (higher is better). Evaluated with `seed=42`:

| Policy | Easy | Medium | Hard |
|--------|:----:|:------:|:----:|
| Threshold | 0.72 | 0.65 | 0.55 |
| Step Scaling | 0.74 | 0.67 | 0.58 |
| Target Tracking | 0.76 | 0.70 | 0.62 |
| **PPO (ours)** | **0.88** | **0.83** | **0.78** |

PPO consistently outperforms all heuristic baselines, especially on hard workloads where proactive planning matters most.

---

## API Endpoints

### OpenEnv Spec Endpoints

| Method | Path | Description |
|:------:|------|-------------|
| `POST` | `/reset?difficulty=<d>&seed=<s>` | Reset environment for a new episode |
| `POST` | `/step?difficulty=<d>&step_scaling=<n>&target_utilization=<f>&capacity_signal=<n>` | Execute one step |
| `GET` | `/state?difficulty=<d>` | Return current observation |
| `GET` | `/tasks` | List available tasks with descriptions |

### Additional Endpoints

| Method | Path | Description |
|:------:|------|-------------|
| `GET` | `/health` | Service health + model status |
| `GET` | `/evaluate?difficulty=<d>&seed=<s>` | Full 200-step PPO evaluation |
| `GET` | `/baselines?difficulty=<d>&seed=<s>` | Evaluate all heuristic baselines |

---

## Quick Start

```bash
# Install
pip install -e .

# Train (curriculum: easy → medium → hard)
python -m src.agent.train --seed 42

# Evaluate locally
python -m src.evaluation.evaluator

# Run LLM inference (set env vars first)
export HF_TOKEN=<your_token>
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference_llm.py

# Run API server
uvicorn app:app --host 0.0.0.0 --port 7860
```

---

## Project Structure

```
proxima-autoscaler/
├── app.py                    # FastAPI application (OpenEnv endpoints)
├── inference_llm.py          # LLM baseline inference script
├── openenv.yaml              # OpenEnv specification
├── Dockerfile                # Container definition
├── pyproject.toml            # Dependencies
├── checkpoints/              # Trained model weights
│   └── ppo_cloud_final.zip
└── src/
    ├── env/
    │   ├── cloud_env.py      # CloudAutoScalingEnv (Gymnasium)
    │   ├── schemas.py        # Observation, Action, Reward models
    │   ├── queue_model.py    # M/M/c queueing model
    │   └── workload.py       # Workload generators
    ├── agent/
    │   ├── networks.py       # GRU+MLP feature extractor
    │   ├── ppo_agent.py      # PPO agent builder
    │   └── train.py          # Curriculum training script
    ├── baselines/
    │   └── heuristics.py     # Threshold, Step, Target Tracking
    └── evaluation/
        ├── evaluator.py      # Multi-baseline evaluator
        └── inference.py      # PPO inference module
```

---

## Tech Stack

Python 3.11 · PyTorch · Stable-Baselines3 · Gymnasium · FastAPI · Pydantic v2 · SciPy (Erlang-C) · OpenAI SDK
