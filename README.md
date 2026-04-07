---
title: Cloud AutoScaling RL (OpenEnv)
emoji: ☁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Proxima Auto-Scaler ☁️

**Reinforcement Learning for Cloud Auto-Scaling** — Meta-PyTorch Hackathon Entry

An RL agent that learns to dynamically scale cloud server capacity using Proximal Policy Optimization (PPO). The environment simulates realistic M/M/c queueing dynamics with multi-phase workloads (sinusoidal, noisy, MMPP bursty) and delayed server boot times, forcing the agent to plan scaling decisions ahead of demand.

## Architecture

| Component | Description |
|-----------|-------------|
| **Environment** | Gymnasium-based cloud simulator with Erlang-C latency model, configurable difficulty tiers, and D_startup boot delay |
| **Agent** | PPO with custom GRU+MLP feature extractor (SB3), MultiDiscrete action space for step scaling, target utilization, and capacity signaling |
| **Baselines** | Threshold, Step Scaling, and Target Tracking (HPA-style) heuristic policies |
| **Training** | Curriculum learning across easy → medium → hard difficulties with observation normalization |

## API Endpoints

The FastAPI service exposes three endpoints:

- **`GET /health`** — Returns service status and whether the PPO model loaded successfully
- **`GET /evaluate?difficulty=<diff>&seed=<int>`** — Runs a full 200-step episode with the trained PPO agent, returning per-step traces and a normalized score
- **`GET /baselines?difficulty=<diff>&seed=<int>`** — Evaluates all three heuristic baselines on the same workload for direct comparison

## Quick Start

```bash
# Install
pip install -e .

# Train (curriculum: easy → medium → hard)
python -m src.agent.train --seed 42

# Evaluate
python -m src.evaluation.evaluator

# Run API server
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Tech Stack

Python 3.11 · PyTorch · Stable-Baselines3 · Gymnasium · FastAPI · Pydantic v2 · SciPy (Erlang-C)
