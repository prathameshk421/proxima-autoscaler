---
title: Cloud AutoScaling RL (OpenEnv)
emoji: ☁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Intelligent Cloud Auto-Scaling using Reinforcement Learning

## Project Motivation
Rule-based autoscaling often reacts too late to bursts or overprovisions during low demand. This project provides a deterministic simulation and PPO-based policy that learns better latency-cost trade-offs under realistic cloud constraints.

## Environment Overview
The environment simulates queueing-inspired cloud dynamics with finite server capacity, startup delays, cooldown periods, non-linear latency under overload, dropped requests, and workload variability across easy, medium, and hard tasks.

## Why This Differs from Rule-Based Autoscaling
- Hybrid action space combines step scaling, target-utilization control, and explicit insufficient-capacity signaling.
- Reward shaping balances SLA quality, cost efficiency, stability, and capacity-awareness behavior.
- Deterministic evaluation across fixed seeds enables reproducible benchmarking.

## Observation Space
Typed observation model includes:
- infrastructure state: current_servers, max_servers, pending_startups, cooldown_remaining
- workload state: incoming_request_rate, demand_trend
- performance state: utilization, queue_length, estimated_latency, dropped_requests
- control context: target_utilization, task_difficulty

## Hybrid Action Space
- Step scaling: -2, -1, 0, +1, +2
- Target utilization: 0.4, 0.5, 0.6, 0.7
- Capacity signal: INSUFFICIENT_CAPACITY

## Reward Design
Reward is normalized to [0,1] and combines:
- latency/SLA quality
- cost efficiency
- stability and oscillation control
- capacity signal correctness

## Task Levels
- Easy: smooth predictable demand, low variance.
- Medium: sinusoidal demand with moderate stochastic variation.
- Hard: noisy demand with seeded burst spikes and stronger saturation pressure.

## Baselines
Three deterministic baselines are included:
- threshold policy
- aggressive step policy
- target-tracking policy

## Baseline Inference Script (OpenAI-Compatible)
The project includes a reproducible baseline inference runner in [baseline_inference.py](baseline_inference.py).

Required environment variables for LLM mode:
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN (or OPENAI_API_KEY)

Run deterministic rule baseline (fully offline):
- python baseline_inference.py --agent threshold --tasks easy medium hard --seeds 42 123 456

Run OpenAI-compatible LLM baseline:
- python baseline_inference.py --agent llm --tasks easy medium hard --seeds 42 123 456

The script prints benchmark-compatible logs:
- [START] task=... env=... model=...
- [STEP] step=... action=... reward=... done=... error=...
- [END] success=... steps=... score=... rewards=...

It also writes a report JSON (default [baseline_inference_report.json](baseline_inference_report.json)).

## Baseline Scores (Deterministic Evaluation Summary)
From [evaluation_report.json](evaluation_report.json) (mean over seeds 42, 123, 456):

| Difficulty | Threshold | Step | Target Tracking | PPO |
| --- | ---: | ---: | ---: | ---: |
| easy | 0.7734 | 0.5605 | 0.1331 | 1.0000 |
| medium | 0.4135 | 0.0000 | 0.0000 | 0.9846 |
| hard | 0.0463 | 0.0000 | 0.0000 | 0.9717 |

## Training Overview
PPO training runs as curriculum phases:
- easy: 50,000 steps
- medium: 100,000 steps
- hard: 150,000 steps

Checkpoints are saved in the checkpoints directory for deployment and evaluation.

## Local Setup
1. Create and activate a Python 3.11 virtual environment.
2. Install project package:
	pip install -e .
3. Run tests:
	pytest -q
4. Train PPO:
	python src/agent/train.py --seed 42

## Docker Run
1. Build image:
	docker build -t cloud-autoscaling-rl .
2. Run container:
	docker run -p 7860:7860 cloud-autoscaling-rl

## API Usage Examples
- Health check:
  GET /health
- PPO evaluation:
  GET /evaluate?difficulty=hard&seed=42
- Baseline comparison:
  GET /baselines?difficulty=easy&seed=42

## Hugging Face Spaces Deployment
This repository is configured for Docker-based Spaces deployment on port 7860. Commit trained checkpoint files in checkpoints for reproducible hosted inference.

## OpenEnv Compliance
The environment exposes reset, step, and state methods with typed Observation, Action, and Reward schemas, deterministic task generation, and reproducible grading.
