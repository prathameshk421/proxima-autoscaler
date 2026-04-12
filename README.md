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

Proxima Autoscalar is our hackathon system for intelligent cloud capacity control. It simulates realistic autoscaling constraints and evaluates how well an RL or LLM policy can balance latency, queue pressure, and infrastructure cost.

## Problem Statement

Cloud systems fail in two opposite ways:

- If scaling is too slow, latency spikes and requests accumulate in queue.
- If scaling is too aggressive, cost rises due to idle servers.

Proxima Autoscalar addresses this as a sequential decision problem where an agent must react to non-stationary demand while operating under operational constraints such as startup delay and cooldown.

## What We Delivered

1. A deterministic autoscaling benchmark with three progressively harder traffic regimes.
2. A complete OpenEnv-compatible environment API with session-level isolation.
3. A simulation stack that models cloud realities: delayed provisioning, cooldown locks, bounded queue, and capacity saturation.
4. A multi-objective reward design that encodes service quality, efficiency, and controller stability.
5. Policy evaluation support for both heuristic and LLM-based decision making.

## Technical Design

### 1. Workload Modeling

We built three workload profiles to stress different behaviors:

- smooth: periodic demand where trend-following policies should perform well.
- fluctuating: noisy traffic where robust controllers must avoid overreaction.
- spiky: burst-heavy traffic where capacity saturation handling becomes critical.

Each task is seed-controlled to keep experiments reproducible.

### 2. Capacity Dynamics

The server pool is modeled with production-like constraints:

- startup delay: new servers are not immediately active.
- scaling cooldown: consecutive scaling operations are rate-limited.
- hard capacity bounds: minimum and maximum server limits are enforced.

This prevents unrealistic instant scaling and creates a meaningful control challenge.

### 3. Queue and Latency Modeling

The environment computes arrivals, service capacity, and queue transitions at each step. Queue overflow is capped and dropped. Latency is estimated with an M/M/c queueing approximation, which captures nonlinear latency escalation near high utilization.

### 4. Decision Interface

The control interface supports three action modes:

- step mode for direct scale-up or scale-down deltas.
- target mode for utilization-driven scaling intent.
- signal mode for explicit insufficient-capacity signaling at saturation.

State observations include active capacity, demand, queue depth, latency, utilization, and normalization terms that simplify policy learning.

### 5. Reward Engineering

Reward is intentionally multi-objective and includes:

- throughput encouragement
- SLO violation penalty
- over-provisioning cost penalty
- queue penalty
- stability penalty for abrupt scaling
- invalid-action penalty (for cooldown and boundary violations)
- correctness bonus for valid insufficient-capacity signaling

The final step reward is clipped to stabilize downstream learning.

## Scenario Matrix

| Task | Episode Length | Capacity Ceiling | Main Evaluation Focus |
|---|---:|---:|---|
| easy | 50 | 10 | Basic tracking of predictable demand |
| medium | 75 | 12 | Noise tolerance and anti-thrashing behavior |
| hard | 100 | 15 | Burst handling, saturation response, and resilience |

## Implementation Outcomes

- Completed OpenEnv interaction contract for reset, step, state, and health endpoints.
- Added UUID-based session management to support parallel evaluation safely.
- Integrated strict action validation with error-aware reward shaping.
- Implemented deterministic task configuration for reproducible judging.
- Added inference and analysis workflow support for policy assessment.
- Added tests that verify core environment validity for valid and invalid actions.

## Why This Is Hackathon-Ready

- Demonstrates full-stack problem ownership: simulation, API surface, policy integration, and evaluation.
- Balances practical engineering with algorithmic rigor.
- Produces measurable outputs that judges can compare across policies and tasks.
- Provides a clear path from benchmark environment to real autoscaling controller research.
