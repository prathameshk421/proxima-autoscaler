# Intelligent Cloud Auto-Scaling Environment using Reinforcement Learning (Enhanced)

---

## 1. Problem Statement

Modern cloud systems must dynamically allocate compute resources to handle fluctuating user demand. Existing auto-scaling mechanisms (e.g., AWS Auto Scaling, Kubernetes HPA) rely on predefined rules, thresholds, or target tracking policies based on limited system metrics such as CPU utilization. These systems are primarily reactive and often fail to optimize across multiple objectives like cost, latency, and stability.

This project proposes an AI-driven environment where a Reinforcement Learning (RL) agent learns to make intelligent scaling decisions in a simulated cloud system.

Unlike traditional approaches, the agent operates using a **hybrid control strategy**, enabling it to:
- Perform direct scaling actions (increase/decrease servers)
- Adjust system-level targets (e.g., utilization thresholds)
- Operate under real-world constraints such as cooldown periods and capacity limits
- Detect and signal when demand exceeds available infrastructure capacity

The goal is to design an adaptive system that learns optimal scaling strategies under dynamic and uncertain workloads.

---

## 2. Objectives

- Simulate a realistic cloud environment with varying workload demand
- Enable an RL agent to take real-time scaling decisions
- Optimize for:
  - Latency (performance)
  - Cost (resource usage)
  - Stability (avoid oscillations)
- Incorporate real-world constraints:
  - Server capacity limits
  - Cooldown periods
  - Instance startup delays
- Allow agent to signal insufficient infrastructure capacity
- Provide a reproducible OpenEnv-compatible environment

---

## 3. System Overview

### 3.1 State Representation

The environment state includes:
- Current number of active servers
- Maximum available servers (capacity limit)
- Incoming request rate (workload)
- Server utilization
- Request queue length
- Estimated latency
- Current target utilization (if set)
- Cooldown timer (time remaining before next scaling allowed)
- Optional: historical demand trend

---

### 3.2 Action Space (Hybrid)

The RL agent uses a structured action space with multiple modes:

#### Mode 1: Step Scaling
- Scale +1
- Scale +2
- Scale -1
- Scale -2
- No change

#### Mode 2: Target Utilization Adjustment
- Set target utilization to one of:
  - 0.4
  - 0.5
  - 0.6
  - 0.7

Environment computes required servers as:
required_servers = workload / target_utilization

#### Mode 3: Capacity Signal
- Emit signal: INSUFFICIENT_CAPACITY
- Used when demand exceeds maximum available servers

---

### 3.3 Action Format

{
  "mode": "step" | "target" | "signal",
  "scale_delta": int,
  "target_utilization": float,
  "signal_type": "INSUFFICIENT_CAPACITY"
}

---

### 3.4 Environment Dynamics

The environment simulates:

- Workload patterns:
  - Predictable (smooth)
  - Fluctuating (random noise)
  - Spiky (sudden bursts)

- Server behavior:
  - Each server has finite capacity
  - New servers have startup delay

- Latency:
  - Increases sharply when demand exceeds capacity

- Queue:
  - Excess requests are queued or dropped

- Cooldown:
  - Scaling actions are restricted for a fixed time after execution

---

## 4. Reward Function

The reward function balances multiple objectives:

### Positive Reward
- Handling requests without latency issues
- Maintaining stable system behavior

### Penalties
- Under-provisioning:
  - High latency
  - Dropped/queued requests

- Over-provisioning:
  - Unnecessary server cost

- Oscillation:
  - Frequent scaling up/down

- Invalid actions:
  - Scaling during cooldown

- Mode switching:
  - Frequent switching between step and target modes

### Bonus
- Correctly signaling insufficient capacity when system saturates

---

## 5. Constraints

- Minimum and maximum server limits enforced
- Cooldown period between scaling actions
- Instance startup delay
- Non-linear latency under overload
- Limited observability (no perfect future knowledge)

---

## 6. Technology Stack

- Python
- NumPy / Pandas
- FastAPI (OpenEnv API)
- Docker
- Hugging Face Spaces
- Optional: Matplotlib dashboard

---

## 7. Tasks

### Easy
- Predictable workload
- Low variance

### Medium
- Fluctuating workload

### Hard
- Sudden spikes and unpredictable demand

Each task:
- Deterministic
- Reproducible scoring

---

## 8. Baselines

- Threshold-based scaling
- Step scaling policies
- Target tracking policies

---

## 9. Key Differentiation

| Existing Systems | Proposed System |
|----------------|---------------|
| Rule-based | Learning-based |
| Reactive | Adaptive |
| Single metric | Multi-objective |
| Static policies | Dynamic decisions |
| No capacity awareness | Detects saturation |

---

## 10. OpenEnv Compliance

- step(action)
- reset()
- state()
- Typed models (Observation, Action, Reward)
- openenv.yaml included
- Baseline inference script with reproducible results

---

## 11. Final Statement

This project demonstrates how reinforcement learning can enable adaptive, efficient, and stable cloud auto-scaling by replacing static rules with intelligent decision-making under real-world constraints.