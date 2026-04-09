from typing import Literal, Optional

from pydantic import BaseModel, Field


class CloudObservation(BaseModel):
    active_servers: int
    max_servers: int
    request_rate: float  # req/s arriving
    utilization: float  # avg across active servers [0,1]
    queue_length: int  # unserved requests waiting
    latency_ms: float  # estimated p99 latency
    target_utilization: float  # current target (mode 2)
    cooldown_remaining: int  # steps until next scale allowed
    step_number: int
    done: bool
    # Normalized fields (future-proof for RL)
    queue_ratio: float = Field(default=0.0)  # queue_length / MAX_QUEUE
    capacity_ratio: float = Field(default=0.0)  # active_servers / max_servers


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
