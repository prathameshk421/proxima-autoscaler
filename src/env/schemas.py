"""Pydantic schemas for the cloud auto-scaling environment."""

from typing import List

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Structured state exposed to heuristic policies and API logs."""

    current_capacity: int = Field(..., ge=1)
    max_capacity: int = Field(..., ge=1)
    queue_length: float = Field(..., ge=0.0)
    avg_utilization: float = Field(..., ge=0.0)
    cooldown_remaining: int = Field(..., ge=0)
    booting_servers: List[int] = Field(default_factory=list)
    demand_rate: float = Field(..., ge=0.0)
    service_rate: float = Field(..., gt=0.0)
    p95_latency: float = Field(..., ge=0.0)
    trend: List[float] = Field(default_factory=list)


class Action(BaseModel):
    """Structured action used by heuristic baselines."""

    step_scaling: int = Field(..., ge=-2, le=2)
    target_utilization: float = Field(..., ge=0.4, le=0.7)
    capacity_signal: int = Field(..., ge=0, le=1)
