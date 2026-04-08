"""Pydantic schemas for the cloud auto-scaling environment.

Defines typed models for the Observation, Action, and Reward types
required by the OpenEnv specification.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Structured environment state exposed to policies and the API.

    Fields mirror the scalar and trend components of the flat 39-dim
    observation vector used for RL training, but in a human-readable
    Pydantic model for heuristic baselines and API logging.
    """

    current_capacity: int = Field(..., ge=1, description="Active servers currently running")
    max_capacity: int = Field(..., ge=1, description="Maximum available server capacity")
    queue_length: float = Field(..., ge=0.0, description="Backlog of unserved requests")
    avg_utilization: float = Field(..., ge=0.0, description="λ / (c × μ)")
    cooldown_remaining: int = Field(..., ge=0, description="Steps until next scaling allowed")
    booting_servers: list[int] = Field(
        default_factory=list, description="Servers in the boot pipeline"
    )
    demand_rate: float = Field(..., ge=0.0, description="Current incoming request rate λ")
    service_rate: float = Field(..., gt=0.0, description="Per-server processing rate μ")
    p95_latency: float = Field(..., ge=0.0, description="Estimated p95 response latency (s)")
    trend: list[float] = Field(default_factory=list, description="Last 8 demand observations")
    util_trend: list[float] = Field(
        default_factory=list, description="Last 8 utilisation observations"
    )
    latency_trend: list[float] = Field(
        default_factory=list, description="Last 8 latency observations"
    )
    queue_trend: list[float] = Field(
        default_factory=list, description="Last 8 queue depth observations"
    )


class Action(BaseModel):
    """Structured action used by heuristic baselines and the step API.

    The three channels map to the environment's
    ``MultiDiscrete([5, 4, 2])`` action space.
    """

    step_scaling: int = Field(..., ge=-2, le=2, description="Servers to add/remove")
    target_utilization: float = Field(..., ge=0.4, le=0.7, description="Utilisation setpoint")
    capacity_signal: int = Field(..., ge=0, le=1, description="Signal insufficient capacity")


class Reward(BaseModel):
    """Structured reward signal for OpenEnv spec compliance."""

    value: float = Field(..., description="Raw reward signal for the current step")
    sla_penalty: float = Field(0.0, description="SLA breach cost component")
    waste_penalty: float = Field(0.0, description="Over-provisioning cost component")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized task score in [0.0, 1.0]",
    )
