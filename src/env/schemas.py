# src/env/schemas.py
"""Pydantic v2 schemas for the Proxima Auto-Scaler RL environment.

Defines the Observation, Action, and Reward data models used across the
OpenEnv-compatible environment, training pipeline, and inference API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D_STARTUP: int = 4  # fixed boot-countdown vector length (tune in config)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
class Observation(BaseModel):
    """Complete state visible to the RL agent at each timestep."""

    current_capacity: int = Field(ge=0, description="Active server count")

    booting_servers: list[int] = Field(
        default_factory=lambda: [0] * D_STARTUP,
        description=(
            "Fixed-length countdown vector of length D_STARTUP. "
            "Each element counts remaining boot ticks for a pending server."
        ),
    )

    next_server_ready: int = Field(
        ge=0,
        description="Ticks until the next booting server becomes active",
    )

    max_capacity: int = Field(ge=1, description="Hard ceiling on server count")

    request_rate: float = Field(
        ge=0.0, description="Incoming request rate (req/s)"
    )

    avg_utilization: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean CPU/memory utilisation across active servers",
    )

    queue_length: int = Field(
        ge=0, description="Requests currently waiting in the queue"
    )

    latency_p95: float = Field(
        ge=0.0, description="95th-percentile response latency (ms)"
    )

    cooldown_remaining: float = Field(
        ge=0.0,
        description="Seconds remaining before another scaling action is permitted",
    )

    target_utilization: float = Field(
        ge=0.4,
        le=0.7,
        description="Current target utilisation setpoint",
    )

    workload_trend: list[float] = Field(
        min_length=5,
        max_length=10,
        description="Sliding window of recent request-rate samples (5–10 values)",
    )

    # -- validators ----------------------------------------------------------

    @field_validator("booting_servers")
    @classmethod
    def _booting_servers_exact_length(cls, v: list[int]) -> list[int]:
        """Enforce that booting_servers is always exactly D_STARTUP elements."""
        if len(v) != D_STARTUP:
            raise ValueError(
                f"booting_servers must have exactly {D_STARTUP} elements "
                f"(D_STARTUP), got {len(v)}"
            )
        return v


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
class Action(BaseModel):
    """Agent output applied to the environment each timestep."""

    step_scaling: int = Field(
        default=0,
        ge=-2,
        le=2,
        description="Discrete scale-in/out delta (−2 … +2 servers)",
    )

    target_utilization: float = Field(
        default=0.5,
        ge=0.4,
        le=0.7,
        description="Desired utilisation setpoint for the current step",
    )

    capacity_signal: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Binary pre-warm flag: 1 = bring extra capacity online now",
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
class Reward(BaseModel):
    """Scalar reward returned to the agent plus a diagnostic breakdown."""

    value: float = Field(description="Aggregated scalar reward for this timestep")

    components: dict[str, float] = Field(
        description=(
            "Breakdown of cost, latency, oscillation, "
            "invalid action, and capacity signal factors"
        ),
    )
