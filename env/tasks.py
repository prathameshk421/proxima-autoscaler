"""
Phase 4 — Task Configurations

Each task defines a workload profile, episode length, server limits, and seed
so that episodes are fully deterministic and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    """Immutable specification for a single environment task."""

    name: str
    profile: str               # WorkloadGenerator profile
    max_steps: int             # episode length
    initial_servers: int       # servers active at reset
    min_servers: int           # scaling floor
    max_servers: int           # scaling ceiling
    seed: int                  # RNG seed for deterministic workloads
    spike_probability: float   # burst probability (spiky profile only)


TASKS: dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name="easy",
        profile="smooth",
        max_steps=50,
        initial_servers=3,
        min_servers=1,
        max_servers=10,
        seed=42,
        spike_probability=0.0,
    ),
    "medium": TaskConfig(
        name="medium",
        profile="fluctuating",
        max_steps=75,
        initial_servers=3,
        min_servers=1,
        max_servers=12,
        seed=42,
        spike_probability=0.0,
    ),
    "hard": TaskConfig(
        name="hard",
        profile="spiky",
        max_steps=100,
        initial_servers=3,
        min_servers=1,
        max_servers=15,
        seed=42,
        spike_probability=0.15,
    ),
}
