"""
Phase 2 — Simulation Core
Deterministic workload generator, server pool, and latency model.
No environment logic yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

SERVER_CAPACITY: int = 100      # req/s per server
STARTUP_DELAY: int = 2          # steps before a new server becomes active
COOLDOWN_STEPS: int = 3         # steps locked after any scaling action
MAX_QUEUE: int = 500            # drop requests beyond this
LATENCY_BASE_MS: float = 10.0
STEP_DURATION_SEC: float = 15.0 # seconds per step (matches Kubernetes HPA scrape interval)
                                # CRITICAL: all rate→count conversions must multiply by this


# ── Latency Model (M/M/c Erlang C) ──────────────────────────────────────────

def compute_mmc_latency(request_rate: float, active_servers: int,
                        latency_base_ms: float = LATENCY_BASE_MS) -> float:
    """Return estimated latency in ms using the M/M/c (Erlang C) formula.

    M/M/c is the industry standard for homogeneous pod/VM clusters.
    It naturally produces asymptotic latency explosion near 100% utilization.
    """
    c = max(1, min(active_servers, 15))   # cap at 15 — prevents factorial overflow
    mu = float(SERVER_CAPACITY)            # service rate per server (req/s)
    rho = request_rate / (c * mu)         # system utilization factor

    # Overload state: theoretical latency → ∞, return timeout cap
    if rho >= 0.99:
        return latency_base_ms + 15000.0

    # Erlang C formula: probability that a request must wait
    crho = c * rho
    numerator = (crho ** c) / (math.factorial(c) * (1 - rho))
    sum_k = sum((crho ** k) / math.factorial(k) for k in range(c))
    p_wait = numerator / (sum_k + numerator)

    # Expected wait time in queue: seconds → milliseconds
    wait_time_ms = (p_wait / (c * mu * (1 - rho))) * 1000.0

    return latency_base_ms + wait_time_ms


# ── WorkloadGenerator ────────────────────────────────────────────────────────

class WorkloadGenerator:
    """Pre-generates a deterministic workload trajectory for a given profile.

    Supported profiles:
        - ``smooth``       – Non-Stationary Poisson Process (pure sinusoidal rate).
        - ``fluctuating``  – Poisson with lognormal rate multiplier (right-skewed).
        - ``spiky``        – fluctuating + Pareto burst events.

    The full trajectory is computed in ``__init__`` so that ``get(step)`` is
    order-independent and fully reproducible given the same *seed*.

    Critical: ALL numpy RNG calls happen inside ``__init__`` using vectorized
    array operations. The seeded ``rng`` is consumed in exactly the same order
    every time ``reset()`` re-creates the generator — guaranteeing determinism.
    """

    VALID_PROFILES = ("smooth", "fluctuating", "spiky")

    def __init__(self, profile: str, seed: int, max_steps: int,
                 base_amp: float = 300.0, period: int = 50,
                 spike_probability: float = 0.0):  # spike_probability kept for API compat, ignored
        rng = np.random.default_rng(seed)
        t = np.arange(max_steps)

        # Base diurnal cycle — sinusoidal, always
        lambda_t = base_amp * np.sin(2 * np.pi * t / period) + base_amp
        lambda_t = np.maximum(10.0, lambda_t)   # minimum traffic floor

        if profile == "smooth":
            # Pure Non-Stationary Poisson Process
            self._workload = rng.poisson(lambda_t).astype(float)

        elif profile == "fluctuating":
            # Lognormal multiplier: right-skewed, never negative
            lognormal_multipliers = rng.lognormal(mean=0.0, sigma=0.25, size=max_steps)
            lambda_noisy = lambda_t * lognormal_multipliers
            self._workload = rng.poisson(lambda_noisy).astype(float)

        elif profile == "spiky":
            lognormal_multipliers = rng.lognormal(mean=0.0, sigma=0.25, size=max_steps)
            lambda_noisy = lambda_t * lognormal_multipliers
            base = rng.poisson(lambda_noisy).astype(float)
            # Pareto spikes: 2% probability, alpha=2.0, capped at 10x base_amp
            spike_mask = rng.random(size=max_steps) < 0.02
            spike_multipliers = np.minimum(rng.pareto(a=2.0, size=max_steps), 10.0)
            spikes = (spike_mask * base_amp * spike_multipliers).astype(float)
            self._workload = base + spikes

        else:
            # Fallback — treat as smooth
            self._workload = rng.poisson(lambda_t).astype(float)

    def get(self, step: int) -> float:
        return float(self._workload[step])

    @property
    def trajectory(self) -> np.ndarray:
        """Full pre-generated workload array (read-only copy)."""
        return self._workload.copy()


# ── ServerPool ───────────────────────────────────────────────────────────────

@dataclass
class _PendingServer:
    """A server that has been requested but is not yet active."""
    arrival_step: int


@dataclass
class ServerPool:
    """Manages a pool of active and pending servers with cooldown logic.

    Parameters
    ----------
    initial : int
        Number of servers that start active immediately.
    min_s : int
        Minimum number of servers (scaling floor).
    max_s : int
        Maximum number of servers (scaling ceiling).
    """

    active: int
    min_s: int
    max_s: int
    pending: List[_PendingServer] = field(default_factory=list)
    cooldown_until: int = 0   # step at which cooldown expires

    # ── construction helper ──────────────────────────────────────────────

    def __init__(self, initial: int, min_s: int, max_s: int) -> None:
        self.active = int(np.clip(initial, min_s, max_s))
        self.min_s = min_s
        self.max_s = max_s
        self.pending: List[_PendingServer] = []
        self.cooldown_until: int = 0

    # ── public API ───────────────────────────────────────────────────────

    def apply_action(self, delta: int, current_step: int) -> int:
        """Request to add/remove *delta* servers.

        Returns the actual delta applied (after cooldown & clamping).

        * Respects cooldown — returns 0 if still in cooldown.
        * Clamps the resulting server count to ``[min_s, max_s]``.
        * New servers are placed in *pending* with
          ``arrival_step = current_step + STARTUP_DELAY``.
        * Removed servers are taken from *active* immediately.
        """
        if current_step < self.cooldown_until:
            return 0

        if delta == 0:
            return 0

        total = self.active + len(self.pending)

        if delta > 0:
            # Scale up — add pending servers
            max_add = self.max_s - total
            actual = min(delta, max_add)
            if actual <= 0:
                return 0
            for _ in range(actual):
                self.pending.append(
                    _PendingServer(arrival_step=current_step + STARTUP_DELAY)
                )
        else:
            # Scale down — remove active servers immediately
            max_remove = self.active - self.min_s
            actual = -min(-delta, max_remove)  # actual is negative
            if actual >= 0:
                return 0
            self.active += actual  # subtracts

        self.cooldown_until = current_step + COOLDOWN_STEPS
        return actual

    def tick(self, current_step: int) -> int:
        """Graduate pending servers whose arrival_step <= current_step.

        Returns the number of servers that became active this tick.
        """
        graduated = 0
        still_pending: List[_PendingServer] = []
        for srv in self.pending:
            if srv.arrival_step <= current_step:
                self.active += 1
                graduated += 1
            else:
                still_pending.append(srv)
        self.pending = still_pending
        return graduated

    @property
    def effective_capacity(self) -> float:
        """Total request-handling capacity of all *active* servers."""
        return float(self.active * SERVER_CAPACITY)

    @property
    def total_servers(self) -> int:
        """Active + pending server count."""
        return self.active + len(self.pending)

    def cooldown_remaining(self, current_step: int) -> int:
        """Steps remaining before a new scaling action is allowed."""
        return max(0, self.cooldown_until - current_step)


# ── Exit-condition script ────────────────────────────────────────────────────

def _print_trace(seed: int, run_label: str) -> None:
    """Print the first 50 steps for each profile with the given seed."""
    print(f"\n{'='*60}")
    print(f"  Run: {run_label}  |  seed={seed}")
    print(f"{'='*60}")
    for profile in WorkloadGenerator.VALID_PROFILES:
        sp = 0.15 if profile == "spiky" else 0.0
        gen = WorkloadGenerator(profile=profile, seed=seed, max_steps=200,
                                spike_probability=sp)
        values = [gen.get(s) for s in range(50)]
        print(f"\n[{profile}] first 50 steps:")
        for i in range(0, 50, 10):
            chunk = values[i:i + 10]
            formatted = "  ".join(f"{v:8.2f}" for v in chunk)
            print(f"  step {i:3d}-{i+9:3d}: {formatted}")


if __name__ == "__main__":
    SEED = 42
    _print_trace(SEED, "Run-1")
    _print_trace(SEED, "Run-2")
    print("\n✅  If both runs are identical the generator is deterministic.")
