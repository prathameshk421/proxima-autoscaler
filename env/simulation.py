"""
Phase 2 — Simulation Core
Deterministic workload generator, server pool, and latency model.
No environment logic yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

SERVER_CAPACITY: int = 100      # req/s per server
STARTUP_DELAY: int = 2          # steps before a new server becomes active
COOLDOWN_STEPS: int = 3         # steps locked after any scaling action
MAX_QUEUE: int = 500            # drop requests beyond this
LATENCY_BASE_MS: float = 10.0
LATENCY_SCALE: float = 5.0


# ── Latency Model (M/D/1 queuing approximation) ─────────────────────────────

def compute_latency(request_rate: float, capacity: float,
                    queue_length: int) -> float:
    """Return estimated latency in ms using a piecewise M/D/1 approximation.

    Below 80 % utilization the relationship is roughly linear; above that
    threshold it grows quadratically to simulate queuing blow-up, with an
    additional term proportional to queue depth.
    """
    utilization = min(request_rate / max(capacity, 1.0), 1.0)
    if utilization < 0.8:
        return LATENCY_BASE_MS + utilization * 20.0
    overflow = (utilization - 0.8) / 0.2
    return (LATENCY_BASE_MS + 20.0
            + overflow ** 2 * 200.0
            + queue_length * 0.5)


# ── WorkloadGenerator ────────────────────────────────────────────────────────

class WorkloadGenerator:
    """Pre-generates a deterministic workload trajectory for a given profile.

    Supported profiles:
        - ``smooth``       – sinusoidal with small Gaussian noise.
        - ``fluctuating``  – smooth + moderate noise.
        - ``spiky``        – fluctuating + random burst events.

    The full trajectory is computed in ``__init__`` so that ``get(step)`` is
    order-independent and fully reproducible given the same *seed*.
    """

    VALID_PROFILES = ("smooth", "fluctuating", "spiky")

    def __init__(
        self,
        profile: str,
        seed: int,
        max_steps: int,
        spike_probability: float = 0.0,
    ) -> None:
        if profile not in self.VALID_PROFILES:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Choose from {self.VALID_PROFILES}."
            )
        rng = np.random.default_rng(seed)
        self._profile = profile
        self._max_steps = max_steps
        self._workload: np.ndarray = self._generate(
            rng, profile, max_steps, spike_probability,
        )

    # ── public API ───────────────────────────────────────────────────────

    def get(self, step: int) -> float:
        """Return the workload value (req/s) for the given *step*."""
        if step < 0 or step >= self._max_steps:
            raise IndexError(
                f"step {step} out of range [0, {self._max_steps})."
            )
        return float(self._workload[step])

    @property
    def trajectory(self) -> np.ndarray:
        """Full pre-generated workload array (read-only copy)."""
        return self._workload.copy()

    # ── internal generation ──────────────────────────────────────────────

    @staticmethod
    def _generate(
        rng: np.random.Generator,
        profile: str,
        max_steps: int,
        spike_probability: float,
    ) -> np.ndarray:
        t = np.arange(max_steps, dtype=np.float64)

        # Base sinusoidal signal: amplitude A, period T
        A = 200.0      # peak amplitude
        T = max_steps   # one full cycle over the episode

        base = A * np.sin(2.0 * np.pi * t / T) + A  # range ≈ [0, 2A]

        # Small Gaussian noise (smooth profile)
        epsilon = rng.normal(0.0, 5.0, size=max_steps)
        smooth = base + epsilon

        if profile == "smooth":
            return np.clip(smooth, 0.0, None)

        # Moderate noise (fluctuating profile)
        moderate_noise = rng.normal(0.0, 30.0, size=max_steps)
        fluctuating = smooth + moderate_noise

        if profile == "fluctuating":
            return np.clip(fluctuating, 0.0, None)

        # Burst events (spiky profile)
        spike_mask = rng.random(max_steps) < spike_probability
        spike_magnitudes = rng.uniform(150.0, 400.0, size=max_steps)
        spiky = fluctuating + spike_mask * spike_magnitudes

        return np.clip(spiky, 0.0, None)


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
