"""Workload generator for the Proxima Auto-Scaler RL environment.

Provides normalised arrival-rate intensity I_norm(t) for three difficulty
tiers, calculated from a sinusoidal base, optional Gaussian noise, and an
MMPP (Markov Modulated Poisson Process) burst layer.

Arrival rate used by the environment::

    λ(t) = μ · c_baseline · I_norm(t)

Normalised intensity (before clamping)::

    I_norm(t) = I(t) / I_max

where I_max is the theoretical peak for each difficulty (see below).
"""

from __future__ import annotations

import math

import numpy as np

_BETA0: float = 0.5
_BETA1: float = 0.2
_BETA2: float = 0.15
_F1: float = 1.0 / 288
_F2: float = 1.0 / 96

_SIGMA: float = 0.08
_BURST_SCALAR: float = 4.0
_P_N2B: float = 0.05
_P_B2N: float = 0.15
_PARETO_SHAPE: float = 1.5
_PARETO_SCALE: float = 5.0

_BASE_MAX: float = _BETA0 + _BETA1 + _BETA2

_I_MAX: dict[str, float] = {
    "easy": _BASE_MAX,
    "medium": _BASE_MAX + 3.0 * _SIGMA,
    "hard": _BASE_MAX * _BURST_SCALAR + 3.0 * _SIGMA,
}


_BOUNDS: dict[str, tuple[float, float]] = {
    "easy": (0.3, 0.8),
    "medium": (0.2, 1.1),
    "hard": (0.1, 1.8),
}

VALID_DIFFICULTIES = frozenset(_BOUNDS)


def _sinusoidal(t: int) -> float:
    """Return the deterministic diurnal base intensity at timestep *t*."""
    return (
        _BETA0
        + _BETA1 * math.sin(2.0 * math.pi * _F1 * t)
        + _BETA2 * math.cos(2.0 * math.pi * _F2 * t)
    )


class WorkloadGenerator:
    """Generate normalised workload intensity for the auto-scaling environment.

    Determinism guarantee
    ---------------------
    All stochastic components (Gaussian noise for medium, MMPP state sequence
    for hard) are drawn lazily and cached in episode order.  Calling
    ``get_intensity(t)`` for any *t* within or beyond the current cache
    transparently extends the simulation from its last position, so replay is
    bit-for-bit reproducible for the same ``(seed, t)`` pair as long as
    ``reset()`` is not called between calls.

    Parameters
    ----------
    difficulty : str
        Complexity tier — one of ``"easy"``, ``"medium"``, ``"hard"``.
    seed : int
        Integer seed for ``numpy.random.default_rng``.
    c_baseline : int
        Nominal server count; scales the arrival rate in
        ``get_arrival_rate``.
    mu : float
        Nominal requests processed per server per timestep.
    """

    def __init__(
        self,
        difficulty: str,
        seed: int,
        c_baseline: int,
        mu: float,
    ) -> None:
        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"difficulty must be one of {sorted(VALID_DIFFICULTIES)}, got {difficulty!r}"
            )
        self.difficulty: str = difficulty
        self.c_baseline: int = c_baseline
        self.mu: float = mu

        self._lo, self._hi = _BOUNDS[difficulty]
        self._i_max: float = _I_MAX[difficulty]

        self._init_rng(seed)

    def _init_rng(self, seed: int) -> None:
        """Reset all RNG state and episode caches."""
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        self._noise_cache: list[float] = []

        self._mmpp_cache: list[float] = []
        self._mmpp_state: int = 0
        self._mmpp_remaining: int = 0

    def _mmpp_advance_session(self) -> None:
        """Transition to a new MMPP state and sample the next session length.

        Called whenever the previous session expires (``_mmpp_remaining <= 0``).
        """
        if self._mmpp_state == 0:
            if self.rng.random() < _P_N2B:
                self._mmpp_state = 1
        else:
            if self.rng.random() < _P_B2N:
                self._mmpp_state = 0

        raw_length = self.rng.pareto(_PARETO_SHAPE) * _PARETO_SCALE + _PARETO_SCALE
        self._mmpp_remaining = max(1, int(raw_length))

    def _ensure_mmpp_cache(self, t: int) -> None:
        """Lazily extend the MMPP multiplier cache through timestep *t*."""
        while len(self._mmpp_cache) <= t:
            if self._mmpp_remaining <= 0:
                self._mmpp_advance_session()

            multiplier = _BURST_SCALAR if self._mmpp_state == 1 else 1.0
            self._mmpp_cache.append(multiplier)
            self._mmpp_remaining -= 1

    def _ensure_noise_cache(self, t: int) -> None:
        """Lazily extend the Gaussian noise cache through timestep *t*."""
        while len(self._noise_cache) <= t:
            self._noise_cache.append(float(self.rng.normal(0.0, _SIGMA)))

    def get_intensity(self, t: int) -> float:
        """Return I_norm(t), the normalised workload intensity at timestep *t*.

        Formula
        -------
        - **easy** :  I(t) = β₀ + β₁·sin(2π·f₁·t) + β₂·cos(2π·f₂·t)
        - **medium**: I(t) = I_easy(t) + N(0, σ²)
        - **hard** :  I(t) = I_easy(t) · mmpp_multiplier(t)

        The raw intensity is then normalised and clamped::

            I_norm(t) = clip( I(t) / I_max,  lo,  hi )

        Parameters
        ----------
        t : int
            Non-negative discrete timestep index.
        """
        if t < 0:
            raise ValueError(f"Timestep t must be >= 0, got {t}")

        base = _sinusoidal(t)

        if self.difficulty == "easy":
            raw = base
        elif self.difficulty == "medium":
            self._ensure_noise_cache(t)
            raw = base + self._noise_cache[t]
        else:
            self._ensure_mmpp_cache(t)
            raw = base * self._mmpp_cache[t]

        normalized = raw / self._i_max
        return float(np.clip(normalized, self._lo, self._hi))

    def get_arrival_rate(self, t: int) -> float:
        """Return λ(t) = μ · c_baseline · I_norm(t).

        This is the value consumed by the environment's step function.

        Parameters
        ----------
        t : int
            Non-negative discrete timestep index.
        """
        return self.mu * self.c_baseline * self.get_intensity(t)

    def reset(self, seed: int, c_baseline: int) -> None:
        """Re-initialise the generator for a new episode.

        Clears all caches and MMPP state so the next call to
        ``get_intensity`` starts a fresh, reproducible trajectory.

        Parameters
        ----------
        seed : int
            New RNG seed.
        c_baseline : int
            Updated baseline server capacity (may differ between runs).
        """
        self.c_baseline = c_baseline
        self._i_max = _I_MAX[self.difficulty]
        self._init_rng(seed)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T = 300
    ts = list(range(T))

    difficulties: list[str] = ["easy", "medium", "hard"]
    palette = {"easy": "#4CAF50", "medium": "#FF9800", "hard": "#EF5350"}

    fig, axes = plt.subplots(
        len(difficulties),
        1,
        figsize=(13, 7),
        sharex=True,
        facecolor="#0d1117",
    )
    fig.suptitle(
        "Proxima Auto-Scaler — Workload Intensity  I_norm(t)",
        fontsize=14,
        fontweight="bold",
        color="white",
        y=0.98,
    )

    print(f"{'Difficulty':<10}  {'min':>8}  {'max':>8}  {'bounds'}")
    print("-" * 42)

    for ax, diff in zip(axes, difficulties):
        gen = WorkloadGenerator(difficulty=diff, seed=42, c_baseline=10, mu=80.0)
        values = [gen.get_intensity(t) for t in ts]
        lo, hi = _BOUNDS[diff]

        color = palette[diff]
        ax.set_facecolor("#161b22")
        ax.plot(ts, values, color=color, linewidth=1.6, label=diff)
        ax.axhline(lo, color="white", linestyle="--", linewidth=0.7, alpha=0.45)
        ax.axhline(hi, color="white", linestyle="--", linewidth=0.7, alpha=0.45)
        ax.fill_between(ts, lo, hi, alpha=0.07, color=color)
        ax.set_ylabel("I_norm(t)", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.legend(
            loc="upper right",
            fontsize=9,
            facecolor="#21262d",
            labelcolor="white",
            edgecolor="#30363d",
        )
        ax.grid(True, color="#21262d", linewidth=0.5)
        ax.set_ylim(0.0, 2.0)

        print(f"{diff:<10}  {min(values):>8.4f}  {max(values):>8.4f}  [{lo}, {hi}]")

    axes[-1].set_xlabel("Timestep  t", color="white", fontsize=9)
    plt.tight_layout()

    out_path = "workload_test.png"
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nPlot saved → {out_path}")
