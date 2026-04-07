"""M/M/c queueing model for the Proxima Auto-Scaler RL environment.

Provides Erlang-C-based estimates of response latency, server utilisation,
and queue depth.  All factorial-heavy terms are computed in log-space via
``scipy.special.gammaln`` and ``logsumexp`` to avoid float64 overflow for
large server counts (c ≫ 20).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, logsumexp

# Sentinel latency (seconds) returned when the queue is in overload.
OVERLOAD_LATENCY: float = 10.0


# ────────────────────────────────────────────────────────────────────────────
# Internal: log-space Erlang C
# ────────────────────────────────────────────────────────────────────────────

def _erlang_c(c: int, a: float, rho: float) -> float:
    """Return the Erlang C probability C(c, a) in fully log-space arithmetic.

    Parameters
    ----------
    c   : number of servers  (≥ 1, precondition)
    a   : offered load  a = λ/μ  (≥ 0)
    rho : traffic intensity  ρ = λ/(c·μ)  < 1  (stability precondition)

    Log-space derivation
    --------------------
    The Erlang C formula from the spec:

        C(c, a) = 1 / ( 1  +  (1−ρ) · (c! / aᶜ) · Σ_{k=0}^{c−1} aᵏ/k! )

    Step 1 — Represent each term of the partial sum in log-space:

        log_term[k]  =  k·log(a) − gammaln(k+1)       [ = log(aᵏ / k!) ]

    Step 2 — Sum them numerically-stably with logsumexp:

        log(S)  =  logsumexp( log_term[0..c−1] )       [ S = Σ aᵏ/k! ]

    Step 3 — Combine into the "extra" denominator factor:

        log_extra  =  log(1−ρ)              (factor 1)
                    + gammaln(c+1)          (factor 2: log c!)
                    − c·log(a)              (factor 3: −log aᶜ)
                    + log(S)                (factor 4: log Σ aᵏ/k!)

        extra  =  exp(log_extra)            (no direct overflow risk now)

    Step 4 — Recover Erlang C:

        C(c, a)  =  1 / (1 + extra)

    This avoids computing c! or aᶜ directly, both of which overflow float64
    beyond c ≈ 170 (gammaln(171) ≈ 709, right at exp overflow, so any larger
    direct computation is undefined).
    """
    if a == 0.0:
        return 0.0  # zero offered load → no waiting at all

    log_a = math.log(a)

    # ── Step 1 & 2: partial-sum S = Σ_{k=0}^{c-1}  aᵏ/k!  in log-space ──
    log_terms = np.array([k * log_a - gammaln(k + 1) for k in range(c)])
    log_S = logsumexp(log_terms)

    # ── Step 3: log_extra = log[(1−ρ)·c!/aᶜ·S]  ──────────────────────────
    log_extra = (
        math.log(1.0 - rho)   # log(1−ρ)
        + gammaln(c + 1)       # log(c!)
        - c * log_a            # −log(aᶜ)
        + log_S                # log(Σ aᵏ/k!)
    )

    # ── Step 4: C = 1/(1+exp(log_extra)) — guard extreme exponents ─────────
    if log_extra > 700.0:     # exp(log_extra) ≫ 1  →  C ≈ 0
        return 0.0
    if log_extra < -700.0:    # exp(log_extra) ≈ 0  →  C ≈ 1
        return 1.0

    return 1.0 / (1.0 + math.exp(log_extra))


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def compute_latency(lam: float, mu: float, c: int) -> float:
    """Return mean response time T (seconds) using the M/M/c formula.

    Formula (stable queue, ρ < 1)
    ----------------------------------
        a  = λ/μ          (offered load, dimensionless)
        ρ  = λ/(c·μ)      (traffic intensity per server)
        T  = C(c,a) / (c·μ − λ)  +  1/μ

    where C(c, a) is the Erlang C waiting probability computed in log-space.

    Parameters
    ----------
    lam : arrival rate λ  [requests/s]
    mu  : per-server service rate μ  [requests/s]
    c   : number of active servers

    Returns
    -------
    T in seconds, capped at OVERLOAD_LATENCY (10.0 s) when the queue is
    overloaded (ρ ≥ 1) or degenerate.
    """
    # ── edge cases ──────────────────────────────────────────────────────────
    if c == 0:
        return OVERLOAD_LATENCY          # no servers → permanent overload

    if mu <= 0.0:
        return OVERLOAD_LATENCY          # undefined service rate

    if lam <= 0.0:
        return 1.0 / mu                  # no arrivals → only service time

    # ── stability ───────────────────────────────────────────────────────────
    rho = lam / (c * mu)
    if rho >= 1.0:
        return OVERLOAD_LATENCY          # queue diverges

    # ── M/M/c response time ─────────────────────────────────────────────────
    a  = lam / mu
    ec = _erlang_c(c, a, rho)
    T  = ec / (c * mu - lam) + 1.0 / mu

    # Numerical guard: for very high ρ the formula is finite but may slightly
    # exceed the sentinel due to floating-point imprecision near ρ→1.
    return min(T, OVERLOAD_LATENCY)


def compute_utilization(lam: float, mu: float, c: int) -> float:
    """Return per-server traffic intensity ρ = λ/(c·μ), clamped to [0, 1].

    Parameters
    ----------
    lam : arrival rate λ  [requests/s]
    mu  : per-server service rate μ  [requests/s]
    c   : number of active servers
    """
    if c == 0 or mu <= 0.0:
        return 1.0                        # degenerate: fully loaded
    rho = lam / (c * mu)
    return float(np.clip(rho, 0.0, 1.0))


def compute_queue_length(lam: float, mu: float, c: int) -> float:
    """Return expected number of requests waiting in queue, Lq (M/M/c).

    Formula
    -------
        Lq = C(c, λ/μ) · ρ / (1 − ρ)

    Parameters
    ----------
    lam : arrival rate λ  [requests/s]
    mu  : per-server service rate μ  [requests/s]
    c   : number of active servers

    Returns
    -------
    Expected queue depth.  Returns ``float('inf')`` for overload / degenerate
    inputs and 0.0 when lam ≤ 0.
    """
    if c == 0 or mu <= 0.0:
        return float("inf")

    if lam <= 0.0:
        return 0.0

    rho = lam / (c * mu)
    if rho >= 1.0:
        return float("inf")

    a  = lam / mu
    ec = _erlang_c(c, a, rho)
    return ec * rho / (1.0 - rho)


# ────────────────────────────────────────────────────────────────────────────
# Unit tests / smoke-test
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    _PASS = "\033[92m PASS\033[0m"
    _FAIL = "\033[91m FAIL\033[0m"
    _results: list[bool] = []

    def _check(name: str, condition: bool, detail: str = "") -> None:
        tag = _PASS if condition else _FAIL
        suffix = f"  ({detail})" if detail else ""
        print(f"[{tag}] {name}{suffix}")
        _results.append(condition)

    print("=" * 55)
    print("  M/M/c queue_model unit tests")
    print("=" * 55)

    # ── 1. M/M/1: c=1, λ=0.5, μ=1.0 → T should be ≈ 2.0 s ───────────────
    T1 = compute_latency(lam=0.5, mu=1.0, c=1)
    _check(
        "M/M/1  T ≈ 2.0 s",
        abs(T1 - 2.0) < 1e-9,
        f"T={T1:.10f}",
    )

    # ── 2. Near-saturation: c=5, λ=4.9, μ=1.0 → returns 10.0 (clamped) ──
    T2 = compute_latency(lam=4.9, mu=1.0, c=5)
    _check(
        "Near-saturation (c=5, λ=4.9) → 10.0 s",
        T2 == OVERLOAD_LATENCY,
        f"T={T2:.4f}",
    )

    # Double-check the pre-clamp value is indeed >> 10
    rho_5 = 4.9 / (5 * 1.0)
    a_5   = 4.9 / 1.0
    ec_5  = _erlang_c(5, a_5, rho_5)
    T2_raw = ec_5 / (5 * 1.0 - 4.9) + 1.0 / 1.0
    _check(
        "Near-saturation raw T >> 10 s (before clamp)",
        T2_raw > 10.0,
        f"raw T={T2_raw:.4f}",
    )

    # ── 3. No overflow for large c: c=200, λ=150, μ=1.0 ──────────────────
    try:
        T3 = compute_latency(lam=150.0, mu=1.0, c=200)
        finite = math.isfinite(T3) and T3 > 0
        _check(
            "No overflow  c=200, λ=150, μ=1.0",
            finite,
            f"T={T3:.6f} s",
        )
    except Exception as exc:
        _check("No overflow  c=200, λ=150, μ=1.0", False, str(exc))

    # ── 4. Zero arrival: T = 1/μ exactly ─────────────────────────────────
    mu_val = 2.5
    T4 = compute_latency(lam=0.0, mu=mu_val, c=3)
    _check(
        "Zero arrival → T = 1/μ exactly",
        T4 == 1.0 / mu_val,
        f"T={T4}, 1/μ={1/mu_val}",
    )

    # ── 5. Edge cases ─────────────────────────────────────────────────────
    _check("c=0             → 10.0 s", compute_latency(1.0, 1.0, 0) == OVERLOAD_LATENCY)
    _check("mu=0            → 10.0 s", compute_latency(1.0, 0.0, 3) == OVERLOAD_LATENCY)
    _check("ρ≥1 (overload)  → 10.0 s", compute_latency(5.0, 1.0, 4) == OVERLOAD_LATENCY)

    # ── 6. Utilisation clamp ──────────────────────────────────────────────
    u1 = compute_utilization(100.0, 1.0, 5)   # ρ = 20, clamp to 1.0
    u2 = compute_utilization(0.0, 1.0, 5)     # ρ = 0
    _check("Utilization clamp high → 1.0", u1 == 1.0, f"ρ={u1}")
    _check("Utilization zero λ → 0.0",     u2 == 0.0, f"ρ={u2}")

    # ── 7. Queue length sanity ────────────────────────────────────────────
    Lq_mm1 = compute_queue_length(lam=0.5, mu=1.0, c=1)
    # M/M/1: Lq = ρ² / (1−ρ) = 0.25/0.5 = 0.5
    _check(
        "M/M/1 Lq ≈ 0.5",
        abs(Lq_mm1 - 0.5) < 1e-9,
        f"Lq={Lq_mm1:.10f}",
    )
    _check(
        "Overload Lq = inf",
        math.isinf(compute_queue_length(5.0, 1.0, 4)),
    )

    print("=" * 55)
    passed = sum(_results)
    total  = len(_results)
    print(f"  {passed}/{total} tests passed")
    print("=" * 55)
    sys.exit(0 if passed == total else 1)
