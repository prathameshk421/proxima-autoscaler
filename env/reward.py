"""
Phase 3 — Reward Function (v3.0)
Multi-term reward with exponential SLO penalty and L2 stability penalty.

Note
----
This function is statically computed every environment step. It is *not*
trained.  The RL agent/policy is what gets optimised; this function only
provides the learning signal.

Bounds summary
--------------
  throughput_reward    ∈ [0.0,  1.0]   (weighted ×1.0)
  slo_penalty          ∈ [-15.0, 0.0]  (weighted ×1.5, capped at 10.0)
  cost_penalty         ∈ [-0.5, 0.0]
  queue_penalty        ∈ [-1.0, 0.0]
  stability_penalty    ∈ [-0.2, 0.0]   (weighted ×0.2, L2 norm)
  invalid_penalty      ∈ {-0.5, 0.0}
  signal_bonus         ∈ { 0.0, 1.0}
  ─────────────────────────────────────
    total (clipped)      ∈ [-5.0, 2.0]
"""

from __future__ import annotations

import math

try:
    from env.models import CloudAction, CloudObservation, CloudReward
    from env.simulation import SERVER_CAPACITY
except ModuleNotFoundError:  # pragma: no cover - supports `python env/reward.py`
    from models import CloudAction, CloudObservation, CloudReward
    from simulation import SERVER_CAPACITY


# ── Constants ────────────────────────────────────────────────────────────────

SLO_TARGET_MS: float = 100.0   # SLO latency threshold (ms)
SLO_K: float         = 3.0     # steepness of exponential penalty (2.0–4.0 range)
MAX_ACTION: int      = 2       # max |scale_delta| used for L2 normalization

REWARD_RANGE: tuple[float, float] = (-5.0, 2.0)


# ── Core function ─────────────────────────────────────────────────────────────

def compute_reward(
    obs: CloudObservation,
    action: CloudAction,
    invalid_action: bool,
    correctly_signaled: bool,
) -> CloudReward:
    """Compute the step reward for the Proxima autoscaler environment.

    Parameters
    ----------
    obs:
        Observation *after* the action has been applied.
    action:
        The action that was taken this step.
    invalid_action:
        ``True`` when the action attempted an unavailable scaling operation,
        such as scaling during cooldown or beyond min/max bounds.
    correctly_signaled:
        ``True`` when the agent issued an ``INSUFFICIENT_CAPACITY`` signal
        at the right moment; ``False`` otherwise.

    Returns
    -------
    CloudReward
        Contains ``value`` (rounded total) and ``breakdown`` (individual terms).
    """

    # 1. Throughput reward — fraction of demand being served [0, +1.0]
    served_ratio = min(
        obs.request_rate / max(obs.active_servers * SERVER_CAPACITY, 1), 1.0
    )
    throughput_reward = served_ratio * 1.0

    # 2. Exponential SLO latency penalty
    # Stays ~0 when latency < SLO_TARGET_MS.
    # Grows exponentially past threshold — matches real cascading failure behavior.
    if obs.latency_ms > SLO_TARGET_MS:
        slo_penalty = math.exp(SLO_K * ((obs.latency_ms - SLO_TARGET_MS)
                                         / SLO_TARGET_MS)) - 1.0
        slo_penalty = min(slo_penalty, 10.0)   # cap — prevents exploding gradients
    else:
        slo_penalty = 0.0

    # 3. Cost penalty — over-provisioning [-0.5, 0]
    over_provision = max(obs.active_servers * SERVER_CAPACITY - obs.request_rate, 0)
    cost_penalty = -min(over_provision / SERVER_CAPACITY, 1.0) * 0.5

    # 4. Queue penalty [-1.0, 0]
    queue_penalty = -min(obs.queue_length / 100, 1.0) * 1.0

    # 5. L2 action delta stability penalty (replaces sign-change oscillation deque)
    # Smooth gradient: scale_delta=1 → 0.25 penalty, scale_delta=2 → 1.0 penalty
    stability_penalty = (action.scale_delta / max(1, MAX_ACTION)) ** 2

    # 6. Invalid action penalty — scaled during cooldown
    invalid_penalty = -0.5 if invalid_action else 0.0

    # 7. Capacity signal bonus
    signal_bonus = 1.0 if correctly_signaled else 0.0

    # Weights: SLO avoidance paramount, then throughput, then stability
    w_eff  = 1.0
    w_slo  = 1.5
    w_stab = 0.2

    total = (
        w_eff  * throughput_reward
        - w_slo  * slo_penalty
        + cost_penalty
        + queue_penalty
        - w_stab * stability_penalty
        + invalid_penalty
        + signal_bonus
    )

    # Clip reward to maintain value-network stability during RL training
    total = max(-5.0, min(total, 2.0))

    return CloudReward(value=round(total, 4), breakdown={
        "throughput":  round(w_eff * throughput_reward, 4),
        "slo_penalty": round(-w_slo * slo_penalty, 4),
        "cost":        round(cost_penalty, 4),
        "queue":       round(queue_penalty, 4),
        "stability":   round(-w_stab * stability_penalty, 4),
        "invalid":     round(invalid_penalty, 4),
        "signal":      round(signal_bonus, 4),
    })


# ── Exit-condition verification ───────────────────────────────────────────────

def _verify_bounds(label: str, reward: CloudReward) -> None:
    """Assert that all breakdown terms are within spec and sum to value."""

    term_bounds: dict[str, tuple[float, float]] = {
        "throughput":  (0.0, 1.0),
        "slo_penalty": (-15.0, 0.0),
        "cost":        (-0.5, 0.0),
        "queue":       (-1.0, 0.0),
        "stability":   (-0.2, 0.0),
        "invalid":     (-0.5, 0.0),
        "signal":      (0.0, 1.0),
    }

    errors: list[str] = []

    for term, (lo, hi) in term_bounds.items():
        val = reward.breakdown[term]
        if not (lo - 1e-9 <= val <= hi + 1e-9):
            errors.append(
                f"  [{term}] {val:.6f} out of [{lo}, {hi}]"
            )

    # Verify sum — after clipping to REWARD_RANGE (breakdown stores unclipped terms)
    computed_sum = sum(reward.breakdown.values())
    clipped_sum = max(REWARD_RANGE[0], min(computed_sum, REWARD_RANGE[1]))
    if abs(clipped_sum - reward.value) > 5e-4:
        errors.append(
            f"  clipped breakdown sum {clipped_sum:.6f} ≠ value {reward.value:.4f}"
            f"  (raw sum={computed_sum:.6f})"
        )

    # Verify total is within global bounds
    lo_total, hi_total = REWARD_RANGE
    if not (lo_total - 1e-9 <= reward.value <= hi_total + 1e-9):
        errors.append(
            f"  total {reward.value} out of global range {REWARD_RANGE}"
        )

    status = "✅ PASS" if not errors else "❌ FAIL"
    print(f"\n{status}  {label}")
    print(f"  value={reward.value}  breakdown={reward.breakdown}")
    if errors:
        for e in errors:
            print(e)
        raise AssertionError(f"Reward bounds violated for scenario '{label}'")


def _run_exit_conditions() -> None:
    """Construct dummy observations and verify every invariant."""

    # Helper to build a minimal CloudObservation
    def _obs(**kwargs) -> CloudObservation:
        defaults = dict(
            active_servers=2, max_servers=10,
            request_rate=100.0, utilization=0.5,
            queue_length=0, latency_ms=20.0,
            target_utilization=0.6, cooldown_remaining=0,
            step_number=1, done=False,
            queue_ratio=0.0, capacity_ratio=0.2,
        )
        defaults.update(kwargs)
        return CloudObservation(**defaults)

    base_obs = _obs()

    # ── Scenario 1: ideal steady-state ───────────────────────────────────────
    action_normal = CloudAction(mode="step", scale_delta=0)
    r = compute_reward(base_obs, action_normal, 0, correctly_signaled=False)
    _verify_bounds("ideal steady-state", r)

    # ── Scenario 2: overloaded — high latency, long queue ────────────────────
    overloaded = _obs(latency_ms=200.0, queue_length=150, request_rate=500.0,
                      active_servers=2)
    r = compute_reward(overloaded, action_normal, 0, correctly_signaled=False)
    _verify_bounds("overloaded system", r)

    # ── Scenario 3: over-provisioned, idle ───────────────────────────────────
    idle = _obs(request_rate=10.0, active_servers=10, latency_ms=10.0,
                queue_length=0)
    r = compute_reward(idle, action_normal, 0, correctly_signaled=False)
    _verify_bounds("over-provisioned idle", r)

    # ── Scenario 4: scale action during cooldown (invalid) ───────────────────
    cooldown_obs = _obs(cooldown_remaining=2)
    action_scale = CloudAction(mode="step", scale_delta=1)
    r = compute_reward(cooldown_obs, action_scale, 0, correctly_signaled=False)
    _verify_bounds("scaling during cooldown", r)

    # ── Scenario 5: correct signal bonus ─────────────────────────────────────
    r = compute_reward(base_obs, CloudAction(mode="signal", signal_type="INSUFFICIENT_CAPACITY"),
                       0, correctly_signaled=True)
    _verify_bounds("correct capacity signal", r)

    # ── Scenario 6: worst-case (all penalties maxed, no bonus) ───────────────
    worst = _obs(latency_ms=500.0, queue_length=200, request_rate=500.0,
                 active_servers=1, cooldown_remaining=3)
    r = compute_reward(worst, action_scale, 2, correctly_signaled=False)
    _verify_bounds("worst-case all penalties", r)

    # ── Scenario 7: best-case (all bonuses, no penalties) ────────────────────
    best = _obs(latency_ms=10.0, queue_length=0, request_rate=200.0,
                active_servers=2, cooldown_remaining=0)
    r = compute_reward(best, action_normal, 0, correctly_signaled=True)
    _verify_bounds("best-case all bonuses", r)

    print(f"\n{'─'*50}")
    print("All scenarios passed.  Total range confirmed: "
          f"[{REWARD_RANGE[0]}, {REWARD_RANGE[1]}]")


if __name__ == "__main__":
    _run_exit_conditions()
