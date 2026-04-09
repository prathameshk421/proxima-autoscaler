"""
Phase 3 — Reward Function
Multi-term reward with all components individually bounded and a predictable
total range.

Note
----
This function is statically computed every environment step. It is *not*
trained.  The RL agent/policy is what gets optimised; this function only
provides the learning signal.

Bounds summary
--------------
  throughput_reward    ∈ [0.0,  1.0]
  latency_penalty      ∈ [-2.0, 0.0]
  cost_penalty         ∈ [-0.5, 0.0]
  queue_penalty        ∈ [-1.0, 0.0]
  oscillation_penalty  ∈ [-1.0, 0.0]  (caller-supplied, must honour bound)
  invalid_penalty      ∈ {-0.5, 0.0}
  signal_bonus         ∈ { 0.0, 1.0}
  ─────────────────────────────────────
  total                ∈ [-5.0, 2.0]
"""

from __future__ import annotations

from env.models import CloudAction, CloudObservation, CloudReward
from env.simulation import SERVER_CAPACITY


# ── Public constant ──────────────────────────────────────────────────────────

REWARD_RANGE: tuple[float, float] = (-5.0, 2.0)


# ── Core function ─────────────────────────────────────────────────────────────

def compute_reward(
    obs: CloudObservation,
    action: CloudAction,
    prev_obs: CloudObservation,           # reserved for future delta-based terms
    oscillation_penalty: float,           # pre-computed by caller ∈ [-1.0, 0.0]
    correctly_signaled: bool,
) -> CloudReward:
    """Compute the step reward for the Proxima autoscaler environment.

    Parameters
    ----------
    obs:
        Observation *after* the action has been applied.
    action:
        The action that was taken this step.
    prev_obs:
        Observation from the *previous* step (kept for future delta terms).
    oscillation_penalty:
        Caller-computed penalty for rapid scale-up/down oscillation.
        Must lie in ``[-1.0, 0.0]``.
    correctly_signaled:
        ``True`` when the agent issued an ``INSUFFICIENT_CAPACITY`` signal
        at the right moment; ``False`` otherwise.

    Returns
    -------
    CloudReward
        Contains ``value`` (rounded total) and ``breakdown`` (individual terms).
    """

    # ── 1. Latency penalty  ∈ [-2.0, 0.0] ───────────────────────────────────
    # No penalty below 50 ms; grows linearly to -2.0 at 100 ms and clamps there.
    if obs.latency_ms < 50:
        latency_penalty: float = 0.0
    else:
        latency_penalty = -min((obs.latency_ms - 50.0) / 50.0, 1.0) * 2.0

    # ── 2. Throughput reward  ∈ [0.0, 1.0] ──────────────────────────────────
    # Fraction of incoming requests that can actually be served.
    effective_capacity = max(obs.active_servers * SERVER_CAPACITY, 1.0)
    served_ratio = min(obs.request_rate / effective_capacity, 1.0)
    throughput_reward: float = served_ratio * 1.0

    # ── 3. Cost penalty (over-provisioning)  ∈ [-0.5, 0.0] ──────────────────
    # Penalises idle capacity; normalised by one server's worth of capacity.
    over_provision = max(obs.active_servers * SERVER_CAPACITY - obs.request_rate, 0.0)
    cost_penalty: float = -min(over_provision / SERVER_CAPACITY, 1.0) * 0.5

    # ── 4. Queue penalty  ∈ [-1.0, 0.0] ─────────────────────────────────────
    # Linearly proportional to queue depth up to 100 requests; clamps thereafter.
    queue_penalty: float = -min(obs.queue_length / 100.0, 1.0) * 1.0

    # ── 5. Oscillation penalty  ∈ [-1.0, 0.0]  (caller-supplied) ────────────
    # Validated here so that any caller bug is caught early.
    if not (-1.0 <= oscillation_penalty <= 0.0):
        raise ValueError(
            f"oscillation_penalty must be in [-1.0, 0.0], got {oscillation_penalty!r}"
        )

    # ── 6. Invalid action penalty  ∈ {-0.5, 0.0} ────────────────────────────
    # Penalises attempting a scaling action while still in cooldown.
    # Uses prev_obs (pre-action state) because a valid action *creates*
    # cooldown — checking post-action obs would wrongly penalise valid actions.
    invalid_penalty: float = (
        -0.5 if (prev_obs.cooldown_remaining > 0 and action.mode != "signal") else 0.0
    )

    # ── 7. Capacity signal bonus  ∈ {0.0, 1.0} ──────────────────────────────
    signal_bonus: float = 1.0 if correctly_signaled else 0.0

    # ── Total  ∈ [-5.0, 2.0] ─────────────────────────────────────────────────
    total = (
        throughput_reward
        + latency_penalty
        + cost_penalty
        + queue_penalty
        + oscillation_penalty
        + invalid_penalty
        + signal_bonus
    )

    return CloudReward(
        value=round(total, 4),
        breakdown={
            "throughput": round(throughput_reward, 4),
            "latency": round(latency_penalty, 4),
            "cost": round(cost_penalty, 4),
            "queue": round(queue_penalty, 4),
            "oscillation": round(oscillation_penalty, 4),
            "invalid": round(invalid_penalty, 4),
            "signal": round(signal_bonus, 4),
        },
    )


# ── Exit-condition verification ───────────────────────────────────────────────

def _verify_bounds(label: str, reward: CloudReward) -> None:
    """Assert that all breakdown terms are within spec and sum to value."""

    term_bounds: dict[str, tuple[float, float]] = {
        "throughput":  (0.0, 1.0),
        "latency":     (-2.0, 0.0),
        "cost":        (-0.5, 0.0),
        "queue":       (-1.0, 0.0),
        "oscillation": (-1.0, 0.0),
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

    # Verify sum (allow for float rounding up to 1 ulp at 4 decimal places)
    computed_sum = sum(reward.breakdown.values())
    if abs(computed_sum - reward.value) > 5e-4:
        errors.append(
            f"  breakdown sum {computed_sum:.6f} ≠ value {reward.value:.4f}"
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
    r = compute_reward(base_obs, action_normal, base_obs,
                       oscillation_penalty=0.0, correctly_signaled=False)
    _verify_bounds("ideal steady-state", r)

    # ── Scenario 2: overloaded — high latency, long queue ────────────────────
    overloaded = _obs(latency_ms=200.0, queue_length=150, request_rate=500.0,
                      active_servers=2)
    r = compute_reward(overloaded, action_normal, base_obs,
                       oscillation_penalty=-1.0, correctly_signaled=False)
    _verify_bounds("overloaded system", r)

    # ── Scenario 3: over-provisioned, idle ───────────────────────────────────
    idle = _obs(request_rate=10.0, active_servers=10, latency_ms=10.0,
                queue_length=0)
    r = compute_reward(idle, action_normal, base_obs,
                       oscillation_penalty=0.0, correctly_signaled=False)
    _verify_bounds("over-provisioned idle", r)

    # ── Scenario 4: scale action during cooldown (invalid) ───────────────────
    cooling_prev = _obs(cooldown_remaining=2)  # prev_obs has cooldown active
    action_scale = CloudAction(mode="step", scale_delta=1)
    r = compute_reward(base_obs, action_scale, cooling_prev,
                       oscillation_penalty=-0.5, correctly_signaled=False)
    _verify_bounds("scaling during cooldown", r)

    # ── Scenario 5: correct signal bonus ─────────────────────────────────────
    r = compute_reward(base_obs, CloudAction(mode="signal", signal_type="INSUFFICIENT_CAPACITY"),
                       base_obs, oscillation_penalty=0.0, correctly_signaled=True)
    _verify_bounds("correct capacity signal", r)

    # ── Scenario 6: worst-case (all penalties maxed, no bonus) ───────────────
    worst = _obs(latency_ms=200.0, queue_length=200, request_rate=500.0,
                 active_servers=1)
    worst_prev = _obs(cooldown_remaining=3)  # prev_obs has cooldown active
    r = compute_reward(worst, action_scale, worst_prev,
                       oscillation_penalty=-1.0, correctly_signaled=False)
    _verify_bounds("worst-case all penalties", r)

    # ── Scenario 7: best-case (all bonuses, no penalties) ────────────────────
    best = _obs(latency_ms=10.0, queue_length=0, request_rate=200.0,
                active_servers=2, cooldown_remaining=0)
    r = compute_reward(best, action_normal, base_obs,
                       oscillation_penalty=0.0, correctly_signaled=True)
    _verify_bounds("best-case all bonuses", r)

    print(f"\n{'─'*50}")
    print("All scenarios passed.  Total range confirmed: "
          f"[{REWARD_RANGE[0]}, {REWARD_RANGE[1]}]")


if __name__ == "__main__":
    _run_exit_conditions()
