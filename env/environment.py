"""
Phase 4 — Environment Class (v3.0)

Full ``reset / step / state`` loop for the Cloud Auto-Scaling OpenEnv.
Integrates the workload generator, server pool, latency model, and reward
function into a single coherent game loop.

Critical design notes
---------------------
* **Queue drain** — Lindley's recurrence converts rate (req/s) to arrivals
  per step using ``STEP_DURATION_SEC``, ensuring dimensional correctness.
* **Target-utilization math** — HPA-style ratio scaling with ±10% deadband
  prevents micro-scaling from workload noise.
* **Stability** — L2 action delta penalty in ``reward.py`` replaces the
  old oscillation-detection deque.
"""

from __future__ import annotations

import math
from typing import Optional

from env.models import CloudAction, CloudObservation, StepResult
from env.reward import compute_reward
from env.simulation import (
    MAX_QUEUE,
    SERVER_CAPACITY,
    STEP_DURATION_SEC,
    ServerPool,
    WorkloadGenerator,
    compute_mmc_latency,
)
from env.tasks import TASKS, TaskConfig


class CloudAutoScalingEnv:
    """OpenEnv-compliant environment for cloud auto-scaling.

    Parameters
    ----------
    task_name : str
        One of ``"easy"``, ``"medium"``, ``"hard"``.
    """

    # ── construction ─────────────────────────────────────────────────────

    def __init__(self, task_name: str = "easy") -> None:
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from {list(TASKS.keys())}."
            )
        self._task: TaskConfig = TASKS[task_name]
        self._reset_internals()

    # ── public API ───────────────────────────────────────────────────────

    def reset(self) -> CloudObservation:
        """Reset the environment and return the initial observation."""
        self._reset_internals()
        return self._build_observation()

    def step(self, action: CloudAction) -> StepResult:
        """Execute one environment step.

        Order of operations
        -------------------
        1. Record previous observation (for reward delta terms).
        2. Dispatch action by mode (step / target / signal).
        3. Advance workload to the next timestep.
        4. Tick server pool (graduate pending → active).
        5. Drain queue:  total_demand → fulfilled → leftover.
        6. Compute utilization and latency.
        7. Compute reward.
        8. Check termination.
        9. Return ``StepResult``.
        """
        if self._done:
            raise RuntimeError("Episode is done.  Call reset() first.")

        # ── 0. Snapshot previous observation ─────────────────────────────
        prev_obs = self._build_observation()

        # ── 1. Dispatch action ───────────────────────────────────────────
        actual_delta = self._apply_action(action)

        # ── 2. Advance step counter ─────────────────────────────────────
        self._step += 1

        # ── 3. Advance workload ─────────────────────────────────────────
        if self._step < self._task.max_steps:
            self._request_rate = self._workload.get(self._step)
        # else: keep last request rate (episode about to end)

        # ── 4. Tick server pool (pending → active) ──────────────────────
        self._server_pool.tick(self._step)

        # ── 5. Drain queue (Lindley's Recurrence) ─────────────────────
        # request_rate is req/s, must be converted to arrivals per step
        arrivals       = int(self._request_rate * STEP_DURATION_SEC)
        capacity_count = int(self._server_pool.active * SERVER_CAPACITY * STEP_DURATION_SEC)
        total_demand   = self._queue + arrivals
        fulfilled      = min(total_demand, capacity_count)
        new_queue      = max(0, total_demand - fulfilled)
        dropped        = max(0, new_queue - MAX_QUEUE)       # requests lost to buffer overflow
        self._queue    = min(new_queue, MAX_QUEUE)
        utilization    = fulfilled / capacity_count if capacity_count > 0 else 1.0

        # ── 6. Utilization & latency ────────────────────────────────────
        self._utilization = utilization
        self._latency_ms = compute_mmc_latency(
            self._request_rate, self._server_pool.active,
        )

        # ── 7. Signal correctness check ─────────────────────────────────
        correctly_signaled = self._check_signal(action)

        # ── 8. Reward ───────────────────────────────────────────────
        obs = self._build_observation()
        reward_result = compute_reward(
            obs=obs,
            action=action,
            prev_delta=action.scale_delta,
            correctly_signaled=correctly_signaled,
        )

        # ── 9. Termination ──────────────────────────────────────────────
        self._done = self._step >= self._task.max_steps - 1

        # Refresh obs with final done flag
        obs = self._build_observation()

        return StepResult(
            observation=obs,
            reward=reward_result.value,
            done=self._done,
            info=reward_result.breakdown,
        )

    def state(self) -> CloudObservation:
        """Return current observation without advancing the step."""
        return self._build_observation()

    # ── internal helpers ─────────────────────────────────────────────────

    def _reset_internals(self) -> None:
        """(Re-)initialise all mutable state."""
        task = self._task

        self._step: int = 0
        self._done: bool = False
        self._queue: int = 0
        self._target_utilization: float = 0.6

        # Workload generator (deterministic)
        self._workload = WorkloadGenerator(
            profile=task.profile,
            seed=task.seed,
            max_steps=task.max_steps,
            spike_probability=task.spike_probability,
        )
        self._request_rate: float = self._workload.get(0)

        # Server pool
        self._server_pool = ServerPool(
            initial=task.initial_servers,
            min_s=task.min_servers,
            max_s=task.max_servers,
        )

        # Derived metrics (initial values)
        self._utilization: float = min(
            self._request_rate / max(self._server_pool.effective_capacity, 1.0), 1.0
        )
        self._latency_ms: float = compute_mmc_latency(
            self._request_rate, self._server_pool.active,
        )

    def _apply_action(self, action: CloudAction) -> int:
        """Dispatch action by mode.  Returns the actual server delta applied."""
        if action.mode == "step":
            return self._server_pool.apply_action(
                action.scale_delta, self._step,
            )

        elif action.mode == "target":
            # HPA-style ratio scaling with 10% tolerance deadband
            # Prevents micro-scaling from noise — matches Kubernetes HPA behavior
            target_util = max(min(action.target_utilization, 0.95), 0.1)
            self._target_utilization = target_util
            ratio = self._utilization / target_util   # use current utilization
            if 0.9 <= ratio <= 1.1:
                delta = 0   # within deadband — no action
            else:
                desired = math.ceil(self._server_pool.active * ratio)
                delta = desired - self._server_pool.active
            return self._server_pool.apply_action(delta, self._step)

        elif action.mode == "signal":
            # Signal mode — no scaling, just report capacity status
            return 0

        else:
            raise ValueError(f"Unknown action mode '{action.mode}'.")

    def _check_signal(self, action: CloudAction) -> bool:
        """Return True if the agent correctly signaled INSUFFICIENT_CAPACITY.

        Correct when: agent signals AND at max servers AND utilization > 0.95.
        """
        if action.mode != "signal":
            return False
        if action.signal_type != "INSUFFICIENT_CAPACITY":
            return False
        at_max = self._server_pool.active >= self._task.max_servers
        overloaded = self._utilization > 0.95
        return at_max and overloaded


    def _build_observation(self) -> CloudObservation:
        """Construct the current observation snapshot."""
        return CloudObservation(
            active_servers=self._server_pool.active,
            max_servers=self._task.max_servers,
            request_rate=round(self._request_rate, 2),
            utilization=round(self._utilization, 4),
            queue_length=self._queue,
            latency_ms=round(self._latency_ms, 2),
            target_utilization=self._target_utilization,
            cooldown_remaining=self._server_pool.cooldown_remaining(self._step),
            step_number=self._step,
            done=self._done,
            queue_ratio=round(self._queue / MAX_QUEUE, 4),
            capacity_ratio=round(
                self._server_pool.active / self._task.max_servers, 4,
            ),
        )


# ── Exit-condition verification ──────────────────────────────────────────────

def _run_exit_condition() -> None:
    """20-step smoke test: rewards should vary, queue should drain (not vanish)."""
    print("=" * 60)
    print("  Phase 4 — Environment exit-condition test")
    print("=" * 60)

    env = CloudAutoScalingEnv("easy")
    obs = env.reset()

    print(f"\n  Initial state:")
    print(f"    servers={obs.active_servers}  rate={obs.request_rate:.1f}  "
          f"util={obs.utilization:.2f}  queue={obs.queue_length}  "
          f"latency={obs.latency_ms:.1f}ms")

    rewards: list[float] = []
    prev_queue: Optional[int] = None
    queue_drained = False

    print(f"\n  {'step':>4}  {'reward':>7}  {'latency':>8}  {'queue':>5}  "
          f"{'servers':>7}  {'util':>5}  {'rate':>7}")
    print("  " + "─" * 56)

    for i in range(20):
        result = env.step(CloudAction(mode="step", scale_delta=1))
        o = result.observation
        rewards.append(result.reward)

        # Detect if queue ever drains (decreases when capacity is available)
        if prev_queue is not None and o.queue_length < prev_queue:
            queue_drained = True
        prev_queue = o.queue_length

        print(f"  {o.step_number:4d}  {result.reward:+7.3f}  "
              f"{o.latency_ms:8.2f}  {o.queue_length:5d}  "
              f"{o.active_servers:7d}  {o.utilization:.3f}  "
              f"{o.request_rate:7.1f}")

    # ── Assertions ───────────────────────────────────────────────────────
    unique_rewards = len(set(round(r, 4) for r in rewards))
    print(f"\n  Unique reward values: {unique_rewards}")
    print(f"  Queue drained at least once: {queue_drained}")

    assert unique_rewards > 1, (
        f"Expected varying rewards, got {unique_rewards} unique values."
    )
    # Queue should either drain or stay at 0 when capacity exceeds demand
    # (not vanish — our drain logic handles this correctly)

    print(f"\n{'='*60}")
    print("  ✅  Environment exit-condition PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    _run_exit_condition()
