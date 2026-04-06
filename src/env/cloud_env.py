# src/env/cloud_env.py
"""Core Gymnasium environment for the Proxima Auto-Scaler RL project.

CloudAutoScalingEnv implements the full M/M/c cloud simulation with:
  - MultiDiscrete action space (step_scaling, target_util, capacity_signal)
  - Booting queue with configurable D_startup delay
  - Erlang-C physics via queue_model
  - Lagrangian reward function
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .schemas import Action, Observation
from .workload import WorkloadGenerator
from .queue_model import (
    OVERLOAD_LATENCY,
    compute_latency,
    compute_queue_length,
    compute_utilization,
)

# ── Action index → value mappings ────────────────────────────────────────────
_STEP_SCALING_MAP = [-2, -1, 0, 1, 2]          # a1: indices 0..4
_TARGET_UTIL_MAP  = [0.4, 0.5, 0.6, 0.7]       # a2: indices 0..3
_CAP_SIGNAL_MAP   = [0, 1]                       # a3: indices 0..1

# ── Reward constants ──────────────────────────────────────────────────────────
_W_COST        = 0.3
_W_LATENCY     = 0.6
_W_OSCILLATION = 0.1
_W_INVALID     = 0.1
_W_SIGNAL      = 0.5
_KAPPA         = 0.05   # oscillation penalty weight

# ── Episode length ────────────────────────────────────────────────────────────
_MAX_STEPS        = 200
_WORKLOAD_TREND_N = 8   # number of historical intensity samples kept


class CloudAutoScalingEnv(gym.Env):
    """Gymnasium cloud auto-scaling environment.

    Action space   : MultiDiscrete([5, 4, 2])
    Observation    : flat float32 Box (length = 10 + D_startup + 8)
    Reward         : Lagrangian sum of cost, latency, oscillation, invalid, signal
    """

    metadata = {"render_modes": []}

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(
        self,
        difficulty: str  = "easy",
        seed: int        = 42,
        c_max: int       = 20,
        mu: float        = 10.0,
        D_startup: int   = 3,
        T_cooldown: int  = 5,
        T_sla: float     = 0.5,
    ) -> None:
        super().__init__()

        self.difficulty  = difficulty
        self._init_seed  = seed
        self.c_max       = c_max
        self.mu          = mu
        self.D_startup   = D_startup
        self.T_cooldown  = T_cooldown
        self.T_sla       = T_sla

        # ── Action space ─────────────────────────────────────────────────────
        self.action_space = spaces.MultiDiscrete([5, 4, 2])

        # ── Observation space ─────────────────────────────────────────────────
        # Layout (total = 10 + D_startup + _WORKLOAD_TREND_N):
        #   [0]                        current_capacity
        #   [1 .. D_startup]           booting_servers   (D_startup elems)
        #   [D_startup+1]              next_server_ready
        #   [D_startup+2]              max_capacity
        #   [D_startup+3]              request_rate
        #   [D_startup+4]              avg_utilization
        #   [D_startup+5]              queue_length
        #   [D_startup+6]              latency_p95
        #   [D_startup+7]              cooldown_remaining
        #   [D_startup+8]              target_utilization
        #   [D_startup+9 .. end]       workload_trend (8 values)
        obs_dim = 1 + D_startup + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + _WORKLOAD_TREND_N
        _large  = np.finfo(np.float32).max
        lo  = np.zeros(obs_dim, dtype=np.float32)
        hi  = np.full(obs_dim, _large, dtype=np.float32)

        d = D_startup
        hi[0]       = c_max              # current_capacity
        hi[1:1+d]   = c_max              # booting_servers
        hi[1+d]     = d                  # next_server_ready
        lo[2+d]     = 1;  hi[2+d] = c_max          # max_capacity
        hi[3+d]     = mu * c_max * 2.0  # request_rate (generous upper bound)
        hi[4+d]     = 1.0               # avg_utilization
        hi[5+d]     = c_max * 20.0      # queue_length
        hi[6+d]     = OVERLOAD_LATENCY * 1.3        # latency_p95
        hi[7+d]     = T_cooldown        # cooldown_remaining
        lo[8+d]     = 0.4; hi[8+d] = 0.7           # target_utilization
        hi[9+d: 9+d+_WORKLOAD_TREND_N] = 1.8       # workload_trend

        self.observation_space = spaces.Box(lo, hi, dtype=np.float32)

        # Internal episode state (populated by reset)
        self._workload_gen:    WorkloadGenerator | None = None
        self.current_capacity: int   = 0
        self.booting_servers:  list[int] = [0] * D_startup
        self.cooldown_remaining: int = 0
        self.t:                int   = 0
        self._target_util:     float = 0.5
        self._last_desired:    int   = 0
        self._c_baseline:      int   = 0
        self._workload_trend:  deque = deque([0.5] * _WORKLOAD_TREND_N,
                                             maxlen=_WORKLOAD_TREND_N)
        # Cached physics (updated each step)
        self._lam:     float = 0.0
        self._rho:     float = 0.0
        self._T:       float = 0.0
        self._ql:      float = 0.0
        self._intensity: float = 0.5

    # ── Private helpers ───────────────────────────────────────────────────────

    def _c_baseline_for_difficulty(self) -> int:
        return int(self.c_max * 0.6) if self.difficulty == "hard" else self.c_max // 2

    def _decode_action(self, action: np.ndarray) -> tuple[int, float, int]:
        return (
            _STEP_SCALING_MAP[int(action[0])],
            _TARGET_UTIL_MAP[int(action[1])],
            _CAP_SIGNAL_MAP[int(action[2])],
        )

    def _next_server_ready(self) -> int:
        """Ticks until the soonest booting server graduates (0 if none booting)."""
        for k, n in enumerate(self.booting_servers):
            if n > 0:
                return k
        return 0

    def _run_physics(self) -> None:
        """Update cached physics values from current state and timestep."""
        self._lam       = self._workload_gen.get_arrival_rate(self.t)
        self._intensity = self._workload_gen.get_intensity(self.t)
        self._rho       = compute_utilization(self._lam, self.mu, self.current_capacity)
        self._T         = compute_latency(self._lam, self.mu, self.current_capacity)
        raw_ql          = compute_queue_length(self._lam, self.mu, self.current_capacity)
        self._ql        = 0.0 if math.isinf(raw_ql) else raw_ql
        self._workload_trend.append(self._intensity)

    def _build_obs(self) -> np.ndarray:
        obs = np.empty(self.observation_space.shape, dtype=np.float32)
        d = self.D_startup
        obs[0]          = self.current_capacity
        obs[1:1+d]      = self.booting_servers
        obs[1+d]        = self._next_server_ready()
        obs[2+d]        = self.c_max
        obs[3+d]        = self._lam
        obs[4+d]        = self._rho
        obs[5+d]        = self._ql
        obs[6+d]        = self._T * 1.3         # latency_p95
        obs[7+d]        = self.cooldown_remaining
        obs[8+d]        = self._target_util
        obs[9+d:9+d+_WORKLOAD_TREND_N] = list(self._workload_trend)
        return obs

    # ── Public API ────────────────────────────────────────────────────────────

    def state(self) -> Observation:
        """Return current state as a validated Pydantic Observation."""
        return Observation(
            current_capacity   = self.current_capacity,
            booting_servers    = list(self.booting_servers),
            next_server_ready  = self._next_server_ready(),
            max_capacity       = self.c_max,
            request_rate       = self._lam,
            avg_utilization    = float(np.clip(self._rho, 0.0, 1.0)),
            queue_length       = max(0, int(self._ql)),
            latency_p95        = float(self._T * 1.3),
            cooldown_remaining = float(self.cooldown_remaining),
            target_utilization = self._target_util,
            workload_trend     = list(self._workload_trend),
        )

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        _seed = seed if seed is not None else self._init_seed

        self._c_baseline    = self._c_baseline_for_difficulty()
        self._workload_gen  = WorkloadGenerator(
            difficulty  = self.difficulty,
            seed        = _seed,
            c_baseline  = self._c_baseline,
            mu          = self.mu,
        )
        self.current_capacity    = self._c_baseline
        self.booting_servers     = [0] * self.D_startup
        self.cooldown_remaining  = 0
        self.t                   = 0
        self._target_util        = 0.5
        self._last_desired       = self._c_baseline
        self._workload_trend     = deque([0.5] * _WORKLOAD_TREND_N,
                                        maxlen=_WORKLOAD_TREND_N)
        self._run_physics()
        obs  = self._build_obs()
        info = {"t": 0, "c_baseline": self._c_baseline}
        return obs, info

    # ── step ──────────────────────────────────────────────────────────────────

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:

        # 1. Decode action ────────────────────────────────────────────────────
        step_scaling, target_util, capacity_signal = self._decode_action(action)
        self._target_util = target_util
        action_obj = Action(
            step_scaling       = step_scaling,
            target_utilization = target_util,
            capacity_signal    = capacity_signal,
        )

        # 2. Graduate booting servers (happens every step) ─────────────────────
        self.current_capacity = min(
            self.current_capacity + self.booting_servers[0],
            self.c_max,
        )
        self.booting_servers = self.booting_servers[1:] + [0]

        # 3. Scaling logic ────────────────────────────────────────────────────
        is_invalid       = False
        P_invalid        = 0.0
        desired_capacity = self.current_capacity

        if self.cooldown_remaining > 0:
            # Cooldown active — ignore all scaling
            is_invalid = True
            P_invalid  = -0.5

        elif step_scaling != 0:
            # Direct step scaling
            desired_capacity = int(np.clip(
                self.current_capacity + step_scaling, 1, self.c_max
            ))
            delta = desired_capacity - self.current_capacity
            if delta > 0:
                self.booting_servers[-1] += delta   # scale-up: enter boot queue
            elif delta < 0:
                self.current_capacity = desired_capacity  # scale-down: instant
            self.cooldown_remaining = self.T_cooldown

        else:
            # Target utilisation logic
            avg_util = compute_utilization(self._lam, self.mu, self.current_capacity)
            if target_util > 0:
                raw = self.current_capacity * (avg_util / target_util)
            else:
                raw = float(self.current_capacity)
            desired_capacity = int(np.clip(math.ceil(raw), 1, self.c_max))
            delta = desired_capacity - self.current_capacity
            if desired_capacity != self.current_capacity:
                if delta > 0:
                    self.booting_servers[-1] += delta
                else:
                    self.current_capacity = desired_capacity
                self.cooldown_remaining = self.T_cooldown

        # 4. Physics ──────────────────────────────────────────────────────────
        self._run_physics()

        # 5. Reward ───────────────────────────────────────────────────────────
        P_cost = -(self.current_capacity + sum(self.booting_servers)) / self.c_max

        if self._T <= self.T_sla:
            P_latency = 0.0
        else:
            P_latency = -((self._T - self.T_sla) / self.T_sla) ** 2

        P_oscillation = -_KAPPA * abs(desired_capacity - self._last_desired)
        self._last_desired = desired_capacity

        # Capacity signal bonus (three exact cases)
        if capacity_signal == 1:
            if self._rho >= 0.95 and self.current_capacity == self.c_max:
                B_signal = +1.0
            else:
                B_signal = -1.0
        else:
            B_signal = 0.0

        reward = (
            _W_COST        * P_cost
            + _W_LATENCY     * P_latency
            + _W_OSCILLATION * P_oscillation
            + _W_INVALID     * P_invalid
            + _W_SIGNAL      * B_signal
        )

        reward_components = {
            "cost":            P_cost,
            "latency":         P_latency,
            "oscillation":     P_oscillation,
            "invalid_action":  P_invalid,
            "capacity_signal": B_signal,
        }

        # 6. Cooldown tick ────────────────────────────────────────────────────
        self.cooldown_remaining = max(0, self.cooldown_remaining - 1)

        # 7. Advance time, check truncation ───────────────────────────────────
        self.t += 1
        truncated  = self.t >= _MAX_STEPS
        terminated = False

        obs = self._build_obs()
        info = {
            "t":                 self.t,
            "current_capacity":  self.current_capacity,
            "booting_servers":   list(self.booting_servers),
            "lam":               self._lam,
            "rho":               self._rho,
            "latency_p95":       self._T * 1.3,
            "queue_length":      self._ql,
            "cooldown":          self.cooldown_remaining,
            "desired_capacity":  desired_capacity,
            "is_invalid":        is_invalid,
            "reward_components": reward_components,
        }
        return obs, float(reward), terminated, truncated, info
