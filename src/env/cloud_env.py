"""Gymnasium environment for cloud auto-scaling simulation."""

from __future__ import annotations

from typing import List, Tuple

import gymnasium as gym
import numpy as np

from src.env.schemas import Observation


class CloudAutoScalingEnv(gym.Env):
    """Queue-based environment with delayed scale-out effects."""

    metadata = {"render_modes": []}

    def __init__(self, difficulty: str = "easy", seed: int = 42):
        super().__init__()
        self._init_seed = int(seed)
        self.rng = np.random.default_rng(self._init_seed)

        self.difficulty = difficulty
        self.max_steps = 200
        self.T_sla = 2.0
        self.cooldown_period = 3

        self._configure_difficulty(difficulty)

        self.action_space = gym.spaces.MultiDiscrete([5, 4, 2])
        self.observation_space = gym.spaces.Box(
            low=-1e6,
            high=1e6,
            shape=(15,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.current_capacity = max(1, self.c_init)
        self.queue_length = 0.0
        self.cooldown_remaining = 0
        self._boot_pipeline: List[Tuple[int, int]] = []
        self._lam = float(self.lam_base)
        self.latency_p95 = 0.0
        self.trend_window = [self._lam] * 8

    def _configure_difficulty(self, difficulty: str) -> None:
        if difficulty == "easy":
            self.c_init = 4
            self.c_max = 12
            self.mu = 5.0
            self.lam_base = 16.0
            self.lam_amp = 6.0
            self.noise_scale = 1.0
            self.boot_delay = 1
        elif difficulty == "medium":
            self.c_init = 5
            self.c_max = 16
            self.mu = 4.8
            self.lam_base = 20.0
            self.lam_amp = 9.0
            self.noise_scale = 1.6
            self.boot_delay = 2
        else:
            self.c_init = 6
            self.c_max = 20
            self.mu = 4.5
            self.lam_base = 24.0
            self.lam_amp = 12.0
            self.noise_scale = 2.2
            self.boot_delay = 3

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._init_seed = int(seed)
        self.rng = np.random.default_rng(self._init_seed)

        self.current_step = 0
        self.current_capacity = max(1, self.c_init)
        self.queue_length = 0.0
        self.cooldown_remaining = 0
        self._boot_pipeline = []

        self._lam = float(self._demand_profile(self.current_step))
        self.latency_p95 = self._estimate_latency(self._lam, self.current_capacity, self.queue_length)
        self.trend_window = [self._lam] * 8

        return self._build_obs(), {}

    def state(self) -> Observation:
        return Observation(
            current_capacity=int(self.current_capacity),
            max_capacity=int(self.c_max),
            queue_length=float(self.queue_length),
            avg_utilization=float(self._utilization(self._lam, self.current_capacity)),
            cooldown_remaining=int(self.cooldown_remaining),
            booting_servers=[int(servers) for servers, _ in self._boot_pipeline],
            demand_rate=float(self._lam),
            service_rate=float(self.mu),
            p95_latency=float(self.latency_p95),
            trend=[float(x) for x in self.trend_window],
        )

    def _build_obs(self) -> np.ndarray:
        total_booting = int(sum(servers for servers, _ in self._boot_pipeline))
        obs = np.array(
            [
                float(self.current_capacity),
                float(self.c_max),
                float(total_booting),
                float(self.cooldown_remaining),
                float(self.queue_length),
                float(self._lam),
                float(self.mu),
            ]
            + self.trend_window,
            dtype=np.float32,
        )
        return obs

    def _demand_profile(self, t: int) -> float:
        sinusoid = self.lam_amp * np.sin(2.0 * np.pi * (t / 40.0))
        burst = 0.0
        if self.difficulty != "easy" and (t % 50 in (0, 1, 2)):
            burst = self.lam_amp * 0.6
        noise = self.rng.normal(0.0, self.noise_scale)
        return float(max(0.0, self.lam_base + sinusoid + burst + noise))

    def _utilization(self, lam: float, capacity: int) -> float:
        if capacity <= 0:
            return 1.0
        return float(lam / max(1e-6, capacity * self.mu))

    def _estimate_latency(self, lam: float, capacity: int, queue_len: float) -> float:
        service = max(1e-6, capacity * self.mu)
        rho = min(1.5, lam / service)
        wait = queue_len / service
        baseline = 0.08 + 0.35 * (rho**2)
        penalty = 1.2 * max(0.0, rho - 1.0)
        return float(max(0.0, baseline + penalty + wait))

    def step(self, action: np.ndarray):
        step_idx = int(np.clip(action[0], 0, 4))
        cap_signal = int(np.clip(action[2], 0, 1))
        step_scaling = step_idx - 2

        if self.cooldown_remaining <= 0 and step_scaling != 0:
            if step_scaling > 0:
                add = min(step_scaling, self.c_max - self.current_capacity)
                if add > 0:
                    self._boot_pipeline.append((add, self.boot_delay))
            else:
                remove = min(abs(step_scaling), self.current_capacity - 1)
                self.current_capacity -= remove
            self.cooldown_remaining = self.cooldown_period
        else:
            self.cooldown_remaining = max(0, self.cooldown_remaining - 1)

        activated = 0
        next_boot: List[Tuple[int, int]] = []
        for servers, remaining in self._boot_pipeline:
            remaining -= 1
            if remaining <= 0:
                activated += servers
            else:
                next_boot.append((servers, remaining))
        self._boot_pipeline = next_boot
        self.current_capacity = min(self.c_max, self.current_capacity + activated)

        self._lam = float(self._demand_profile(self.current_step))
        service_capacity = self.current_capacity * self.mu
        served = min(self.queue_length + self._lam, service_capacity)
        self.queue_length = max(0.0, self.queue_length + self._lam - served)

        self.latency_p95 = self._estimate_latency(self._lam, self.current_capacity, self.queue_length)
        util = self._utilization(self._lam, self.current_capacity)

        self.trend_window.pop(0)
        self.trend_window.append(self._lam)

        # Reward shaping
        sla_breach = max(0.0, self.latency_p95 - self.T_sla)
        reward = 0.5 - 12.0 * sla_breach

        excess_capacity = max(0.0, self.current_capacity * self.mu - self._lam)
        reward -= 0.008 * (excess_capacity ** 2)          # quadratic waste penalty
        reward -= 0.04 * abs(step_scaling)                 # action magnitude penalty
        reward -= 0.15 * max(0.0, util - 1.0)             # congestion penalty

        if cap_signal == 1 and util < 0.9:
            reward -= 0.05                                 # false capacity-signal penalty

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "latency_p95": float(self.latency_p95),
            "lam": float(self._lam),
            "current_capacity": int(self.current_capacity),
            "utilization": float(util),
        }
        return self._build_obs(), float(reward), terminated, truncated, info
