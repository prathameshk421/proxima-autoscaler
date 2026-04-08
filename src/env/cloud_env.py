"""Gymnasium environment for cloud auto-scaling simulation.

Implements a queue-based cloud environment with delayed server boot effects,
multi-channel trend observations, and a shaped reward function balancing
SLA compliance against resource waste.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from src.env.schemas import Observation


class CloudAutoScalingEnv(gym.Env):
    """Queue-based cloud auto-scaling environment with delayed scale-out.

    The environment simulates a pool of servers handling incoming requests
    characterised by a time-varying arrival rate λ(t).  Scaling decisions
    take effect after a configurable boot delay, and a cooldown period
    prevents rapid oscillation.

    Parameters
    ----------
    difficulty : str
        Difficulty tier — ``"easy"``, ``"medium"``, or ``"hard"``.
    seed : int
        Random seed for reproducible workload generation.

    Attributes
    ----------
    max_steps : int
        Episode length (fixed at 200 timesteps).
    T_sla : float
        SLA latency threshold in seconds (2.0 s).
    cooldown_period : int
        Minimum steps between consecutive scaling actions.
    """

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
            shape=(39,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.current_capacity = max(1, self.c_init)
        self.queue_length = 0.0
        self.cooldown_remaining = 0
        self._boot_pipeline: list[tuple[int, int]] = []
        self._lam = float(self.lam_base)
        self.latency_p95 = 0.0
        self.trend_window = [self._lam] * 8
        init_util = self._utilization(self._lam, self.current_capacity)
        init_lat = self._estimate_latency(self._lam, self.current_capacity, 0.0)
        self.util_trend = [init_util] * 8
        self.latency_trend = [init_lat] * 8
        self.queue_trend = [0.0] * 8

    def _configure_difficulty(self, difficulty: str) -> None:
        """Set environment parameters for the given difficulty tier.

        Parameters
        ----------
        difficulty : str
            One of ``"easy"``, ``"medium"``, or ``"hard"``.
        """
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
        """Reset the environment for a new episode.

        Parameters
        ----------
        seed : int | None
            Optional new RNG seed.  If provided, replaces the stored seed.
        options : dict | None
            Unused; accepted for Gymnasium API compatibility.

        Returns
        -------
        tuple[np.ndarray, dict]
            Initial observation vector and an empty info dict.
        """
        if seed is not None:
            self._init_seed = int(seed)
        self.rng = np.random.default_rng(self._init_seed)

        self.current_step = 0
        self.current_capacity = max(1, self.c_init)
        self.queue_length = 0.0
        self.cooldown_remaining = 0
        self._boot_pipeline = []

        self._lam = float(self._demand_profile(self.current_step))
        self.latency_p95 = self._estimate_latency(
            self._lam, self.current_capacity, self.queue_length
        )
        self.trend_window = [self._lam] * 8
        init_util = self._utilization(self._lam, self.current_capacity)
        self.util_trend = [init_util] * 8
        self.latency_trend = [self.latency_p95] * 8
        self.queue_trend = [0.0] * 8

        return self._build_obs(), {}

    def state(self) -> Observation:
        """Return the current environment state as a structured Pydantic model."""
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
            util_trend=[float(x) for x in self.util_trend],
            latency_trend=[float(x) for x in self.latency_trend],
            queue_trend=[float(x) for x in self.queue_trend],
        )

    def step(self, action: np.ndarray):
        """Advance the environment by one timestep.

        Parameters
        ----------
        action : np.ndarray
            MultiDiscrete action array ``[step_idx, util_idx, cap_signal]``.

        Returns
        -------
        tuple
            ``(obs, reward, terminated, truncated, info)`` following the
            Gymnasium step API.
        """
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
        next_boot: list[tuple[int, int]] = []
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

        self.latency_p95 = self._estimate_latency(
            self._lam, self.current_capacity, self.queue_length
        )
        util = self._utilization(self._lam, self.current_capacity)

        self.trend_window.pop(0)
        self.trend_window.append(self._lam)
        self.util_trend.pop(0)
        self.util_trend.append(util)
        self.latency_trend.pop(0)
        self.latency_trend.append(self.latency_p95)
        self.queue_trend.pop(0)
        self.queue_trend.append(self.queue_length)

        sla_breach = max(0.0, self.latency_p95 - self.T_sla)
        reward = 0.5 - 12.0 * sla_breach

        excess_capacity = max(0.0, self.current_capacity * self.mu - self._lam)
        reward -= 0.008 * (excess_capacity**2)
        reward -= 0.04 * abs(step_scaling)
        reward -= 0.15 * max(0.0, util - 1.0)

        if cap_signal == 1 and util < 0.9:
            reward -= 0.05

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

    def _build_obs(self) -> np.ndarray:
        """Construct the flat observation vector from current state.

        Returns a 39-dimensional float32 array: 7 scalar features followed
        by 4 normalised trend channels of 8 timesteps each.
        """
        total_booting = int(sum(servers for servers, _ in self._boot_pipeline))

        lam_scale = max(1.0, self.lam_base + self.lam_amp)
        latency_scale = max(1.0, self.T_sla * 2.0)
        queue_scale = max(1.0, self.c_max * self.mu)
        norm_demand = [x / lam_scale for x in self.trend_window]
        norm_util = self.util_trend
        norm_latency = [x / latency_scale for x in self.latency_trend]
        norm_queue = [x / queue_scale for x in self.queue_trend]
        return np.array(
            [
                float(self.current_capacity),
                float(self.c_max),
                float(total_booting),
                float(self.cooldown_remaining),
                float(self.queue_length),
                float(self._lam),
                float(self.mu),
            ]
            + norm_demand
            + norm_util
            + norm_latency
            + norm_queue,
            dtype=np.float32,
        )

    def _demand_profile(self, t: int) -> float:
        """Return the synthetic demand rate λ(t) at timestep *t*."""
        sinusoid = self.lam_amp * np.sin(2.0 * np.pi * (t / 40.0))
        burst = 0.0
        if self.difficulty != "easy" and (t % 50 in (0, 1, 2)):
            burst = self.lam_amp * 0.6
        noise = self.rng.normal(0.0, self.noise_scale)
        return float(max(0.0, self.lam_base + sinusoid + burst + noise))

    def _utilization(self, lam: float, capacity: int) -> float:
        """Return server utilisation ρ = λ / (c · μ), clamped above zero."""
        if capacity <= 0:
            return 1.0
        return float(lam / max(1e-6, capacity * self.mu))

    def _estimate_latency(self, lam: float, capacity: int, queue_len: float) -> float:
        """Return an estimated p95 latency from a simplified queueing model."""
        service = max(1e-6, capacity * self.mu)
        rho = min(1.5, lam / service)
        wait = queue_len / service
        baseline = 0.08 + 0.35 * (rho**2)
        penalty = 1.2 * max(0.0, rho - 1.0)
        return float(max(0.0, baseline + penalty + wait))
