"""Gym wrapper converting MultiDiscrete → Discrete for DQN compatibility."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class DiscreteActionWrapper(gym.ActionWrapper):
    """Flatten ``MultiDiscrete([5, 4, 2])`` → ``Discrete(40)`` for value-based methods.

    Encoding::

        flat_action = a0 * (4*2) + a1 * 2 + a2

    Decoding::

        a2 = flat_action % 2
        a1 = (flat_action // 2) % 4
        a0 = flat_action // (4*2)
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        nvec = env.action_space.nvec
        self._nvec = nvec.copy()
        self._n = int(np.prod(nvec))
        self.action_space = gym.spaces.Discrete(self._n)

    def action(self, flat_action: int) -> np.ndarray:
        """Decode a flat integer into the original MultiDiscrete components."""
        acts = []
        remainder = int(flat_action)
        for dim_size in reversed(self._nvec):
            acts.append(remainder % dim_size)
            remainder //= dim_size
        return np.array(list(reversed(acts)), dtype=np.int64)

    def reverse_action(self, multi_action: np.ndarray) -> int:
        """Encode a MultiDiscrete action into a flat integer."""
        flat = 0
        for i, a in enumerate(multi_action):
            multiplier = int(np.prod(self._nvec[i + 1 :]))
            flat += int(a) * multiplier
        return flat
