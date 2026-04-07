"""Pytest suite for deterministic task/workload behavior."""

import numpy as np

from src.env.cloud_env import CloudAutoScalingEnv
from src.env.workload import WorkloadGenerator


def test_workload_patterns_differ_by_difficulty() -> None:
    """Easy/medium/hard workload series should not be identical patterns."""
    easy = WorkloadGenerator("easy", episode_length=120, seed=42, c_baseline=4, mu_per_server=50.0).generate()
    medium = WorkloadGenerator("medium", episode_length=120, seed=42, c_baseline=4, mu_per_server=50.0).generate()
    hard = WorkloadGenerator("hard", episode_length=120, seed=42, c_baseline=4, mu_per_server=50.0).generate()

    assert not np.allclose(easy, medium)
    assert not np.allclose(medium, hard)
    assert not np.allclose(easy, hard)


def test_same_seed_gives_same_series() -> None:
    """Same difficulty and seed should reproduce exactly the same workload."""
    gen_a = WorkloadGenerator("hard", episode_length=160, seed=77, c_baseline=4, mu_per_server=50.0)
    gen_b = WorkloadGenerator("hard", episode_length=160, seed=77, c_baseline=4, mu_per_server=50.0)

    assert np.allclose(gen_a.generate(), gen_b.generate())


def test_observation_reflects_difficulty_label() -> None:
    """reset observation should carry the selected task difficulty label."""
    for difficulty in ("easy", "medium", "hard"):
        env = CloudAutoScalingEnv(difficulty=difficulty, seed=21)
        obs, _ = env.reset(seed=21)
        assert obs.task_difficulty == difficulty
