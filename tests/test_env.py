"""Pytest suite for CloudAutoScalingEnv contract and dynamics."""

from src.env.cloud_env import CloudAutoScalingEnv
from src.env.schemas import Action, Observation


def test_reset_returns_valid_observation() -> None:
	"""reset should return valid Observation with expected difficulty label."""
	env = CloudAutoScalingEnv(difficulty="easy", seed=42)
	obs, info = env.reset()

	assert isinstance(obs, Observation)
	assert isinstance(info, dict)
	assert obs.task_difficulty == "easy"
	assert env.min_servers <= obs.current_servers <= env.max_servers


def test_step_returns_observation_reward_done_info() -> None:
	"""step should return observation, reward, done flag, and info dictionary."""
	env = CloudAutoScalingEnv(difficulty="medium", seed=99)
	env.reset()
	obs, reward, done, info = env.step(Action(mode="step", scale_delta=1))

	assert isinstance(obs, Observation)
	assert isinstance(reward, float)
	assert isinstance(done, bool)
	assert isinstance(info, dict)


def test_cooldown_prevents_immediate_repeated_scaling() -> None:
	"""Second immediate infra-changing action should be blocked by cooldown."""
	env = CloudAutoScalingEnv(difficulty="easy", seed=7, cooldown_steps=2)
	env.reset()

	env.step(Action(mode="step", scale_delta=1))
	_, _, _, info = env.step(Action(mode="step", scale_delta=1))

	assert info["cooldown_blocked"] is True


def test_min_max_server_constraints_are_enforced() -> None:
	"""Environment should never exceed [min_servers, max_servers]."""
	env = CloudAutoScalingEnv(
		difficulty="hard",
		seed=5,
		min_servers=1,
		max_servers=3,
		startup_delay_steps=1,
		cooldown_steps=0,
	)
	env.reset()

	for _ in range(10):
		obs, _, _, _ = env.step(Action(mode="step", scale_delta=2))
		assert env.min_servers <= obs.current_servers <= env.max_servers

	for _ in range(10):
		obs, _, _, _ = env.step(Action(mode="step", scale_delta=-2))
		assert env.min_servers <= obs.current_servers <= env.max_servers


def test_determinism_for_same_seed_and_difficulty() -> None:
	"""Two env instances with same difficulty/seed should produce identical trajectory."""
	env_a = CloudAutoScalingEnv(difficulty="medium", seed=123)
	env_b = CloudAutoScalingEnv(difficulty="medium", seed=123)

	obs_a, _ = env_a.reset(seed=123)
	obs_b, _ = env_b.reset(seed=123)
	assert obs_a.model_dump() == obs_b.model_dump()

	scripted_actions = [
		Action(mode="step", scale_delta=1),
		Action(mode="target", target_utilization=0.6),
		Action(mode="signal", signal_type="INSUFFICIENT_CAPACITY"),
	]

	for idx in range(12):
		action = scripted_actions[idx % len(scripted_actions)]
		obs_a, reward_a, done_a, info_a = env_a.step(action)
		obs_b, reward_b, done_b, info_b = env_b.step(action)

		assert obs_a.model_dump() == obs_b.model_dump()
		assert reward_a == reward_b
		assert done_a == done_b
		assert info_a == info_b
