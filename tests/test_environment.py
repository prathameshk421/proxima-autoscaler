from env.environment import CloudAutoScalingEnv
from env.models import CloudAction


def test_valid_step_scale_is_not_marked_invalid() -> None:
    env = CloudAutoScalingEnv("easy")
    env.reset()

    result = env.step(CloudAction(mode="step", scale_delta=1))

    assert result.info["invalid"] == 0.0
    assert result.info["last_action_error"] is None


def test_step_during_cooldown_is_marked_invalid() -> None:
    env = CloudAutoScalingEnv("easy")
    env.reset()

    env.step(CloudAction(mode="step", scale_delta=1))
    result = env.step(CloudAction(mode="step", scale_delta=1))

    assert result.info["invalid"] == -0.5
    assert result.info["last_action_error"] == "cooldown_active"
