"""Pytest suite for deterministic task grading behavior."""

from src.env.grader import grade_episode


def _sample_metrics() -> dict[str, float]:
	"""Return stable synthetic episode metrics for grading tests."""
	return {
		"steps": 200.0,
		"avg_latency": 165.0,
		"sla_threshold": 220.0,
		"avg_servers": 5.5,
		"min_servers": 1.0,
		"max_servers": 10.0,
		"oscillation_count": 14.0,
		"signal_tp": 9.0,
		"signal_fp": 3.0,
		"signal_fn": 4.0,
		"saturation_events": 13.0,
		"avg_reward": 0.65,
	}


def test_grade_episode_in_range() -> None:
	"""grade_episode should always return normalized score in [0,1]."""
	score = grade_episode(_sample_metrics(), difficulty="medium")
	assert 0.0 <= score <= 1.0


def test_grader_is_deterministic() -> None:
	"""Same metrics and difficulty should produce identical score repeatedly."""
	metrics = _sample_metrics()
	score_a = grade_episode(metrics, difficulty="easy")
	score_b = grade_episode(metrics, difficulty="easy")
	assert score_a == score_b


def test_hard_weights_signal_more_than_easy() -> None:
	"""Hard grading should benefit more from strong saturation signaling."""
	metrics = {
		"steps": 200.0,
		"avg_latency": 190.0,
		"sla_threshold": 220.0,
		"avg_servers": 6.5,
		"min_servers": 1.0,
		"max_servers": 10.0,
		"oscillation_count": 20.0,
		"signal_tp": 12.0,
		"signal_fp": 0.0,
		"signal_fn": 0.0,
		"saturation_events": 12.0,
		"avg_reward": 0.55,
	}

	easy_score = grade_episode(metrics, difficulty="easy")
	hard_score = grade_episode(metrics, difficulty="hard")
	assert hard_score > easy_score
