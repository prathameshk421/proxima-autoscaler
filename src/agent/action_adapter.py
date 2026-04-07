"""Discrete-to-structured action adapter for PPO policies."""

from __future__ import annotations

from src.env.schemas import Action


_ACTION_TABLE: tuple[Action, ...] = (
	Action(mode="step", scale_delta=-2),
	Action(mode="step", scale_delta=-1),
	Action(mode="step", scale_delta=0),
	Action(mode="step", scale_delta=1),
	Action(mode="step", scale_delta=2),
	Action(mode="target", target_utilization=0.4),
	Action(mode="target", target_utilization=0.5),
	Action(mode="target", target_utilization=0.6),
	Action(mode="target", target_utilization=0.7),
	Action(mode="signal", signal_type="INSUFFICIENT_CAPACITY"),
)


def action_id_to_action(action_id: int) -> Action:
	"""Convert discrete action id to structured Action model."""
	if not isinstance(action_id, int):
		raise ValueError(f"action_id must be int, got {type(action_id).__name__}")

	if action_id < 0 or action_id >= len(_ACTION_TABLE):
		raise ValueError(f"Invalid action_id={action_id}; expected range [0, {len(_ACTION_TABLE) - 1}]")

	return _ACTION_TABLE[action_id]


def num_actions() -> int:
	"""Return number of discrete actions available to RL policy."""
	return len(_ACTION_TABLE)
