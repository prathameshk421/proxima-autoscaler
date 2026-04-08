"""Ablation study configurations for Proxima Auto-Scaler.

Defines 4 ablation variants to quantify the contribution of each
design component:

1. **no_gru**          — Flat MLP replaces the GRU trend encoder
2. **no_curriculum**   — Train only on hard difficulty (skip easy/medium)
3. **symmetric_reward** — Equal SLA/waste penalty weights
4. **no_boot_delay**   — Remove server startup delay
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment.

    Attributes
    ----------
    name : str
        Human-readable experiment name.
    description : str
        What this ablation tests.
    curriculum : list[tuple[str, int]]
        Training phases as ``(difficulty, timesteps)`` pairs.
    use_gru : bool
        Whether to use the GRU feature extractor (False = flat MLP).
    sla_penalty_weight : float
        Weight for SLA breach penalty in the reward function.
    waste_penalty_weight : float
        Weight for over-provisioning penalty.
    override_boot_delay : int | None
        If set, overrides the per-difficulty boot delay.
    """

    name: str
    description: str
    curriculum: list[tuple[str, int]]
    use_gru: bool = True
    sla_penalty_weight: float = 12.0
    waste_penalty_weight: float = 0.008
    override_boot_delay: int | None = None

    @property
    def tag(self) -> str:
        """Return a filesystem-safe tag derived from the experiment name."""
        return self.name.replace(" ", "_").lower()


FULL_MODEL = AblationConfig(
    name="Full Model",
    description="Complete Proxima system with all components (control)",
    curriculum=[
        ("easy", 150_000),
        ("medium", 250_000),
        ("hard", 600_000),
    ],
)


NO_GRU = AblationConfig(
    name="No GRU",
    description="Replace GRU trend encoder with flat MLP — tests value of temporal reasoning",
    curriculum=[
        ("easy", 150_000),
        ("medium", 250_000),
        ("hard", 600_000),
    ],
    use_gru=False,
)


NO_CURRICULUM = AblationConfig(
    name="No Curriculum",
    description="Train only on hard difficulty — tests value of progressive training",
    curriculum=[
        ("hard", 1_000_000),
    ],
)


SYMMETRIC_REWARD = AblationConfig(
    name="Symmetric Reward",
    description="Equal SLA/waste penalty (both 1.0) — tests value of asymmetric shaping",
    curriculum=[
        ("easy", 150_000),
        ("medium", 250_000),
        ("hard", 600_000),
    ],
    sla_penalty_weight=1.0,
    waste_penalty_weight=1.0,
)


NO_BOOT_DELAY = AblationConfig(
    name="No Boot Delay",
    description="Set boot_delay=0 for all tiers — tests value of delay-aware planning",
    curriculum=[
        ("easy", 150_000),
        ("medium", 250_000),
        ("hard", 600_000),
    ],
    override_boot_delay=0,
)


ALL_ABLATIONS: dict[str, AblationConfig] = {
    "full_model": FULL_MODEL,
    "no_gru": NO_GRU,
    "no_curriculum": NO_CURRICULUM,
    "symmetric_reward": SYMMETRIC_REWARD,
    "no_boot_delay": NO_BOOT_DELAY,
}
