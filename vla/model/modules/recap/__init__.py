"""RECAP advantage estimation and policy conditioning utilities."""

from .advantage import (
    AdvantageCondition,
    compute_advantage_condition,
    compute_advantages,
    episode_rewards,
    estimate_task_thresholds,
    values_from_logits,
)
from .value_model import QwenValueModel, ValueModelOutput

__all__ = [
    "AdvantageCondition",
    "QwenValueModel",
    "ValueModelOutput",
    "QwenRecap",
    "QwenRecapDefaultConfig",
    "compute_advantage_condition",
    "compute_advantages",
    "episode_rewards",
    "estimate_task_thresholds",
    "values_from_logits",
]


def __getattr__(name):
    # Keep the tensor-only advantage API free of config/vision imports.
    if name in ("QwenRecap", "QwenRecapDefaultConfig"):
        from .recap import QwenRecap, QwenRecapDefaultConfig

        return {"QwenRecap": QwenRecap, "QwenRecapDefaultConfig": QwenRecapDefaultConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
