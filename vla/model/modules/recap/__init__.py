"""RECAP advantage estimation and policy conditioning utilities."""

from .advantage import AdvantageCondition, AdvantageModule
from .value_model import QwenValueModel, ValueModelOutput

__all__ = [
    "AdvantageCondition",
    "AdvantageModule",
    "QwenValueModel",
    "ValueModelOutput",
    "QwenRecap",
    "QwenRecapDefaultConfig",
]


def __getattr__(name):
    # Keep the tensor-only advantage API free of config/vision imports.
    if name in ("QwenRecap", "QwenRecapDefaultConfig"):
        from .recap import QwenRecap, QwenRecapDefaultConfig

        return {"QwenRecap": QwenRecap, "QwenRecapDefaultConfig": QwenRecapDefaultConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
