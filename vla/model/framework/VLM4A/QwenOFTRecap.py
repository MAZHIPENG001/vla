"""OFT policy conditioned on RECAP's binary improvement indicator."""

from dataclasses import dataclass, field

import torch
from omegaconf import OmegaConf

from vla.model.framework.share_tools import _to_omegaconf, merge_framework_config
from vla.model.modules.recap.advantage import AdvantageCondition
from vla.model.tools import FRAMEWORK_REGISTRY
from .QwenOFT import QwenOFTDefaultConfig, Qwenvl_OFT


@dataclass
class QwenOFTRecapDefaultConfig(QwenOFTDefaultConfig):
    """Policy config lives under framework; the separate critic uses recap."""

    name: str = "QwenOFTRecap"
    action_model: dict = field(default_factory=lambda: {
        **QwenOFTDefaultConfig().action_model, "action_horizon": 8,
    })
    advantage_conditioning: dict = field(default_factory=lambda: {
        "enabled": True,
        "freeze_critic": True,
        "value_loss_weight": 1.0,
        "dropout_probability": 0.3,
        "inference_condition": "positive",
    })


@FRAMEWORK_REGISTRY.register("QwenOFTRecap")
class Qwenvl_OFT_Recap(Qwenvl_OFT):
    """Train OFT with positive/negative/unconditional action contexts.

    When enabled, compute conditions online from sampled trajectory targets.
    File labels and caller-supplied training conditions are not supported.
    Disable conditioning for ordinary OFT training without a critic.
    Deployment uses a fixed condition without executing the critic.
    """

    def __init__(self, config=None, **kwargs):
        # Preserve tracked configs when supplied by the trainer.
        if not hasattr(config, "_cfg"):
            config = OmegaConf.merge({"datasets": {"vla_data": {}}}, _to_omegaconf(config))
        config = merge_framework_config(QwenOFTRecapDefaultConfig, config)
        cfg = config.framework.advantage_conditioning
        for name in ("enabled", "freeze_critic"):
            if not isinstance(cfg[name], bool):
                raise ValueError(f"advantage_conditioning.{name} must be bool")
        for obsolete in ("source", "sft_positive"):
            if obsolete in cfg:
                raise ValueError(f"remove advantage_conditioning.{obsolete}; enabled advantage conditioning is online only")
        if not 0 <= cfg.dropout_probability <= 1:
            raise ValueError("advantage_conditioning.dropout_probability must lie in [0, 1]")
        if cfg.inference_condition not in ("positive", "negative", "unconditional"):
            raise ValueError("inference_condition must be positive, negative or unconditional")
        super().__init__(config, **kwargs)
        self.online_advantage = None
        if cfg.enabled:
            from vla.model.modules.recap.recap import QwenRecap
            self.online_advantage = QwenRecap(self.config, online=True)

    def forward(self, examples=None, **kwargs):
        if kwargs.get("advantage_condition") is not None or kwargs.get("advantage") is not None:
            raise ValueError("VLA training computes its own advantage condition online")
        online = None
        if self.online_advantage is not None:
            online = self.online_advantage(examples, online=True, generator=kwargs.get("generator"))
            kwargs["advantage_condition"] = online["condition"]
        output = super().forward(examples, **kwargs)
        if online is not None:
            output["advantage_mean"] = online["advantages"].mean()
            if "value_loss" in online:
                output["value_loss"] = online["value_loss"]
                weight = self.config.framework.advantage_conditioning.value_loss_weight
                output["loss"] = output["action_loss"] + weight * online["value_loss"]
        return output

    def _condition_texts(self, examples, *, inference, advantage_condition=None, advantage=None):
        cfg = self.config.framework.advantage_conditioning
        if not cfg.enabled:
            return [""] * len(examples)
        if inference:
            if advantage_condition is not None:
                raise ValueError("inference uses advantage='positive', 'negative' or 'unconditional'")
            selected = cfg.inference_condition if advantage is None else advantage
            if selected not in ("positive", "negative", "unconditional"):
                raise ValueError("advantage must be positive, negative or unconditional")
            text = "" if selected == "unconditional" else f"Advantage: {selected}"
            return [text] * len(examples)
        if advantage is not None:
            raise ValueError("training requires an internally computed online condition")
        if not isinstance(advantage_condition, AdvantageCondition):
            raise ValueError("training requires an internally computed online AdvantageCondition")
        for tensor in (advantage_condition.indicator, advantage_condition.conditioning_mask):
            if tensor.dtype != torch.bool or tensor.shape != (len(examples),):
                raise ValueError("condition indicator and mask must be bool tensors of shape [batch]")
        # Interventions and dropout are handled once by QwenRecap.
        return advantage_condition.to_text()

    def _build_action_inputs(
        self, images, instructions, examples, *, inference,
        advantage_condition=None, advantage=None, generator=None, **kwargs,
    ):
        conditions = self._condition_texts(
            examples, inference=inference, advantage_condition=advantage_condition,
            advantage=advantage,
        )
        action_tokens = self.action_token * self.chunk_len
        action_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        suffixes = []
        for example, condition in zip(examples, conditions):
            subtask = example.get("subtask", "")
            if not isinstance(subtask, str):
                raise ValueError("subtask must be text")
            suffix = f"\nSubtask: {subtask}" if subtask else ""
            if condition:
                suffix += f"\n{condition}"
            suffixes.append(suffix + action_suffix)
        return self.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=instructions, instruction_suffixes=suffixes,
        )
