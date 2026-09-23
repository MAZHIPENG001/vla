"""OFT policy conditioned on RECAP's binary improvement indicator."""

from dataclasses import dataclass, field

import numpy as np
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
        "dropout_probability": 0.3,
        "sft_positive": False,
        "inference_condition": "positive",
    })


def _bool_label(value, name):
    """Reject raw advantage scores; binarization belongs to the critic pipeline."""
    if isinstance(value, torch.Tensor) and value.dtype == torch.bool and value.numel() == 1:
        return value.item()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be a boolean label, not a raw advantage score")


@FRAMEWORK_REGISTRY.register("QwenOFTRecap")
class Qwenvl_OFT_Recap(Qwenvl_OFT):
    """Train OFT with positive/negative/unconditional action contexts.

    Each example carries a bool advantage_indicator, optionally is_intervention
    and subtask. Alternatively pass advantage_condition=AdvantageCondition with
    shape [B]; its mask is authoritative and is never dropped a second time.
    Missing labels raise unless sft_positive=True or the action is a correction.
    Normal OFT image/lang/state/action fields and action_loss remain unchanged.
    The model owns only the policy backbone, never a critic.
    """

    def __init__(self, config=None, **kwargs):
        # Preserve tracked configs when supplied by the trainer.
        if not hasattr(config, "_cfg"):
            config = OmegaConf.merge({"datasets": {"vla_data": {}}}, _to_omegaconf(config))
        config = merge_framework_config(QwenOFTRecapDefaultConfig, config)
        cfg = config.framework.advantage_conditioning
        if not 0 <= cfg.dropout_probability <= 1:
            raise ValueError("advantage_conditioning.dropout_probability must lie in [0, 1]")
        if not isinstance(cfg.sft_positive, bool):
            raise ValueError("advantage_conditioning.sft_positive must be bool")
        if cfg.inference_condition not in ("positive", "negative", "unconditional"):
            raise ValueError("inference_condition must be positive, negative or unconditional")
        super().__init__(config, **kwargs)

    def _condition_texts(self, examples, *, inference, advantage_condition=None, advantage=None, generator=None):
        cfg = self.config.framework.advantage_conditioning
        if inference:
            if advantage_condition is not None:
                raise ValueError("inference uses advantage='positive', 'negative' or 'unconditional'")
            selected = cfg.inference_condition if advantage is None else advantage
            if selected not in ("positive", "negative", "unconditional"):
                raise ValueError("advantage must be positive, negative or unconditional")
            text = "" if selected == "unconditional" else f"Advantage: {selected}"
            return [text] * len(examples)
        if advantage is not None:
            raise ValueError("training requires advantage_indicator labels or advantage_condition")

        interventions = torch.tensor([
            _bool_label(example.get("is_intervention", False), "is_intervention") for example in examples
        ], dtype=torch.bool)
        if advantage_condition is not None:
            if not isinstance(advantage_condition, AdvantageCondition):
                raise ValueError("advantage_condition must be an AdvantageCondition")
            for tensor in (advantage_condition.indicator, advantage_condition.conditioning_mask):
                if tensor.dtype != torch.bool or tensor.shape != (len(examples),):
                    raise ValueError("condition indicator and mask must be bool tensors of shape [batch]")
            indicator = advantage_condition.indicator.detach().cpu() | interventions
            retained = advantage_condition.conditioning_mask.detach().cpu()
        else:
            labels = []
            for index, example in enumerate(examples):
                if cfg.sft_positive:
                    positive = True
                elif "advantage_indicator" in example:
                    positive = _bool_label(example["advantage_indicator"], "advantage_indicator")
                elif interventions[index]:
                    positive = True
                else:
                    raise ValueError("training example is missing advantage_indicator; label the data or enable sft_positive")
                labels.append(positive)
            indicator = torch.tensor(labels, dtype=torch.bool) | interventions
            retained = torch.ones(len(examples), dtype=torch.bool)
            if self.training and cfg.dropout_probability > 0:
                # Label preparation uses CPU RNG; a supplied generator must be CPU.
                retained = torch.rand(len(examples), generator=generator) >= cfg.dropout_probability
        return AdvantageCondition(indicator, retained).to_text()

    def _build_action_inputs(
        self, images, instructions, examples, *, inference,
        advantage_condition=None, advantage=None, generator=None, **kwargs,
    ):
        conditions = self._condition_texts(
            examples, inference=inference, advantage_condition=advantage_condition,
            advantage=advantage, generator=generator,
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
