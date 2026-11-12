"""Layer-wise flow-matching policy conditioned on RECAP's improvement indicator."""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

from deployment.model_server.tools.image_tools import to_pil_preserve
from vla.model.framework.share_tools import _to_omegaconf, merge_framework_config
from vla.model.modules.recap.advantage import AdvantageCondition
from vla.model.tools import FRAMEWORK_REGISTRY
from vla.training.trainer_utils.trainer_tools import resize_images
from .QwenPI import QwenPIDefaultConfig, Qwen_PI


@dataclass
class QwenPIRecapDefaultConfig(QwenPIDefaultConfig):
    """Policy config lives under framework; the separate critic uses recap."""

    name: str = "QwenPI_Recap"
    advantage_conditioning: dict = field(default_factory=lambda: {
        "enabled": True,
        "freeze_critic": True,
        "dropout_probability": 0.3,
        "inference_condition": "positive",
    })


@FRAMEWORK_REGISTRY.register("QwenPI_Recap")
class Qwen_PI_Recap(Qwen_PI):
    """Train PI with online RECAP conditions and a frozen pretrained critic.

    Reuse the original PI backbone/action-head setup. Override the encoding,
    training and sampling paths to add condition suffixes without changing PI.
    Deployment uses a fixed condition without executing the critic.
    """

    def __init__(self, config=None, **kwargs):
        if not hasattr(config, "_cfg"):
            config = OmegaConf.merge({"datasets": {"vla_data": {}}}, _to_omegaconf(config))
        config = merge_framework_config(QwenPIRecapDefaultConfig, config)
        cfg = config.framework.advantage_conditioning
        for name in ("enabled", "freeze_critic"):
            if not isinstance(cfg[name], bool):
                raise ValueError(f"advantage_conditioning.{name} must be bool")
        for obsolete in ("source", "sft_positive"):
            if obsolete in cfg:
                raise ValueError(f"remove advantage_conditioning.{obsolete}; enabled advantage conditioning is online only")
        if "value_loss_weight" in cfg:
            raise ValueError("remove advantage_conditioning.value_loss_weight; train the critic separately with train_recap_value")
        if cfg.enabled and not cfg.freeze_critic:
            raise ValueError("advantage conditioning requires freeze_critic=true; train the critic separately with train_recap_value")
        if not 0 <= cfg.dropout_probability <= 1:
            raise ValueError("advantage_conditioning.dropout_probability must lie in [0, 1]")
        if cfg.inference_condition not in ("positive", "negative", "unconditional"):
            raise ValueError("inference_condition must be positive, negative or unconditional")
        super().__init__(config, **kwargs)
        self.online_advantage = None
        if cfg.enabled:
            from vla.model.modules.recap.recap import QwenRecap
            self.online_advantage = QwenRecap(self.config, online=True)

    def _condition_texts(self, examples, *, inference, advantage_condition=None, advantage=None):
        """Use RECAP's online binary labels for training and a fixed label at inference."""
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
        # QwenRecap applies task thresholds, intervention overrides and dropout once.
        return advantage_condition.to_text()

    def _build_action_inputs(
        self, images, instructions, examples, *, inference,
        advantage_condition=None, advantage=None,
    ):
        if not self.config.framework.advantage_conditioning.enabled:
            return self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
        conditions = self._condition_texts(
            examples, inference=inference, advantage_condition=advantage_condition, advantage=advantage,
        )
        suffixes = []
        for example, condition in zip(examples, conditions):
            subtask = example.get("subtask", "")
            if not isinstance(subtask, str):
                raise ValueError("subtask must be text")
            suffix = f"\nSubtask: {subtask}" if subtask else ""
            if condition:
                suffix += f"\n{condition}"
            suffixes.append(suffix)
        # Append after the complete task/CoT template so it reaches every VLM layer.
        return self.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=instructions, instruction_suffixes=suffixes,
        )

    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str], *, examples=None,
        inference=False, advantage_condition=None, advantage=None,
    ) -> tuple:
        """Run QwenVL and return (layer-wise hidden states, attention_mask) for the Action DiT."""
        qwen_inputs = self._build_action_inputs(
            batch_images, instructions, examples, inference=inference,
            advantage_condition=advantage_condition, advantage=advantage,
        )
        attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(qwenvl_outputs.hidden_states[-expected_layers:])
        return vl_embs_list, attention_mask

    def forward(
            self,
            examples: List[dict] = None,
            **kwargs,
    ) -> dict:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
                When RECAP is enabled, the online DataLoader also supplies
                task_key, advantage_reward and advantage_bootstrap.
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        if kwargs.get("advantage_condition") is not None or kwargs.get("advantage") is not None:
            raise ValueError("VLA training computes its own advantage condition online")
        online = None
        if self.online_advantage is not None:
            online = self.online_advantage(examples, online=True, generator=kwargs.get("generator"))

        batch_images = [example["image"] for example in examples]  # [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Step 1: encode through QwenVL
        vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states(
            batch_images, instructions, examples=examples,
            advantage_condition=online["condition"] if online is not None else None,
        )
        base_hidden = vl_embs_list[-1]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float16):
            # Label alignment: take the last chunk_len segment
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon:, :]  # (B, action_horizon, action_dim)

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            repeated_diffusion_steps = 2  # NO repeat for big action FM
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            # Repeat features for each layer
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
                encoder_attention_mask=backbone_attention_mask,
            )  # (B, chunk_len, action_dim)

        output = {"action_loss": action_loss}
        if online is not None:
            output["loss"] = action_loss
            output["advantage_mean"] = online["advantages"].mean()
        return output

    @torch.inference_mode()
    def predict_action(  # TODO align  predict_action with forward, make api more flexible
            self,
            examples: List[dict] = None,
            **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Sample normalized actions with optional training-free RTC flow guidance.
        With RECAP enabled, advantage selects positive/negative/unconditional
        (default: configured inference_condition); the critic is not executed.

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]  # [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: encode through QwenVL
        vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states(
            batch_images, instructions, examples=examples, inference=True,
            advantage_condition=kwargs.get("advantage_condition"), advantage=kwargs.get("advantage"),
        )
        base_hidden = vl_embs_list[-1]
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float16):
            pred_actions = self.action_model.predict_action(
                vl_embs_list, state, encoder_attention_mask=backbone_attention_mask,
                rtc=kwargs.get("rtc"),
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

