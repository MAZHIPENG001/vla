"""Qwen3-VL distributional critic for RECAP, trained with Monte Carlo returns."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ValueModelOutput:
    logits: Tensor
    values: Tensor
    loss: Tensor | None = None


class QwenValueModel(nn.Module):
    """Images + task instruction -> a categorical distribution over value bins.

    Reuses _QWen_VL_Interface for loading, chat formatting and visual inputs.
    The final nonpadding token summarizes the causal multimodal context. A new
    linear head maps it to 201 bins by default (RECAP Sec. IV-A). This head must
    be trained on normalized return-to-go targets before generating advantages;
    the pretrained instruct checkpoint alone is not a trained critic.

    Each sample is one observation with one or more camera images. Callers should
    only submit valid observations, then scatter predictions into [B, T] for
    compute_advantages. The critic input must not include advantage labels or
    future observations/actions.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        *,
        attn_implementation: str = "sdpa",
        num_bins: int = 201,
        value_min: float = -1.0,
        value_max: float = 0.0,
        freeze_backbone: bool = False,
        config=None,
    ) -> None:
        super().__init__()
        if not isinstance(num_bins, int) or isinstance(num_bins, bool) or num_bins < 2:
            raise ValueError("num_bins must be an integer >= 2")
        if not (math.isfinite(value_min) and math.isfinite(value_max) and value_min < value_max):
            raise ValueError("value bounds must be finite and value_min < value_max")

        # Lazy imports keep the standalone advantage helpers usable without a VLM.
        from omegaconf import OmegaConf
        from vla.model.modules.vlm import get_vlm_model

        # Only recap.qwenvl configures the critic. Qwen's existing factory expects
        # framework.qwenvl, so adapt into a private config without changing the
        # full shared config or accidentally loading the policy backbone.
        if config is None:
            backbone_config = OmegaConf.create({
                "framework": {"qwenvl": {
                    "base_vlm": model_id,
                    "attn_implementation": attn_implementation,
                }},
                "datasets": {"vla_data": {}},
            })
        else:
            from vla.model.framework.share_tools import _to_omegaconf

            config = _to_omegaconf(getattr(config, "_cfg", config))
            qwenvl = OmegaConf.select(config, "recap.qwenvl")
            if qwenvl is None:
                raise ValueError("value model config requires recap.qwenvl; framework.qwenvl belongs to the policy")
            # Resolve while nodes still belong to the full config, so references
            # to shared top-level fields (e.g. a model root) continue to work.
            datasets = OmegaConf.select(config, "datasets", default=OmegaConf.create({}))
            backbone_config = OmegaConf.create({
                "framework": {"qwenvl": OmegaConf.to_container(qwenvl, resolve=True)},
                "datasets": OmegaConf.to_container(datasets, resolve=True),
            })
            backbone_config = OmegaConf.merge({"datasets": {"vla_data": {}}}, backbone_config)
        self.qwen_vl_interface = get_vlm_model(backbone_config)
        if self.qwen_vl_interface is None:
            raise ValueError("RECAP requires a Qwen VLM backbone")
        backbone = self.qwen_vl_interface.model
        self.value_head = nn.Linear(backbone.config.hidden_size, num_bins)
        self.value_head.to(device=backbone.device)  # Keep the small head in float32.
        self.register_buffer("bin_values", torch.linspace(value_min, value_max, num_bins, device=backbone.device))
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.qwen_vl_interface.requires_grad_(False)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.qwen_vl_interface.eval()
        return self

    def discretize_returns(self, returns: Tensor) -> Tensor:
        """Nearest-bin labels; reject out-of-support returns instead of clipping.

        Targets are remaining episode returns sum(r[t:]), including the terminal
        reward, even if the policy's advantage mode uses the Appendix F episode
        total. Use the same reward normalization as the advantage computation.
        """
        returns = returns.detach().to(device=self.bin_values.device, dtype=torch.float32)
        if not torch.isfinite(returns).all():
            raise ValueError("returns must be finite")
        if ((returns < self.bin_values[0]) | (returns > self.bin_values[-1])).any():
            raise ValueError("returns lie outside the value support; check reward normalization or value bounds")
        scaled = (returns - self.bin_values[0]) / (self.bin_values[-1] - self.bin_values[0])
        return (scaled * (self.bin_values.numel() - 1)).round().long()

    def forward(self, images, instructions, *, returns: Tensor | None = None) -> ValueModelOutput:
        """Train/evaluate on images: list[list[PIL.Image]], instructions: list[str].

        Optional returns has shape [batch] and enables the categorical CE loss.
        Gradients flow through the value head and, unless frozen, the Qwen VLM.
        """
        if len(images) == 0 or len(images) != len(instructions):
            raise ValueError("images and instructions must have the same nonzero batch length")
        targets = None
        if returns is not None:
            if returns.shape != (len(images),):
                raise ValueError("returns must have shape [batch]")
            targets = self.discretize_returns(returns)
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
        outputs = self.qwen_vl_interface.forward_features(**inputs)
        hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].bool()
        if not attention_mask.any(dim=1).all():
            raise ValueError("each observation must have at least one nonpadding token")
        # Unlike sum(mask)-1, this works with Qwen's left padding and right padding.
        positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(attention_mask)
        last = positions.masked_fill(~attention_mask, -1).max(dim=1).values
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        logits = self.value_head(pooled.to(dtype=self.value_head.weight.dtype)).float()
        values = (logits.softmax(dim=-1) * self.bin_values.float()).sum(dim=-1)
        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return ValueModelOutput(logits=logits, values=values, loss=loss)

    @torch.no_grad()
    def predict_values(self, images, instructions) -> Tensor:
        """Deterministic eval-mode values for advantage labels; restore train mode."""
        was_training = self.training
        try:
            self.eval()
            return self(images, instructions).values
        finally:
            self.train(was_training)
