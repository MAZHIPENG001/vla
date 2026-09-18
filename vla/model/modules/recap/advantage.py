"""Advantage conditioning from pi*0.6 / RECAP (docs/pistar06.pdf).

This module consumes predictions from a separately trained value model. All
label computations are detached; rewards and values must use the same scale.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor


def _float(tensor: Tensor) -> Tensor:
    # Accumulate rewards and quantiles in at least float32 under mixed precision.
    return tensor.detach().to(dtype=torch.float64 if tensor.dtype == torch.float64 else torch.float32)


def _finite(tensor: Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")


def _lengths(lengths: Tensor, batch: int, steps: int) -> None:
    if lengths.shape != (batch,) or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must be an integer tensor of shape [batch]")
    if ((lengths < 1) | (lengths > steps)).any():
        raise ValueError("lengths must lie in [1, steps]")


def _mask(mask: Tensor | None, reference: Tensor, name: str) -> Tensor:
    if mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if mask.dtype != torch.bool or mask.shape != reference.shape or mask.device != reference.device:
        raise ValueError(f"{name} must be bool with the same shape/device as its input")
    return mask


@torch.no_grad()
def values_from_logits(logits: Tensor, bin_values: Tensor | None = None) -> Tensor:
    """Return E[V] from distributional value logits [..., bins] (Sec. IV-A).

    By default bins are uniformly spaced in [-1, 0]; the paper uses 201 bins.
    Pass explicit bin_values when the value model uses another support.
    """
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("logits must have a final dimension of at least two bins")
    logits = _float(logits)
    _finite(logits, "logits")
    if bin_values is None:
        bin_values = torch.linspace(-1, 0, logits.shape[-1], device=logits.device, dtype=logits.dtype)
    else:
        if bin_values.shape != (logits.shape[-1],):
            raise ValueError("bin_values must have shape [bins]")
        bin_values = bin_values.to(device=logits.device, dtype=logits.dtype)
        _finite(bin_values, "bin_values")
        if not (bin_values[1:] > bin_values[:-1]).all():
            raise ValueError("bin_values must be strictly increasing")
    return (logits.softmax(dim=-1) * bin_values).sum(dim=-1)


@torch.no_grad()
def episode_rewards(
    lengths: Tensor,
    success: Tensor,
    *,
    max_steps: int,
    failure_penalty: float,
    normalization: float | Tensor = 1.0,
) -> Tensor:
    """Construct Eq. (5) rewards [batch, max_steps] for complete episodes.

    lengths includes the terminal sample: nonterminal rewards are -1, the last
    reward is 0 on success or -failure_penalty on failure. Padding is zero.
    normalization is a positive scalar or [batch] task-specific divisor. It must
    match the value model's normalization. No clipping is performed.
    """
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    _lengths(lengths, lengths.numel(), max_steps)
    _mask(success, lengths, "success")
    if not math.isfinite(failure_penalty) or failure_penalty <= 0:
        raise ValueError("failure_penalty must be finite and positive")
    scale = torch.as_tensor(normalization, dtype=torch.float32, device=lengths.device)
    if scale.ndim != 0 and scale.shape != lengths.shape:
        raise ValueError("normalization must be scalar or have shape [batch]")
    _finite(scale, "normalization")
    if (scale <= 0).any():
        raise ValueError("normalization must be positive")
    times = torch.arange(max_steps, device=lengths.device)[None, :]
    terminal = times == lengths[:, None] - 1
    rewards = -(times < lengths[:, None] - 1).float()
    rewards = rewards - terminal * (~success[:, None]) * failure_penalty
    return rewards / (scale if scale.ndim == 0 else scale[:, None])


@torch.no_grad()
def compute_advantages(
    rewards: Tensor,
    values: Tensor,
    *,
    lengths: Tensor | None = None,
    mode: Literal["posttrain", "pretrain", "return_to_go"] = "posttrain",
    n_steps: int = 50,
) -> Tensor:
    """Compute undiscounted advantages for padded, complete episodes [B, T].

    posttrain: sum(r[t:t+N]) + V[t+N] - V[t], N=50 by default.
    pretrain: sum(r[0:length]) - V[t], literally following Appendix F.
    return_to_go: sum(r[t:length]) - V[t], the conventional MC alternative.

    Each row is one complete episode, including its terminal reward. Windows
    reaching the end use a zero absorbing value. Values at the terminal sample
    itself are still predictions, not forcibly zeroed (failure has a penalty).
    Padded rewards/values are ignored, and padded advantages are zero. Truncated
    rollout fragments requiring a final bootstrap value are not supported.
    """
    if rewards.ndim != 2 or rewards.shape != values.shape or min(rewards.shape) < 1:
        raise ValueError("rewards and values must have the same nonempty shape [batch, steps]")
    if rewards.device != values.device:
        raise ValueError("rewards and values must be on the same device")
    if mode not in ("posttrain", "pretrain", "return_to_go"):
        raise ValueError("mode must be posttrain, pretrain, or return_to_go")
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    batch, steps = rewards.shape
    if lengths is None:
        lengths = torch.full((batch,), steps, dtype=torch.long, device=rewards.device)
    _lengths(lengths, batch, steps)
    lengths = lengths.to(rewards.device)
    times = torch.arange(steps, device=rewards.device)[None, :]
    valid = times < lengths[:, None]
    rewards = _float(rewards).masked_fill(~valid, 0)
    values = _float(values).masked_fill(~valid, 0)
    _finite(rewards, "valid rewards")
    _finite(values, "valid values")
    prefix = torch.cat((rewards.new_zeros((batch, 1)), rewards.cumsum(dim=1)), dim=1)
    if mode == "pretrain":
        targets = prefix[:, -1:]
    elif mode == "return_to_go":
        targets = prefix[:, -1:] - prefix[:, :-1]
    else:
        ends = torch.minimum(times + min(n_steps, steps), lengths[:, None])
        absorbing_values = torch.cat((values, values.new_zeros((batch, 1))), dim=1)
        bootstrap = absorbing_values.gather(1, ends).masked_fill(ends >= lengths[:, None], 0)
        targets = prefix.gather(1, ends) - prefix[:, :-1] + bootstrap
    return (targets - values).masked_fill(~valid, 0)


def _task_ids(task_ids: Tensor, advantages: Tensor) -> None:
    if task_ids.shape != advantages.shape or task_ids.device != advantages.device:
        raise ValueError("task_ids must have the same shape/device as advantages")
    if task_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("task_ids must be an integer tensor")


@torch.no_grad()
def estimate_task_thresholds(
    advantages: Tensor,
    task_ids: Tensor,
    *,
    stage: Literal["pretrain", "posttrain"] = "posttrain",
    positive_fraction: float | None = None,
    valid_mask: Tensor | None = None,
) -> dict[int, float]:
    """Fit epsilon per task on a calibration dataset (Appendix F).

    Defaults target 30% positives for pretraining or 40% for post-training;
    positive_fraction=0.1 implements the stricter folding-task setting. The
    threshold is quantile(1 - positive_fraction) of valid advantages. Strict
    comparison and ties mean the resulting fraction is only approximate.
    Fit once on representative data (paper: 10k pretraining samples), not on
    every policy minibatch. Use evaluation rollouts for post-training fitting.
    """
    if stage not in ("pretrain", "posttrain"):
        raise ValueError("stage must be pretrain or posttrain")
    if positive_fraction is None:
        positive_fraction = 0.3 if stage == "pretrain" else 0.4
    if not 0 < positive_fraction < 1:
        raise ValueError("positive_fraction must lie strictly between 0 and 1")
    _task_ids(task_ids, advantages)
    valid = _mask(valid_mask, advantages, "valid_mask")
    samples = _float(advantages)[valid]
    tasks = task_ids[valid]
    if samples.numel() == 0:
        raise ValueError("at least one valid calibration sample is required")
    _finite(samples, "valid advantages")
    return {
        int(task.item()): float(torch.quantile(samples[tasks == task], 1 - positive_fraction).item())
        for task in tasks.unique()
    }


@dataclass(frozen=True)
class AdvantageCondition:
    """Separate positive/negative labels from whether conditioning is present."""

    indicator: Tensor
    conditioning_mask: Tensor

    def to_text(self) -> list[str]:
        """Flatten in row-major order; omitted/padded conditions become ''."""
        indicators = self.indicator.flatten().tolist()
        retained = self.conditioning_mask.flatten().tolist()
        return [
            ("Advantage: positive" if positive else "Advantage: negative") if keep else ""
            for positive, keep in zip(indicators, retained)
        ]


@torch.no_grad()
def compute_advantage_condition(
    advantages: Tensor,
    task_ids: Tensor,
    thresholds: Mapping[int, float],
    *,
    intervention_mask: Tensor | None = None,
    valid_mask: Tensor | None = None,
    training: bool = True,
    dropout_probability: float = 0.3,
    generator: torch.Generator | None = None,
) -> AdvantageCondition:
    """Compute I = (A > epsilon_task), force corrections positive, then dropout.

    Dropout removes the condition rather than changing a positive into a negative
    label. It applies to interventions too. training=False disables dropout; for
    policy deployment, directly supply 'Advantage: positive' without a critic.
    A supplied RNG generator must be compatible with the input tensor's device.
    """
    if not 0 <= dropout_probability <= 1:
        raise ValueError("dropout_probability must lie in [0, 1]")
    _task_ids(task_ids, advantages)
    valid = _mask(valid_mask, advantages, "valid_mask")
    advantages = _float(advantages)
    _finite(advantages[valid], "valid advantages")
    indicator = torch.zeros_like(valid)
    for task in task_ids[valid].unique().tolist():
        if task not in thresholds:
            raise ValueError(f"missing calibrated threshold for task {task}")
        threshold = thresholds[task]
        if not math.isfinite(threshold):
            raise ValueError(f"threshold for task {task} must be finite")
        selected = valid & (task_ids == task)
        indicator[selected] = advantages[selected] > threshold
    if intervention_mask is not None:
        indicator |= _mask(intervention_mask, advantages, "intervention_mask") & valid
    retained = valid.clone()
    if training and dropout_probability > 0:
        retained &= torch.rand(advantages.shape, device=advantages.device, generator=generator) >= dropout_probability
    return AdvantageCondition(indicator=indicator, conditioning_mask=retained)
