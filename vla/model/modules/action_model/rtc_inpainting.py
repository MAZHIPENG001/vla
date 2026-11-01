"""Training-free RTC inpainting (RTC paper, Equations 2--5).

All actions here are in the model's *normalized* training action space.  A
prefix is the unconsumed part of the previous chunk aligned to the observation
that started this inference, not to its eventual completion time.  The paper's
"frozen" prefix is a full-weight guidance constraint, not an exact projection:
finite-step PiGDM can leave a residual error (see the paper's Section 3.2).
"""

import math
from numbers import Integral
from typing import Callable

import torch


def rtc_prefix_weights(
    horizon: int,
    overlap: int,
    delay: int,
    *,
    device=None,
) -> torch.Tensor:
    """Return Eq. 5's soft mask, shaped [1, H, 1], in float32.

    ``overlap = H - s`` is the number of old actions available at request
    submission.  The first ``delay`` available actions have weight one; the
    remaining overlap decays exponentially, and the new tail has weight zero.
    An underestimated delay affects guidance, never the controller's indexing.

    Equation 5 assumes ``delay <= overlap``. If inference exceeds that supply
    budget, the controller warns and holds after exhausting its old actions.
    Here only available old rows can be constrained; this bounded mask is an
    exhaustion fallback, not an extension of the paper's continuity guarantee.
    """
    if not isinstance(horizon, Integral) or horizon <= 0:
        raise ValueError("RTC horizon must be a positive integer")
    if not isinstance(overlap, Integral) or not 0 <= overlap <= horizon:
        raise ValueError("RTC overlap must be an integer in [0, horizon]")
    if not isinstance(delay, Integral) or delay < 0:
        raise ValueError("RTC delay must be a non-negative integer")
    weights = torch.zeros(horizon, device=device, dtype=torch.float32)
    frozen = min(delay, overlap)
    weights[:frozen] = 1
    if frozen < overlap:
        indices = torch.arange(frozen, overlap, device=device, dtype=torch.float32)
        c = (overlap - indices) / (overlap - frozen + 1)
        weights[frozen:overlap] = c * torch.expm1(c) / math.expm1(1.0)
    return weights.reshape(1, horizon, 1)


def rtc_guided_euler_step(
    actions: torch.Tensor,
    velocity_fn: Callable[[torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    time: float,
    dt: float,
    beta: float,
) -> torch.Tensor:
    """Integrate one PiGDM-guided flow step using the *full* endpoint VJP.

    ``f(x) = x + (1-time) v(x)`` estimates the final clean actions.  We compute
    ``g = J_f(x).T @ ((target-f(x)) * weights)`` and update by
    ``dt * (v(x) + min(beta, (1-time)/(time*r_time**2))*g)``.
    At time zero the unclipped coefficient has limit infinity, hence beta.
    The residual used as ``grad_outputs`` is detached: differentiating it again
    would compute a different algorithm.  Gradients are only requested for the
    current noisy trajectory; model parameter .grad fields remain untouched.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("RTC guidance beta must be finite and non-negative")
    if not math.isfinite(time) or not 0 <= time < 1:
        raise ValueError("RTC flow time must be in [0, 1)")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("RTC Euler step size must be positive and finite")
    if actions.ndim != 3 or target.shape != actions.shape:
        raise ValueError("RTC target must match the [B, H, D] action trajectory")
    if weights.shape != (1, actions.shape[1], 1):
        raise ValueError("RTC weights must have shape [1, H, 1]")

    # The framework and websocket callers may both be in inference_mode.  Merely
    # using enable_grad() does not override it, and inference tensors cannot be
    # saved for backward.  Clone inside the disabled inference-mode context.
    with torch.inference_mode(False), torch.enable_grad():
        current = actions.detach().clone().requires_grad_(True)
        velocity = velocity_fn(current)
        if velocity.shape != current.shape:
            raise ValueError("RTC velocity must match the action trajectory")
        # Endpoint error and integration use float32 even with fp16/bf16 DiT.
        endpoint = current.float() + (1.0 - time) * velocity.float()
        error = (target.detach().float() - endpoint).detach() * weights.detach().float()
        correction = torch.autograd.grad(endpoint, current, grad_outputs=error)[0]
        scale = beta if time == 0 else min(
            beta, (time * time + (1.0 - time) ** 2) / (time * (1.0 - time))
        )
        updated = current.float() + dt * (velocity.float() + scale * correction.float())
        if not torch.isfinite(updated).all():
            raise FloatingPointError("RTC inpainting produced non-finite actions")
        # Match the ordinary Euler update's dtype promotion.  In QwenPI the
        # VLM noise can start as bf16 while the autocast action velocity is
        # fp16, whose sum is fp32; retaining bf16 would also break numpy output.
        integration_dtype = torch.promote_types(actions.dtype, velocity.dtype)
        return updated.to(integration_dtype).detach()
