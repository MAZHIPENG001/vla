"""RTC scheduling in controller time; no simulator or renderer calls live here.

Every chunk row has an absolute control timestamp: ``origin + row``. Only the
bootstrap chunk is rebased after the initial stationary wait. Inference workers
return values and never mutate the active chunk or episode state.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import math
import time

import numpy as np


@dataclass(frozen=True)
class RTCConfig:
    stride: int = 4
    delay: int = 2
    beta: float = 0.5
    inpainting: bool = True
    guidance_beta: float = 5.0
    timeout: float = 10.0

    def validate(self, horizon):
        if not isinstance(horizon, int) or horizon < 2:
            raise ValueError("RTC requires action_chunk_size >= 2 from server metadata")
        if not isinstance(self.stride, int) or not 0 < self.stride < horizon:
            raise ValueError("RTC requires 0 < rtc_stride < action_chunk_size")
        if not isinstance(self.delay, int) or not 0 <= self.delay < horizon:
            raise ValueError("RTC requires 0 <= rtc_delay < action_chunk_size")
        if not math.isfinite(self.beta) or not 0 <= self.beta <= 1:
            raise ValueError("rtc_beta must be finite and in [0, 1]")
        if not math.isfinite(self.guidance_beta) or self.guidance_beta <= 0:
            raise ValueError("rtc_guidance_beta must be finite and positive")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("rtc_timeout must be finite and positive")


@dataclass
class RTCStats:
    requests: int = 0
    inference_ms: list = field(default_factory=list)
    delay_steps: list = field(default_factory=list)
    wait_steps: int = 0
    blend_actions: int = 0
    buffer_exhaustions: int = 0
    stale_chunks: int = 0


@dataclass
class _Request:
    future: object
    step: int
    submitted: float
    bootstrap: bool
    generation: int


def blend_overlap(old, new, beta):
    """Blend aligned LIBERO OSC delta commands, preserving discrete grippers.

    For L overlapping, *unexecuted* rows, old weight w(j)=beta*(L-j)/L.
    Translation and scaled, small axis-angle rotation deltas are vectors in the
    same controller tangent coordinates, so both use (1-w)*new+w*old. These are
    per-step commands, not absolute orientations. Gripper selects a source and
    retains the existing threshold/encoding in eval_libero._decode_action.
    """
    length = len(old)
    if length == 0:
        return new.copy()
    weights = beta * (length - np.arange(length, dtype=np.float64)) / length
    mixed = new.copy()
    mixed[:, :6] = weights[:, None] * old[:, :6] + (1 - weights[:, None]) * new[:, :6]
    mixed[:, 6] = np.where(weights >= 0.5, old[:, 6], new[:, 6])
    return mixed


class RTCController:
    def __init__(self, client, config, *, executor=None, clock=None):
        self.horizon = client.action_chunk_size
        config.validate(self.horizon)
        if config.inpainting and not client.supports_rtc_inpainting:
            raise ValueError(
                "Server/model does not support RTC prefix inpainting; explicitly set "
                "rtc_inpainting=False for asynchronous chunk blending only"
            )
        self.client, self.config = client, config
        self.clock = clock or time.monotonic
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="libero-rtc")
        self._owns_executor = executor is None
        self.generation = 0
        self.closed = False
        self._clear()

    def _clear(self):
        self.chunk = None
        self.blended = None
        self.origin = None
        self.pending = None
        self.last_step = -1
        self.had_chunk = False
        self.exhausted = False
        self.delays = deque([self.config.delay], maxlen=10)
        self.stats = RTCStats()

    def _infer(self, example, rtc):
        started = self.clock()
        chunk = self.client.predict_chunk(example=example, rtc=rtc)
        completed = self.clock()
        return chunk, (completed - started) * 1000, completed

    def _result(self, request, **kwargs):
        chunk, inference_ms, completed = request.future.result(**kwargs)
        # Include worker scheduling, image preparation and send time in the
        # deadline, even when a slow env.step delayed the next polling tick.
        # A timely result polled late remains valid (subject to control age).
        if completed - request.submitted > self.config.timeout:
            raise TimeoutError(f"RTC inference exceeded {self.config.timeout}s (request_step={request.step})")
        return chunk, inference_ms

    def _submit(self, example, step):
        prefix = None
        if self.chunk is not None:
            offset = step - self.origin
            if 0 <= offset < self.horizon:
                prefix = self.chunk[offset:].copy()
        rtc = None
        if self.config.inpainting and prefix is not None and len(prefix):
            estimated_delay = max(self.delays)
            if estimated_delay > len(prefix):
                logging.warning(
                    "RTC delay estimate=%d exceeds remaining prefix=%d; "
                    "continuity cannot be guaranteed after buffer exhaustion",
                    estimated_delay, len(prefix),
                )
            # The env executes binary grippers. Condition on the same command,
            # expressed in the model's existing open=1/closed=0 convention.
            prefix[:, 6] = (prefix[:, 6] > 0.5).astype(prefix.dtype)
            rtc = {
                "prefix_actions": prefix[None],
                "delay": min(estimated_delay, len(prefix)),
                "beta": self.config.guidance_beta,
            }
        submitted = self.clock()
        future = self.executor.submit(self._infer, example=example, rtc=rtc)
        self.pending = _Request(future, step, submitted, not self.had_chunk, self.generation)
        self.stats.requests += 1
        logging.debug("RTC request=%d generation=%d request_step=%d", self.stats.requests, self.generation, step)

    def _poll(self, step):
        request = self.pending
        if request is None:
            return
        if not request.future.done():
            if self.clock() - request.submitted >= self.config.timeout:
                raise TimeoutError(f"RTC inference exceeded {self.config.timeout}s (request_step={request.step})")
            return
        chunk, inference_ms = self._result(request)  # Do not hide worker failures.
        self.pending = None
        delay = step - request.step
        self.stats.inference_ms.append(inference_ms)
        self.stats.delay_steps.append(delay)
        self.delays.append(delay)
        logging.info(
            "RTC inference: request_step=%d return_step=%d delay_steps=%d "
            "inference_ms=%.3f roundtrip_ms=%.3f",
            request.step, step, delay, inference_ms, (self.clock() - request.submitted) * 1000,
        )
        if request.generation != self.generation:
            self.stats.stale_chunks += 1
            return
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape != (self.horizon, 7) or not np.isfinite(chunk).all():
            raise ValueError(f"Invalid RTC chunk: expected ({self.horizon}, 7), got {chunk.shape}, finite={np.isfinite(chunk).all()}")
        # The initial observation remains stationary during safe startup holds.
        # All replans retain request time, even if the old buffer runs out.
        origin = step if request.bootstrap else request.step
        index = step - origin
        if index >= self.horizon:
            self.stats.stale_chunks += 1
            logging.warning("RTC discarded expired chunk: origin=%d return_step=%d H=%d", origin, step, self.horizon)
            return
        chunk = chunk.copy()
        blended = np.zeros(self.horizon, dtype=bool)
        if self.chunk is not None:
            overlap_end = min(self.origin + self.horizon, origin + self.horizon)
            length = max(0, overlap_end - step)
            if length:
                old_index = step - self.origin
                chunk[index:index + length] = blend_overlap(
                    self.chunk[old_index:old_index + length], chunk[index:index + length], self.config.beta,
                )
                if self.config.beta > 0:
                    blended[index:index + length] = True
        self.chunk, self.origin = chunk, origin
        self.blended = blended
        self.had_chunk = True
        self.exhausted = False

    def next_action(self, example, control_step):
        """Nonblocking tick; consume each absolute timestamp exactly once."""
        if self.closed:
            raise RuntimeError("RTC controller is closed")
        if control_step != self.last_step + 1:
            raise ValueError("RTC control_step must advance by exactly one on every env control cycle")
        self.last_step = control_step
        self._poll(control_step)
        if self.pending is None and (self.chunk is None or control_step - self.origin >= self.config.stride):
            self._submit(example, control_step)
            self._poll(control_step)  # Immediate results need no artificial wait frame.
        if self.chunk is not None:
            index = control_step - self.origin
            if 0 <= index < self.horizon:
                self.stats.blend_actions += int(self.blended[index])
                return self.chunk[index].copy()
        self.stats.wait_steps += 1
        if self.had_chunk and not self.exhausted:
            self.stats.buffer_exhaustions += 1
            self.exhausted = True
            logging.warning("RTC action buffer exhausted at control_step=%d; holding pose and last gripper", control_step)
        return None

    def _drain(self):
        request, self.pending = self.pending, None
        if request is not None:
            # Called only outside the control loop, e.g. after terminal success.
            # The socket has its own receive timeout; this is a second bound.
            remaining = max(0, self.config.timeout - (self.clock() - request.submitted))
            _, inference_ms = self._result(request, timeout=remaining + 0.1)
            self.stats.inference_ms.append(inference_ms)
            logging.info(
                "RTC inference drained after episode: request_step=%d inference_ms=%.3f "
                "control_cycles_until_termination=%d",
                request.step, inference_ms, max(0, self.last_step + 1 - request.step),
            )

    def reset(self):
        """Drain before reuse; late old results cannot populate a new episode."""
        try:
            self._drain()
        except BaseException:
            # A timed-out worker cannot be forcibly stopped by Future.cancel().
            # Do not permit reuse of its connection after a failed reset.
            self.closed = True
            if self._owns_executor:
                self.executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            self.generation += 1
            self._clear()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._drain()
        finally:
            self.chunk = None
            self.blended = None
            self.origin = None
            self.generation += 1
            if self._owns_executor:
                self.executor.shutdown(wait=False, cancel_futures=True)
