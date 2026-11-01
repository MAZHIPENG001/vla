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
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class RTCConfig:
    # Algorithm 1: s_min (auto selects one cycle) and initial d estimate.
    stride: int | Literal["auto"] = "auto"
    delay: int | Literal["auto"] = "auto"
    beta: float = 5.0  # Eq. 2 guidance clipping, Table 4.
    # Deployment bound; not a hyperparameter of the paper's algorithm.
    timeout: float = 10.0

    def validate(self, horizon):
        if not isinstance(horizon, int) or horizon < 2:
            raise ValueError("RTC requires action_chunk_size >= 2 from server metadata")
        if self.stride != "auto" and (type(self.stride) is not int or not 0 < self.stride < horizon):
            raise ValueError("RTC requires rtc_stride='auto' or 0 < rtc_stride < action_chunk_size")
        if self.delay != "auto" and (type(self.delay) is not int or not 0 <= self.delay < horizon):
            raise ValueError("RTC requires rtc_delay='auto' or 0 <= rtc_delay < action_chunk_size")
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("rtc_beta must be finite and positive (paper Eq. 2 guidance clipping)")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("inference_timeout must be finite and positive")


@dataclass
class RTCStats:
    requests: int = 0
    inference_ms: list = field(default_factory=list)
    delay_steps: list = field(default_factory=list)
    estimated_delay_steps: list = field(default_factory=list)
    execution_horizons: list = field(default_factory=list)
    wait_steps: int = 0
    buffer_exhaustions: int = 0
    stale_chunks: int = 0


@dataclass
class _Request:
    future: object
    step: int
    submitted: float
    bootstrap: bool
    generation: int


class RTCController:
    def __init__(self, client, config, *, executor=None, clock=None):
        self.horizon = client.action_chunk_size
        config.validate(self.horizon)
        if not client.supports_rtc_inpainting:
            raise ValueError(
                "Server/model does not support the flow prefix inpainting required by RTC; "
                "set RTC_ENABLED=false for this model"
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
        self.origin = None
        self.pending = None
        self.last_step = -1
        self.had_chunk = False
        self.exhausted = False
        # Auto has no fabricated startup latency. The first unconditioned
        # request measures it while the controller holds safely. Numeric delay
        # retains the original initial-seed behavior and then ages out.
        self.delays = deque([] if self.config.delay == "auto" else [self.config.delay], maxlen=10)
        self.stats = RTCStats()

    @property
    def estimated_delay(self):
        """Forecast from observed *control cycles*, including client overhead."""
        return max(self.delays, default=0)

    @property
    def minimum_execution_horizon(self):
        """Algorithm 1 line 13 uses s_min, not the forecast inference delay.

        Auto selects the smallest positive controller interval. The actual s
        is measured at submission (line 14): after initialization, a single
        in-flight request naturally gives s=max(s_min, previous actual delay).
        """
        return 1 if self.config.stride == "auto" else self.config.stride

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
        estimated_delay = self.estimated_delay  # Algorithm 1 line 17.
        execution_horizon = 0 if self.origin is None else step - self.origin  # Line 14, s=t.
        prefix = None
        if self.chunk is not None:
            offset = step - self.origin
            if 0 <= offset < self.horizon:
                prefix = self.chunk[offset:].copy()
        rtc = None
        if prefix is not None and len(prefix):
            if estimated_delay > len(prefix):
                logging.warning(
                    "RTC delay estimate=%d exceeds remaining prefix=%d; "
                    "continuity cannot be guaranteed after buffer exhaustion",
                    estimated_delay, len(prefix),
                )
            # Algorithm 1 line 15 takes A_cur[s:] verbatim, with no extra
            # controller-side rotation/gripper processing or output blending.
            rtc = {
                "prefix_actions": prefix[None],
                "delay": estimated_delay,
                "beta": self.config.beta,
            }
        elif self.had_chunk:
            logging.warning(
                "RTC has no overlapping prefix at request_step=%d (s=%d H=%d); "
                "outside the paper's continuous-supply region, holding until fresh actions arrive",
                step, execution_horizon, self.horizon,
            )
        submitted = self.clock()
        future = self.executor.submit(self._infer, example=example, rtc=rtc)
        self.pending = _Request(future, step, submitted, not self.had_chunk, self.generation)
        self.stats.requests += 1
        self.stats.estimated_delay_steps.append(estimated_delay)
        self.stats.execution_horizons.append(execution_horizon)
        logging.info(
            "RTC request=%d generation=%d request_step=%d H=%d delay_mode=%s stride_mode=%s "
            "estimated_delay_steps=%d minimum_execution_horizon=%d execution_horizon=%d",
            self.stats.requests, self.generation, step, self.horizon,
            self.config.delay, self.config.stride, estimated_delay,
            self.minimum_execution_horizon, execution_horizon,
        )

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
        # Algorithm 1 lines 20--21: A_cur=A_new, t=t-s. The guidance in the
        # sampler supplies continuity. Do not average or overwrite any rows.
        self.chunk, self.origin = chunk.copy(), origin
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
        if self.pending is None and (self.chunk is None or control_step - self.origin >= self.minimum_execution_horizon):
            self._submit(example, control_step)
            self._poll(control_step)  # Immediate results need no artificial wait frame.
        if self.chunk is not None:
            index = control_step - self.origin
            if 0 <= index < self.horizon:
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
            self.origin = None
            self.generation += 1
            if self._owns_executor:
                self.executor.shutdown(wait=False, cancel_futures=True)
