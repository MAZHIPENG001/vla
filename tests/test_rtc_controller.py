"""Deterministic controller-time tests, without LIBERO, sockets or GPU."""

import threading
import unittest
from unittest.mock import patch

import numpy as np

from examples.LIBERO.eval_files import eval_libero as evaluation
from examples.LIBERO.eval_files.rtc_controller import RTCConfig, RTCController
from test_libero_realtime import Clock, Environment


def chunk(base, horizon=8):
    rows = np.zeros((horizon, 7), dtype=np.float32)
    rows[:, :6] = base + np.arange(horizon)[:, None]
    rows[:, 6] = 0 if base == 0 else 1
    return rows


class ChunkModel:
    action_chunk_size = 8
    supports_rtc_inpainting = True

    def __init__(self):
        self.calls = []
        self.threads = []

    def set_inference_timeout(self, seconds):
        self.timeout = seconds

    def predict_chunk(self, example, rtc=None):
        self.threads.append(threading.get_ident())
        self.calls.append((example, rtc))
        return chunk((len(self.calls) - 1) * 100)


class VirtualFuture:
    def __init__(self, executor, delay, fn, kwargs):
        self.executor, self.fn, self.kwargs = executor, fn, kwargs
        self.ready = executor.clock.now + delay
        self.resolved = False

    def done(self):
        return self.executor.clock.now >= self.ready

    def result(self, timeout=None):
        remaining = max(0, self.ready - self.executor.clock.now)
        if timeout is not None and remaining > timeout:
            raise TimeoutError("virtual inference timeout")
        self.executor.clock.now += remaining
        if not self.resolved:
            self.value = self.fn(**self.kwargs)
            self.resolved = True
        return self.value


class VirtualExecutor:
    def __init__(self, clock, delays):
        self.clock, self.delays = clock, delays
        self.futures, self.submissions = [], []

    def submit(self, fn, **kwargs):
        delay = self.delays[min(len(self.futures), len(self.delays) - 1)]
        future = VirtualFuture(self, delay, fn, kwargs)
        self.futures.append(future)
        self.submissions.append((self.clock.now, kwargs))
        return future


class ControllerTests(unittest.TestCase):
    def make_controller(self, delays=(0, 2, 100), **kwargs):
        clock, model = Clock(), ChunkModel()
        executor = VirtualExecutor(clock, delays)
        controller = RTCController(
            model, RTCConfig(stride=2, timeout=1000, **kwargs), executor=executor, clock=clock.monotonic,
        )
        return controller, clock, model, executor

    def ticks(self, controller, clock, start, end):
        result = []
        for step in range(start, end):
            clock.now = step
            result.append(controller.next_action({"image": [], "lang": f"frame-{step}"}, step))
        return result

    def test_request_before_exhaustion_and_old_motion_continues(self):
        ctl, clock, model, executor = self.make_controller()
        actions = self.ticks(ctl, clock, 0, 6)
        self.assertEqual([a[0] for a in actions], [0, 1, 2, 3, 102, 103])
        self.assertEqual([s[0] for s in executor.submissions[:2]], [0, 2])
        self.assertLess(executor.submissions[1][0], model.action_chunk_size)
        self.assertEqual(ctl.stats.delay_steps[:2], [0, 2])
        self.assertEqual(ctl.stats.wait_steps, 0)
        ctl.close()

    def test_alignment_and_immediate_swap_for_different_actual_delays(self):
        for delay in (0, 1, 2, 5, 6, 7):
            with self.subTest(delay=delay):
                ctl, clock, _, _ = self.make_controller(delays=(0, delay, 100), beta=0.5)
                handoff = 2 + delay
                actions = self.ticks(ctl, clock, 0, handoff + 1)
                self.assertEqual(ctl.origin, 2)
                # Algorithm 1 swaps Acur=Anew with t reset to the actual delay.
                # No post-sampling weighted average is part of the algorithm.
                self.assertEqual(actions[-1][0], 100 + delay)
                self.assertEqual(ctl.stats.delay_steps[-1], delay)
                ctl.close()

    def test_beta_is_guidance_cap_and_never_interpolates_controller_output(self):
        for beta in (0.5, 5., 20.):
            with self.subTest(beta=beta):
                ctl, clock, model, _ = self.make_controller(beta=beta)
                actions = self.ticks(ctl, clock, 0, 5)
                np.testing.assert_array_equal(actions[-1], chunk(100)[2])
                self.assertEqual(model.calls[1][1]["beta"], beta)
                ctl.close()

    def test_never_replay_expired_prefix_or_entire_expired_chunk(self):
        ctl, clock, _, _ = self.make_controller(delays=(0, 9, 100))
        actions = self.ticks(ctl, clock, 0, 12)
        self.assertEqual([a[0] for a in actions[:8]], list(range(8)))
        self.assertTrue(all(a is None for a in actions[8:]))
        self.assertEqual(ctl.stats.stale_chunks, 1)
        self.assertEqual(ctl.stats.buffer_exhaustions, 1)
        self.assertEqual(ctl.stats.wait_steps, 4)
        ctl.close()

    def test_bootstrap_is_stationary_then_initial_chunk_starts_at_zero(self):
        ctl, clock, _, _ = self.make_controller(delays=(3, 100))
        actions = self.ticks(ctl, clock, 0, 5)
        self.assertEqual(actions[:3], [None] * 3)
        self.assertEqual([a[0] for a in actions[3:]], [0, 1])
        self.assertEqual(ctl.origin, 3)
        self.assertEqual(ctl.stats.buffer_exhaustions, 0)
        ctl.close()

    def test_prefix_is_remaining_committed_actions_at_request_not_completion(self):
        ctl, clock, model, executor = self.make_controller(delay=1, beta=7)
        self.ticks(ctl, clock, 0, 5)
        request = executor.submissions[1][1]
        self.assertEqual(request["example"]["lang"], "frame-2")
        rtc = model.calls[1][1]
        np.testing.assert_array_equal(rtc["prefix_actions"], chunk(0)[None, 2:])
        self.assertEqual(rtc["delay"], 1)
        self.assertEqual(rtc["beta"], 7)
        ctl.close()

    def test_prefix_preserves_raw_gripper_values_in_policy_coordinates(self):
        ctl, clock, model, executor = self.make_controller()
        original_predict = model.predict_chunk

        def predict_fractional_gripper(**kwargs):
            result = original_predict(**kwargs)
            result[:, 6] = .35 if len(model.calls) == 1 else .65
            return result

        model.predict_chunk = predict_fractional_gripper
        self.ticks(ctl, clock, 0, 5)
        np.testing.assert_array_equal(executor.submissions[1][1]["rtc"]["prefix_actions"][..., 6], np.float32(.35))
        ctl.close()

    def test_reset_drains_then_clears_actions_and_pending_state(self):
        ctl, clock, model, executor = self.make_controller(delays=(0, 2, 0))
        self.ticks(ctl, clock, 0, 3)
        self.assertIsNotNone(ctl.pending)
        ctl.reset()
        self.assertTrue(executor.futures[1].resolved)
        self.assertIsNone(ctl.chunk)
        self.assertIsNone(ctl.pending)
        self.assertEqual(ctl.last_step, -1)
        self.assertEqual(ctl.stats.requests, 0)
        # No result from the previous episode is installed after reset.
        action = ctl.next_action({"lang": "new episode"}, 0)
        self.assertEqual(action[0], 200)
        self.assertIsNone(model.calls[-1][1])
        ctl.close()

    def test_timeout_and_failed_reset_make_failure_visible(self):
        ctl, clock, _, _ = self.make_controller(delays=(100,))
        ctl.config = RTCConfig(stride=2, timeout=2)
        self.ticks(ctl, clock, 0, 2)
        clock.now = 2
        with self.assertRaisesRegex(TimeoutError, "RTC inference exceeded"):
            ctl.next_action({}, 2)
        with self.assertRaises(TimeoutError):
            ctl.reset()
        self.assertTrue(ctl.closed)
        self.assertIsNone(ctl.pending)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            ctl.next_action({}, 0)

    def test_completed_after_total_deadline_is_rejected_even_if_polled_when_done(self):
        ctl, clock, _, _ = self.make_controller(delays=(2,))
        ctl.config = RTCConfig(stride=2, timeout=1)
        ctl.next_action({}, 0)
        clock.now = 2
        with self.assertRaisesRegex(TimeoutError, "RTC inference exceeded"):
            ctl.next_action({}, 1)

    def test_timely_completion_polled_late_uses_control_age_not_wall_age(self):
        ctl, clock, _, executor = self.make_controller(delays=(0.5, 100))
        ctl.config = RTCConfig(stride=2, timeout=1)
        ctl.next_action({}, 0)
        clock.now = 0.5
        executor.futures[0].result()  # Worker completed before the deadline.
        clock.now = 2
        self.assertEqual(ctl.next_action({}, 1)[0], 0)
        ctl.close()

    def test_inference_errors_surface_during_tick_and_terminal_drain(self):
        for drain in (False, True):
            with self.subTest(drain=drain):
                ctl, clock, model, _ = self.make_controller(delays=(1,))
                model.predict_chunk = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("broken inference"))
                ctl.next_action({}, 0)
                with self.assertRaisesRegex(RuntimeError, "broken inference"):
                    if drain:
                        ctl.close()
                    else:
                        clock.now = 1
                        ctl.next_action({}, 1)

    def test_capability_and_config_are_checked_before_request(self):
        model = ChunkModel()
        model.supports_rtc_inpainting = False
        with self.assertRaisesRegex(ValueError, "does not support"):
            RTCController(model, RTCConfig())
        self.assertEqual(model.calls, [])
        self.assertEqual(RTCConfig().beta, 5.)
        RTCConfig(beta=20.).validate(8)
        for config in (RTCConfig(stride=8), RTCConfig(stride=0), RTCConfig(delay=8),
                       RTCConfig(beta=0), RTCConfig(beta=-1), RTCConfig(beta=float("nan"))):
            with self.subTest(config=config), self.assertRaises(ValueError):
                RTCController(ChunkModel(), config)


class EpisodeTests(unittest.TestCase):
    def run_virtual(self, max_steps=14, success_at=None, delays=(0, 1, 100)):
        clock, model, env = Clock(), ChunkModel(), Environment(success_at=success_at)
        executor = VirtualExecutor(clock, delays)
        controllers = []

        def make_controller(client, config):
            ctl = RTCController(client, config, executor=executor, clock=clock.monotonic)
            controllers.append(ctl)
            return ctl

        args = evaluation.Args(num_steps_wait=0, control_freq=1, rtc_enabled=True, rtc_stride=2,
                               inference_timeout=1000)
        with (
            patch.object(evaluation.time, "monotonic", clock.monotonic),
            patch.object(evaluation.time, "sleep", clock.sleep),
            patch("examples.LIBERO.eval_files.rtc_controller.RTCController", side_effect=make_controller),
        ):
            frames, done = evaluation._run_episode(env, model, env.obs(), "task", args, max_steps)
        return env, model, controllers[0], executor, frames, done

    def test_exhaustion_holds_pose_and_preserves_last_gripper(self):
        env, _, ctl, _, frames, done = self.run_virtual()
        self.assertFalse(done)
        self.assertEqual(len(frames), 14)
        for action in env.actions[10:]:
            self.assertEqual(action[:6], [0] * 6)
            self.assertEqual(action[6], -1)  # Latest returned chunk requests open.
        self.assertEqual(ctl.stats.buffer_exhaustions, 1)
        self.assertEqual(ctl.stats.wait_steps, 4)

    def test_early_success_during_inference_does_not_step_or_install_after_done(self):
        env, _, ctl, executor, frames, done = self.run_virtual(success_at=3, delays=(0, 5))
        self.assertTrue(done)
        self.assertEqual(len(env.actions), 3)
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(f.resolved for f in executor.futures))
        self.assertTrue(ctl.closed)
        self.assertIsNone(ctl.chunk)
        self.assertIsNone(ctl.pending)

    def test_real_worker_continues_control_and_env_stays_on_main_thread(self):
        release = threading.Event()

        class GatedModel(ChunkModel):
            def predict_chunk(self, **kwargs):
                if not release.wait(timeout=2):
                    raise TimeoutError("control loop failed to release inference worker")
                return super().predict_chunk(**kwargs)

        class ReleasingEnv(Environment):
            def step(self, action):
                result = super().step(action)
                if len(self.actions) == 4:
                    release.set()
                return result

        model, env = GatedModel(), ReleasingEnv()
        args = evaluation.Args(num_steps_wait=0, control_freq=100, rtc_enabled=True, rtc_stride=2)
        frames, _ = evaluation._run_episode(env, model, env.obs(), "task", args, 18)
        self.assertEqual(len(frames), 18)
        self.assertEqual(set(env.threads), {threading.get_ident()})
        self.assertTrue(model.threads)
        self.assertNotIn(threading.get_ident(), model.threads)
        self.assertEqual(env.actions[:4], [evaluation.LIBERO_DUMMY_ACTION] * 4)
        self.assertTrue(any(any(a[:6]) for a in env.actions[4:]))


if __name__ == "__main__":
    unittest.main()
