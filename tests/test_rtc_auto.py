"""Algorithm 1 timing and delay-buffer tests, without a simulator or GPU."""

import unittest

import numpy as np

from examples.LIBERO.eval_files.eval_libero import Args
from examples.LIBERO.eval_files.rtc_controller import RTCConfig, RTCController
from test_libero_realtime import Clock
from test_rtc_controller import ChunkModel, VirtualExecutor


class AutomaticTimingTests(unittest.TestCase):
    def make_controller(self, delays=(0,), **config):
        clock, model = Clock(), ChunkModel()
        executor = VirtualExecutor(clock, delays)
        controller = RTCController(model, RTCConfig(timeout=1000, **config),
                                   executor=executor, clock=clock.monotonic)
        self.addCleanup(controller.close)
        return controller, clock, model, executor

    @staticmethod
    def tick(controller, clock, step):
        clock.now = step
        return controller.next_action({"image": [], "lang": f"frame-{step}"}, step)

    def ticks(self, controller, clock, until):
        return [self.tick(controller, clock, step) for step in range(controller.last_step + 1, until)]

    def test_defaults_have_one_paper_beta_and_need_no_manual_delay(self):
        self.assertEqual((Args().rtc_stride, Args().rtc_delay), ("auto", "auto"))
        self.assertEqual((RTCConfig().stride, RTCConfig().delay), ("auto", "auto"))
        self.assertEqual((Args().rtc_beta, RTCConfig().beta), (5., 5.))
        self.assertFalse(hasattr(Args(), "rtc_guidance_beta"))
        self.assertFalse(hasattr(RTCConfig(), "guidance_beta"))
        self.assertFalse(hasattr(RTCConfig(), "inpainting"))
        ctl, _, _, _ = self.make_controller()
        self.assertEqual(ctl.estimated_delay, 0)
        self.assertEqual(ctl.minimum_execution_horizon, 1)
        self.assertEqual(list(ctl.delays), [])

    def test_bootstrap_measures_delay_without_turning_forecast_into_smin(self):
        ctl, clock, model, executor = self.make_controller(delays=(3, 2, 100))
        actions = self.ticks(ctl, clock, 4)
        self.assertEqual(actions[:3], [None] * 3)
        self.assertEqual(actions[3][0], 0)
        self.assertEqual(ctl.estimated_delay, 3)
        self.assertEqual(ctl.minimum_execution_horizon, 1)
        self.assertEqual(len(executor.submissions), 1)
        # Algorithm 1 waits t>=s_min, not t>=max(Q): replan at 4, not 6.
        self.tick(ctl, clock, 4)
        self.assertEqual([s[0] for s in executor.submissions], [0, 4])
        self.assertIsNone(executor.submissions[0][1]["rtc"])
        prefix = executor.submissions[1][1]["rtc"]
        self.assertEqual(prefix["delay"], 3)
        self.assertEqual(prefix["prefix_actions"].shape, (1, 7, 7))
        self.assertEqual(prefix["prefix_actions"][0, 0, 0], 1)
        # Submitted at 4, returned at 6: immediately consume Anew[2].
        self.assertEqual(self.ticks(ctl, clock, 7)[-1][0], 102)
        self.assertEqual(model.calls[1][1]["delay"], 3)
        self.assertEqual(ctl.stats.execution_horizons, [0, 1, 2])
        self.assertEqual(ctl.stats.estimated_delay_steps, [0, 3, 3])

    def test_actual_execution_horizon_follows_previous_delay_and_smin(self):
        ctl, clock, _, executor = self.make_controller(delays=(0, 1, 3, 0, 100), stride=2)
        self.ticks(ctl, clock, 10)
        self.assertEqual([s[0] for s in executor.submissions], [0, 2, 4, 7, 9])
        self.assertEqual(ctl.stats.execution_horizons, [0, 2, 2, 3, 2])
        self.assertEqual(ctl.stats.estimated_delay_steps[-1], 3)
        self.assertEqual(ctl.minimum_execution_horizon, 2)
        for (_, request), actual_s in zip(executor.submissions[1:], ctl.stats.execution_horizons[1:]):
            self.assertEqual(request["rtc"]["prefix_actions"].shape[1], ctl.horizon - actual_s)

    def test_latency_increase_updates_delay_mask_without_duplicate_requests(self):
        ctl, clock, _, executor = self.make_controller(delays=(1, 3, 100))
        actions = self.ticks(ctl, clock, 6)
        self.assertEqual([s[0] for s in executor.submissions], [0, 2, 5])
        self.assertEqual([a[0] for a in actions[1:]], [0, 1, 2, 3, 103])
        self.assertEqual(ctl.estimated_delay, 3)
        self.assertEqual(ctl.minimum_execution_horizon, 1)
        self.assertEqual(executor.submissions[2][1]["rtc"]["delay"], 3)
        self.assertEqual(ctl.stats.estimated_delay_steps, [0, 1, 3])
        self.assertEqual(ctl.stats.execution_horizons, [0, 1, 3])

    def test_zero_latency_uses_one_tick_minimum_without_artificial_waits(self):
        ctl, clock, _, executor = self.make_controller()
        actions = self.ticks(ctl, clock, 6)
        self.assertEqual([s[0] for s in executor.submissions], list(range(6)))
        self.assertEqual([a[0] for a in actions], [0, 100, 200, 300, 400, 500])
        self.assertEqual(ctl.stats.wait_steps, 0)
        self.assertEqual(ctl.stats.execution_horizons, [0, 1, 1, 1, 1, 1])
        self.assertEqual(ctl.stats.estimated_delay_steps, [0] * 6)

    def test_delay_spike_is_not_clamped_and_expired_result_is_discarded(self):
        ctl, clock, _, executor = self.make_controller(delays=(1, 9, 100))
        self.ticks(ctl, clock, 11)
        self.assertEqual([s[0] for s in executor.submissions], [0, 2])
        with self.assertLogs(level="WARNING"):
            action = self.tick(ctl, clock, 11)
        self.assertIsNone(action)
        self.assertEqual(ctl.stats.stale_chunks, 1)
        self.assertEqual(ctl.estimated_delay, 9)
        self.assertEqual(ctl.minimum_execution_horizon, 1)
        self.assertEqual([s[0] for s in executor.submissions], [0, 2, 11])
        self.assertEqual(ctl.stats.estimated_delay_steps[-1], 9)

    def test_slow_bootstrap_does_not_truncate_recorded_delay_or_prefix_request(self):
        ctl, clock, _, executor = self.make_controller(delays=(10, 0))
        actions = self.ticks(ctl, clock, 11)
        self.assertEqual(actions[:10], [None] * 10)
        self.assertEqual(actions[10][0], 0)
        self.assertEqual(ctl.estimated_delay, 10)
        self.assertEqual(ctl.minimum_execution_horizon, 1)
        with self.assertLogs(level="WARNING"):
            self.tick(ctl, clock, 11)
        self.assertEqual([s[0] for s in executor.submissions], [0, 11])
        rtc = executor.submissions[1][1]["rtc"]
        self.assertEqual(rtc["delay"], 10)
        self.assertEqual(rtc["prefix_actions"].shape[1], 7)

    def test_rolling_max_decreases_when_slow_sample_leaves_paper_delay_buffer(self):
        ctl, clock, _, executor = self.make_controller(delays=(3, 1))
        self.ticks(ctl, clock, 16)
        self.assertEqual(list(ctl.delays), [1] * 10)
        self.assertEqual(ctl.estimated_delay, 1)
        self.assertEqual(ctl.minimum_execution_horizon, 1)
        self.assertIn(3, ctl.stats.estimated_delay_steps)
        self.assertEqual(ctl.stats.estimated_delay_steps[-1], 1)
        self.assertEqual(executor.submissions[-1][1]["rtc"]["delay"], 1)

    def test_reset_clears_delay_history_and_does_not_import_drained_delay(self):
        ctl, clock, _, executor = self.make_controller(delays=(2, 4, 0))
        self.ticks(ctl, clock, 5)
        self.assertEqual(ctl.estimated_delay, 2)
        self.assertIsNotNone(ctl.pending)
        ctl.reset()
        self.assertTrue(executor.futures[1].resolved)
        self.assertEqual(list(ctl.delays), [])
        self.assertEqual(ctl.estimated_delay, 0)
        self.assertEqual(ctl.minimum_execution_horizon, 1)
        self.assertEqual(ctl.stats.estimated_delay_steps, [])
        self.assertEqual(ctl.stats.execution_horizons, [])
        self.assertEqual(ctl.next_action({"lang": "next episode"}, 0)[0], 200)
        self.assertIsNone(executor.submissions[-1][1]["rtc"])

    def test_manual_stride_sets_paper_minimum_with_measured_delay(self):
        ctl, clock, _, executor = self.make_controller(delays=(1, 2, 100), stride=4)
        self.ticks(ctl, clock, 8)
        self.assertEqual([s[0] for s in executor.submissions], [0, 5])
        self.assertEqual(ctl.estimated_delay, 2)
        self.assertEqual(ctl.minimum_execution_horizon, 4)
        self.assertEqual(executor.submissions[1][1]["rtc"]["delay"], 1)
        self.assertEqual(ctl.stats.execution_horizons, [0, 4])

    def test_manual_delay_is_initial_seed_and_never_changes_smin(self):
        ctl, clock, _, executor = self.make_controller(delay=3)
        self.assertEqual(ctl.estimated_delay, 3)
        self.assertEqual(ctl.minimum_execution_horizon, 1)
        self.ticks(ctl, clock, 2)
        self.assertEqual([s[0] for s in executor.submissions], [0, 1])
        self.assertEqual(executor.submissions[1][1]["rtc"]["delay"], 3)
        self.assertEqual(ctl.stats.estimated_delay_steps, [3, 3])
        self.ticks(ctl, clock, 12)
        self.assertEqual(ctl.estimated_delay, 0)
        np.testing.assert_array_equal(ctl.delays, [0] * 10)

    def test_manual_smin_and_initial_delay_keep_existing_chunk_schedule(self):
        ctl, clock, _, executor = self.make_controller(delays=(0, 2, 100), stride=2, delay=2)
        actions = self.ticks(ctl, clock, 6)
        self.assertEqual([a[0] for a in actions], [0, 1, 2, 3, 102, 103])
        self.assertEqual([s[0] for s in executor.submissions], [0, 2, 4])
        self.assertEqual(ctl.stats.execution_horizons, [0, 2, 2])

    def test_invalid_auto_tokens_and_numeric_types_fail_before_request(self):
        for key in ("stride", "delay"):
            for invalid in ("AUTO", "4", "", None, True, 1.5):
                with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                    RTCConfig(**{key: invalid}).validate(8)


if __name__ == "__main__":
    unittest.main()
