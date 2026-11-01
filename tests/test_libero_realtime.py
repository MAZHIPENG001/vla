"""Timing/rollout regression tests; no LIBERO, GPU or policy server required."""

import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from examples.LIBERO.eval_files import eval_libero as evaluation


def response(step):
    return {"raw_action": {
        "world_vector": [0.1 + step, 0.2, 0.3],
        "rotation_delta": [0.4, 0.5, 0.6],
        "open_gripper": [0.0],
    }}


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class DelayedFuture:
    def __init__(self, clock, delay, fn, kwargs):
        self.clock = clock
        self.ready_at = clock.now + delay
        self.fn, self.kwargs = fn, kwargs
        self.resolved = False

    def done(self):
        return self.clock.now >= self.ready_at

    def result(self):
        if not self.resolved:
            # Draining a pending request after termination may advance wall
            # time, but must never step the environment or start a new request.
            self.clock.now = max(self.clock.now, self.ready_at)
            self.value = self.fn(**self.kwargs)
            self.resolved = True
        return self.value


class DelayedExecutor:
    def __init__(self, clock, delay):
        self.clock, self.delay = clock, delay
        self.futures = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, **kwargs):
        future = DelayedFuture(self.clock, self.delay, fn, kwargs)
        self.futures.append(future)
        return future


class Environment:
    def __init__(self, success_at=None):
        self.actions = []
        self.threads = []
        self.success_at = success_at

    def obs(self):
        img = np.full((2, 2, 3), len(self.actions), dtype=np.uint8)
        return {"agentview_image": img, "robot0_eye_in_hand_image": img.copy()}

    def step(self, action):
        self.threads.append(threading.get_ident())
        self.actions.append(list(action))
        return self.obs(), 0, len(self.actions) == self.success_at, {}


class Model:
    def __init__(self):
        self.steps = []
        self.observation_indices = []
        self.cached = False

    def needs_inference(self, step):
        return not self.cached or step % 2 == 0

    def step(self, example, step):
        self.steps.append(step)
        self.observation_indices.append(int(example["image"][0][0, 0, 0]))
        self.cached = True
        return response(step)


class RealtimeTests(unittest.TestCase):
    def run_virtual(self, max_steps=10, delay=0.12, realtime=True, success_at=None, model=None):
        clock = Clock()
        executor = DelayedExecutor(clock, delay)
        env, model = Environment(success_at), model or Model()
        args = evaluation.Args(num_steps_wait=0, control_freq=20, realtime=realtime)
        with (
            patch.object(evaluation.time, "monotonic", clock.monotonic),
            patch.object(evaluation.time, "sleep", clock.sleep),
            patch.object(evaluation, "ThreadPoolExecutor", return_value=executor),
        ):
            frames, done = evaluation._run_episode(env, model, env.obs(), "task", args, max_steps)
        return env, model, frames, done, clock, executor

    def test_inference_wait_repeats_complete_last_action_and_preserves_cache_indices(self):
        env, model, frames, done, clock, executor = self.run_virtual()
        self.assertFalse(done)
        self.assertEqual(len(frames), 10)
        self.assertAlmostEqual(clock.now, 0.5)
        self.assertEqual(model.steps, [0, 1, 2, 3])
        self.assertEqual(len(executor.futures), 2)
        # First inference waits for three ticks with the startup command.
        for action in env.actions[:3]:
            self.assertEqual(action, evaluation.LIBERO_DUMMY_ACTION)
        np.testing.assert_allclose(env.actions[3], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1])
        # The second request repeats ALL SEVEN values of the last cache action.
        for action in env.actions[5:8]:
            self.assertEqual(action, env.actions[4])
        self.assertNotEqual(env.actions[8], env.actions[4])
        # Request 2 captures the latest frame at submission, before waiting.
        self.assertEqual(model.observation_indices, [0, 4, 5, 9])
        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], list(range(1, 11)))

    def test_timeout_counts_wait_steps_and_drains_inference(self):
        env, model, frames, done, clock, executor = self.run_virtual(max_steps=4, delay=2)
        self.assertFalse(done)
        self.assertEqual(len(env.actions), 4)
        self.assertEqual(len(frames), 4)
        self.assertEqual(model.steps, [0])
        self.assertTrue(all(future.resolved for future in executor.futures))
        self.assertAlmostEqual(clock.now, 2)

    def test_success_during_inference_records_terminal_frame_and_stops(self):
        env, model, frames, done, _, executor = self.run_virtual(success_at=2, delay=2)
        self.assertTrue(done)
        self.assertEqual(len(env.actions), 2)
        self.assertEqual(len(frames), 2)
        self.assertEqual(model.steps, [0])
        self.assertTrue(executor.futures[0].resolved)

    def test_immediate_responses_add_no_artificial_wait_ticks(self):
        env, model, frames, _, clock, _ = self.run_virtual(max_steps=6, delay=0)
        self.assertEqual(model.steps, list(range(6)))
        self.assertEqual(len(frames), 6)
        self.assertNotIn(evaluation.LIBERO_DUMMY_ACTION, env.actions)
        self.assertAlmostEqual(clock.now, 0.3)

    def test_synchronous_mode_still_consumes_one_action_per_tick(self):
        _, model, frames, _, clock, executor = self.run_virtual(max_steps=6, realtime=False)
        self.assertEqual(model.steps, list(range(6)))
        self.assertEqual(len(frames), 6)
        self.assertEqual(executor.futures, [])
        self.assertEqual(clock.now, 0)

    def test_video_labels_match_inference_waits_and_cached_execution(self):
        with patch.object(evaluation, "_annotate_video_frame", wraps=evaluation._annotate_video_frame) as draw:
            self.run_virtual()
        self.assertEqual(
            [call.args[1] for call in draw.call_args_list],
            [True, True, True, False, False, True, True, True, False, False],
        )
        with patch.object(evaluation, "_annotate_video_frame", wraps=evaluation._annotate_video_frame) as draw:
            self.run_virtual(max_steps=4, realtime=False)
        self.assertEqual([call.args[1] for call in draw.call_args_list], [False] * 4)

    def test_model_failures_propagate_including_when_draining(self):
        class BrokenModel(Model):
            def step(self, **kwargs):
                raise RuntimeError("inference failed")

        for delay in (0, 2):
            with self.subTest(delay=delay), self.assertRaisesRegex(RuntimeError, "inference failed"):
                self.run_virtual(max_steps=3, delay=delay, model=BrokenModel())

    def test_camera_buffers_are_copied_before_worker_uses_them(self):
        obs = Environment().obs()
        example = evaluation._make_model_example(obs, "task")
        obs["agentview_image"][:] = 42
        obs["robot0_eye_in_hand_image"][:] = 42
        for image in example["image"]:
            self.assertTrue(image.flags.c_contiguous)
            self.assertTrue((image == 0).all())

    def test_real_worker_does_not_block_environment_or_use_renderer_thread(self):
        class SlowModel(Model):
            def __init__(self):
                super().__init__()
                self.worker_threads = []

            def step(self, example, step):
                if self.needs_inference(step):
                    self.worker_threads.append(threading.get_ident())
                    time.sleep(0.08)
                return super().step(example, step)

        env, model = Environment(), SlowModel()
        args = evaluation.Args(num_steps_wait=0, control_freq=50)
        started = time.monotonic()
        frames, _ = evaluation._run_episode(env, model, env.obs(), "task", args, 12)
        self.assertGreaterEqual(time.monotonic() - started, 12 / 50)
        self.assertEqual(len(frames), 12)
        self.assertEqual(set(env.threads), {threading.get_ident()})
        self.assertTrue(model.worker_threads)
        self.assertNotIn(threading.get_ident(), model.worker_threads)
        self.assertLess(len(model.steps), len(env.actions))


if __name__ == "__main__":
    unittest.main()
