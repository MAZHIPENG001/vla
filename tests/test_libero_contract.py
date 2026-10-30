import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from examples.LIBERO.eval_files.libero_contract import gripper_to_libero, libero_images, libero_state

# Exercise the evaluator with fake transport/environment, without simulator dependencies.
with patch.dict(sys.modules, {
    "deployment.model_server.tools.websocket_policy_client": SimpleNamespace(WebsocketClientPolicy=None),
}):
    import examples.LIBERO.eval_files.model2libero_interface as adapter
    import examples.LIBERO.eval_files.eval_libero as evaluation
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient
    from examples.LIBERO.eval_files.eval_libero import Args, _resolve_replan_interval, _run_episode


class FakePolicy:
    def __init__(self, horizon=8, returned_horizon=None, include_state=False):
        self.horizon = horizon
        self.returned_horizon = returned_horizon or horizon
        self.include_state = include_state
        self.requests = []
        self.closed = False

    def get_server_metadata(self):
        return {"action_chunk_size": self.horizon, "include_state": self.include_state}

    def predict_action(self, request):
        self.requests.append(request)
        actions = np.zeros((1, self.returned_horizon, 7), np.float32)
        actions[0, :, 0] = 100 * len(self.requests) + np.arange(self.returned_horizon)
        return {"data": {"actions": actions}}

    def close(self):
        self.closed = True


def make_client(fake):
    with patch.object(adapter, "WebsocketClientPolicy", return_value=fake):
        return ModelClient("127.0.0.1", 10093)


class FakeEnv:
    def __init__(self, success_after=None):
        self.success_after = success_after
        self.closed = False

    def reset(self):
        self.actions = []

    def set_init_state(self, state):
        self.init_state = state
        return self.observation()

    def observation(self):
        return {
            "agentview_image": np.full((16, 16, 3), len(self.actions), np.uint8),
            "robot0_eye_in_hand_image": np.zeros((16, 16, 3), np.uint8),
            "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0., 0., 0., 1.]),
            "robot0_gripper_qpos": np.array([0.02, -0.02]),
        }

    def step(self, action):
        self.actions.append(action)
        executed = sum(a[0] >= 100 for a in self.actions)
        done = self.success_after is not None and executed >= self.success_after
        return self.observation(), 0, done, {}

    def close(self):
        self.closed = True


class LiberoContractTests(unittest.TestCase):
    def test_gripper_encodings_match_environment_sign(self):
        for encoding, opened, closed in (
            ("binary_open", 1, 0), ("binary_close", 0, 1), ("signed_close", -1, 1),
        ):
            self.assertEqual(gripper_to_libero(opened, encoding).item(), -1)
            self.assertEqual(gripper_to_libero(closed, encoding).item(), 1)
        self.assertEqual(gripper_to_libero(0, "signed_close").item(), 0)
        for invalid in ([], [0, 1], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                gripper_to_libero(invalid)

    def test_replan_intervals_execute_prefix_and_use_latest_observation(self):
        for interval in (1, 2, 4, 8):
            with self.subTest(interval=interval):
                fake = FakePolicy()
                env = FakeEnv()
                args = Args(replan_interval=interval, num_steps_wait=3)
                success, replay = _run_episode(env, make_client(fake), "init", "pick", args, 24)
                self.assertFalse(success)
                self.assertEqual(env.init_state, "init")
                np.testing.assert_array_equal(env.actions[:3], [[0]*6+[-1]]*3)
                actual = np.asarray(env.actions[3:])
                expected = [100*(i//interval+1)+i%interval for i in range(24)]
                np.testing.assert_array_equal(actual[:, 0], expected)
                np.testing.assert_array_equal(actual[:, 6], np.ones(24))
                self.assertEqual(len(replay), 24)
                self.assertEqual(len(fake.requests), 24//interval)
                for i, request in enumerate(fake.requests):
                    example = request["examples"][0]
                    self.assertEqual(example["lang"], "pick")
                    self.assertNotIn("state", example)
                    self.assertEqual(example["image"][0][0, 0, 0], 3+i*interval)

    def test_episode_local_queue_discards_unused_predictions(self):
        fake = FakePolicy()
        client = make_client(fake)
        env = FakeEnv(success_after=2)
        args = Args(replan_interval=8, num_steps_wait=0)
        for task, first in (("pick", 100), ("pick", 200), ("place", 300)):
            success, replay = _run_episode(env, client, "init", task, args, 24)
            self.assertTrue(success)
            self.assertEqual(len(replay), 2)
            np.testing.assert_array_equal(np.asarray(env.actions)[:, 0], [first, first+1])

    def test_short_prediction_refreshes_before_exhaustion(self):
        env = FakeEnv()
        client = make_client(FakePolicy(returned_horizon=2))
        _run_episode(env, client, None, "pick", Args(replan_interval=8, num_steps_wait=0), 5)
        np.testing.assert_array_equal(np.asarray(env.actions)[:, 0], [100, 101, 200, 201, 300])

    def test_invalid_interval_and_response(self):
        self.assertEqual(_resolve_replan_interval(None, 8), 8)
        for interval in (0, -1, 9):
            with self.assertRaises(ValueError):
                _resolve_replan_interval(interval, 8)
        for bad in (np.zeros((8, 7)), np.zeros((1, 0, 7)), np.zeros((1, 8, 8)),
                    np.full((1, 8, 7), np.nan)):
            fake = FakePolicy()
            fake.predict_action = lambda _, bad=bad: {"data": {"actions": bad}}
            with self.assertRaises(ValueError):
                make_client(fake).infer({"image": [], "lang": "a"})

    def test_cameras_rotation_resize_and_state_match_training(self):
        rng = np.random.default_rng(42)
        primary = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        wrist = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        images = libero_images({"agentview_image": primary, "robot0_eye_in_hand_image": wrist})
        fake = FakePolicy()
        make_client(fake).infer({"image": images, "lang": "pick", "state": np.ones((1, 8))})
        received = fake.requests[0]["examples"][0]
        self.assertNotIn("state", received)
        for raw, actual in zip((primary, wrist), received["image"]):
            expected = np.asarray(Image.fromarray(np.rot90(raw, 2)).resize((224, 224)))
            np.testing.assert_array_equal(actual, expected)

    def test_state_is_sent_only_when_checkpoint_requests_it(self):
        fake = FakePolicy(include_state=True)
        env = FakeEnv()
        _run_episode(env, make_client(fake), None, "pick", Args(num_steps_wait=0), 1)
        state = fake.requests[0]["examples"][0]["state"]
        np.testing.assert_allclose(state, [[0, 0, 0, 0, 0, 0, 0.02, -0.02]])
        obs = env.observation()
        obs["robot0_eef_quat"] = np.array([0., 0., np.sin(0.2), np.cos(0.2)])
        original = obs["robot0_eef_quat"].copy()
        np.testing.assert_allclose(libero_state(obs)[3:6], [0, 0, 0.4])
        np.testing.assert_array_equal(obs["robot0_eef_quat"], original)

    def test_evaluation_closes_environment_and_connection_on_success_or_error(self):
        task = SimpleNamespace(language="pick")
        suite = SimpleNamespace(n_tasks=1, get_task=lambda _: task, get_task_init_states=lambda _: ["init"])
        benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_goal": lambda: suite})
        for fails in (False, True):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as output:
                fake = FakePolicy()
                client = make_client(fake)
                env = FakeEnv(success_after=1)
                if fails:
                    fake.predict_action = lambda _: {"data": {"actions": np.zeros((1, 0, 7))}}
                modules = {
                    "libero": SimpleNamespace(), "libero.libero": SimpleNamespace(benchmark=benchmark),
                    "imageio": SimpleNamespace(mimwrite=lambda *args, **kwargs: None),
                }
                with patch.dict(sys.modules, modules), patch.object(evaluation, "ModelClient", return_value=client), \
                     patch.object(evaluation, "_get_libero_env", return_value=(env, "pick")):
                    args = Args(num_trials_per_task=1, num_steps_wait=0, video_out_path=output)
                    if fails:
                        with self.assertRaises(ValueError):
                            evaluation.eval_libero(args)
                    else:
                        evaluation.eval_libero(args)
                self.assertTrue(env.closed)
                self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main()
