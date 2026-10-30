import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from examples.LIBERO.eval_files.libero_contract import gripper_to_libero, libero_images

# These are scheduling/observation tests, independent of websocket/msgpack.
with patch.dict(sys.modules, {
    "deployment.model_server.tools.websocket_policy_client": SimpleNamespace(WebsocketClientPolicy=None),
}):
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient


class FakePolicy:
    def __init__(self, horizon=8, returned_horizon=None):
        self.horizon = horizon
        self.returned_horizon = returned_horizon or horizon
        self.requests = []

    def get_server_metadata(self):
        return {"action_chunk_size": self.horizon, "include_state": False}

    def predict_action(self, request):
        self.requests.append(request)
        actions = np.zeros((1, self.returned_horizon, 7), np.float32)
        actions[0, :, 0] = 100 * len(self.requests) + np.arange(self.returned_horizon)
        return {"data": {"actions": actions}}


def make_client(fake, **kwargs):
    with patch("examples.LIBERO.eval_files.model2libero_interface.WebsocketClientPolicy", return_value=fake):
        return ModelClient(**kwargs)


class LiberoContractTests(unittest.TestCase):
    def test_gripper_encodings_match_environment_sign(self):
        for encoding, opened, closed in (
            ("binary_open", 1, 0), ("binary_close", 0, 1), ("signed_close", -1, 1),
        ):
            self.assertEqual(gripper_to_libero(opened, encoding).item(), -1)
            self.assertEqual(gripper_to_libero(closed, encoding).item(), 1)
        for invalid in ([], [0, 1], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                gripper_to_libero(invalid)

    def test_replan_intervals_start_each_prediction_at_zero(self):
        for interval in (1, 2, 4, 8):
            with self.subTest(interval=interval):
                fake = FakePolicy()
                client = make_client(fake, replan_interval=interval)
                # A nonzero initial global step must still execute action 0.
                actual = [client.step({"lang": "pick"}, step=s)["raw_action"]["world_vector"][0]
                          for s in range(5, 29)]
                expected = [100 * (i // interval + 1) + i % interval for i in range(24)]
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(len(fake.requests), 24 // interval)

    def test_reset_task_change_discontinuity_and_omitted_step(self):
        fake = FakePolicy()
        client = make_client(fake, replan_interval=4)
        def action(lang="a", **kw):
            return client.step({"lang": lang}, **kw)["raw_action"]["world_vector"][0]
        self.assertEqual(action(), 100)
        self.assertEqual(action(), 101)
        client.reset("a")
        self.assertEqual(action(step=10), 200)
        self.assertEqual(action(step=11), 201)
        self.assertEqual(action(step=20), 300)
        self.assertEqual(action("b", step=21), 400)

    def test_short_prediction_refreshes_before_exhaustion(self):
        client = make_client(FakePolicy(returned_horizon=2), replan_interval=8)
        actual = [client.step({"lang": "a"})["raw_action"]["world_vector"][0] for _ in range(5)]
        np.testing.assert_array_equal(actual, [100, 101, 200, 201, 300])

    def test_invalid_interval_and_response(self):
        for interval in (0, -1, 9):
            with self.assertRaises(ValueError):
                make_client(FakePolicy(), replan_interval=interval)
        for bad in (np.zeros((8, 7)), np.zeros((1, 0, 7)), np.zeros((1, 8, 8)),
                    np.full((1, 8, 7), np.nan)):
            fake = FakePolicy()
            fake.predict_action = lambda _, bad=bad: {"data": {"actions": bad}}
            with self.assertRaises(ValueError):
                make_client(fake).step({"lang": "a"})

    def test_cameras_rotation_resize_and_state_match_training(self):
        rng = np.random.default_rng(42)
        primary = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        wrist = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        images = libero_images({"agentview_image": primary, "robot0_eye_in_hand_image": wrist})
        fake = FakePolicy()
        client = make_client(fake)
        client.step({"image": images, "lang": "pick", "state": np.ones((1, 8))})
        received = fake.requests[0]["examples"][0]
        self.assertNotIn("state", received)
        for raw, actual in zip((primary, wrist), received["image"]):
            # This is the training _pack_sample RGB resize call (default filter).
            expected = np.asarray(Image.fromarray(np.rot90(raw, 2)).resize((224, 224)))
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
