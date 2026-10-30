import unittest
from unittest.mock import patch

import numpy as np

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.model_server.policy_wrapper import PolicyServerWrapper


class PolicyNormTests(unittest.TestCase):
    def processor(self):
        self.low = np.array([-2, -4, -6, -8, -10, -12, 0], np.float32)
        self.high = np.array([4, 8, 12, 16, 20, 24, 1], np.float32)
        action = {
            "min": self.low.tolist(), "max": self.high.tolist(),
            # Deliberately different from min/max to catch accidental q99 inversion.
            "q01": (self.low * 0.2).tolist(), "q99": (self.high * 0.2).tolist(),
            "mean": [0.] * 7, "std": [1.] * 7, "mask": [True] * 6 + [False],
        }
        cfg = {"datasets": {"vla_data": {"data_mix": "libero_all"}}}
        state = {k: [v] * 8 for k, v in (("min", -1), ("max", 1), ("mean", 0),
                                       ("std", 1), ("q01", -0.9), ("q99", 0.9))}
        with patch("deployment.model_server.policy_norm_processor.read_mode_config",
                   return_value=(cfg, {"franka": {"action": action, "state": state}})):
            return PolicyNormProcessor("unused.pt")

    def test_min_max_inverse_and_untouched_gripper(self):
        proc = self.processor()
        predicted = np.array([[-1]*6+[0.2], [0]*6+[0.5], [1]*6+[0.8]], np.float32)
        actual = proc.unapply_actions(predicted)
        expected = np.stack([self.low, (self.low+self.high)/2, self.high])
        expected[:, 6] = predicted[:, 6]
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_training_forward_then_server_inverse(self):
        proc = self.processor()
        raw = np.stack([self.low, (self.low+self.high)/2, self.high])
        # Run the actual training transform, including tensor conversion.
        normalized = proc.transform({k: raw[:, i:i+1].copy() for i, k in enumerate(proc.action_keys)})
        packed = np.concatenate([normalized[k].numpy() for k in proc.action_keys], axis=1)
        np.testing.assert_allclose(proc.unapply_actions(packed), raw, atol=1e-6)

    def test_server_omits_state_for_no_state_checkpoint(self):
        proc = self.processor()
        server = PolicyServerWrapper.__new__(PolicyServerWrapper)
        server._default_unnorm_key = "franka"
        server._include_state = False
        server._get_processor = lambda key: proc
        class Framework:
            def predict_action(self, examples, **kwargs):
                if "state" in examples[0]:
                    raise AssertionError("Unexpected state changes the OFT prompt")
                return {"normalized_actions": np.zeros((1, 8, 7), np.float32)}
        server._framework = Framework()
        example = {"image": [], "lang": "pick", "state": np.ones((1, 8))}
        self.assertEqual(server.predict_action([example])["actions"].shape, (1, 8, 7))
        self.assertIn("state", example)  # Do not mutate caller's observations.


if __name__ == "__main__":
    unittest.main()
