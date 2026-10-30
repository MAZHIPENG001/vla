import unittest
from unittest.mock import patch

import numpy as np
import torch
from types import SimpleNamespace

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.model_server.policy_wrapper import PolicyServerWrapper
from vla.model.framework.share_tools import include_state_enabled
from vla.dataloader.gr00t_lerobot.data_config import Libero4in1DataConfig
from vla.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from vla.model.framework.VLM4A.QwenOFT import Qwenvl_OFT


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

    def test_missing_state_setting_keeps_state_through_training_and_server(self):
        for config in (None, {}, {"include_state": True}, {"include_state": "True"}):
            self.assertTrue(include_state_enabled(config))
        for value in (False, "False", "false"):
            self.assertFalse(include_state_enabled({"include_state": value}))
        config = Libero4in1DataConfig()
        self.assertEqual(config.modality_config()["state"].delta_indices, [0])
        dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
        dataset.data_cfg = {}
        dataset._modality_keys = {k: v.modality_keys for k, v in config.modality_config().items()}
        dataset.tag = config.embodiment_tag
        raw = {k: np.zeros((1, 4, 4, 3), np.uint8) for k in config.video_keys}
        raw.update({k: np.zeros((8, 1), np.float32) for k in config.action_keys})
        raw.update({k: np.array([[i/10]], np.float32) for i, k in enumerate(config.state_keys)})
        raw[config.language_keys[0]] = ["pick"]
        sample = dataset._pack_sample(raw)
        self.assertEqual(sample["state"].shape, (1, 8))
        self.assertEqual(sample["state"].dtype, np.float16)

        proc = self.processor()
        seen = []
        framework = SimpleNamespace(predict_action=lambda examples, **_: (
            seen.extend(examples) or {"normalized_actions": np.zeros((1, 8, 7))}
        ))
        cfg = {"datasets": {"vla_data": {"data_mix": "libero_all"}},
               "framework": {"action_model": {"action_horizon": 8}}}
        with patch("deployment.model_server.policy_wrapper.baseframework.from_pretrained") as loader, \
             patch("deployment.model_server.policy_wrapper.read_mode_config", return_value=(cfg, {"franka": {}})), \
             patch.object(PolicyServerWrapper, "_get_processor", return_value=proc):
            loader.return_value.to.return_value.eval.return_value = framework
            server = PolicyServerWrapper("unused.pt", device="cpu")
            self.assertTrue(server.metadata["include_state"])
            server.predict_action([sample])
            np.testing.assert_array_equal(seen[0]["state"], sample["state"])
            disabled = PolicyServerWrapper("unused.pt", device="cpu", include_state=False)
            self.assertFalse(disabled.metadata["include_state"])

    def test_oft_bfloat16_head_returns_numpy_actions(self):
        # Exercise the real inference method with lightweight backbone outputs.
        # This reproduces the server's --use_bf16 path without a 4B allocation.
        model = Qwenvl_OFT.__new__(Qwenvl_OFT)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(datasets=SimpleNamespace(vla_data=SimpleNamespace()))
        model.action_token_id = 1
        model.chunk_len = 8
        model.qwen_vl_interface = lambda **_: SimpleNamespace(
            hidden_states=[torch.zeros((1, 8, 4), dtype=torch.bfloat16)],
        )
        class Head(torch.nn.Linear):
            predict_action = torch.nn.Linear.forward
        model.action_model = Head(4, 7, dtype=torch.bfloat16)
        model._build_action_inputs = lambda *_, **__: {"input_ids": torch.ones((1, 8), dtype=torch.long)}
        result = model.predict_action([{"image": [], "lang": "pick"}])["normalized_actions"]
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (1, 8, 7))
        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
