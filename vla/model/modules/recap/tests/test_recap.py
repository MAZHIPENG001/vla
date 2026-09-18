"""Config, example-batch API, checkpoint, and trajectory-label integration tests."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import BatchFeature

from vla.model.modules.recap import QwenRecap, QwenValueModel
from vla.model.modules.recap.tests.test_value_model import tiny_qwen


class RecapTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.processor = MagicMock()
        self.processor.apply_chat_template.side_effect = self._process
        patches = [
            patch("vla.model.modules.vlm.Qwen.Qwen3VLForConditionalGeneration.from_pretrained", side_effect=lambda *a, **k: tiny_qwen()),
            patch("vla.model.modules.vlm.Qwen.AutoProcessor.from_pretrained", return_value=self.processor),
        ]
        self.load_model, self.load_processor = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)
        self.batch = [
            {"image": [Image.new("RGB", (8, 8))], "lang": "close the box", "return": -0.5},
            {"image": np.zeros((8, 8, 3), dtype=np.uint8), "lang": "fold shirt",
             "return": -0.2, "state": np.zeros((1, 7))},
        ]

    @staticmethod
    def _process(messages, **kwargs):
        batch = len(messages)
        return BatchFeature({
            "input_ids": torch.tensor([[1, 5, 3, 6, 7]]).expand(batch, -1),
            "attention_mask": torch.ones(batch, 5, dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 0, 1, 0, 0]]).expand(batch, -1),
            "pixel_values": torch.ones(batch * 4, 12),
            "image_grid_thw": torch.tensor([[1, 2, 2]]).expand(batch, -1),
        })

    def test_yaml_overrides_defaults_and_loads_requested_backbone(self):
        cfg = {
            "recap": {
                "qwenvl": {"base_vlm": "/models/Qwen-local", "attn_implementation": "eager"},
                "value_model": {"num_bins": 11, "hidden_size": 999},
                "custom_key": "preserved",
            },
            "datasets": {"vla_data": {"CoT_prompt": "Task: {instruction}", "obs_image_size": [12, 10]}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            OmegaConf.save(OmegaConf.create(cfg), path)
            model = QwenRecap(path)
        self.assertEqual(self.load_model.call_args.args[0], "/models/Qwen-local")
        self.assertEqual(self.load_model.call_args.kwargs["attn_implementation"], "eager")
        self.assertEqual(model.config.recap.value_model.hidden_size, 32)
        self.assertEqual(model.config.recap.value_model.num_bins, 11)
        self.assertEqual(model.config.recap.advantage.n_steps, 50)
        self.assertEqual(model.config.recap.custom_key, "preserved")
        model.predict_value(self.batch)
        content = self.processor.apply_chat_template.call_args.args[0][0][0]["content"]
        self.assertEqual(content[0]["image"].size, (12, 10))
        self.assertEqual(content[-1]["text"], "Task: close the box")

    def test_default_and_shipped_config(self):
        model = QwenRecap()
        self.assertEqual(model.config.recap.name, "QwenRecap")
        self.assertEqual(self.load_model.call_args.args[0], "Qwen/Qwen3-VL-2B-Instruct")
        config_path = Path(__file__).resolve().parents[5] / "examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
        task_config = OmegaConf.load(config_path)
        from_yaml = QwenRecap(config_path)
        self.assertEqual(from_yaml.config.recap.value_model.num_bins, 201)
        self.assertEqual(self.load_model.call_args.args[0], task_config.recap.qwenvl.base_vlm)
        self.assertEqual(from_yaml.config.framework, task_config.framework)
        self.assertEqual(from_yaml.config.datasets, task_config.datasets)
        self.assertEqual(from_yaml.config.trainer, task_config.trainer)
        self.assertTrue(torch.isfinite(from_yaml(self.batch)["value_loss"]))

    def test_shared_config_runs_policy_and_recap_in_either_load_order(self):
        from vla.model.framework.VLM4A.QwenOFT import Qwenvl_OFT

        self.processor.tokenizer.return_value = {"input_ids": [7]}
        for recap_first in (True, False):
            with self.subTest(recap_first=recap_first):
                cfg = OmegaConf.create({
                    "framework": {
                        "name": "QwenOFT",
                        "qwenvl": {"base_vlm": "Qwen/policy-4B", "attn_implementation": "eager"},
                        "action_model": {"action_horizon": 1, "action_dim": 2},
                    },
                    "recap": {"qwenvl": {"base_vlm": "Qwen/critic-2B", "attn_implementation": "sdpa"}},
                    "datasets": {"vla_data": {"obs_image_size": [12, 10]}},
                    "trainer": {"learning_rate": 1e-5},
                })
                self.load_model.reset_mock()
                if recap_first:
                    before = OmegaConf.to_container(cfg, resolve=True)
                    recap = QwenRecap(cfg)
                    self.assertEqual(OmegaConf.to_container(cfg, resolve=True), before)
                    policy = Qwenvl_OFT(cfg)
                else:
                    policy = Qwenvl_OFT(cfg)
                    before = OmegaConf.to_container(cfg, resolve=True)
                    recap = QwenRecap(cfg)
                    self.assertEqual(OmegaConf.to_container(cfg, resolve=True), before)
                expected = ["Qwen/critic-2B", "Qwen/policy-4B"] if recap_first else ["Qwen/policy-4B", "Qwen/critic-2B"]
                self.assertEqual([call.args[0] for call in self.load_model.call_args_list], expected)
                self.assertEqual(OmegaConf.to_container(recap.config.framework), before["framework"])
                self.assertEqual(recap.config.trainer, cfg.trainer)
                self.assertEqual(policy.config.framework.qwenvl.base_vlm, "Qwen/policy-4B")
                self.assertEqual(recap.qwen_vl_interface.config.framework.qwenvl.base_vlm, "Qwen/critic-2B")
                batch = [{**example, "action": np.zeros((1, 2), dtype=np.float32)} for example in self.batch]
                self.assertTrue(torch.isfinite(policy(batch)["action_loss"]))
                self.assertTrue(torch.isfinite(recap(batch)["value_loss"]))

    def test_policy_only_config_does_not_select_policy_backbone(self):
        cfg = OmegaConf.create({"framework": {"name": "QwenOFT", "qwenvl": {"base_vlm": "Qwen/policy-4B"}}})
        snapshot = OmegaConf.to_container(cfg)
        model = QwenRecap(cfg)
        self.assertEqual(self.load_model.call_args.args[0], "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(OmegaConf.to_container(cfg), snapshot)
        self.assertEqual(model.config.framework.name, "QwenOFT")
        with self.assertRaisesRegex(ValueError, "recap.qwenvl"):
            QwenValueModel(config=cfg)

    def test_shared_interpolations_and_tracked_config(self):
        from vla.training.trainer_utils.config_tracker import AccessTrackedConfig

        cfg = OmegaConf.create({
            "model_root": "/models",
            "framework": {"name": "QwenOFT", "qwenvl": {"base_vlm": "${model_root}/Qwen-policy"}},
            "recap": {"qwenvl": {"base_vlm": "${model_root}/Qwen-critic"}},
            "prompt_prefix": "Task",
            "datasets": {"vla_data": {"CoT_prompt": "${prompt_prefix}: {instruction}"}},
        })
        snapshot = OmegaConf.to_container(cfg, resolve=False)
        model = QwenRecap(AccessTrackedConfig(cfg))
        self.assertEqual(self.load_model.call_args.args[0], "/models/Qwen-critic")
        model.predict_value(self.batch)
        messages = self.processor.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0][0]["content"][-1]["text"], "Task: close the box")
        self.assertEqual(OmegaConf.to_container(cfg, resolve=False), snapshot)

    def test_examples_training_and_prediction_with_optional_state(self):
        model = QwenRecap()
        losses = model(self.batch)
        self.assertEqual(set(losses), {"value_loss"})
        self.assertEqual(losses["value_loss"].ndim, 0)
        losses["value_loss"].backward()
        self.assertGreater(model.value_model.value_head.weight.grad.abs().sum().item(), 0)
        messages = self.processor.apply_chat_template.call_args.args[0]
        self.assertNotIn("[STATE]", messages[0][0]["content"][-1]["text"])
        self.assertIn("[STATE]", messages[1][0]["content"][-1]["text"])
        prediction = model.predict_value(self.batch[0])["values"]
        self.assertEqual(prediction.shape, (1,))
        self.assertFalse(prediction.requires_grad)
        self.assertTrue(model.training)

    def test_trajectory_scatter_chunking_and_configured_estimate(self):
        model = QwenRecap({"recap": {"advantage": {"n_steps": 1, "prediction_batch_size": 2}}})
        episodes = [self.batch, [self.batch[0]]]
        rewards = torch.tensor([[-0.1, 0.0, float("nan")], [-0.8, float("nan"), float("nan")]])
        with patch.object(model, "predict_value", side_effect=[
            {"values": torch.tensor([-0.6, -0.3])}, {"values": torch.tensor([-0.5])},
        ]) as predict:
            result = model.predict_advantages(episodes, rewards)
        self.assertEqual([len(call.args[0]) for call in predict.call_args_list], [2, 1])
        torch.testing.assert_close(result["values"], torch.tensor([[-0.6, -0.3, 0], [-0.5, 0, 0]]))
        torch.testing.assert_close(result["advantages"], torch.tensor([[0.2, 0.3, 0], [-0.3, 0, 0]]))
        self.assertEqual(result["valid_mask"].tolist(), [[True, True, False], [True, False, False]])
        self.assertEqual(result["lengths"].tolist(), [2, 1])

    def test_threshold_config_roundtrip_and_dropout_modes(self):
        model = QwenRecap({"recap": {"advantage": {"mode": "pretrain", "dropout_probability": 1}}})
        advantages = torch.arange(10.0)
        tasks = torch.zeros(10, dtype=torch.long)
        thresholds = model.fit_thresholds(advantages, tasks)
        self.assertAlmostEqual(thresholds[0], 6.3, places=5)
        self.assertFalse(model.make_condition(advantages, tasks).conditioning_mask.any())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibrated.yaml"
            OmegaConf.save(model.config, path)
            restored = QwenRecap(path).eval()
        condition = restored.make_condition(advantages, tasks)
        self.assertEqual(condition.indicator.sum().item(), 3)
        self.assertTrue(condition.conditioning_mask.all())

    def test_checkpoint_loading_restores_trained_head(self):
        model = QwenRecap({"recap": {"value_model": {"num_bins": 5, "freeze_backbone": True}}})
        with torch.no_grad():
            model.value_model.value_head.bias.copy_(torch.arange(5.0))
        expected = model.predict_value(self.batch)["values"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "critic.pt"
            torch.save(model.value_model.state_dict(), path)
            cfg = OmegaConf.create(OmegaConf.to_container(model.config))
            cfg.recap.value_model.checkpoint = str(path)
            restored = QwenRecap(cfg)
            torch.testing.assert_close(restored.predict_value(self.batch)["values"], expected)
            self.assertFalse(restored.qwen_vl_interface.training)
            cfg.recap.value_model.value_min = -2
            with self.assertRaisesRegex(ValueError, "support"):
                QwenRecap(cfg)

    def test_bad_config_rejected_before_loading_weights(self):
        for config in (
            {"recap": {"name": "QwenOFT"}},
            {"recap": {"advantage": {"mode": "unknown"}}},
            {"recap": {"advantage": {"prediction_batch_size": 0}}},
            {"recap": {"advantage": {"dropout_probability": -1}}},
            {"recap": {"advantage": {"positive_fraction": 2}}},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                QwenRecap(config)
        self.load_model.assert_not_called()

    def test_missing_labels_and_inconsistent_episodes_fail(self):
        model = QwenRecap()
        with self.assertRaisesRegex(ValueError, "return"):
            model([{"image": self.batch[0]["image"], "lang": "test"}])
        with self.assertRaises(ValueError):
            model.predict_advantages([self.batch], torch.zeros(1, 1))
        with self.assertRaises(ValueError):
            model.predict_advantages([[]], torch.zeros(1, 1))
        with self.assertRaises(ValueError):
            model.make_condition(torch.ones(1), torch.zeros(1, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
