"""Exercise the existing Qwen wrapper with a tiny, real Qwen3-VL transformer."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch
from transformers import BatchFeature, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from vla.model.modules.recap import QwenValueModel, compute_advantages


def tiny_qwen():
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True},
        ),
        vision_config=dict(
            depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
            patch_size=2, temporal_patch_size=1, spatial_merge_size=2,
            out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[0, 1],
        ),
        image_token_id=3, video_token_id=4, vision_start_token_id=5, vision_end_token_id=6,
    )
    return Qwen3VLForConditionalGeneration(config)


class ValueModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.backbone = tiny_qwen()
        self.processor = MagicMock()
        # One image token replaces four 2x2 patches after spatial merging.
        self.processor.apply_chat_template.return_value = BatchFeature({
            "input_ids": torch.tensor([[0, 1, 5, 3, 6, 7], [1, 5, 3, 6, 7, 8]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
            "mm_token_type_ids": torch.tensor([[0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0]]),
            "pixel_values": torch.randn(8, 12),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
        })
        self.images = [[object()], [object()]]
        self.instructions = ["fold the shirt", "close the box"]
        model_patch = patch(
            "vla.model.modules.vlm.Qwen.Qwen3VLForConditionalGeneration.from_pretrained",
            return_value=self.backbone,
        )
        processor_patch = patch(
            "vla.model.modules.vlm.Qwen.AutoProcessor.from_pretrained", return_value=self.processor,
        )
        self.load_model = model_patch.start()
        self.load_processor = processor_patch.start()
        self.addCleanup(model_patch.stop)
        self.addCleanup(processor_patch.stop)

    def test_default_id_and_multimodal_training(self):
        model = QwenValueModel()
        self.assertEqual(self.load_model.call_args.args[0], "Qwen/Qwen3-VL-2B-Instruct")
        self.load_processor.assert_called_once_with("Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(self.processor.tokenizer.padding_side, "left")
        # No language vocabulary projection is needed for the critic.
        with patch.object(self.backbone.lm_head, "forward", side_effect=AssertionError("unused LM head")):
            result = model(self.images, self.instructions, returns=torch.tensor([-0.8, -0.2]))
        self.assertEqual(result.logits.shape, (2, 201))
        self.assertEqual(result.values.shape, (2,))
        self.assertTrue(((result.values >= -1) & (result.values <= 0)).all())
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        self.assertGreater(model.value_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.backbone.model.language_model.embed_tokens.weight.grad.abs().sum().item(), 0)
        vision_grads = [p.grad for p in self.backbone.model.visual.parameters() if p.grad is not None]
        self.assertTrue(any(g.abs().sum() > 0 for g in vision_grads))
        messages = self.processor.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0][0]["content"][-1]["text"], self.instructions[0])

    def test_pool_last_nonpadding_token_for_either_padding_side(self):
        model = QwenValueModel(num_bins=3)
        hidden = torch.randn(2, 6, 32)
        self.processor.apply_chat_template.return_value["attention_mask"] = torch.tensor([
            [0, 0, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0],
        ])
        with patch.object(model.qwen_vl_interface, "forward_features", return_value=SimpleNamespace(last_hidden_state=hidden)):
            result = model(self.images, self.instructions)
        expected = model.value_head(torch.stack([hidden[0, 5], hidden[1, 2]]))
        torch.testing.assert_close(result.logits, expected)

    def test_frozen_backbone_stays_in_eval_and_head_trains(self):
        model = QwenValueModel(freeze_backbone=True)
        model.train()
        self.assertFalse(model.qwen_vl_interface.training)
        self.assertTrue(model.value_head.training)
        result = model(self.images, self.instructions, returns=torch.tensor([-1.0, 0.0]))
        result.loss.backward()
        self.assertIsNotNone(model.value_head.weight.grad)
        self.assertTrue(all(p.grad is None for p in self.backbone.parameters()))

    def test_prediction_connects_to_advantages_and_restores_mode(self):
        model = QwenValueModel()
        values = model.predict_values(self.images, self.instructions)
        self.assertFalse(values.requires_grad)
        self.assertTrue(model.training)
        self.assertTrue(model.qwen_vl_interface.training)
        result = compute_advantages(torch.tensor([[-0.1, 0.0]]), values.reshape(1, 2))
        torch.testing.assert_close(result, torch.tensor([[-0.1, 0.0]]) - values.reshape(1, 2))
        model.eval()
        torch.testing.assert_close(model.predict_values(self.images, self.instructions), values)
        self.assertFalse(model.training)

    def test_return_bins_and_invalid_targets(self):
        model = QwenValueModel(num_bins=5)
        self.assertEqual(model.discretize_returns(torch.tensor([-1.0, -0.75, -0.5, -0.25, 0.0])).tolist(), [0, 1, 2, 3, 4])
        for targets in (torch.tensor([-1.01, 0.0]), torch.tensor([float("nan"), 0.0]), torch.ones(2, 1)):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                model(self.images, self.instructions, returns=targets)
        with self.assertRaises(ValueError):
            model([], [])

    def test_checkpoint_contains_head_backbone_and_bins(self):
        model = QwenValueModel(num_bins=5, value_min=-2)
        expected = model.predict_values(self.images, self.instructions)
        state = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            model.value_head.bias.add_(torch.arange(5.0))
        model.load_state_dict(state)
        torch.testing.assert_close(model.predict_values(self.images, self.instructions), expected)
        self.assertIn("bin_values", state)
        self.assertIn("value_head.weight", state)
        self.assertTrue(any(key.startswith("qwen_vl_interface.model.") for key in state))


if __name__ == "__main__":
    unittest.main()
