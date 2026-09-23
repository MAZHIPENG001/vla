"""OFT RECAP policy tests with a tiny real Qwen3-VL and deterministic image inputs."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import BatchFeature

from vla.model.framework.base_framework import build_framework
from vla.model.framework.VLM4A.QwenOFT import Qwenvl_OFT
from vla.model.framework.VLM4A.QwenOFTRecap import Qwenvl_OFT_Recap
from vla.model.modules.recap import AdvantageCondition, compute_advantage_condition
from vla.model.modules.recap.tests.test_value_model import tiny_qwen


class OFTRecapTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.backbone = tiny_qwen()
        self.processor = MagicMock()
        self.processor.tokenizer.return_value = {"input_ids": [7]}
        self.processor.apply_chat_template.side_effect = self._process
        for target, value in (
            ("Qwen3VLForConditionalGeneration", self.backbone),
            ("AutoProcessor", self.processor),
        ):
            patcher = patch("vla.model.modules.vlm.Qwen." + target + ".from_pretrained", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.cfg = OmegaConf.create({
            "framework": {
                "name": "QwenOFTRecap",
                "qwenvl": {"base_vlm": "Qwen/policy", "attn_implementation": "sdpa"},
                "action_model": {"action_horizon": 2, "action_dim": 2},
                "advantage_conditioning": {"dropout_probability": 0.0},
            },
            "recap": {"qwenvl": {"base_vlm": "Qwen/independent-critic"}},
            "datasets": {"vla_data": {"CoT_prompt": "Task: {instruction}. Template end."}},
        })
        self.batch = [{
            "image": [Image.new("RGB", (8, 8))], "lang": "fold shirt",
            "action": np.zeros((3, 2), dtype=np.float32),
            "state": np.zeros((1, 7), dtype=np.float32),
            "subtask": "grasp sleeve", "advantage_indicator": flag,
        } for flag in (True, False)]

    @staticmethod
    def _process(messages, **kwargs):
        sequences = []
        for message in messages:
            prompt = message[0]["content"][-1]["text"]
            condition = [8] if "Advantage: positive" in prompt else [9] if "Advantage: negative" in prompt else []
            sequences.append([1, 5, 3, 6] + condition + [7] * prompt.count("🔍"))
        width = max(map(len, sequences))
        ids = torch.tensor([[0] * (width - len(sequence)) + sequence for sequence in sequences])
        batch = len(messages)
        return BatchFeature({
            "input_ids": ids, "attention_mask": (ids != 0).long(),
            "mm_token_type_ids": (ids == 3).long(), "pixel_values": torch.ones(batch * 4, 12),
            "image_grid_thw": torch.tensor([[1, 2, 2]]).expand(batch, -1),
        })

    def prompts(self):
        return [m[0]["content"][-1]["text"] for m in self.processor.apply_chat_template.call_args.args[0]]

    def test_factory_prompt_order_and_training_gradients(self):
        model = build_framework(self.cfg)
        self.assertIsInstance(model, Qwenvl_OFT_Recap)
        self.assertEqual(model.config.recap.qwenvl.base_vlm, "Qwen/independent-critic")
        self.assertFalse(hasattr(model, "value_model"))
        result = model(self.batch)
        self.assertEqual(set(result), {"action_loss"})
        self.assertTrue(torch.isfinite(result["action_loss"]))
        result["action_loss"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.action_model.parameters()))
        self.assertGreater(self.backbone.model.language_model.embed_tokens.weight.grad.abs().sum().item(), 0)
        for prompt, label in zip(self.prompts(), ("positive", "negative")):
            self.assertLess(prompt.index("[STATE]"), prompt.index("Template end."))
            self.assertLess(prompt.index("Template end."), prompt.index("Subtask:"))
            self.assertLess(prompt.index("Subtask:"), prompt.index("Advantage:"))
            self.assertLess(prompt.index("Advantage:"), prompt.index("<action>"))
            self.assertEqual(prompt.count("Advantage:"), 1)
            self.assertIn("Advantage: " + label, prompt)

    def test_dropout_omits_condition_and_preserves_inputs(self):
        self.cfg.framework.advantage_conditioning.dropout_probability = 1.0
        model = Qwenvl_OFT_Recap(self.cfg)
        model(self.batch)
        self.assertTrue(all("Advantage:" not in p for p in self.prompts()))
        self.assertEqual([e["advantage_indicator"] for e in self.batch], [True, False])
        model.eval()
        model(self.batch)
        self.assertIn("Advantage: negative", self.prompts()[1])

    def test_recap_mask_is_authoritative_not_dropped_twice(self):
        self.cfg.framework.advantage_conditioning.dropout_probability = 1.0
        model = Qwenvl_OFT_Recap(self.cfg)
        condition = compute_advantage_condition(
            torch.tensor([1.0, -1.0]), torch.tensor([0, 0]), {0: 0.0}, training=False,
        )
        model(self.batch, advantage_condition=condition)
        self.assertIn("Advantage: positive", self.prompts()[0])
        self.assertIn("Advantage: negative", self.prompts()[1])
        condition = AdvantageCondition(torch.tensor([True, False]), torch.tensor([False, True]))
        model(self.batch, advantage_condition=condition)
        self.assertNotIn("Advantage:", self.prompts()[0])
        self.assertIn("Advantage: negative", self.prompts()[1])

    def test_missing_labels_raw_scores_and_invalid_masks_fail(self):
        model = Qwenvl_OFT_Recap(self.cfg)
        for bad in (None, 0.1, "positive", 1):
            examples = deepcopy(self.batch)
            if bad is None:
                del examples[0]["advantage_indicator"]
            else:
                examples[0]["advantage_indicator"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model(examples)
        invalid = AdvantageCondition(torch.ones(2, 1, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            model(self.batch, advantage_condition=invalid)

    def test_sft_and_interventions_force_positive(self):
        examples = deepcopy(self.batch)
        examples[1]["is_intervention"] = True
        model = Qwenvl_OFT_Recap(self.cfg)
        model(examples)
        self.assertTrue(all("Advantage: positive" in p for p in self.prompts()))
        model.config.framework.advantage_conditioning.sft_positive = True
        del examples[0]["advantage_indicator"]
        examples[1]["is_intervention"] = False
        model(examples)
        self.assertTrue(all("Advantage: positive" in p for p in self.prompts()))

    def test_inference_positive_negative_and_unconditional_without_critic(self):
        self.cfg.framework.advantage_conditioning.dropout_probability = 1.0
        model = Qwenvl_OFT_Recap(self.cfg).eval()
        predictions = {}
        for mode in ("positive", "negative", "unconditional"):
            predictions[mode] = model.predict_action(self.batch, advantage=mode)["normalized_actions"]
            self.assertEqual(predictions[mode].shape, (2, 2, 2))
            if mode == "unconditional":
                self.assertTrue(all("Advantage:" not in p for p in self.prompts()))
            else:
                self.assertTrue(all("Advantage: " + mode in p for p in self.prompts()))
        self.assertFalse(np.allclose(predictions["positive"], predictions["negative"]))
        model.predict_action(self.batch)  # Ignore stale negative training labels at deployment.
        self.assertTrue(all("Advantage: positive" in p for p in self.prompts()))

    def test_dropout_seed_and_legacy_oft_checkpoint_compatibility(self):
        model = Qwenvl_OFT_Recap(self.cfg)
        model.config.framework.advantage_conditioning.dropout_probability = 0.3
        examples = self.batch * 200
        first = model._condition_texts(examples, inference=False, generator=torch.Generator().manual_seed(42))
        second = model._condition_texts(examples, inference=False, generator=torch.Generator().manual_seed(42))
        self.assertEqual(first, second)
        self.assertGreater(first.count(""), 70)
        self.assertLess(first.count(""), 170)
        legacy_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg))
        legacy_cfg.framework.name = "QwenOFT"
        original = Qwenvl_OFT(legacy_cfg)
        model.load_state_dict(original.state_dict(), strict=True)
        original(self.batch)
        self.assertTrue(all("Advantage:" not in p for p in self.prompts()))

    def test_external_task_config(self):
        path = Path(__file__).resolve().parents[5] / "examples/LIBERO/train_files/starvla_oft_recap_libero.yaml"
        cfg = OmegaConf.load(path)
        cfg.framework.qwenvl.base_vlm = "Qwen/policy"
        model = build_framework(cfg)
        self.assertIsInstance(model, Qwenvl_OFT_Recap)
        self.assertEqual(model.config.recap.value_model.num_bins, 201)


if __name__ == "__main__":
    unittest.main()
