"""CPU integration checks: regular loader -> online critic -> OFT prompt/loss."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from vla.dataloader.recap_targets import online_advantage_target
from vla.model.framework.VLM4A.QwenOFTRecap import Qwenvl_OFT_Recap
from vla.model.modules.recap.advantage import AdvantageCondition
from vla.model.modules.recap.recap import QwenRecap
from vla.model.modules.recap.tests.test_value_pipeline import (
    ParquetEpisodes, TinyInterface, config, fixture_loader,
)
from vla.training.trainer_utils.trainer_tools import build_param_lr_groups
from vla.training.trainer_utils.config_tracker import wrap_config


class TinyPolicyInterface(TinyInterface):
    def __init__(self, config):
        super().__init__(config)
        self.processor = SimpleNamespace(tokenizer=lambda *args, **kwargs: {"input_ids": [31]})
        self.suffixes = []

    def build_qwenvl_inputs(self, images, instructions, instruction_suffixes=None):
        self.suffixes = instruction_suffixes or [""] * len(instructions)
        tokens = [[31 if char == "🔍" else ord(char) % 30 + 1 for char in text + suffix]
                  for text, suffix in zip(instructions, self.suffixes)]
        size = max(map(len, tokens))
        ids = torch.tensor([[0] * (size - len(row)) + row for row in tokens])
        return {"input_ids": ids, "attention_mask": ids != 0}

    def forward(self, input_ids, attention_mask, **kwargs):
        return SimpleNamespace(hidden_states=[self.forward_features(input_ids, attention_mask).last_hidden_state])


class OnlinePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = config(self.root)
        self.cfg.framework = {
            "name": "QwenOFTRecap", "action_model": {"action_horizon": 2, "action_dim": 2},
            "advantage_conditioning": {
                "enabled": True, "source": "online", "freeze_critic": True,
                "dropout_probability": 0.0, "value_loss_weight": 0.5,
            },
        }
        self.cfg.trainer = {"learning_rate": {"base": 0.01, "online_advantage": 0.02}, "freeze_modules": ""}
        self.cfg.recap.advantage.n_steps = 2
        self.cfg.recap.advantage.task_thresholds = {"fixture:0": -0.1}
        self.cfg.datasets.vla_data.num_workers = 0
        self.cfg.datasets.vla_data.save_statistics = False
        self.ds = ParquetEpisodes(self.root / "data")
        self.encoder = patch("vla.model.modules.recap.value_model.get_vlm_model", TinyInterface)
        self.encoder.start()
        self.policy_encoder = patch("vla.model.framework.VLM4A.QwenOFT.get_vlm_model", TinyPolicyInterface)
        self.policy_encoder.start()
        # Seed critic weights only; no network/model download is needed.
        checkpoint = self.root / "critic.pt"
        torch.save(QwenRecap(self.cfg).value_model.state_dict(), checkpoint)
        self.cfg.recap.value_model.checkpoint = str(checkpoint)

    def tearDown(self):
        self.policy_encoder.stop()
        self.encoder.stop()
        self.temp.cleanup()

    def batch(self):
        loader = fixture_loader(self.cfg, [self.ds], profile=None)
        examples = [self.ds.get_sample(2, frame) for frame in (0, 3, 4)]
        for example in examples:
            example["action"] = np.zeros((2, 2), dtype=np.float32)
        return loader, examples

    def test_loader_builds_bootstrap_without_sidecar_or_future_actions(self):
        loader, examples = self.batch()
        self.assertEqual(loader.dataset.sampling_mode, "random")
        self.assertEqual(self.ds._advantage_labels.source, "none")
        self.assertNotIn("advantage_indicator", examples[0])
        self.assertAlmostEqual(examples[0]["advantage_reward"], -0.2)
        self.assertIn("frame 2", examples[0]["advantage_bootstrap"]["lang"])
        self.assertEqual(set(examples[0]["advantage_bootstrap"]), {"image", "lang"})
        self.assertIsNone(examples[1]["advantage_bootstrap"])
        self.assertIsNone(examples[2]["advantage_bootstrap"])
        self.cfg.datasets.vla_data.num_workers = 1
        self.cfg.datasets.vla_data.persistent_workers = False
        self.assertIn("advantage_reward", next(iter(fixture_loader(self.cfg, [self.ds], profile=None)))[0])

    def test_sampled_formulas_match_complete_episode_for_all_modes(self):
        self.batch()
        for mode in ("posttrain", "pretrain", "return_to_go"):
            for episode in self.ds.value_targets.episodes:
                values = torch.linspace(-0.7, -0.2, episode.length)
                expected = AdvantageCondition.compute_advantages(
                    episode.rewards[None], values[None], mode=mode, n_steps=2,
                )[0]
                for frame in range(episode.length):
                    reward, bootstrap = online_advantage_target(episode, frame, mode=mode, n_steps=2)
                    actual = reward - values[frame] + (values[bootstrap] if bootstrap is not None else 0)
                    torch.testing.assert_close(actual, expected[frame])

    def test_online_predictions_match_full_episode_and_condition_prompt(self):
        _, examples = self.batch()
        policy = Qwenvl_OFT_Recap(self.cfg).train()
        critic = policy.online_advantage.critic
        episode = self.ds.value_targets.by_episode[(self.ds.dataset_name, 2)]
        whole = [self.ds.get_sample(2, frame) for frame in range(episode.length)]
        expected = critic.predict_advantages([whole], episode.rewards[None])["advantages"][0]
        online = policy.online_advantage(examples)
        torch.testing.assert_close(online["advantages"], expected[[0, 3, 4]])
        output = policy(examples)
        expected_text = online["condition"].to_text()
        for suffix, text in zip(policy.qwen_vl_interface.suffixes, expected_text):
            self.assertIn(text, suffix)
        output["action_loss"].backward()
        self.assertTrue(any(p.grad is not None for p in policy.action_model.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in critic.parameters()))
        self.assertFalse(critic.training)
        groups = build_param_lr_groups(policy, self.cfg)
        self.assertFalse(any(group["name"] == "online_advantage" for group in groups))
        # Deployment uses a requested positive condition; no future data or critic forward.
        policy.eval()
        with patch.object(critic, "predict_value", side_effect=AssertionError("deployment called critic")):
            actions = policy.predict_action([{key: examples[0][key] for key in ("image", "lang")}])
        self.assertEqual(actions["normalized_actions"].shape, (1, 2, 2))
        self.assertIn("Advantage: positive", policy.qwen_vl_interface.suffixes[0])

    def test_unfrozen_critic_updates_head_or_full_backbone_and_roundtrips(self):
        _, examples = self.batch()
        for freeze_backbone in (True, False):
            with self.subTest(freeze_backbone=freeze_backbone):
                self.cfg.framework.advantage_conditioning.freeze_critic = False
                self.cfg.recap.value_model.freeze_backbone = freeze_backbone
                policy = Qwenvl_OFT_Recap(self.cfg).train()
                output = policy(examples)
                torch.testing.assert_close(output["loss"], output["action_loss"] + 0.5 * output["value_loss"])
                output["loss"].backward()
                value = policy.online_advantage.critic.value_model
                self.assertGreater(value.value_head.weight.grad.abs().sum().item(), 0)
                gradient = value.qwen_vl_interface.model.embedding.weight.grad
                self.assertEqual(gradient is None, freeze_backbone)
                if not freeze_backbone:
                    self.assertGreater(gradient.abs().sum().item(), 0)
                groups = build_param_lr_groups(policy, self.cfg)
                critic_ids = {id(p) for group in groups if group["name"] == "online_advantage" for p in group["params"]}
                self.assertIn(id(value.value_head.weight), critic_ids)
                before = value.value_head.weight.detach().clone()
                torch.optim.SGD(groups).step()
                self.assertFalse(torch.equal(before, value.value_head.weight))
                restored = Qwenvl_OFT_Recap(self.cfg)
                restored.load_state_dict(policy.state_dict(), strict=True)
                torch.testing.assert_close(restored.online_advantage.critic.value_model.value_head.weight, value.value_head.weight)

    def test_online_interventions_and_dropout_apply_once(self):
        _, examples = self.batch()
        self.cfg.recap.advantage.task_thresholds = {"fixture:0": 100.0}
        examples[0]["is_intervention"] = True
        policy = Qwenvl_OFT_Recap(self.cfg).train()
        policy(examples)
        self.assertIn("Advantage: positive", policy.qwen_vl_interface.suffixes[0])
        self.assertIn("Advantage: negative", policy.qwen_vl_interface.suffixes[1])
        policy.online_advantage.dropout_probability = 1.0
        policy(examples)
        self.assertTrue(all("Advantage:" not in suffix for suffix in policy.qwen_vl_interface.suffixes))
        policy.eval()
        condition = policy.online_advantage(examples)["condition"]
        self.assertTrue(condition.conditioning_mask.all())

    def test_disabled_mode_needs_neither_labels_nor_critic(self):
        self.cfg.framework.advantage_conditioning.enabled = False
        self.cfg.recap.value_model.checkpoint = "/missing/critic.pt"
        loader = fixture_loader(self.cfg, [self.ds], profile=None)
        self.assertIsNone(loader.dataset.value_targets)
        with patch("vla.model.modules.recap.online.QwenRecap", side_effect=AssertionError("loaded critic")):
            policy = Qwenvl_OFT_Recap(self.cfg)
        _, examples = self.batch()
        output = policy(examples)
        self.assertTrue(torch.isfinite(output["action_loss"]))
        self.assertTrue(all("Advantage:" not in suffix for suffix in policy.qwen_vl_interface.suffixes))

    def test_offline_mode_keeps_labels_dropout_and_external_conditions(self):
        self.cfg.framework.advantage_conditioning.source = "offline"
        policy = Qwenvl_OFT_Recap(self.cfg)
        examples = [{"advantage_indicator": True}, {"advantage_indicator": False}]
        self.assertEqual(policy._condition_texts(examples, inference=False),
                         ["Advantage: positive", "Advantage: negative"])
        policy.config.framework.advantage_conditioning.dropout_probability = 1.0
        self.assertEqual(policy._condition_texts(examples, inference=False), ["", ""])
        condition = AdvantageCondition(torch.tensor([True, False]), torch.tensor([True, True]))
        self.assertEqual(policy._condition_texts(examples, inference=False, advantage_condition=condition), condition.to_text())
        with self.assertRaisesRegex(ValueError, "missing advantage_indicator"):
            policy._condition_texts([{}], inference=False)

    def test_online_requires_checkpoint_calibration_and_nonzero_training_loss(self):
        self.cfg.recap.advantage.task_thresholds = {}
        with self.assertRaisesRegex(ValueError, "task_thresholds"):
            Qwenvl_OFT_Recap(self.cfg)
        self.cfg.recap.advantage.task_thresholds = {"fixture:0": 0.0}
        self.cfg.recap.value_model.checkpoint = None
        with self.assertRaisesRegex(ValueError, "trained"):
            Qwenvl_OFT_Recap(self.cfg)
        self.cfg.framework.advantage_conditioning.freeze_critic = False
        self.cfg.framework.advantage_conditioning.value_loss_weight = 0
        with self.assertRaisesRegex(ValueError, "value_loss_weight"):
            Qwenvl_OFT_Recap(self.cfg)

    def test_trainer_tracked_config_and_value_profile(self):
        tracked = wrap_config(self.cfg)
        loader = fixture_loader(tracked, [self.ds], profile=None)
        self.assertTrue(loader.dataset.data_cfg.return_online_advantage)
        policy = Qwenvl_OFT_Recap(tracked)
        self.assertIsNotNone(policy.online_advantage)
        # Independent value training/export must not inherit online policy inputs.
        value_loader = fixture_loader(tracked, [self.ds], profile="value")
        self.assertNotIn("advantage_bootstrap", value_loader.dataset[0])


if __name__ == "__main__":
    unittest.main()
