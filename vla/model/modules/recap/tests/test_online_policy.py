"""CPU regression checks for staged RECAP training; no model downloads."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn

from vla.model.framework.VLM4A.QwenOFT import Qwenvl_OFT
from vla.model.framework.VLM4A.QwenOFTRecap import Qwenvl_OFT_Recap
from vla.model.modules.recap.recap import QwenRecap


class TinyVLM(nn.Module):
    """Replace the pretrained encoder, retaining the real value head and CE."""

    def __init__(self):
        super().__init__()
        self.model = nn.Embedding(2, 4)
        self.model.config = SimpleNamespace(hidden_size=4)
        self.model.device = torch.device("cpu")

    def build_qwenvl_inputs(self, *, images, instructions):
        tokens = torch.ones(len(images), 2, dtype=torch.long)
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}

    def forward_features(self, *, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.model(input_ids))


class OnlinePolicyTest(unittest.TestCase):
    def setUp(self):
        self.encoder_patch = patch(
            "vla.model.modules.recap.value_model.get_vlm_model",
            side_effect=lambda config: TinyVLM(),
        )
        self.encoder_patch.start()
        self.addCleanup(self.encoder_patch.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cfg = OmegaConf.create({
            "framework": {"advantage_conditioning": {
                "enabled": True, "freeze_critic": True,
                "dropout_probability": 0.0, "inference_condition": "positive",
            }},
            "recap": {"advantage": {"task_thresholds": {"demo:0": 0.0}}},
        })
        critic = QwenRecap(self.cfg)
        nn.init.zeros_(critic.value_model.value_head.weight)
        nn.init.zeros_(critic.value_model.value_head.bias)
        checkpoint = Path(directory.name) / "critic.pt"
        torch.save(critic.value_model.state_dict(), checkpoint)
        self.cfg.recap.value_model = {"checkpoint": str(checkpoint)}
        observation = {"image": [Image.new("RGB", (8, 8))], "lang": "Fold."}
        self.examples = [
            {**observation, "task_key": "demo:0", "advantage_reward": -0.1,
             "advantage_bootstrap": observation, "is_intervention": False},
            {**observation, "task_key": "demo:0", "advantage_reward": -0.1,
             "advantage_bootstrap": None, "is_intervention": False},
            {**observation, "task_key": "demo:0", "advantage_reward": -0.1,
             "advantage_bootstrap": observation, "is_intervention": True},
        ]

    def test_online_labels_use_bootstrap_terminal_and_intervention(self):
        critic = QwenRecap(self.cfg, online=True)
        critic.train()
        with patch.object(critic, "_forward_value", side_effect=AssertionError("value training")):
            result = critic(self.examples, online=True)
        torch.testing.assert_close(result["advantages"], torch.tensor([-0.1, 0.4, -0.1]))
        self.assertEqual(result["condition"].to_text(), [
            "Advantage: negative", "Advantage: positive", "Advantage: positive",
        ])
        self.assertNotIn("value_loss", result)
        self.assertFalse(result["advantages"].requires_grad)
        self.assertFalse(critic.value_model.training)
        self.assertTrue(all(not p.requires_grad for p in critic.parameters()))

    def test_condition_dropout_follows_policy_mode_with_critic_in_eval(self):
        self.cfg.framework.advantage_conditioning.dropout_probability = 1.0
        critic = QwenRecap(self.cfg, online=True)
        critic.train()
        self.assertEqual(critic(self.examples, online=True)["condition"].to_text(), [""] * 3)
        critic.eval()
        self.assertEqual(critic(self.examples, online=True)["condition"].to_text(), [
            "Advantage: negative", "Advantage: positive", "Advantage: positive",
        ])
        self.assertFalse(critic.value_model.training)

    def test_standalone_value_training_still_updates_value_head(self):
        critic = QwenRecap(self.cfg)
        critic.train()
        output = critic([{**self.examples[0], "return": -0.25}])
        output["value_loss"].backward()
        gradient = critic.value_model.value_head.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in critic.qwen_vl_interface.parameters()))

    def test_policy_loss_only_updates_action_parameters(self):
        # Replace the CUDA OFT encoder/head with a small conditional regression
        # while exercising the actual RECAP constructor, conditions and forward.
        def init_policy(policy, config, **kwargs):
            nn.Module.__init__(policy)
            policy.config = config
            policy.action_model = nn.Embedding(3, 1)
            nn.init.constant_(policy.action_model.weight, 0.5)

        def action_forward(policy, examples, **kwargs):
            texts = policy._condition_texts(
                examples, inference=False,
                advantage_condition=kwargs.get("advantage_condition"),
            )
            codes = {"": 0, "Advantage: negative": 1, "Advantage: positive": 2}
            inputs = torch.tensor([codes[text] for text in texts])
            return {"action_loss": policy.action_model(inputs).abs().mean()}

        with patch.object(Qwenvl_OFT, "__init__", init_policy), \
             patch.object(Qwenvl_OFT, "forward", action_forward):
            policy = Qwenvl_OFT_Recap(self.cfg)
            policy.train()
            before = {k: v.clone() for k, v in policy.online_advantage.state_dict().items()}
            optimizer = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=0.1)
            output = policy(self.examples)
            self.assertIs(output["loss"], output["action_loss"])
            self.assertNotIn("value_loss", output)
            output["loss"].backward()
            gradients = policy.action_model.weight.grad[:, 0]
            self.assertEqual(gradients[0].item(), 0.0)
            self.assertGreater(gradients[1].item(), 0.0)
            self.assertGreater(gradients[2].item(), 0.0)
            optimizer.step()
            self.assertTrue(all(p.grad is None for p in policy.online_advantage.parameters()))
            for key, value in policy.online_advantage.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            self.assertFalse(policy.online_advantage.value_model.training)

    def test_joint_training_configs_fail_before_loading_policy(self):
        for option, value, error in (
            ("freeze_critic", False, "freeze_critic=true"),
            ("value_loss_weight", 1.0, "remove advantage_conditioning.value_loss_weight"),
        ):
            with self.subTest(option=option):
                config = OmegaConf.create(OmegaConf.to_container(self.cfg))
                config.framework.advantage_conditioning[option] = value
                with self.assertRaisesRegex(ValueError, error):
                    QwenRecap(config, online=True)
                with patch.object(Qwenvl_OFT, "__init__", side_effect=AssertionError("policy loading")):
                    with self.assertRaisesRegex(ValueError, error):
                        Qwenvl_OFT_Recap(config)


if __name__ == "__main__":
    unittest.main()
