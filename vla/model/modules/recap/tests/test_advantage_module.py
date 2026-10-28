"""Numerical and lifecycle checks for the standalone advantage component."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
from torch import nn

from vla.model.modules.recap import AdvantageCondition, AdvantageModule
from vla.model.modules.recap.recap import QwenRecap


class AdvantageModuleTests(unittest.TestCase):
    def test_condition_class_methods_from_rewards_to_text(self):
        lengths = torch.tensor([3, 2])
        rewards = AdvantageCondition.episode_rewards(
            lengths, torch.tensor([True, False]),
            max_steps=3, failure_penalty=4, normalization=10,
        )
        torch.testing.assert_close(rewards, torch.tensor([[-0.1, -0.1, 0.], [-0.1, -0.4, 0.]]))
        values = AdvantageCondition.values_from_logits(
            torch.zeros(2, 3, 3, requires_grad=True), torch.tensor([-1., -0.5, 0.]),
        )
        torch.testing.assert_close(values, torch.full((2, 3), -0.5))
        self.assertFalse(values.requires_grad)
        advantages = AdvantageCondition.compute_advantages(rewards, values, lengths=lengths)
        torch.testing.assert_close(advantages, torch.tensor([[0.3, 0.4, 0.5], [0., 0.1, 0.]]))
        valid = torch.arange(3)[None, :] < lengths[:, None]
        task_ids = torch.tensor([[0] * 3, [1] * 3])
        thresholds = AdvantageCondition.estimate_task_thresholds(
            advantages, task_ids, positive_fraction=0.5, valid_mask=valid,
        )
        condition = AdvantageCondition.compute_advantage_condition(
            advantages, task_ids, thresholds, valid_mask=valid, training=False,
        )
        self.assertIsInstance(condition, AdvantageCondition)
        torch.testing.assert_close(condition.indicator, torch.tensor([[False, False, True], [False, True, False]]))
        self.assertEqual(condition.to_text(), [
            "Advantage: negative", "Advantage: negative", "Advantage: positive",
            "Advantage: negative", "Advantage: positive", "",
        ])

    def test_padded_episode_formulas_and_detached_mixed_precision(self):
        rewards = torch.tensor(
            [[-1, -1, -1, 0], [-1, -5, float("nan"), float("nan")]],
            dtype=torch.float16, requires_grad=True,
        )
        values = torch.tensor(
            [[-4, -3, -2, -1], [-7, -4, float("nan"), float("nan")]],
            dtype=torch.float16, requires_grad=True,
        )
        expected = {
            "posttrain": [[0, 0, 1, 1], [1, -1, 0, 0]],
            "pretrain": [[1, 0, -1, -2], [1, -2, 0, 0]],
            "return_to_go": [[1, 1, 1, 1], [1, -1, 0, 0]],
        }
        for mode, result in expected.items():
            with self.subTest(mode=mode):
                module = AdvantageModule(mode=mode, n_steps=2)
                actual = module(rewards, values, lengths=torch.tensor([4, 2]))
                torch.testing.assert_close(actual, torch.tensor(result, dtype=torch.float32))
                self.assertFalse(actual.requires_grad)

    def test_calibration_interventions_and_train_eval_dropout(self):
        advantages = torch.tensor([[0., 1., 2., 3.], [10., 20., 30., 40.], [float("nan")] * 4])
        task_ids = torch.tensor([[0] * 4, [1] * 4, [99] * 4])
        valid = task_ids != 99
        interventions = torch.zeros_like(valid)
        interventions[0, 0] = True
        module = AdvantageModule(dropout_probability=1.0)
        thresholds = module.fit_thresholds(advantages, task_ids, valid_mask=valid)
        self.assertAlmostEqual(thresholds[0], 1.8, places=5)
        self.assertAlmostEqual(thresholds[1], 28., places=5)
        thresholds[0] = 100.  # Returned thresholds must not mutate fitted state.
        condition = module.make_condition(
            advantages, task_ids, valid_mask=valid, intervention_mask=interventions,
        )
        expected = torch.tensor([[True, False, True, True], [False, False, True, True], [False] * 4])
        torch.testing.assert_close(condition.indicator, expected)
        self.assertFalse(condition.conditioning_mask.any())
        self.assertEqual(condition.to_text(), [""] * 12)

        module.eval()
        condition = module.make_condition(
            advantages, task_ids, valid_mask=valid, intervention_mask=interventions,
        )
        torch.testing.assert_close(condition.indicator, expected)
        torch.testing.assert_close(condition.conditioning_mask, valid)
        self.assertEqual(condition.to_text()[:2], ["Advantage: positive", "Advantage: negative"])

    def test_stage_defaults_and_explicit_positive_fraction(self):
        advantages = torch.arange(10, dtype=torch.float32)
        task_ids = torch.zeros(10, dtype=torch.long)
        for mode, expected in (("posttrain", 5.4), ("pretrain", 6.3), ("return_to_go", 6.3)):
            with self.subTest(mode=mode):
                module = AdvantageModule(mode=mode)
                self.assertAlmostEqual(module.fit_thresholds(advantages, task_ids)[0], expected, places=5)
        module = AdvantageModule(positive_fraction=0.1)
        self.assertAlmostEqual(module.fit_thresholds(advantages, task_ids)[0], 8.1, places=5)

    def test_configured_thresholds_and_missing_calibration(self):
        advantages = torch.tensor([0., 1.])
        task_ids = torch.tensor([3, 3])
        with self.assertRaisesRegex(ValueError, "missing calibrated threshold"):
            AdvantageModule().make_condition(advantages, task_ids)
        thresholds = {"3": 0.0}
        module = AdvantageModule(thresholds=thresholds).eval()
        thresholds["3"] = 100.
        condition = module.make_condition(advantages, task_ids)
        torch.testing.assert_close(condition.indicator, torch.tensor([False, True]))

    def test_invalid_configuration(self):
        for settings in (
            {"mode": "invalid"}, {"n_steps": 0}, {"n_steps": True},
            {"positive_fraction": 1.0}, {"positive_fraction": float("nan")},
            {"dropout_probability": -0.1}, {"thresholds": {0: float("inf")}},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                AdvantageModule(**settings)


class _ValueModelStub(nn.Module):
    """Avoid loading a VLM when checking component composition."""

    def __init__(self, **kwargs):
        super().__init__()
        self.value_head = nn.Linear(4, kwargs["num_bins"])


class QwenRecapAdvantageTests(unittest.TestCase):
    def test_prediction_mode_propagation_and_calibrated_config_roundtrip(self):
        config = {"recap": {"advantage": {
            "n_steps": 1, "prediction_batch_size": 2, "dropout_probability": 1.0,
        }}}
        with patch("vla.model.modules.recap.recap.QwenValueModel", _ValueModelStub):
            recap = QwenRecap(config)
            self.assertIsInstance(recap.advantage, AdvantageModule)
            rewards = torch.tensor([[-1., -1., 0.], [-1., 0., float("nan")]])
            episodes = [[{}] * 3, [{}] * 2]
            with patch.object(recap, "predict_value", side_effect=lambda examples: {
                "values": torch.full((len(examples),), -2.),
            }) as predict:
                result = recap.predict_advantages(episodes, rewards)
            self.assertEqual([len(call.args[0]) for call in predict.call_args_list], [2, 2, 1])
            torch.testing.assert_close(
                result["advantages"], torch.tensor([[-1., -1., 2.], [-1., 2., 0.]]),
            )
            task_ids = torch.zeros_like(rewards, dtype=torch.long)
            advantages, valid = result["advantages"], result["valid_mask"]
            thresholds = recap.fit_thresholds(advantages, task_ids, valid_mask=valid)
            self.assertEqual(dict(recap.config.recap.advantage.thresholds), thresholds)
            self.assertFalse(recap.make_condition(advantages, task_ids, valid_mask=valid).conditioning_mask.any())
            recap.eval()
            self.assertFalse(recap.advantage.training)
            condition = recap.make_condition(advantages, task_ids, valid_mask=valid)
            torch.testing.assert_close(condition.conditioning_mask, valid)

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "calibrated.yaml"
                OmegaConf.save(recap.config, path)
                restored = QwenRecap(path).eval()
            self.assertEqual(restored.advantage.thresholds, thresholds)
            restored_condition = restored.make_condition(advantages, task_ids, valid_mask=valid)
            torch.testing.assert_close(restored_condition.indicator, condition.indicator)
            recap.train()
            self.assertTrue(recap.advantage.training)


if __name__ == "__main__":
    unittest.main()
