"""Numerical checks against hand calculations and a scalar n-step reference."""

import unittest

import torch

from vla.model.modules.recap import (
    compute_advantage_condition,
    compute_advantages,
    episode_rewards,
    estimate_task_thresholds,
    values_from_logits,
)


class AdvantageTests(unittest.TestCase):
    def test_distribution_expectation(self):
        logits = torch.tensor([[0.25, 0.5, 0.25], [0.1, 0.2, 0.7]]).log().requires_grad_()
        values = values_from_logits(logits)
        torch.testing.assert_close(values, torch.tensor([-0.5, -0.2]))
        self.assertFalse(values.requires_grad)
        torch.testing.assert_close(
            values_from_logits(logits, torch.tensor([-4.0, -2.0, 0.0])), values * 4
        )
        torch.testing.assert_close(values_from_logits(torch.zeros(2, 201)), torch.full((2,), -0.5))

    def test_rewards_success_failure_and_padding(self):
        actual = episode_rewards(
            torch.tensor([4, 2, 1]), torch.tensor([True, False, True]),
            max_steps=5, failure_penalty=8, normalization=torch.tensor([4, 10, 1]),
        )
        torch.testing.assert_close(actual, torch.tensor([
            [-0.25, -0.25, -0.25, 0, 0], [-0.1, -0.8, 0, 0, 0], [0, 0, 0, 0, 0],
        ]))

    def test_hand_computed_modes(self):
        rewards = torch.tensor([[-1.0, -1.0, 0.0]])
        values = torch.tensor([[-2.5, -1.5, -0.2]], requires_grad=True)
        expected = {
            "posttrain": [[0.3, 0.5, 0.2]],
            "pretrain": [[0.5, -0.5, -1.8]],
            "return_to_go": [[0.5, 0.5, 0.2]],
        }
        for mode, target in expected.items():
            with self.subTest(mode=mode):
                actual = compute_advantages(rewards, values, mode=mode, n_steps=2)
                torch.testing.assert_close(actual, torch.tensor(target))
                self.assertFalse(actual.requires_grad)

    def test_n_step_against_scalar_reference(self):
        rng = torch.Generator().manual_seed(12)
        rewards = torch.randn(3, 75, generator=rng, dtype=torch.float64)
        values = torch.randn(3, 75, generator=rng, dtype=torch.float64)
        lengths = torch.tensor([75, 51, 1])
        for n_steps in (1, 2, 50, 100):
            expected = torch.zeros_like(rewards)
            for b, length in enumerate(lengths.tolist()):
                for t in range(length):
                    end = min(t + n_steps, length)
                    bootstrap = values[b, end] if end < length else 0
                    expected[b, t] = rewards[b, t:end].sum() + bootstrap - values[b, t]
            actual = compute_advantages(rewards, values, lengths=lengths, n_steps=n_steps)
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            compute_advantages(rewards, values, lengths=lengths),
            compute_advantages(rewards, values, lengths=lengths, n_steps=50),
        )

    def test_terminal_penalty_not_dropped_and_nan_padding_ignored(self):
        rewards = torch.tensor([[-1.0, -8.0, float("nan")]])
        values = torch.tensor([[-5.0, -6.0, float("nan")]])
        lengths = torch.tensor([2])
        torch.testing.assert_close(
            compute_advantages(rewards, values, lengths=lengths, n_steps=1),
            torch.tensor([[-2.0, -2.0, 0.0]]),
        )
        torch.testing.assert_close(
            compute_advantages(rewards, values, lengths=lengths),
            torch.tensor([[-4.0, -2.0, 0.0]]),
        )
        for mode in ("pretrain", "return_to_go"):
            self.assertTrue(torch.isfinite(compute_advantages(rewards, values, lengths=lengths, mode=mode)).all())

    def test_half_precision_accumulates_in_float32(self):
        rewards = torch.full((1, 4096), -1.0, dtype=torch.float16)
        values = torch.zeros_like(rewards)
        result = compute_advantages(rewards, values, n_steps=1)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.full((1, 4096), -1.0))

    def test_threshold_fractions_and_task_separation(self):
        advantages = torch.cat((torch.arange(100.0), torch.arange(100.0) + 200))
        tasks = torch.tensor([0] * 100 + [7] * 100)
        for stage, fraction, count in (("pretrain", None, 30), ("posttrain", None, 40), ("posttrain", 0.1, 10)):
            thresholds = estimate_task_thresholds(advantages, tasks, stage=stage, positive_fraction=fraction)
            condition = compute_advantage_condition(advantages, tasks, thresholds, training=False)
            for task in (0, 7):
                self.assertEqual(condition.indicator[tasks == task].sum().item(), count)

    def test_strict_comparison_and_interventions(self):
        advantages = torch.tensor([-1.0, 0.0, 1.0, -2.0])
        tasks = torch.zeros(4, dtype=torch.long)
        condition = compute_advantage_condition(
            advantages, tasks, {0: 0}, training=False,
            intervention_mask=torch.tensor([False, False, False, True]),
        )
        self.assertEqual(condition.indicator.tolist(), [False, False, True, True])
        self.assertEqual(condition.to_text(), ["Advantage: negative"] * 2 + ["Advantage: positive"] * 2)
        ties = torch.ones(10)
        task_ids = torch.zeros(10, dtype=torch.long)
        thresholds = estimate_task_thresholds(ties, task_ids)
        self.assertFalse(compute_advantage_condition(ties, task_ids, thresholds).indicator.any())

    def test_dropout_removes_condition_without_relabeling(self):
        advantages = torch.tensor([-1.0, 1.0])
        tasks = torch.zeros(2, dtype=torch.long)
        dropped = compute_advantage_condition(advantages, tasks, {0: 0}, dropout_probability=1)
        self.assertEqual(dropped.indicator.tolist(), [False, True])
        self.assertEqual(dropped.to_text(), ["", ""])
        retained = compute_advantage_condition(advantages, tasks, {0: 0}, training=False, dropout_probability=1)
        self.assertTrue(retained.conditioning_mask.all())
        advantages = torch.ones(10000)
        tasks = torch.zeros(10000, dtype=torch.long)
        results = [compute_advantage_condition(
            advantages, tasks, {0: 0}, generator=torch.Generator().manual_seed(23)
        ) for _ in range(2)]
        self.assertTrue(torch.equal(results[0].conditioning_mask, results[1].conditioning_mask))
        self.assertAlmostEqual(results[0].conditioning_mask.float().mean().item(), 0.7, delta=0.02)

    def test_padding_excluded_from_calibration_and_conditions(self):
        advantages = torch.tensor([[0.0, 1.0, float("nan")]])
        tasks = torch.tensor([[3, 3, -1]])
        valid = torch.tensor([[True, True, False]])
        thresholds = estimate_task_thresholds(advantages, tasks, valid_mask=valid)
        self.assertEqual(set(thresholds), {3})
        condition = compute_advantage_condition(
            advantages, tasks, thresholds, valid_mask=valid, training=False,
            intervention_mask=torch.ones_like(valid),
        )
        self.assertEqual(condition.indicator.tolist(), [[True, True, False]])
        self.assertEqual(condition.to_text(), ["Advantage: positive", "Advantage: positive", ""])

    def test_invalid_inputs(self):
        advantages = torch.ones(2)
        tasks = torch.zeros(2, dtype=torch.long)
        cases = [
            lambda: values_from_logits(torch.tensor([float("nan"), 1.0])),
            lambda: values_from_logits(torch.zeros(3), torch.tensor([0.0, -1.0, -2.0])),
            lambda: compute_advantages(torch.ones(2, 3), torch.ones(2, 4)),
            lambda: compute_advantages(torch.ones(2, 3), torch.ones(2, 3), n_steps=0),
            lambda: compute_advantages(torch.ones(2, 3), torch.ones(2, 3), lengths=torch.tensor([0, 3])),
            lambda: compute_advantages(torch.ones(2, 3), torch.ones(2, 3), mode="invalid"),
            lambda: estimate_task_thresholds(advantages, tasks, positive_fraction=0),
            lambda: estimate_task_thresholds(advantages, tasks, valid_mask=torch.zeros(2, dtype=torch.bool)),
            lambda: compute_advantage_condition(advantages, tasks, {}),
            lambda: compute_advantage_condition(advantages, tasks, {0: float("nan")}),
            lambda: compute_advantage_condition(advantages, tasks, {0: 0}, dropout_probability=2),
            lambda: compute_advantage_condition(advantages, tasks.float(), {0: 0}),
            lambda: episode_rewards(torch.tensor([2]), torch.tensor([True]), max_steps=2, failure_penalty=3, normalization=0),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                case()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matches_cpu(self):
        rewards = torch.tensor([[-1.0, -1.0, 0.0]])
        values = torch.tensor([[-2.5, -1.5, -0.2]])
        cpu = compute_advantages(rewards, values)
        gpu = compute_advantages(rewards.cuda(), values.cuda())
        torch.testing.assert_close(gpu.cpu(), cpu)
        tasks = torch.zeros_like(gpu, dtype=torch.long)
        thresholds = estimate_task_thresholds(gpu, tasks)
        condition = compute_advantage_condition(gpu, tasks, thresholds, training=False)
        self.assertEqual(condition.indicator.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
