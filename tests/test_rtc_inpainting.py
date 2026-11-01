"""CPU checks for the paper's PiGDM equations and real PI sampler wiring."""

import math
import unittest

import torch

from vla.model.modules.action_model.rtc_inpainting import (
    rtc_guided_euler_step,
    rtc_prefix_weights,
)


class RTCPrefixWeightsTests(unittest.TestCase):
    def test_paper_soft_mask_uses_committed_prefix_overlap_and_new_tail(self):
        weights = rtc_prefix_weights(8, 6, 2).flatten()
        c = torch.tensor([4, 3, 2, 1], dtype=torch.float32) / 5
        expected = torch.cat((torch.ones(2), c * torch.expm1(c) / math.expm1(1), torch.zeros(2)))
        torch.testing.assert_close(weights, expected)
        self.assertTrue(torch.all(weights[2:5] > weights[3:6]))

    def test_zero_delay_and_delay_larger_than_old_cache(self):
        self.assertTrue(torch.all(rtc_prefix_weights(4, 0, 3) == 0))
        torch.testing.assert_close(
            rtc_prefix_weights(4, 2, 10).flatten(), torch.tensor([1., 1., 0., 0.])
        )
        no_frozen = rtc_prefix_weights(4, 2, 0).flatten()
        self.assertTrue(0 < no_frozen[1] < no_frozen[0] < 1)

    def test_invalid_mask_arguments_fail(self):
        for args in ((0, 0, 0), (4, 5, 1), (4, -1, 1), (4, 1, -1), (4, 1, 1.5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                rtc_prefix_weights(*args)


class RTCGuidanceTests(unittest.TestCase):
    def test_full_endpoint_vjp_changes_unmasked_future(self):
        # v_0 depends on future x_1.  J_f.T must propagate the prefix error to
        # x_1 even though its mask is zero; post-hoc prefix replacement cannot.
        actions = torch.zeros(1, 2, 1)
        target = torch.tensor([[[1.], [0.]]])
        weights = torch.tensor([[[1.], [0.]]])

        def velocity(x):
            return torch.stack((x[:, 1], torch.zeros_like(x[:, 1])), dim=1)

        actual = rtc_guided_euler_step(
            actions, velocity, target, weights, time=.5, dt=.25, beta=.5
        )
        torch.testing.assert_close(actual, torch.tensor([[[.125], [.0625]]]))
        self.assertFalse(actual.requires_grad)
        self.assertNotEqual(actual[0, 0, 0].item(), 1.)

    def test_inference_mode_override_and_no_parameter_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor(2.))
        with torch.inference_mode():
            initial = torch.ones(1, 2, 1)
            target = torch.zeros_like(initial)
            weights = rtc_prefix_weights(2, 1, 1)
            actual = rtc_guided_euler_step(
                initial, lambda x: parameter * x,
                target, weights, time=0., dt=.25, beta=.5,
            )
        # f(x)=3*x, error=-3, VJP=-9; v=2.  time=0 clips at beta.
        torch.testing.assert_close(actual, torch.tensor([[[.375], [1.5]]]))
        self.assertIsNone(parameter.grad)
        self.assertFalse(actual.is_inference())

    def test_zero_mask_and_zero_beta_have_unconditioned_euler_result(self):
        initial = torch.tensor([[[.25], [-.5]]])
        target = torch.zeros_like(initial)
        velocity = lambda x: x * .5 + .25
        expected = initial + .25 * velocity(initial)
        for weights, beta in ((torch.zeros(1, 2, 1), 5.), (torch.ones(1, 2, 1), 0.)):
            actual = rtc_guided_euler_step(
                initial, velocity, target, weights, time=.25, dt=.25, beta=beta
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_guidance_scale_uses_paper_equation_when_not_clipped(self):
        initial = torch.zeros(1, 1, 1)
        actual = rtc_guided_euler_step(
            initial, torch.zeros_like, torch.ones_like(initial), torch.ones_like(initial),
            time=.25, dt=.1, beta=100.,
        )
        torch.testing.assert_close(actual, torch.full_like(initial, 1 / 3))

    def test_paper_beta_five_is_a_clipping_bound_not_an_interpolation_weight(self):
        parameter = torch.nn.Parameter(torch.tensor(2.))
        initial = torch.zeros(1, 1, 1)
        # At tau=0 the unclipped scale is infinite, hence the paper's beta=5
        # is used directly. f(x)=3*x, so J_f.T @ error=3 at x=0,Y=1.
        actual = rtc_guided_euler_step(
            initial, lambda x: parameter * x,
            torch.ones_like(initial), torch.ones_like(initial),
            time=0., dt=.1, beta=5.,
        )
        torch.testing.assert_close(actual, torch.full_like(initial, 1.5))
        self.assertIsNone(parameter.grad)

    def test_bf16_noise_and_fp16_velocity_promote_like_original_sampler(self):
        initial = torch.ones(1, 1, 1, dtype=torch.bfloat16)
        actual = rtc_guided_euler_step(
            initial, lambda x: (2 * x).to(torch.float16),
            torch.zeros_like(initial), torch.zeros(1, 1, 1), time=0., dt=.25, beta=.5,
        )
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, torch.tensor([[[1.5]]]))
        actual.numpy()  # QwenPI returns numpy actions to the policy wrapper.

    def test_invalid_and_non_finite_guidance_fail(self):
        initial = torch.zeros(1, 1, 1)
        for beta in (-1., math.inf, math.nan):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                rtc_guided_euler_step(
                    initial, torch.zeros_like, initial, initial, time=0, dt=.25, beta=beta
                )
        with self.assertRaises(FloatingPointError):
            rtc_guided_euler_step(
                initial, lambda x: x * math.nan, initial, initial, time=0, dt=.25, beta=1
            )


class PIHeadIntegrationTests(unittest.TestCase):
    @staticmethod
    def make_head():
        from omegaconf import OmegaConf
        from vla.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead

        config = OmegaConf.create({"framework": {"action_model": {
            "action_dim": 7, "action_horizon": 4, "state_dim": 7,
            "num_inference_timesteps": 2, "num_target_vision_tokens": 1,
            "add_pos_embed": True, "max_seq_len": 8,
            "noise_beta_alpha": 1.5, "noise_beta_beta": 1., "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                "num_layers": 1, "input_embedding_dim": 8, "attention_head_dim": 4,
                "num_attention_heads": 2, "cross_attention_dim": 8,
                "dropout": 0., "final_dropout": False, "norm_type": "ada_norm",
                "positional_embeddings": None,
            },
        }}})
        return LayerwiseFlowmatchingActionHead(config).eval()

    def test_real_head_guidance_works_with_inference_conditioning_and_state(self):
        head = self.make_head()
        with torch.inference_mode():
            hidden = [torch.randn(1, 3, 8)]
            state = torch.randn(1, 1, 7)
            attention_mask = torch.ones(1, 3, dtype=torch.bool)
            prediction = head.predict_action(
                hidden, state, attention_mask,
                rtc={"prefix_actions": torch.ones(1, 2, 7), "delay": 1, "beta": 5.},
            )
        self.assertEqual(prediction.shape, (1, 4, 7))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))

    def test_disabled_and_zero_beta_match_with_identical_noise(self):
        head = self.make_head()
        hidden = [torch.randn(1, 3, 8)]
        torch.manual_seed(42)
        original = head.predict_action(hidden)
        torch.manual_seed(42)
        zero_beta = head.predict_action(
            hidden, rtc={"prefix_actions": torch.ones(1, 2, 7), "delay": 1, "beta": 0.}
        )
        torch.testing.assert_close(original, zero_beta, rtol=0, atol=0)

    def test_sampler_uses_checkpoint_n_and_rejects_invalid_integration_count(self):
        head = self.make_head()
        self.assertEqual(head.num_inference_timesteps, 2)
        head.num_inference_timesteps = 0
        with self.assertRaisesRegex(ValueError, "denoising steps n"):
            head.predict_action([torch.randn(1, 3, 8)])


if __name__ == "__main__":
    unittest.main()
