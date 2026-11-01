"""RTC wire protocol and checkpoint normalization tests; no model or GPU needed."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from deployment.model_server.policy_norm_processor import PolicyNormProcessor, _build_dataset_metadata
from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools import msgpack_numpy
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from examples.LIBERO.eval_files.model2libero_interface import ModelClient
from vla.dataloader.gr00t_lerobot.data_config import Libero4in1DataConfig


def norm_processor():
    """Bind the actual LIBERO training transform to simple checkpoint stats."""
    proc = PolicyNormProcessor.__new__(PolicyNormProcessor)
    config = Libero4in1DataConfig()
    proc._action_keys = list(config.action_keys)
    proc._state_keys = []
    proc._action_key_dims = {k: 1 for k in config.action_keys}
    low, high = [-2.] * 6 + [0.], [4.] * 6 + [1.]
    stats = {"action": {
        "min": low, "max": high, "q01": low, "q99": high,
        "mean": [1.] * 6 + [0.5], "std": [1.] * 7,
    }}
    metadata = _build_dataset_metadata(
        stats, config.embodiment_tag, config.action_keys, [], proc._action_key_dims, {}
    )
    proc._transform = config.transform()
    proc._transform.set_metadata(metadata)
    proc._transform.eval()
    return proc


class NormalizationTests(unittest.TestCase):
    def test_prefix_uses_actual_training_transform_and_keeps_gripper_encoding(self):
        proc = norm_processor()
        raw = np.array([[-2., 1., 4., -1., 2., 3., 0.], [4., 1., -2., 3., 2., -1., 1.]])
        original = raw.copy()
        normalized = proc.apply_actions(raw)
        np.testing.assert_allclose(normalized[:, :6], (raw[:, :6] + 2.) / 3. - 1., atol=1e-6)
        np.testing.assert_array_equal(normalized[:, 6], raw[:, 6])
        np.testing.assert_allclose(proc.unapply_actions(normalized), raw, atol=1e-6)
        np.testing.assert_array_equal(raw, original)

    def test_server_converts_prefix_before_framework_and_reverses_model_output(self):
        proc = norm_processor()
        framework = Mock(supports_rtc_inpainting=True)
        framework.predict_action.return_value = {"normalized_actions": np.zeros((1, 4, 7))}
        wrapper = PolicyServerWrapper.__new__(PolicyServerWrapper)
        wrapper._framework = framework
        wrapper._action_chunk_size = 4
        wrapper._include_state = True
        wrapper._default_unnorm_key = "libero"
        wrapper._get_processor = lambda _: proc
        raw_prefix = np.ones((1, 2, 7), dtype=np.float32)
        rtc = {"prefix_actions": raw_prefix, "delay": 1, "beta": 3.}
        output = wrapper.predict_action([{"lang": "task"}], rtc=rtc)
        received = framework.predict_action.call_args.kwargs["rtc"]
        np.testing.assert_allclose(received["prefix_actions"][0], proc.apply_actions(raw_prefix[0]))
        self.assertEqual((received["delay"], received["beta"]), (1, 3.))
        np.testing.assert_array_equal(rtc["prefix_actions"], raw_prefix)
        np.testing.assert_allclose(output["actions"][0, :, :6], 1.)
        framework.supports_rtc_inpainting = False
        with self.assertRaisesRegex(ValueError, "does not support"):
            wrapper.predict_action([{}], rtc=rtc)
        framework.supports_rtc_inpainting = True
        for invalid in (dict(rtc, delay=-1), dict(rtc, beta=float("nan")), dict(rtc, prefix_actions=np.zeros((1, 5, 7)))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                wrapper.predict_action([{}], rtc=invalid)

    def test_paper_beta_and_observed_delay_are_forwarded_without_blend_limits(self):
        proc = norm_processor()
        framework = Mock(supports_rtc_inpainting=True)
        framework.predict_action.return_value = {"normalized_actions": np.zeros((1, 4, 7))}
        wrapper = PolicyServerWrapper.__new__(PolicyServerWrapper)
        wrapper._framework = framework
        wrapper._action_chunk_size = 4
        wrapper._include_state = True
        wrapper._default_unnorm_key = "libero"
        wrapper._get_processor = lambda _: proc
        # A delay beyond prefix capacity is an exhausted controller budget, not
        # permission for transport to silently change the observed estimate.
        wrapper.predict_action([{}], rtc={
            "prefix_actions": np.ones((1, 2, 7)), "delay": 6, "beta": 5.,
        })
        received = framework.predict_action.call_args.kwargs["rtc"]
        self.assertEqual(received["delay"], 6)
        self.assertEqual(received["beta"], 5.)


class Socket:
    def __init__(self, route):
        self.route = route
        self.closed = False
        self.timeout = None

    def send(self, data):
        self.sent = msgpack_numpy.unpackb(data)

    def recv(self, timeout=None):
        self.timeout = timeout
        return msgpack_numpy.packb(self.route(self.sent))

    def close(self):
        self.closed = True


def transport(socket):
    client = WebsocketClientPolicy.__new__(WebsocketClientPolicy)
    client._packer = msgpack_numpy.Packer()
    client._request_sequence = 0
    client._inference_timeout = None
    client._failed = False
    client._ws = socket
    return client


class ProtocolTests(unittest.TestCase):
    def test_client_server_request_ids_errors_and_latency(self):
        policy = Mock()
        policy.predict_action.return_value = {"actions": np.ones((1, 4, 7))}
        server = WebsocketPolicyServer(policy)
        sock = Socket(server._route_message)
        client = transport(sock)
        client.set_inference_timeout(0.25)
        output = client.predict_action({"examples": [{"lang": "task"}]})
        self.assertEqual(output["request_id"], "1")
        self.assertGreaterEqual(output["server_inference_ms"], 0.)
        self.assertEqual(sock.timeout, 0.25)
        policy.predict_action.assert_called_once_with(examples=[{"lang": "task"}])
        policy.predict_action.side_effect = ValueError("prefix failure")
        with self.assertLogs(level="ERROR"), self.assertRaisesRegex(RuntimeError, "prefix failure"):
            client.predict_action({"examples": [{}]})

    def test_mismatched_reply_is_rejected_and_socket_never_reused(self):
        sock = Socket(lambda _: {"request_id": "old-episode", "ok": True})
        client = transport(sock)
        with self.assertRaisesRegex(RuntimeError, "expected"):
            client.predict_action({})
        self.assertTrue(sock.closed)
        with self.assertRaisesRegex(RuntimeError, "unusable"):
            client.predict_action({})

    def test_receive_timeout_closes_connection_and_propagates(self):
        sock = Socket(lambda _: {})
        sock.recv = Mock(side_effect=TimeoutError("inference timed out"))
        client = transport(sock)
        client.set_inference_timeout(0.1)
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            client.predict_action({})
        sock.recv.assert_called_once_with(timeout=0.1)
        self.assertTrue(sock.closed)
        for invalid in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=invalid), self.assertRaises(ValueError):
                client.set_inference_timeout(invalid)


class ModelClientTests(unittest.TestCase):
    def make_client(self, supports=True):
        socket_client = Mock()
        socket_client.get_server_metadata.return_value = {"action_chunk_size": 4, "rtc_inpainting": supports}
        actions = np.arange(28, dtype=np.float32).reshape(1, 4, 7)
        socket_client.predict_action.return_value = {"data": {"actions": actions}, "ok": True}
        with patch("examples.LIBERO.eval_files.model2libero_interface.WebsocketClientPolicy", return_value=socket_client), patch("builtins.print"):
            client = ModelClient(action_ensemble=False, image_size=(4, 4))
        return client, socket_client, actions

    def test_predict_chunk_does_not_mutate_episode_or_cache(self):
        client, socket, actions = self.make_client()
        client.reset("old episode")
        cached = np.zeros((4, 7))
        client.raw_actions = cached
        state = np.array([[1., 2., 3.]])
        prefix = {"prefix_actions": actions[:, :2], "delay": 1, "beta": 5.}
        predicted = client.predict_chunk({"lang": "new episode", "image": [np.zeros((8, 8, 3), dtype=np.uint8)], "state": state}, rtc=prefix)
        self.assertIs(client.raw_actions, cached)
        self.assertEqual(client.task_description, "old episode")
        self.assertEqual(len(client.image_history), 0)
        np.testing.assert_array_equal(predicted, actions[0])
        self.assertFalse(np.shares_memory(predicted, actions))
        request = socket.predict_action.call_args.args[0]
        self.assertIs(request["rtc"], prefix)
        self.assertIs(request["examples"][0]["state"], state)
        self.assertEqual(request["examples"][0]["image"][0].shape, (4, 4, 3))

    def test_legacy_step_still_requests_only_at_chunk_boundaries(self):
        client, socket, actions = self.make_client()
        for step in range(8):
            output = client.step({"lang": "task"}, step=step)["raw_action"]
            np.testing.assert_array_equal(output["world_vector"], actions[0, step % 4, :3])
            np.testing.assert_array_equal(output["open_gripper"], actions[0, step % 4, 6:7])
        self.assertEqual(socket.predict_action.call_count, 2)

    def test_unsupported_inpainting_and_invalid_chunk_fail_explicitly(self):
        client, socket, _ = self.make_client(supports=False)
        with self.assertRaisesRegex(ValueError, "does not support"):
            client.predict_chunk({}, rtc={"prefix_actions": np.zeros((1, 2, 7))})
        for shape in ((1, 3, 7), (1, 4, 8)):
            socket.predict_action.return_value = {"data": {"actions": np.zeros(shape)}}
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "Expected finite"):
                client.predict_chunk({})
        socket.predict_action.return_value = {"ok": False, "error": {"message": "broken"}}
        with self.assertRaisesRegex(RuntimeError, "broken"):
            client.predict_chunk({})


if __name__ == "__main__":
    unittest.main()
