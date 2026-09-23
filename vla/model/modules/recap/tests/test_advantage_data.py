"""Real packing/collation paths on temporary LeRobot-shaped trajectory tables."""
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from vla.dataloader.advantage_labels import AdvantageLabelError, AdvantageLabelSource, write_advantage_labels
from vla.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from vla.dataloader.lerobot_datasets import collate_fn
from vla.model.framework.base_framework import build_framework
from vla.model.modules.recap.tests import test_oft_recap as policy_test_fixtures
from vla.model.modules.recap.tests.test_value_model import tiny_qwen


class FixtureDataset(LeRobotSingleDataset):
    """Skip unrelated metadata/video initialization; use real __getitem__/_pack_sample."""
    def __init__(self, root, source):
        self._dataset_path = root
        self._dataset_name = root.name
        self.data_cfg = {"advantage_labels": {"source": source}, "include_state": False}
        self._advantage_labels = AdvantageLabelSource(root, self.data_cfg["advantage_labels"])
        self._trajectory_ids = np.array([7])
        self._trajectory_lengths = np.array([3])
        self._all_steps = [(7, 2), (7, 0), (7, 1)]  # Intentionally not frame order.
        self._modality_keys = {"video": ["video.image"], "language": ["annotation.task"], "action": ["action.joints"]}
        self._lerobot_info_meta = {"total_videos": 0}
        self.tag = "test_robot"
        self.transforms = lambda data: data
        self.curr_traj_data = None

    def get_step_data(self, trajectory_id, base_index):
        self.curr_traj_data = pd.read_json(self.dataset_path / "episode.json", orient="records")
        return {
            "video.image": np.zeros((1, 8, 8, 3), dtype=np.uint8),
            "annotation.task": ["fold shirt"],
            "action.joints": np.full((2, 2), base_index, dtype=np.float32),
        }


class AdvantageDataTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.records = [
            {"episode_index": 7, "frame_index": 0, "advantage_indicator": False},
            {"episode_index": 7, "frame_index": 1, "advantage_indicator": True},
            {"episode_index": 7, "frame_index": 2, "advantage_indicator": False, "is_intervention": True},
        ]
        self.a = self.make_dataset("a", self.records)
        inverted = [{**record, "advantage_indicator": not record["advantage_indicator"], "is_intervention": False} for record in self.records]
        self.b = self.make_dataset("b", inverted)

    def make_dataset(self, name, records):
        root = self.root / name
        root.mkdir()
        frame = pd.DataFrame(records).fillna({"is_intervention": False})
        frame.to_json(root / "episode.json", orient="records")
        write_advantage_labels(root / "meta/advantage_labels.jsonl", records)
        return root

    def mixture(self, datasets):
        with patch.object(LeRobotMixtureDataset, "update_metadata"):
            return LeRobotMixtureDataset([(dataset, 1.0) for dataset in datasets], mode="train")

    def test_single_anchors_labels_to_original_frame_before_action_chunk(self):
        dataset = FixtureDataset(self.a, "sidecar")
        first, second = dataset[0], dataset[1]
        self.assertEqual((first["episode_index"], first["frame_index"]), (7, 2))
        self.assertIs(first["advantage_indicator"], True)
        self.assertIs(first["is_intervention"], True)
        self.assertEqual(second["frame_index"], 0)
        self.assertIs(second["advantage_indicator"], False)
        np.testing.assert_array_equal(second["action"], np.zeros((2, 2)))
        self.assertEqual(second["dataset_name"], "a")
        self.assertEqual(second["lang"], "fold shirt")

    def test_mixture_random_sampling_keeps_dataset_and_frame_identity(self):
        a, b = FixtureDataset(self.a, "sidecar"), FixtureDataset(self.b, "sidecar")
        mixture = self.mixture([a, b])
        with patch.object(mixture, "sample_step", side_effect=[(a, 7, 0), (b, 7, 0), (a, 7, 1)]):
            batch = next(iter(DataLoader(mixture, batch_size=3, collate_fn=collate_fn)))
        self.assertEqual([sample["dataset_name"] for sample in batch], ["a", "b", "a"])
        self.assertEqual([sample["advantage_indicator"] for sample in batch], [False, True, True])
        for sample in batch:
            self.assertIs(type(sample["advantage_indicator"]), bool)

    def test_parquet_and_sidecar_produce_same_labels(self):
        sidecar = FixtureDataset(self.a, "sidecar")
        parquet = FixtureDataset(self.a, "parquet")
        for index in range(3):
            for field in ("advantage_indicator", "is_intervention", "episode_index", "frame_index"):
                self.assertEqual(sidecar[index][field], parquet[index][field])

    def test_disabled_source_preserves_legacy_samples(self):
        sample = FixtureDataset(self.a, "none")[0]
        self.assertEqual(set(sample), {"image", "action", "lang", "robot_tag"})

    def test_missing_label_is_not_retried_or_replaced_by_another_sample(self):
        dataset = FixtureDataset(self.a, "sidecar")
        del dataset._advantage_labels.labels[(7, 1)]
        mixture = self.mixture([dataset])
        with patch.object(mixture, "sample_step", return_value=(dataset, 7, 1)) as sample_step:
            with patch("vla.dataloader.gr00t_lerobot.datasets.random.randint") as random_retry:
                with self.assertRaisesRegex(AdvantageLabelError, "frame_index=1"):
                    mixture[0]
                sample_step.assert_called_once()
                random_retry.assert_not_called()

    def test_validation_missing_file_duplicate_and_wrong_types(self):
        with self.assertRaises(AdvantageLabelError):
            AdvantageLabelSource(self.root / "absent", {"source": "sidecar"})
        path = self.root / "new_labels.jsonl"
        for record in ({**self.records[0], "advantage_indicator": value} for value in (0.5, 1, "positive", None)):
            with self.assertRaises(AdvantageLabelError):
                write_advantage_labels(path, [record])
            self.assertFalse(path.exists())
        with self.assertRaises(AdvantageLabelError):
            write_advantage_labels(path, [self.records[0], self.records[0]])
        write_advantage_labels(path, self.records)
        with self.assertRaises(FileExistsError):
            write_advantage_labels(path, self.records)
        (self.a / "meta/advantage_labels.jsonl").write_text(path.read_text() + path.read_text())
        with self.assertRaisesRegex(AdvantageLabelError, "duplicate"):
            AdvantageLabelSource(self.a, {"source": "sidecar"})

    def test_parquet_rejects_wrong_frame_or_missing_label(self):
        source = AdvantageLabelSource(self.a, {"source": "parquet"})
        frame = pd.read_json(self.a / "episode.json", orient="records")
        frame.loc[0, "frame_index"] = 9
        with self.assertRaisesRegex(AdvantageLabelError, "frame_index"):
            source.get(7, 0, frame)
        with self.assertRaisesRegex(AdvantageLabelError, "column"):
            source.get(7, 0, frame.drop(columns=["advantage_indicator"]))

    def test_label_source_is_picklable_for_workers(self):
        original = AdvantageLabelSource(self.a, {"source": "sidecar"})
        restored = pickle.loads(pickle.dumps(original))
        self.assertEqual(restored.get(7, 0), original.get(7, 0))
        sample = restored.get(7, 0)
        sample["advantage_indicator"] = True
        self.assertFalse(restored.get(7, 0)["advantage_indicator"])

    def test_dataloader_batch_reaches_conditioned_policy_and_backpropagates(self):
        dataset = FixtureDataset(self.a, "sidecar")
        batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_fn)))
        processor = MagicMock()
        processor.tokenizer.return_value = {"input_ids": [7]}
        processor.apply_chat_template.side_effect = policy_test_fixtures.OFTRecapTests._process
        config = OmegaConf.create({
            "framework": {
                "name": "QwenOFTRecap",
                "qwenvl": {"base_vlm": "Qwen/test", "attn_implementation": "sdpa"},
                "action_model": {"action_horizon": 2, "action_dim": 2},
                "advantage_conditioning": {"dropout_probability": 0},
            }, "datasets": {"vla_data": {}},
        })
        with patch("vla.model.modules.vlm.Qwen.Qwen3VLForConditionalGeneration.from_pretrained", return_value=tiny_qwen()):
            with patch("vla.model.modules.vlm.Qwen.AutoProcessor.from_pretrained", return_value=processor):
                model = build_framework(config)
        loss = model(batch)["action_loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.action_model.parameters()))
        prompts = [m[0]["content"][-1]["text"] for m in processor.apply_chat_template.call_args.args[0]]
        self.assertIn("Advantage: positive", prompts[0])
        self.assertIn("Advantage: negative", prompts[1])


if __name__ == "__main__":
    unittest.main()
