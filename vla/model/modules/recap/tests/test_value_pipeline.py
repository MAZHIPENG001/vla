"""CPU checks of trajectory targets, real value-head training, resume and export.

Only the expensive Qwen encoder/video decoding are replaced. The production
value model, trainer, Accelerate checkpoints and sidecar reader run unchanged.
"""
from dataclasses import asdict
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Subset

from vla.dataloader.advantage_labels import AdvantageLabelSource
from vla.dataloader import build_dataloader
from vla.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from vla.dataloader.recap_targets import RecapEpisodes
from vla.model.modules.recap.recap import QwenRecap, QwenRecapDefaultConfig
from vla.training.label_recap_advantages import label
from vla.training.recap_utils import PIPELINE_DEFAULTS, VALUE_DATA_PROFILE, validate_checkpoint
from vla.training.train_recap_value import train


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(32, 8)
        self.dropout = nn.Dropout(0.25)
        self.lm_head = nn.Linear(8, 32)

    @property
    def device(self):
        return self.embedding.weight.device


class TinyInterface(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = TinyBackbone()

    def build_qwenvl_inputs(self, images, instructions):
        tokens = [[ord(char) % 31 + 1 for char in text] for text in instructions]
        size = max(map(len, tokens))
        ids = torch.tensor([[0] * (size - len(row)) + row for row in tokens], device=self.model.device)
        return {"input_ids": ids, "attention_mask": ids != 0}

    def forward_features(self, input_ids, attention_mask):
        # A causal-style context summary that varies with episode/frame identity.
        embeddings = self.model.embedding(input_ids)
        embeddings = embeddings * attention_mask[..., None]
        return SimpleNamespace(last_hidden_state=self.model.dropout(embeddings.cumsum(1)))


class IdentityTransform:
    def eval(self):
        return self

    def __call__(self, value):
        return value


class ParquetEpisodes(LeRobotSingleDataset):
    """Minimal decoding fixture exposing the existing LeRobot adapter contract."""
    def __init__(self, path, lengths=(1, 2, 5, 3, 4), write=True):
        self._dataset_path = Path(path)
        self._dataset_name = "fixture"
        self._trajectory_ids = np.arange(len(lengths))
        self._trajectory_lengths = np.asarray(lengths)
        self._modality_keys = {"video": ["video.image"], "language": ["language.task"]}
        self._delta_indices = {"video.image": [0], "language.task": [0]}
        self._all_steps = [(int(ep), frame) for ep, length in zip(self.trajectory_ids, lengths) for frame in range(length)]
        self._lerobot_info_meta = {"total_videos": 0}
        self.delete_pause_frame = False
        self.data_cfg = OmegaConf.create({})
        self.value_targets = None
        self.transforms = IdentityTransform()
        self.curr_traj_id = self.curr_traj_data = None
        if not write:
            return
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        for ep, length in enumerate(lengths):
            pd.DataFrame({"episode_index": [ep] * length, "frame_index": range(length),
                          "task_index": [0] * length, "success": [ep % 2 == 0] * length,
                          "truncated": [False] * length,
                          "intervention": [i == 0 and ep == 1 for i in range(length)]}).to_parquet(self.dataset_path / f"{ep}.parquet")

    def configure(self, data_cfg):
        self.data_cfg = data_cfg
        self._advantage_labels = AdvantageLabelSource(self.dataset_path, data_cfg.get("advantage_labels", {"source": "none"}))
        if data_cfg.get("sampling_mode") == "sequential":
            self._all_steps = self._get_all_steps()
        self.value_targets = None
        if data_cfg.get("return_value_targets", False):
            self.value_targets = RecapEpisodes([self], data_cfg.value_target_config)
        return self

    def get_trajectory_data(self, trajectory_id):
        return pd.read_parquet(self.dataset_path / f"{trajectory_id}.parquet")

    def get_step_data(self, episode, frame):
        return {"image": [Image.new("RGB", (4, 4))], "lang": f"task episode {episode} frame {frame}",
                "action": "future data must not reach critic"}

    def _pack_sample(self, sample):
        return sample


def config(root, output="run", freeze=True):
    return OmegaConf.merge(
        {"seed": 42, "recap": asdict(QwenRecapDefaultConfig())}, {"recap": PIPELINE_DEFAULTS},
        {"datasets": {"vla_data": {"profiles": {"value": VALUE_DATA_PROFILE}}}},
        {"recap": {
            "value_model": {"num_bins": 11, "freeze_backbone": freeze},
            "data": {"success_source": "column", "intervention_column": "intervention"},
            "reward": {"failure_penalty": 2, "normalization": 10},
            "training": {"output_dir": str(Path(root) / output), "max_steps": 4, "batch_size": 2,
                         "gradient_accumulation_steps": 3, "mixed_precision": "bf16",
                         "gradient_checkpointing": False, "eval_interval": 1, "save_interval": 2,
                         "learning_rate": 0.01, "validation_fraction": 0.4},
            "labeling": {"output_dir": str(Path(root) / "labels")},
        }, "datasets": {"vla_data": {"data_root_dir": str(Path(root) / "data"), "data_mix": "fixture",
                                           "dataset_py": "lerobot_datasets", "per_device_batch_size": 2,
                                           "advantage_labels": {"source": "sidecar", "path": "meta/advantage_labels.jsonl"}}}},
    )


def fixture_loader(cfg, datasets, *, profile="value"):
    """Use the production factory/mixture; only file/video setup is a fixture."""
    from vla.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES

    by_name = {ds.dataset_name: ds for ds in datasets}
    def make_single(data_root_dir, data_name, robot_type, delete_pause_frame=False, data_cfg=None):
        return by_name[data_name].configure(data_cfg)
    with patch.dict(DATASET_NAMED_MIXTURES, {"fixture": [(name, 1.0, "fixture_robot") for name in by_name]}), \
         patch("vla.dataloader.lerobot_datasets.make_LeRobotSingleDataset", side_effect=make_single), \
         patch.object(LeRobotMixtureDataset, "update_metadata"):
        return build_dataloader(cfg, "lerobot_datasets", profile=profile)


class ValuePipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = config(self.root)
        self.ds = ParquetEpisodes(self.root / "data")
        self.loader = fixture_loader(self.cfg, [self.ds])
        self.store = self.loader.dataset.value_targets
        self.encoder = patch("vla.model.modules.recap.value_model.get_vlm_model", TinyInterface)
        self.encoder.start()

    def tearDown(self):
        self.encoder.stop()
        self.temp.cleanup()

    def test_frame_metadata_is_opt_in_for_standard_vla_samples(self):
        cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        cfg.datasets.vla_data.advantage_labels = {"source": "none"}
        cfg.datasets.vla_data.num_workers = 0
        cfg.datasets.vla_data.save_statistics = False
        ds = ParquetEpisodes(self.root / "ordinary")
        ordinary = fixture_loader(cfg, [ds], profile=None)
        sample = ordinary.dataset[3]
        self.assertEqual(set(sample), {"image", "lang", "action"})
        self.assertIsNone(ordinary.dataset.value_targets)
        cfg.datasets.vla_data.return_frame_info = True
        with_info = fixture_loader(cfg, [ds], profile=None)
        for i in range(len(with_info.dataset)):
            sample = with_info.dataset[i]
            _, ep, frame = with_info.dataset.sample_step(i)
            self.assertEqual(sample["episode_index"], ep)
            self.assertEqual(sample["frame_index"], frame)
            self.assertEqual(sample["episode_length"], int(ds.trajectory_lengths[ep]))
            self.assertNotIn("return", sample)
        cfg.datasets.vla_data.return_value_targets = True
        cfg.datasets.vla_data.value_target_config = cfg.recap
        with_targets = fixture_loader(cfg, [ds], profile=None)
        self.assertEqual(with_targets.dataset.sampling_mode, "random")
        for i in range(len(with_targets.dataset)):
            sample = with_targets.dataset[i]
            ep = with_targets.dataset.value_targets.by_episode[(sample["dataset_name"], sample["episode_index"])]
            self.assertEqual(sample["return"], float(ep.returns[sample["frame_index"]]))

    def test_sequential_mixture_visits_every_frame_and_ignores_filtered_cache(self):
        second = ParquetEpisodes(self.root / "second", (2, 1))
        second._dataset_name = "second"
        # A policy cache may have removed entire episodes/frames. Value indexing
        # must enumerate original frame IDs without consuming that cache.
        meta = second.dataset_path / "meta"
        meta.mkdir()
        (meta / "steps_data_index.pkl").write_bytes(pickle.dumps({"steps": [(0, 0)]}))
        loader = fixture_loader(self.cfg, [self.ds, second])
        expected = [(ds.dataset_name, int(ep), frame)
                    for ds in (self.ds, second) for ep, length in zip(ds.trajectory_ids, ds.trajectory_lengths)
                    for frame in range(length)]
        actual = [(row["dataset_name"], row["episode_index"], row["frame_index"])
                  for i in range(len(loader.dataset)) for row in [loader.dataset[i]]]
        self.assertEqual(actual, expected)
        self.assertEqual(len(set(actual)), 18)
        indices = loader.dataset.value_targets.frame_indices([5, 6])
        self.assertEqual(indices, [15, 16, 17])
        self.assertEqual([loader.dataset[i]["dataset_name"] for i in indices], ["second"] * 3)

    def test_sequential_decode_failure_keeps_original_frame_identity(self):
        with patch.object(self.ds, "get_step_data", side_effect=OSError("broken frame")) as decode:
            with self.assertRaisesRegex(OSError, "broken frame"):
                self.loader.dataset[2]
            decode.assert_called_once_with(1, 1)

    def test_shared_loader_supports_workers_and_validation_subset(self):
        self.cfg.datasets.vla_data.profiles.value.num_workers = 1
        self.cfg.datasets.vla_data.profiles.value.persistent_workers = False
        ids = self.store.frame_indices([1, 3])
        loader = build_dataloader(self.cfg, "lerobot_datasets", profile="value", mode="val",
                                  dataset=Subset(self.loader.dataset, ids))
        samples = [sample for batch in loader for sample in batch]
        self.assertEqual([(row["episode_index"], row["frame_index"]) for row in samples],
                         [(1, 0), (1, 1), (3, 0), (3, 1), (3, 2)])

    def test_variable_lengths_returns_split_and_no_future_data(self):
        self.assertEqual([ep.length for ep in self.store.episodes], [1, 2, 5, 3, 4])
        torch.testing.assert_close(self.store.episodes[0].returns, torch.tensor([0.]))
        torch.testing.assert_close(self.store.episodes[1].returns, torch.tensor([-0.3, -0.2]))
        torch.testing.assert_close(self.store.episodes[2].returns, torch.tensor([-0.4, -0.3, -0.2, -0.1, 0.]))
        dataset = Subset(self.loader.dataset, self.store.frame_indices(range(5)))
        self.assertEqual(len(dataset), 15)
        torch.testing.assert_close(torch.tensor([dataset[i]["return"] for i in range(15)]),
                                   torch.cat([ep.returns for ep in self.store.episodes]))
        self.assertEqual(dataset[1]["episode_index"], 1)
        self.assertEqual(dataset[1]["frame_index"], 0)
        self.assertEqual(dataset[1]["episode_length"], 2)
        train_ids, val_ids = self.store.split(0.4, 42)
        self.assertFalse(set(train_ids) & set(val_ids))

    def test_invalid_and_truncated_episodes(self):
        frame = self.ds.get_trajectory_data(1)
        frame.loc[1, "frame_index"] = 3
        frame.to_parquet(self.ds.dataset_path / "1.parquet")
        with self.assertRaisesRegex(ValueError, "contiguous"):
            RecapEpisodes([self.ds], self.cfg.recap)
        frame.loc[1, "frame_index"] = 1
        frame.loc[1, "truncated"] = True
        frame.to_parquet(self.ds.dataset_path / "1.parquet")
        self.cfg.recap.data.truncated_column = "truncated"
        with self.assertRaisesRegex(ValueError, "truncated"):
            RecapEpisodes([self.ds], self.cfg.recap)

    def test_read_existing_reward_column_without_outcomes(self):
        self.cfg.recap.reward.source = "column"
        self.cfg.recap.reward.failure_penalty = None
        self.cfg.recap.data.success_source = "manifest"  # No manifest is needed for column rewards.
        for episode in self.ds.trajectory_ids:
            frame = self.ds.get_trajectory_data(episode)
            frame["next.reward"] = -1.0
            frame.to_parquet(self.ds.dataset_path / f"{episode}.parquet")
        store = RecapEpisodes([self.ds], self.cfg.recap)
        torch.testing.assert_close(store.episodes[1].returns, torch.tensor([-0.2, -0.1]))
        frame["next.reward"] = float("nan")
        frame.to_parquet(self.ds.dataset_path / f"{episode}.parquet")
        with self.assertRaisesRegex(ValueError, "finite"):
            RecapEpisodes([self.ds], self.cfg.recap)

    def test_default_frozen_backbone_and_full_weight_export(self):
        model = QwenRecap(self.cfg)
        self.assertTrue(QwenRecapDefaultConfig().value_model["freeze_backbone"])
        model.train()
        self.assertFalse(model.qwen_vl_interface.training)
        self.assertFalse(any(p.requires_grad for p in model.qwen_vl_interface.parameters()))
        before = {k: v.clone() for k, v in model.qwen_vl_interface.state_dict().items()}
        output = model(self.loader.dataset[self.store.frame_indices([2])[0]])
        output["value_loss"].backward()
        self.assertIsNotNone(model.value_model.value_head.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.qwen_vl_interface.parameters()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.qwen_vl_interface.state_dict().items()))
        support = model.value_model.bin_values.clone()
        model.bfloat16()
        torch.testing.assert_close(model.value_model.bin_values, support, rtol=0, atol=0)
        checkpoint = train(self.cfg, self.loader, "cpu")
        state = torch.load(checkpoint, weights_only=True)
        self.assertIn("qwen_vl_interface.model.embedding.weight", state)
        self.assertIn("qwen_vl_interface.model.lm_head.weight", state)
        self.assertIn("value_head.weight", state)
        self.assertTrue((checkpoint.parent / "training_state/optimizer.bin").is_file())
        metrics = json.loads((checkpoint.parent.parent / "metrics.jsonl").read_text().splitlines()[-1])
        for prefix in ("train", "validation"):
            for key in ("value_loss", "predicted_value", "true_return", "value_mae"):
                self.assertTrue(np.isfinite(metrics[f"{prefix}_{key}"]))

    def test_exact_resume_across_epochs_and_optimizer_state(self):
        self.cfg.recap.value_model.freeze_backbone = False
        original = train(self.cfg, self.loader, "cpu")
        self.cfg.recap.training.output_dir = str(self.root / "resumed")
        self.cfg.recap.training.resume_from_checkpoint = str(original.parent.parent / "step_0000002")
        resumed = train(self.cfg, self.loader, "cpu")
        expected, actual = torch.load(original, weights_only=True), torch.load(resumed, weights_only=True)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        original_metrics = [json.loads(row) for row in (original.parent.parent / "metrics.jsonl").read_text().splitlines()]
        resumed_metrics = [json.loads(row) for row in (resumed.parent.parent / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(original_metrics[2:], resumed_metrics)
        self.cfg.recap.training.max_steps = 5
        self.cfg.recap.training.resume_from_checkpoint = str(resumed.parent)
        self.assertEqual(train(self.cfg, self.loader, "cpu").parent.name, "final_step_0000005")
        self.cfg.recap.training.max_steps = 6
        self.cfg.recap.training.batch_size = 3
        with self.assertRaisesRegex(ValueError, "resume config"):
            train(self.cfg, self.loader, "cpu")

    def test_accumulation_matches_larger_batch(self):
        ds = ParquetEpisodes(self.root / "accum_data", (1, 2, 5))
        self.cfg.recap.training.validation_fraction = 0
        self.cfg.recap.training.gradient_accumulation_steps = 1
        self.cfg.recap.training.batch_size = 4
        self.cfg.recap.training.max_steps = 3
        loader = fixture_loader(self.cfg, [ds])
        large = train(self.cfg, loader, "cpu")
        self.cfg.recap.training.output_dir = str(self.root / "accumulated")
        self.cfg.recap.training.gradient_accumulation_steps = 2
        self.cfg.recap.training.batch_size = 2
        small = train(self.cfg, loader, "cpu")
        large, small = torch.load(large, weights_only=True), torch.load(small, weights_only=True)
        # CPU bf16 GEMMs can round differently for different batch shapes.
        torch.testing.assert_close(small["value_head.weight"], large["value_head.weight"], atol=0.002, rtol=0.01)

    def test_train_checkpoint_to_labels_roundtrip(self):
        self.cfg.recap.training.max_steps = 1
        checkpoint = train(self.cfg, self.loader, "cpu")
        self.cfg.recap.value_model.checkpoint = str(checkpoint)
        validate_checkpoint(self.cfg, checkpoint)
        targets = label(self.cfg, self.loader, "cpu")
        source = AdvantageLabelSource(self.ds.dataset_path, self.cfg.datasets.vla_data.advantage_labels)
        scores = [json.loads(row) for row in (self.root / "labels/scores.jsonl").read_text().splitlines()]
        self.assertEqual(len(scores), 15)
        self.assertEqual(len(targets), 1)
        for row in scores:
            result = source.get(row["episode_index"], row["frame_index"])
            self.assertEqual(result["advantage_indicator"], row["advantage_indicator"])
        self.assertTrue(source.get(1, 0)["advantage_indicator"])
        with self.assertRaises(FileExistsError):
            label(self.cfg, self.loader, "cpu")


if __name__ == "__main__":
    unittest.main()
