"""End-to-end critic training/checkpoint/labeling on tiny real Qwen3-VL."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from examples.LIBERO import train_recap_value, label_recap_advantages
from examples.LIBERO.recap_utils import load_config, parser_for, validate_checkpoint
from vla.dataloader.advantage_labels import AdvantageLabelSource
from vla.dataloader.recap_dataset import RecapEpisodes, ReturnDataset, build_episode_store
from vla.model.modules.recap.tests.test_advantage_data import FixtureDataset
from vla.model.modules.recap.tests import test_recap as critic_fixtures
from vla.model.modules.recap.tests.test_value_model import tiny_qwen


class IdentityTransform:
    def eval(self):
        pass

    def __call__(self, raw):
        return raw


class PipelineDataset(FixtureDataset):
    """Use actual sample packing with deterministic video stubs, JSON table I/O."""
    def __init__(self, root, tables):
        super().__init__(root, "none")
        self.tables = tables
        self._trajectory_ids = np.array(list(tables))
        self._trajectory_lengths = np.array([len(table) for table in tables.values()])
        self._delta_indices = {key: np.array([0]) for keys in self.modality_keys.values() for key in keys}
        self.curr_traj_id = None
        self.transforms = IdentityTransform()

    def get_trajectory_data(self, trajectory_id):
        return self.tables[trajectory_id]

    def get_step_data(self, trajectory_id, base_index):
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        return {
            "video.image": np.zeros((1, 8, 8, 3), dtype=np.uint8),
            "annotation.task": ["fold shirt"],
            "action.joints": np.full((2, 2), base_index, dtype=np.float32),
        }


class RecapPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config_path = self.root / "task.yaml"
        OmegaConf.save(OmegaConf.create({
            "seed": 4,
            "framework": {"name": "QwenOFTRecap", "qwenvl": {"base_vlm": "Qwen/policy"}},
            "datasets": {"vla_data": {"advantage_labels": {"source": "sidecar", "path": "meta/advantage_labels.jsonl"}}},
            "recap": {
                "qwenvl": {"base_vlm": "Qwen/tiny", "attn_implementation": "eager"},
                "value_model": {"num_bins": 11},
                "data": {"success_source": "manifest", "intervention_column": "is_intervention"},
                "reward": {"failure_penalty": 4, "normalization": 10},
                "training": {
                    "output_dir": str(self.root / "train"), "max_steps": 2, "batch_size": 2,
                    "gradient_accumulation_steps": 2, "validation_fraction": 0.5,
                    "save_interval": 1, "eval_interval": 1, "gradient_checkpointing": False,
                },
                "labeling": {"output_dir": str(self.root / "labels")},
                "advantage": {"n_steps": 2, "prediction_batch_size": 2},
            },
        }), self.config_path)
        self.cfg = load_config(parser_for("test").parse_args(["--config_yaml", str(self.config_path)]))
        self.datasets = [self.dataset("a"), self.dataset("b")]
        processor = MagicMock()
        processor.apply_chat_template.side_effect = critic_fixtures.RecapTests._process
        for patcher in (
            patch("vla.model.modules.vlm.Qwen.Qwen3VLForConditionalGeneration.from_pretrained", side_effect=lambda *a, **k: tiny_qwen()),
            patch("vla.model.modules.vlm.Qwen.AutoProcessor.from_pretrained", return_value=processor),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def dataset(self, name):
        root = self.root / name
        (root / "meta").mkdir(parents=True)
        rows = [{"episode_index": ep, "success": success, "complete": True} for ep, success in ((7, True), (9, False))]
        (root / "meta/recap_episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        tables = {
            ep: pd.DataFrame({
                "episode_index": [ep] * length, "frame_index": np.arange(length),
                "task_index": [0] * length, "is_intervention": [False] * (length - 1) + [True],
            }) for ep, length in ((7, 3), (9, 4))
        }
        return PipelineDataset(root, tables)

    def store(self):
        return RecapEpisodes(self.datasets, self.cfg)

    def test_complete_trajectories_returns_split_and_task_namespacing(self):
        store = self.store()
        self.assertEqual(store.task_ids, {"a:0": 0, "b:0": 1})
        torch.testing.assert_close(store.episodes[0].returns, torch.tensor([-0.2, -0.1, 0.0]))
        torch.testing.assert_close(store.episodes[1].returns, torch.tensor([-0.7, -0.6, -0.5, -0.4]))
        train, val = store.split(0.5, 42)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(len(train), 2)
        self.assertEqual(len(val), 2)
        dataset = ReturnDataset(store, [0, 1])
        self.assertEqual(len(dataset), 7)  # bypass deliberately incomplete/shuffled all_steps
        self.assertAlmostEqual(dataset[3]["return"], -0.7)
        self.assertEqual(set(dataset[0]), {"image", "lang", "return"})

    def test_unknown_incomplete_duplicate_outcomes_are_rejected(self):
        path = self.datasets[0].dataset_path / "meta/recap_episodes.jsonl"
        for rows in (
            [{"episode_index": 7, "success": True, "complete": True}],
            [{"episode_index": 7, "success": True, "complete": False}],
            [{"episode_index": 7, "success": "true", "complete": True}],
            [{"episode_index": 7, "success": True, "complete": True}] * 2,
        ):
            with self.subTest(rows=rows):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaises(ValueError):
                    self.store()

    def test_frame_gaps_future_inputs_and_invalid_scale_are_rejected(self):
        self.datasets[0].tables[7].loc[1, "frame_index"] = 5
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.store()
        self.datasets[0].tables[7].loc[1, "frame_index"] = 1
        self.datasets[0]._delta_indices["video.image"] = np.array([1])
        with self.assertRaisesRegex(ValueError, "future"):
            self.store()
        self.datasets[0]._delta_indices["video.image"] = np.array([0])
        self.cfg.recap.reward.normalization = 1
        with self.assertRaisesRegex(ValueError, "outside value support"):
            self.store()

    def test_column_outcomes_truncation_and_explicit_success_only_mode(self):
        self.cfg.recap.data.success_source = "column"
        for ds in self.datasets:
            for table in ds.tables.values():
                table["success"] = False
                table["truncated"] = False
        store = self.store()
        self.assertAlmostEqual(float(store.episodes[0].returns[-1]), -0.4)
        self.cfg.recap.data.truncated_column = "truncated"
        self.datasets[0].tables[7].loc[2, "truncated"] = True
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.store()
        self.cfg.recap.data.truncated_column = None
        self.cfg.recap.data.success_source = "all_success"
        self.assertEqual(float(self.store().episodes[1].returns[-1]), 0)

    def test_factory_disables_old_labels_without_mutating_shared_config(self):
        before = OmegaConf.to_container(self.cfg, resolve=True)
        with patch("vla.dataloader.lerobot_datasets.get_vla_dataset") as factory:
            factory.return_value.datasets = self.datasets
            build_episode_store(self.cfg)
        data = factory.call_args.args[0]
        self.assertEqual(data.advantage_labels.source, "none")
        self.assertFalse(data.delete_pause_frame)
        self.assertEqual(before, OmegaConf.to_container(self.cfg, resolve=True))

    def test_cli_train_reload_label_and_consume(self):
        store = self.store()
        argv = ["--config_yaml", str(self.config_path), "--device", "cpu"]
        with patch.object(train_recap_value, "build_episode_store", return_value=store):
            train_recap_value.main(argv)
        checkpoint = self.root / "train/final/critic.pt"
        self.assertTrue(checkpoint.is_file())
        self.assertTrue((self.root / "train/step_0000001/critic.pt").is_file())
        metrics = [json.loads(line) for line in (self.root / "train/metrics.jsonl").read_text().splitlines()]
        self.assertEqual([m["step"] for m in metrics], [1, 2])
        self.assertTrue(all(np.isfinite(m["validation_value_mae"]) and m["grad_norm"] > 0 for m in metrics))
        initial = torch.load(self.root / "train/step_0000001/critic.pt", weights_only=True)
        final = torch.load(checkpoint, weights_only=True)
        self.assertFalse(torch.equal(initial["value_head.weight"], final["value_head.weight"]))
        with patch.object(label_recap_advantages, "build_episode_store", return_value=store):
            label_recap_advantages.main(argv)
        scores = [json.loads(line) for line in (self.root / "labels/scores.jsonl").read_text().splitlines()]
        self.assertEqual(len(scores), 14)
        meta = json.loads((self.root / "labels/metadata.json").read_text())
        self.assertEqual(meta["critic_step"], 2)
        for ds in self.datasets:
            source = AdvantageLabelSource(ds.dataset_path, {"source": "sidecar"})
            self.assertEqual(len(source.labels), 7)
            for row in [r for r in scores if r["dataset_name"] == ds.dataset_name]:
                expected = row["advantage"] > meta["thresholds"][row["task_key"]] or row["is_intervention"]
                actual = source.get(row["episode_index"], row["frame_index"])
                self.assertEqual(actual["advantage_indicator"], expected)
        # Terminal advantage is terminal reward minus V; no bootstrap past the end.
        terminal = next(row for row in scores if row["dataset_name"] == "a" and row["episode_index"] == 9 and row["frame_index"] == 3)
        self.assertAlmostEqual(terminal["advantage"], -0.4 - terminal["value"], places=5)
        with self.assertRaises(FileExistsError):
            label_recap_advantages.label(self.cfg, store, "cpu")
        changed = OmegaConf.create(OmegaConf.to_container(self.cfg))
        changed.recap.reward.normalization = 20
        with self.assertRaisesRegex(ValueError, "config mismatch"):
            validate_checkpoint(changed, checkpoint)

    def test_check_data_does_not_load_model_or_require_checkpoint(self):
        store = self.store()
        argv = ["--config_yaml", str(self.config_path), "--check_data"]
        for module in (train_recap_value, label_recap_advantages):
            with patch.object(module, "build_episode_store", return_value=store), patch.object(module, "QwenRecap") as model:
                module.main(argv)
                model.assert_not_called()
        self.assertFalse((self.root / "train").exists())


if __name__ == "__main__":
    unittest.main()
