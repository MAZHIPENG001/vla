"""Run with torchrun --standalone --nproc_per_node=2 -m ...distributed_value_smoke --output_dir /tmp/new-run."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from accelerate import Accelerator
import torch


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    accelerator = Accelerator(cpu=True, mixed_precision="bf16")
    # Initialize CPU/Gloo before project logging creates a PartialState.
    from torch.utils.data import Subset
    from vla.model.modules.recap.recap import QwenRecap
    from vla.model.modules.recap.tests.test_value_pipeline import config, ParquetEpisodes, TinyInterface, fixture_loader
    from vla.training.train_recap_value import metric_rows, train
    root = Path(args.output_dir)
    cfg = config(root, freeze=False)
    if accelerator.is_main_process:
        ParquetEpisodes(root / "data")
    torch.distributed.barrier()
    ds = ParquetEpisodes(root / "data", write=False)
    loader = fixture_loader(cfg, [ds])
    store = loader.dataset.value_targets
    with patch("vla.model.modules.recap.value_model.get_vlm_model", TinyInterface):
        checkpoint = train(cfg, loader, "cpu")
        cfg.recap.training.output_dir = str(root / "resumed")
        cfg.recap.training.resume_from_checkpoint = str(root / "run/step_0000002")
        resumed = train(cfg, loader, "cpu")
        cfg.recap.value_model.freeze_backbone = True
        cfg.recap.training.output_dir = str(root / "frozen")
        cfg.recap.training.resume_from_checkpoint = None
        cfg.recap.training.max_steps = 1
        frozen = train(cfg, loader, "cpu")
        if accelerator.is_main_process:
            expected = torch.load(checkpoint, weights_only=True)
            actual = torch.load(resumed, weights_only=True)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            frozen_weights = torch.load(frozen, weights_only=True)
            assert "qwen_vl_interface.model.lm_head.weight" in frozen_weights
            cfg.recap.value_model.checkpoint = str(checkpoint)
            critic = QwenRecap(cfg).eval()
            _, ids = store.split(cfg.recap.training.validation_fraction, cfg.seed)
            dataset = Subset(loader.dataset, store.frame_indices(ids))
            rows = []
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                for i in range(0, len(dataset), 2):
                    rows.append(metric_rows(critic([dataset[j] for j in range(i, min(i + 2, len(dataset)))])))
            means = torch.cat(rows).mean(0)
            log = json.loads((root / "run/metrics.jsonl").read_text().splitlines()[-1])
            torch.testing.assert_close(means[0], torch.tensor(log["validation_value_loss"]), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(means[-1], torch.tensor(log["validation_true_return"]), atol=1e-7, rtol=1e-6)
            print(f"Distributed save/resume and unpadded validation passed ({len(dataset)} validation frames).", flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
