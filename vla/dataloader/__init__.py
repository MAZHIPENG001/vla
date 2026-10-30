import json
import logging
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from pathlib import Path
from vla.dataloader.vlm_datasets import make_vlm_dataloader

logger = logging.getLogger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe", *, dataset=None, profile=None,
                     mode="train", batch_size=None):
    """Shared VLA/Value loader; optional YAML profiles override dataset options.

    A supplied Dataset/Subset reuses the same decoder and loader configuration,
    which lets training and validation share the already loaded LeRobot mixture.
    """
    if dataset_py == "lerobot_datasets":
        from vla.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

        # Resolve interpolations while still attached to the full task config.
        full_cfg = getattr(cfg, "_cfg", cfg)
        data_cfg = OmegaConf.create(OmegaConf.to_container(full_cfg, resolve=True)["datasets"]["vla_data"])
        profiles = data_cfg.pop("profiles", {})
        if profile is not None:
            if profile not in profiles:
                raise ValueError(f"unknown datasets.vla_data profile: {profile}")
            data_cfg = OmegaConf.merge(data_cfg, profiles[profile])
        elif OmegaConf.select(full_cfg, "framework.name") == "QwenOFTRecap":
            conditioning = OmegaConf.select(full_cfg, "framework.advantage_conditioning", default={})
            if not conditioning.get("enabled", True):
                data_cfg.return_online_advantage = False
            else:
                # Worker processes prepare scalar rewards and a bootstrap
                # observation. Only the model process loads/runs the critic.
                recap = OmegaConf.select(full_cfg, "recap")
                data_cfg.return_value_targets = True
                data_cfg.value_target_config = OmegaConf.to_container(recap, resolve=True)
                data_cfg.return_online_advantage = True
                data_cfg.online_advantage_config = {
                    "mode": recap.advantage.get("mode", "posttrain"),
                    "n_steps": recap.advantage.get("n_steps", 50),
                }
        if dataset is None:
            dataset = get_vla_dataset(
                data_cfg=data_cfg, mode=mode, seed=cfg.get("seed", 42),
                balance_dataset_weights=data_cfg.get("balance_dataset_weights", False),
                balance_trajectory_weights=data_cfg.get("balance_trajectory_weights", False),
            )
        workers = data_cfg.get("num_workers", 16)
        worker_options = {}
        if workers > 0:
            worker_options = {
                "persistent_workers": data_cfg.get("persistent_workers", True),
                "prefetch_factor": data_cfg.get("prefetch_factor", 4),
            }
        loader = DataLoader(
            dataset, batch_size=batch_size if batch_size is not None else data_cfg.per_device_batch_size,
            collate_fn=collate_fn, num_workers=workers,
            pin_memory=data_cfg.get("pin_memory", True),
            shuffle=data_cfg.get("shuffle", False) if mode == "train" else False,
            generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
            **worker_options,
        )
        if (data_cfg.get("save_statistics", True)
                and (not dist.is_initialized() or dist.get_rank() == 0)):
            # Standard Subset/ConcatDataset views need no separate statistics.
            if hasattr(dataset, "save_dataset_statistics"):
                output_dir = Path(cfg.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return loader
    elif dataset_py == "vlm_datasets":
        return make_vlm_dataloader(cfg)["train_dataloader"]
    raise ValueError(f"unsupported dataset_py: {dataset_py}")
