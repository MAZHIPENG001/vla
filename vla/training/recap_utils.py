"""Shared CLI/config/checkpoint helpers for the two RECAP stages."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
from omegaconf import OmegaConf
import torch

from vla.model.modules.recap.recap import QwenRecapDefaultConfig

VALUE_DATA_PROFILE = {
    "return_frame_info": True, "return_value_targets": True,
    "sampling_mode": "sequential", "transform_mode": "eval", "delete_pause_frame": False,
    "value_target_config": {
        "data": "${recap.data}", "reward": "${recap.reward}",
        "value_model": {"value_min": "${recap.value_model.value_min}", "value_max": "${recap.value_model.value_max}"},
    },
    "per_device_batch_size": "${recap.training.batch_size}",
    "num_workers": "${recap.training.num_workers}", "shuffle": True,
    "pin_memory": False, "save_statistics": False,
}

PIPELINE_DEFAULTS = {
    "data": {
        "success_source": "manifest", "outcome_path": "meta/recap_episodes.jsonl",
        "success_column": "success", "task_column": "task_index",
        "truncated_column": None, "intervention_column": None,
    },
    "reward": {"source": "episode", "column": "next.reward", "failure_penalty": None,
               "normalization": None, "task_overrides": {}},
    "training": {
        "output_dir": "playground/Checkpoints/recap_value",
        "batch_size": 2, "num_workers": 0, "max_steps": 10000,
        "gradient_accumulation_steps": 4, "learning_rate": 1e-5, "weight_decay": 0.01,
        "max_grad_norm": 1.0, "validation_fraction": 0.1,
        "eval_interval": 500, "save_interval": 1000, "logging_steps": 10,
        "gradient_checkpointing": True,
        "mixed_precision": "bf16", "resume_from_checkpoint": None,
    },
}


def parser_for(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config_yaml", required=True, help="Shared task YAML for the selected environment.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_id", help="Override only the critic backbone.")
    parser.add_argument("--checkpoint", help="Trained QwenValueModel state_dict; training uses this as a warm start.")
    parser.add_argument("--resume_from_checkpoint", help="Training checkpoint directory (restores optimizer, RNG and data position).")
    parser.add_argument("--check_data", action="store_true", help="Validate trajectory metadata/returns without loading Qwen.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="OmegaConf overrides on the shared YAML.")
    return parser


def load_config(args):
    cfg = OmegaConf.merge(
        {"seed": 42, "recap": asdict(QwenRecapDefaultConfig())},
        {"recap": PIPELINE_DEFAULTS},
        {"datasets": {"vla_data": {"profiles": {"value": VALUE_DATA_PROFILE}}}},
        OmegaConf.load(args.config_yaml), OmegaConf.from_dotlist(args.set),
    )
    if args.model_id:
        cfg.recap.qwenvl.base_vlm = args.model_id
    if args.checkpoint:
        cfg.recap.value_model.checkpoint = args.checkpoint
    if args.resume_from_checkpoint:
        cfg.recap.training.resume_from_checkpoint = args.resume_from_checkpoint
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def single_process_only():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Advantage export is single-process; use python -m vla.training.label_recap_advantages.")


def synchronize(accelerator):
    # Accelerate 1.15 passes CUDA device_ids even for CPU/Gloo barriers. With a
    # CUDA-enabled PyTorch build this can select an invalid GPU on CPU ranks.
    if accelerator.device.type == "cpu":
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    else:
        accelerator.wait_for_everyone()


def config_contract(cfg):
    """Settings that change the critic's value scale or observation interpretation."""
    resolved = OmegaConf.to_container(cfg, resolve=True)
    recap = resolved["recap"]
    data = resolved["datasets"]["vla_data"]
    data = {**data, **data.get("profiles", {}).get("value", {})}
    return {
        "reward": recap["reward"],
        "base_vlm": recap["qwenvl"]["base_vlm"],
        "value_support": {key: recap["value_model"][key] for key in ("num_bins", "value_min", "value_max")},
        "observations": {key: data.get(key) for key in (
            "include_state", "obs_image_size", "CoT_prompt", "action_type", "action_mode", "data_mix",
        )},
        "task_column": recap["data"]["task_column"],
        "trajectory_data": recap["data"],
    }


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, payload):
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"))


def validate_checkpoint(cfg, checkpoint):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Train the critic first; checkpoint not found: {checkpoint}")
    metadata_path = checkpoint.parent / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"missing checkpoint provenance: {metadata_path}; use train_recap_value outputs")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("step", 0) < 1:
        raise ValueError("checkpoint has no completed optimizer steps")
    if metadata["contract"] != config_contract(cfg):
        raise ValueError("critic training/labeling config mismatch: reward scale, value support, backbone or observations changed")
    return metadata


def save_checkpoint(directory, critic, accelerator, step, cfg, split, progress):
    """Collective save: every rank must participate, including ZeRO-3 gathers."""
    directory = Path(directory)
    # Publish only complete checkpoints. An interrupted save leaves a hidden
    # directory and cannot be mistaken for a usable resume checkpoint.
    temporary = directory.with_name(f".{directory.name}.incomplete")
    if directory.exists() or temporary.exists():
        raise FileExistsError(f"checkpoint already exists: {directory} or {temporary}")
    synchronize(accelerator)
    if accelerator.is_main_process:
        temporary.mkdir(parents=True, exist_ok=False)
    synchronize(accelerator)
    accelerator.save_state(str(temporary / "training_state"), safe_serialization=False)
    state = accelerator.get_state_dict(critic)
    checkpoint = directory / "critic.pt"
    if accelerator.is_main_process:
        weights = {key.removeprefix("value_model."): value for key, value in state.items()
                   if key.startswith("value_model.")}
        if not weights:
            raise ValueError("checkpoint contains no value model weights")
        atomic_write(temporary / "critic.pt", lambda tmp: torch.save(weights, tmp))
        write_json(temporary / "progress.json", {"step": step, **progress})
        write_json(temporary / "metadata.json", {"step": step, "contract": config_contract(cfg), "split": split})
        saved = OmegaConf.create(OmegaConf.to_container(accelerator.unwrap_model(critic).config, resolve=True))
        saved.recap.value_model.checkpoint = str(checkpoint.resolve())
        OmegaConf.save(saved, temporary / "config.yaml")
    synchronize(accelerator)
    if accelerator.is_main_process:
        os.replace(temporary, directory)
    synchronize(accelerator)
    return checkpoint


def describe_store(store):
    print(json.dumps({
        "episodes": len(store.episodes), "frames": sum(ep.length for ep in store.episodes),
        "tasks": store.task_ids,
        "return_min": min(float(ep.returns.min()) for ep in store.episodes),
        "return_max": max(float(ep.returns.max()) for ep in store.episodes),
    }, ensure_ascii=False), flush=True)
