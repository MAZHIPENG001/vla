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

PIPELINE_DEFAULTS = {
    "data": {
        "success_source": "manifest", "outcome_path": "meta/recap_episodes.jsonl",
        "success_column": "success", "task_column": "task_index",
        "truncated_column": None, "intervention_column": None,
    },
    "reward": {"failure_penalty": None, "normalization": None, "task_overrides": {}},
    "training": {
        "output_dir": "playground/Checkpoints/recap_value",
        "batch_size": 2, "num_workers": 0, "max_steps": 10000,
        "gradient_accumulation_steps": 4, "learning_rate": 1e-5, "weight_decay": 0.01,
        "max_grad_norm": 1.0, "validation_fraction": 0.1,
        "eval_interval": 500, "save_interval": 1000, "logging_steps": 10,
        "gradient_checkpointing": True,
    },
    "labeling": {"output_dir": "playground/Checkpoints/recap_labels"},
}


def parser_for(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config_yaml", required=True, help="Shared task YAML for the selected environment.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_id", help="Override only the critic backbone.")
    parser.add_argument("--checkpoint", help="Trained QwenValueModel state_dict; training uses this as a warm start.")
    parser.add_argument("--check_data", action="store_true", help="Validate trajectory metadata/returns without loading Qwen.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="OmegaConf overrides on the shared YAML.")
    return parser


def load_config(args):
    cfg = OmegaConf.merge(
        {"seed": 42, "recap": asdict(QwenRecapDefaultConfig())},
        {"recap": PIPELINE_DEFAULTS},
        OmegaConf.load(args.config_yaml), OmegaConf.from_dotlist(args.set),
    )
    if args.model_id:
        cfg.recap.qwenvl.base_vlm = args.model_id
    if args.checkpoint:
        cfg.recap.value_model.checkpoint = args.checkpoint
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def single_process_only():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("These entrypoints are single-process; use python -m, not torchrun/accelerate launch.")


def config_contract(cfg):
    """Settings that change the critic's value scale or observation interpretation."""
    resolved = OmegaConf.to_container(cfg, resolve=True)
    recap = resolved["recap"]
    data = resolved["datasets"]["vla_data"]
    return {
        "reward": recap["reward"],
        "base_vlm": recap["qwenvl"]["base_vlm"],
        "value_support": {key: recap["value_model"][key] for key in ("num_bins", "value_min", "value_max")},
        "observations": {key: data.get(key) for key in (
            "include_state", "obs_image_size", "CoT_prompt", "action_type", "action_mode", "data_mix",
        )},
        "task_column": recap["data"]["task_column"],
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


def save_checkpoint(directory, critic, optimizer, step, cfg, split):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = directory / "critic.pt"
    atomic_write(checkpoint, lambda tmp: torch.save(critic.value_model.state_dict(), tmp))
    # Kept separately from inference weights; --checkpoint is a warm start, not exact resume.
    atomic_write(directory / "optimizer.pt", lambda tmp: torch.save({"step": step, "optimizer": optimizer.state_dict()}, tmp))
    write_json(directory / "metadata.json", {"step": step, "contract": config_contract(cfg), "split": split})
    saved = OmegaConf.create(OmegaConf.to_container(critic.config, resolve=True))
    saved.recap.value_model.checkpoint = str(checkpoint.resolve())
    OmegaConf.save(saved, directory / "config.yaml")
    return checkpoint


def describe_store(store):
    print(json.dumps({
        "episodes": len(store.episodes), "frames": sum(ep.length for ep in store.episodes),
        "tasks": store.task_ids,
        "return_min": min(float(ep.returns.min()) for ep in store.episodes),
        "return_max": max(float(ep.returns.max()) for ep in store.episodes),
    }, ensure_ascii=False), flush=True)
