"""Train a RECAP critic: python -m vla.training.train_recap_value."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin
from omegaconf import OmegaConf
import torch
from torch.utils.data import Subset

from vla.training.recap_utils import (
    describe_store, load_config, parser_for, save_checkpoint, seed_everything,
    synchronize, validate_checkpoint, write_json,
)
from vla.dataloader import build_dataloader
from vla.model.modules.recap.recap import QwenRecap


def metric_rows(output):
    values, targets = output["values"].detach().float(), output["returns"].detach().float()
    errors = values - targets
    return torch.stack((output["value_losses"].detach().float(), errors.abs(),
                        errors.square(), values, targets), dim=1)


def metric_means(total):
    count = float(total[-1])
    if not count:
        return {}
    return dict(zip(("value_loss", "value_mae", "value_mse", "predicted_value", "true_return"),
                    (total[:-1] / count).tolist()))


@torch.no_grad()
def evaluate(critic, loader, accelerator):
    previous = critic.training
    critic.eval()
    total = torch.zeros(6, device=accelerator.device, dtype=torch.float64)
    try:
        for batch in loader:
            with accelerator.autocast():
                rows = metric_rows(critic(batch))
            # Drop replicated samples added by distributed batch padding.
            rows = accelerator.gather_for_metrics(rows)
            total[:-1] += rows.double().sum(0)
            total[-1] += len(rows)
    finally:
        critic.train(previous)
    return metric_means(total)


def resume_contract(cfg, store, split, world_size):
    """Detect changes that would silently alter the resumed data stream."""
    digest = hashlib.sha256()
    for ep in store.episodes:
        digest.update(json.dumps([ep.dataset_name, ep.episode_index, ep.task_key, ep.length]).encode())
        digest.update(ep.rewards.numpy().tobytes())
        digest.update(ep.returns.numpy().tobytes())
    settings = cfg.recap.training
    return {
        "data_sha256": digest.hexdigest(), "split": split, "seed": cfg.seed,
        "dataset_config": OmegaConf.to_container(cfg.datasets.vla_data, resolve=True),
        "qwenvl_config": OmegaConf.to_container(cfg.recap.qwenvl, resolve=True),
        "world_size": world_size, "freeze_backbone": cfg.recap.value_model.freeze_backbone,
        "training": {key: settings[key] for key in (
            "batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay",
            "max_grad_norm", "mixed_precision", "gradient_checkpointing",
        )},
    }


def make_accelerator(settings, device):
    accelerator = Accelerator(
        cpu=torch.device(device).type == "cpu", mixed_precision=settings.mixed_precision,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=settings.gradient_accumulation_steps, sync_with_dataloader=False,
        ),
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=False, use_seedable_sampler=True,
        ),
        # The Qwen language output head is unused by the feature-only critic.
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.state.deepspeed_plugin is not None:
        config = accelerator.state.deepspeed_plugin.deepspeed_config
        # The shared training YAML controls batch size, accumulation and clipping.
        config.update({
            "train_micro_batch_size_per_gpu": settings.batch_size,
            "gradient_accumulation_steps": settings.gradient_accumulation_steps,
            "train_batch_size": settings.batch_size * settings.gradient_accumulation_steps * accelerator.num_processes,
            "gradient_clipping": settings.max_grad_norm,
        })
        config.setdefault("zero_optimization", {})["stage3_gather_16bit_weights_on_model_save"] = True
    return accelerator


def train(cfg, data_loader, device):
    settings = cfg.recap.training
    for key in ("batch_size", "max_steps", "gradient_accumulation_steps", "eval_interval", "save_interval", "logging_steps"):
        if not isinstance(settings[key], int) or isinstance(settings[key], bool) or settings[key] < 1:
            raise ValueError(f"recap.training.{key} must be a positive integer")
    if (not isinstance(settings.num_workers, int) or settings.num_workers < 0
            or not 0 < settings.learning_rate < float("inf")
            or not 0 < settings.max_grad_norm < float("inf")
            or not 0 <= settings.weight_decay < float("inf")):
        raise ValueError("invalid worker count, learning rate, weight decay or gradient clipping")
    if settings.mixed_precision not in ("no", "bf16"):
        raise ValueError("recap.training.mixed_precision must be no or bf16")
    accelerator = make_accelerator(settings, device)
    seed_everything(cfg.seed)
    dataset = data_loader.dataset
    store = getattr(dataset, "value_targets", None)
    if store is None or dataset.sampling_mode != "sequential":
        raise ValueError("value training requires return_value_targets: true and sampling_mode: sequential")
    train_ids, val_ids = store.split(settings.validation_fraction, cfg.seed)
    training_data = Subset(dataset, store.frame_indices(train_ids))
    validation_data = Subset(dataset, store.frame_indices(val_ids))
    split = {
        label: [[store.episodes[i].dataset_name, store.episodes[i].episode_index] for i in ids]
        for label, ids in (("train", train_ids), ("validation", val_ids))
    }
    contract = resume_contract(cfg, store, split, accelerator.num_processes)
    resume = settings.resume_from_checkpoint
    progress = {"step": 0, "epoch": 0, "batches_in_epoch": 0}
    if resume:
        resume = Path(resume).resolve()
        validate_checkpoint(cfg, resume / "critic.pt")
        progress = json.loads((resume / "progress.json").read_text())
        if progress["resume_contract"] != contract:
            raise ValueError("resume config/data/world size changed; use --checkpoint for a warm start")
        if not (resume / "training_state").is_dir():
            raise ValueError("resume checkpoint has no Accelerate training state")
        if settings.max_steps <= progress["step"]:
            raise ValueError("max_steps must exceed the checkpoint's completed steps")
    elif cfg.recap.value_model.checkpoint:
        validate_checkpoint(cfg, cfg.recap.value_model.checkpoint)
    output = Path(settings.output_dir)
    if output.exists() and (not resume or not (output / "split.json").is_file()
                            or json.loads((output / "split.json").read_text()) != split):
        raise FileExistsError(f"output already exists or has an incompatible split: {output}")
    synchronize(accelerator)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=bool(resume))
        write_json(output / "split.json", split)
        OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    synchronize(accelerator)
    # Accelerate restores the prepared model; a warm-start path may have moved.
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if resume:
        model_cfg.recap.value_model.checkpoint = None
    critic = QwenRecap(model_cfg)
    if settings.gradient_checkpointing and not cfg.recap.value_model.freeze_backbone:
        backbone = critic.qwen_vl_interface.model
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.config.use_cache = False
    optimizer = torch.optim.AdamW(
        [p for p in critic.parameters() if p.requires_grad],
        lr=settings.learning_rate, weight_decay=settings.weight_decay,
    )
    dataset_py = cfg.datasets.vla_data.get("dataset_py", "lerobot_datasets")
    loader = build_dataloader(cfg, dataset_py, dataset=training_data, profile="value",
                              batch_size=settings.batch_size)
    validation_loader = build_dataloader(cfg, dataset_py, dataset=validation_data, profile="value",
                                         mode="val", batch_size=settings.batch_size)
    # PIL images and strings are encoded and placed by the existing Qwen processor.
    critic, optimizer, loader = accelerator.prepare(critic, optimizer, loader)
    validation_loader = accelerator.prepare_data_loader(validation_loader, device_placement=False)
    if resume:
        accelerator.load_state(str(resume / "training_state"))
    epoch, offset = progress["epoch"], progress["batches_in_epoch"]
    loader.set_epoch(epoch)
    iterator = iter(accelerator.skip_first_batches(loader, offset) if offset else loader)
    critic.train()
    optimizer.zero_grad(set_to_none=True)
    accelerator.print(json.dumps({
        "world_size": accelerator.num_processes, "mixed_precision": settings.mixed_precision,
        "global_batch_size": settings.batch_size * settings.gradient_accumulation_steps * accelerator.num_processes,
        "freeze_backbone": cfg.recap.value_model.freeze_backbone, "resumed_step": progress["step"],
    }))
    log_context = (output / "metrics.jsonl").open("a" if resume else "x", encoding="utf-8") if accelerator.is_main_process else nullcontext()
    with log_context as log:
        for step in range(progress["step"] + 1, settings.max_steps + 1):
            total = torch.zeros(6, device=accelerator.device, dtype=torch.float64)
            for _ in range(settings.gradient_accumulation_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch, offset = epoch + 1, 0
                    loader.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
                offset += 1
                with accelerator.accumulate(critic):
                    with accelerator.autocast():
                        result = critic(batch)
                        loss = result["value_loss"]
                    finite = accelerator.reduce(torch.isfinite(loss).int(), reduction="sum")
                    if int(finite) != accelerator.num_processes:
                        raise FloatingPointError(f"nonfinite value loss at step {step}")
                    rows = metric_rows(result)
                    total[:-1] += rows.double().sum(0)
                    total[-1] += len(rows)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(critic.parameters(), settings.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            metrics = {"step": step, **{f"train_{key}": value for key, value in
                       metric_means(accelerator.reduce(total, reduction="sum")).items()}}
            if len(validation_data) and (step % settings.eval_interval == 0 or step == settings.max_steps):
                metrics.update({f"validation_{key}": value for key, value in evaluate(critic, validation_loader, accelerator).items()})
            if accelerator.is_main_process:
                log.write(json.dumps(metrics) + "\n")
                log.flush()
            if step == 1 or step % settings.logging_steps == 0 or step == settings.max_steps:
                accelerator.print(json.dumps(metrics), flush=True)
            if step % settings.save_interval == 0 and step != settings.max_steps:
                save_checkpoint(output / f"step_{step:07d}", critic, accelerator, step, cfg, split,
                                {"epoch": epoch, "batches_in_epoch": offset, "resume_contract": contract})
        final = output / "final"
        if final.exists():
            final = output / f"final_step_{step:07d}"
        checkpoint = save_checkpoint(final, critic, accelerator, step, cfg, split,
                                     {"epoch": epoch, "batches_in_epoch": offset, "resume_contract": contract})
    accelerator.print(f"Saved trained critic: {checkpoint}", flush=True)
    return checkpoint


def main(argv=None):
    parser = parser_for(__doc__)
    args = parser.parse_args(argv)
    cfg = load_config(args)
    seed_everything(cfg.seed)
    loader = build_dataloader(cfg, cfg.datasets.vla_data.get("dataset_py", "lerobot_datasets"), profile="value")
    describe_store(loader.dataset.value_targets)
    if not args.check_data:
        train(cfg, loader, args.device)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
