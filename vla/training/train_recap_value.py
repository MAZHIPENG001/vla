"""Train a RECAP critic: python -m vla.training.train_recap_value."""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vla.training.recap_utils import (
    describe_store, load_config, parser_for, save_checkpoint, seed_everything,
    single_process_only, validate_checkpoint, write_json,
)
from vla.dataloader.recap_dataset import ReturnDataset, build_episode_store, collate_examples
from vla.model.modules.recap.recap import QwenRecap


@torch.no_grad()
def evaluate(critic, loader):
    previous = critic.training
    critic.eval()
    total_loss = total_error = count = 0
    try:
        for batch in loader:
            loss = critic(batch)["value_loss"]
            values = critic.predict_value(batch)["values"].float().cpu()
            targets = torch.tensor([example["return"] for example in batch])
            total_loss += float(loss) * len(batch)
            total_error += float((values - targets).abs().sum())
            count += len(batch)
    finally:
        critic.train(previous)
    return {"value_loss": total_loss / count, "value_mae": total_error / count} if count else {}


def train(cfg, store, device):
    settings = cfg.recap.training
    for key in ("batch_size", "max_steps", "gradient_accumulation_steps", "eval_interval", "save_interval", "logging_steps"):
        if not isinstance(settings[key], int) or isinstance(settings[key], bool) or settings[key] < 1:
            raise ValueError(f"recap.training.{key} must be a positive integer")
    if settings.num_workers < 0 or settings.learning_rate <= 0 or settings.max_grad_norm <= 0:
        raise ValueError("invalid worker count, learning rate or gradient clipping")
    train_ids, val_ids = store.split(settings.validation_fraction, cfg.seed)
    training_data, validation_data = ReturnDataset(store, train_ids), ReturnDataset(store, val_ids)
    split = {
        label: [[store.episodes[i].dataset_name, store.episodes[i].episode_index] for i in ids]
        for label, ids in (("train", train_ids), ("validation", val_ids))
    }
    if cfg.recap.value_model.checkpoint:
        validate_checkpoint(cfg, cfg.recap.value_model.checkpoint)
    output = Path(settings.output_dir)
    output.mkdir(parents=True, exist_ok=False)  # protect previous training runs
    write_json(output / "split.json", split)
    critic = QwenRecap(cfg).to(device)
    if settings.gradient_checkpointing and not cfg.recap.value_model.freeze_backbone:
        backbone = critic.qwen_vl_interface.model
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.config.use_cache = False
    optimizer = torch.optim.AdamW(
        [p for p in critic.parameters() if p.requires_grad],
        lr=settings.learning_rate, weight_decay=settings.weight_decay,
    )
    loader = DataLoader(
        training_data, batch_size=settings.batch_size, shuffle=True, num_workers=settings.num_workers,
        collate_fn=collate_examples, generator=torch.Generator().manual_seed(cfg.seed),
    )
    validation_loader = DataLoader(
        validation_data, batch_size=settings.batch_size, num_workers=settings.num_workers,
        collate_fn=collate_examples,
    )
    iterator = iter(loader)
    critic.train()
    with (output / "metrics.jsonl").open("x", encoding="utf-8") as log:
        for step in range(1, settings.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(settings.gradient_accumulation_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                loss = critic(batch)["value_loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite value loss at step {step}")
                (loss / settings.gradient_accumulation_steps).backward()
                loss_sum += float(loss.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), settings.max_grad_norm, error_if_nonfinite=True,
            )
            optimizer.step()
            metrics = {"step": step, "train_value_loss": loss_sum / settings.gradient_accumulation_steps, "grad_norm": float(grad_norm)}
            if len(validation_data) and (step % settings.eval_interval == 0 or step == settings.max_steps):
                metrics.update({f"validation_{key}": value for key, value in evaluate(critic, validation_loader).items()})
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            if step == 1 or step % settings.logging_steps == 0 or step == settings.max_steps:
                print(json.dumps(metrics), flush=True)
            if step % settings.save_interval == 0 and step != settings.max_steps:
                save_checkpoint(output / f"step_{step:07d}", critic, optimizer, step, cfg, split)
        checkpoint = save_checkpoint(output / "final", critic, optimizer, step, cfg, split)
    print(f"Saved trained critic: {checkpoint}", flush=True)
    return checkpoint


def main(argv=None):
    parser = parser_for(__doc__)
    args = parser.parse_args(argv)
    single_process_only()
    cfg = load_config(args)
    seed_everything(cfg.seed)
    store = build_episode_store(cfg)
    describe_store(store)
    if not args.check_data:
        train(cfg, store, args.device)


if __name__ == "__main__":
    main()
