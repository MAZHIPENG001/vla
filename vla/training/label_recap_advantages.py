"""Label complete trajectories: python -m vla.training.label_recap_advantages."""
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch

from vla.training.recap_utils import (
    atomic_write, describe_store, load_config, parser_for, seed_everything,
    single_process_only, validate_checkpoint, write_json,
)
from vla.dataloader.advantage_labels import write_advantage_labels
from vla.dataloader.recap_dataset import build_episode_store, relative_path
from vla.model.modules.recap.advantage import compute_advantages
from vla.model.modules.recap.recap import QwenRecap


@torch.no_grad()
def label(cfg, store, device, *, overwrite=False):
    checkpoint = cfg.recap.value_model.checkpoint
    if checkpoint is None:
        checkpoint = str(Path(cfg.recap.training.output_dir) / "final/critic.pt")
    provenance = validate_checkpoint(cfg, checkpoint)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.recap.value_model.checkpoint = str(Path(checkpoint).resolve())
    labels_config = cfg.datasets.vla_data.get("advantage_labels", {})
    if labels_config.get("source", "none") != "sidecar":
        raise ValueError("set datasets.vla_data.advantage_labels.source: sidecar for label export/policy training")
    relative = relative_path(labels_config.get("path", "meta/advantage_labels.jsonl"))
    targets = [ds.dataset_path / relative for ds in store.datasets]
    for path in targets:
        if path.exists() and not overwrite:
            raise FileExistsError(f"labels already exist: {path}; use --overwrite_labels for explicit replacement")
    output = Path(cfg.recap.labeling.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    critic = QwenRecap(cfg).to(device).eval()
    settings = critic.config.recap.advantage
    results = []
    for i, episode in enumerate(store.episodes):
        # Bounded image memory: keep only scalar values across the trajectory.
        values = torch.cat([
            critic.predict_value(batch)["values"].float().cpu()
            for batch in store.batches(episode, settings.prediction_batch_size)
        ])
        advantages = compute_advantages(
            episode.rewards[None], values[None],
            mode=settings.mode, n_steps=settings.n_steps,
        )[0]
        if not torch.isfinite(advantages).all():
            raise FloatingPointError(f"nonfinite advantages in {episode.dataset_name}/{episode.episode_index}")
        results.append((episode, values, advantages))
        print(f"[value] episode {i + 1}/{len(store.episodes)}: {episode.dataset_name}/{episode.episode_index}", flush=True)
    # One calibration over all valid frames per task, never one threshold per batch.
    advantages = torch.cat([scores for _, _, scores in results])
    task_ids = torch.cat([
        torch.full((ep.length,), store.task_ids[ep.task_key], dtype=torch.long) for ep, _, _ in results
    ])
    critic.fit_thresholds(advantages, task_ids)
    interventions = torch.cat([ep.interventions for ep, _, _ in results])
    condition = critic.make_condition(advantages, task_ids, intervention_mask=interventions)
    assert condition.conditioning_mask.all(), "offline labels must not use conditioning dropout"
    indicators = condition.indicator.split([ep.length for ep, _, _ in results])
    with (output / "scores.jsonl").open("x", encoding="utf-8") as scores_file:
        for (ep, values, scores), labels in zip(results, indicators):
            for frame in range(ep.length):
                scores_file.write(json.dumps({
                    "dataset_name": ep.dataset_name, "episode_index": ep.episode_index,
                    "frame_index": frame, "task_key": ep.task_key,
                    "value": float(values[frame]), "return": float(ep.returns[frame]),
                    "advantage": float(scores[frame]), "advantage_indicator": bool(labels[frame]),
                    "is_intervention": bool(ep.interventions[frame]),
                }, ensure_ascii=False) + "\n")
    for dataset_index, path in enumerate(targets):
        def records():
            for (ep, _, _), labels in zip(results, indicators):
                if ep.dataset_index != dataset_index:
                    continue
                for frame in range(ep.length):
                    yield {
                        "episode_index": ep.episode_index, "frame_index": frame,
                        "advantage_indicator": bool(labels[frame]),
                        "is_intervention": bool(ep.interventions[frame]),
                    }
        # Fully validate a temporary file before replacing the dataset's label file.
        atomic_write(path, lambda temporary: write_advantage_labels(temporary, records(), overwrite=True))
        print(f"Saved labels: {path}", flush=True)
    fractions = {
        key: float(condition.indicator[task_ids == task_id].float().mean())
        for key, task_id in store.task_ids.items()
    }
    write_json(output / "metadata.json", {
        "critic_checkpoint": str(Path(checkpoint).resolve()), "critic_step": provenance["step"],
        "task_ids": store.task_ids,
        "thresholds": {key: float(critic.config.recap.advantage.thresholds[task_id]) for key, task_id in store.task_ids.items()},
        "positive_fractions_including_interventions": fractions,
        "label_paths": [str(path.resolve()) for path in targets],
    })
    OmegaConf.save(critic.config, output / "config.yaml")
    print(f"Saved scores and thresholds: {output}", flush=True)
    return targets


def main(argv=None):
    parser = parser_for(__doc__)
    parser.add_argument("--overwrite_labels", action="store_true", help="Explicitly replace existing dataset label files.")
    args = parser.parse_args(argv)
    single_process_only()
    cfg = load_config(args)
    seed_everything(cfg.seed)
    store = build_episode_store(cfg)
    describe_store(store)
    if not args.check_data:
        label(cfg, store, args.device, overwrite=args.overwrite_labels)


if __name__ == "__main__":
    main()
