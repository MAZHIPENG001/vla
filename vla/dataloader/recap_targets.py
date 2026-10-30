"""Scalar trajectory rewards/returns; observations use the regular VLA Dataset."""
from dataclasses import dataclass, replace
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from vla.model.modules.recap.advantage import AdvantageCondition


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or str(path) == ".":
        raise ValueError("metadata/label paths must be relative to each dataset root")
    return path


def boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean, got {value!r}")
    return bool(value)


def index(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


@dataclass
class Episode:
    dataset_index: int
    dataset_name: str
    episode_index: int
    task_key: str
    rewards: torch.Tensor
    returns: torch.Tensor
    interventions: torch.Tensor

    @property
    def length(self):
        return len(self.rewards)


def online_advantage_target(episode, frame, *, mode="posttrain", n_steps=50):
    """Scalar reward target and optional bootstrap frame within this episode."""
    frame = index(frame, "frame_index")
    if frame >= episode.length:
        raise ValueError("frame_index must be inside the episode")
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if mode == "posttrain":
        end = min(frame + n_steps, episode.length)
        return float(episode.rewards[frame:end].sum()), end if end < episode.length else None
    if mode == "pretrain":
        return float(episode.rewards.sum()), None
    if mode == "return_to_go":
        return float(episode.returns[frame]), None
    raise ValueError("advantage mode must be posttrain, pretrain or return_to_go")


class RecapEpisodes:
    """Validate complete trajectory tables and cache scalar targets only.

    Episodes must be complete. The final stored row is the terminal sample,
    matching AdvantageCondition.episode_rewards. Only scalar arrays stay in RAM;
    This class neither reads observations nor implements a PyTorch Dataset.
    """
    def __init__(self, datasets, cfg, *, reuse_cached=False):
        self.datasets = list(datasets)
        self.episodes = []
        self.cfg = cfg
        data_cfg, reward_cfg = cfg.data, cfg.reward
        reward_source = reward_cfg.get("source", "episode")
        if reward_source not in ("episode", "column"):
            raise ValueError("recap.reward.source must be episode or column")
        names = [ds.dataset_name for ds in self.datasets]
        if len(names) != len(set(names)):
            raise ValueError("dataset names must be unique")
        if data_cfg.success_source not in ("manifest", "column", "all_success"):
            raise ValueError("success_source must be manifest, column or all_success")
        for dataset_index, ds in enumerate(self.datasets):
            cached = getattr(ds, "value_targets", None)
            if reuse_cached and cached is not None:
                self.episodes.extend(replace(ep, dataset_index=dataset_index) for ep in cached.episodes)
                continue
            for modality in ("video", "state", "language"):
                for key in ds.modality_keys.get(modality, []):
                    if (np.asarray(ds.delta_indices[key]) > 0).any():
                        raise ValueError(f"critic observation {key} contains future frames")
            outcomes = self._outcomes(ds) if reward_source == "episode" and data_cfg.success_source == "manifest" else {}
            seen = set()
            for trajectory_id, length in zip(ds.trajectory_ids, ds.trajectory_lengths):
                trajectory_id = index(trajectory_id, "episode_index")
                length = index(length, "episode length")
                if not length or trajectory_id in seen:
                    raise ValueError("episodes must be nonempty with unique IDs")
                seen.add(trajectory_id)
                frame = ds.get_trajectory_data(trajectory_id)
                if len(frame) != length:
                    raise ValueError(f"{ds.dataset_name}/{trajectory_id}: incomplete trajectory")
                if "frame_index" not in frame or not np.array_equal(frame["frame_index"].to_numpy(), np.arange(length)):
                    raise ValueError(f"{ds.dataset_name}/{trajectory_id}: frames must be contiguous and ordered from zero")
                if "episode_index" in frame and not (frame["episode_index"] == trajectory_id).all():
                    raise ValueError("trajectory contains a different episode")
                column = data_cfg.task_column
                if column not in frame or frame[column].nunique(dropna=False) != 1:
                    raise ValueError(f"{column} must contain one task ID per episode")
                task_key = f"{ds.dataset_name}:{index(frame[column].iloc[0], column)}"
                if data_cfg.truncated_column:
                    column = data_cfg.truncated_column
                    if column not in frame:
                        raise ValueError(f"missing truncated column: {column}")
                    if any(boolean(v, column) for v in frame[column]):
                        raise ValueError("truncated episodes cannot use terminal Monte Carlo targets")
                if reward_source == "column":
                    success = None  # Existing rewards do not require success metadata.
                elif data_cfg.success_source == "manifest":
                    if trajectory_id not in outcomes:
                        raise ValueError(f"missing outcome for {ds.dataset_name}/{trajectory_id}")
                    success = outcomes[trajectory_id]
                elif data_cfg.success_source == "column":
                    column = data_cfg.success_column
                    if column not in frame:
                        raise ValueError(f"missing success column: {column}")
                    success = boolean(frame[column].iloc[-1], column)
                else:
                    success = True  # explicit opt-in for known complete successful demos
                overrides = reward_cfg.task_overrides.get(task_key, {})
                penalty = overrides.get("failure_penalty", reward_cfg.failure_penalty)
                scale = overrides.get("normalization", reward_cfg.normalization)
                if scale is None or not np.isfinite(float(scale)) or float(scale) <= 0:
                    raise ValueError(f"set positive recap.reward.normalization or task_overrides[{task_key!r}]")
                if reward_source == "column":
                    column = reward_cfg.column
                    if column not in frame:
                        raise ValueError(f"missing reward column: {column}")
                    rewards = torch.as_tensor(frame[column].to_numpy(dtype=np.float32)) / float(scale)
                else:
                    if penalty is None:
                        raise ValueError(f"set recap.reward.failure_penalty or task_overrides[{task_key!r}]")
                    rewards = AdvantageCondition.episode_rewards(
                        torch.tensor([length]), torch.tensor([success]), max_steps=length,
                        failure_penalty=float(penalty), normalization=float(scale),
                    )[0]
                if not torch.isfinite(rewards).all():
                    raise ValueError(f"{task_key}/{trajectory_id}: rewards must be finite")
                returns = rewards.flip(0).cumsum(0).flip(0)
                bounds = cfg.value_model
                if ((returns < bounds.value_min) | (returns > bounds.value_max)).any():
                    raise ValueError(f"{task_key}/{trajectory_id}: returns outside value support; adjust fixed task normalization")
                column = data_cfg.intervention_column
                if column:
                    if column not in frame:
                        raise ValueError(f"missing intervention column: {column}")
                    interventions = torch.tensor([boolean(v, column) for v in frame[column]])
                else:
                    interventions = torch.zeros(length, dtype=torch.bool)
                self.episodes.append(Episode(
                    dataset_index, ds.dataset_name, trajectory_id, task_key, rewards, returns, interventions,
                ))
            if set(outcomes) - seen:
                raise ValueError(f"outcome manifest has unknown episodes in {ds.dataset_name}")
        if not self.episodes:
            raise ValueError("no complete trajectories found")
        self.task_ids = {key: i for i, key in enumerate(sorted({ep.task_key for ep in self.episodes}))}
        self.by_episode = {(ep.dataset_name, ep.episode_index): ep for ep in self.episodes}
        self.offsets = np.cumsum([0] + [ep.length for ep in self.episodes]).tolist()

    def _outcomes(self, ds):
        path = ds.dataset_path / relative_path(self.cfg.data.outcome_path)
        outcomes = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    episode = index(row["episode_index"], "episode_index")
                    if episode in outcomes:
                        raise ValueError("duplicate episode_index")
                    if not boolean(row["complete"], "complete"):
                        raise ValueError("incomplete episode")
                    outcomes[episode] = boolean(row["success"], "success")
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        return outcomes

    def frame_indices(self, episode_indices):
        """Indices into the regular mixture in sampling_mode=sequential."""
        return [frame for i in episode_indices for frame in range(self.offsets[i], self.offsets[i + 1])]

    def split(self, validation_fraction, seed):
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        rng = np.random.default_rng(seed)
        train, validation = [], []
        for task_key in self.task_ids:
            group = [i for i, ep in enumerate(self.episodes) if ep.task_key == task_key]
            rng.shuffle(group)
            count = min(len(group) - 1, max(1, round(len(group) * validation_fraction))) if validation_fraction else 0
            validation.extend(group[:count])
            train.extend(group[count:])
        if validation_fraction and not validation:
            raise ValueError("validation needs two episodes in a task; add data or explicitly set validation_fraction: 0")
        return train, validation

