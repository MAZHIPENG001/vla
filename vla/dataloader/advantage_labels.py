"""Strict per-frame RECAP labels, separate from image/action transformations."""
import json
from numbers import Integral
from pathlib import Path

import numpy as np


class AdvantageLabelError(ValueError):
    """Invalid/missing labels must stop training, not trigger random resampling."""


def _index(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise AdvantageLabelError(f"{name} must be a nonnegative integer")
    return int(value)


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise AdvantageLabelError(f"{name} must be boolean; raw scores, nulls and strings are not labels")
    return bool(value)


def _record(record):
    if not isinstance(record, dict):
        raise AdvantageLabelError("each label record must be an object")
    try:
        episode = _index(record["episode_index"], "episode_index")
        frame = _index(record["frame_index"], "frame_index")
        indicator = _boolean(record["advantage_indicator"], "advantage_indicator")
    except KeyError as exc:
        raise AdvantageLabelError(f"label record is missing {exc.args[0]}") from exc
    intervention = _boolean(record.get("is_intervention", False), "is_intervention")
    return {
        "episode_index": episode, "frame_index": frame,
        "advantage_indicator": indicator or intervention, "is_intervention": intervention,
    }


def write_advantage_labels(path, records, *, overwrite=False):
    """Export undropped labels for ONE dataset; do not mix dataset-local IDs.

    records contains episode_index, frame_index, bool advantage_indicator and
    optional bool is_intervention. Validate all records before creating the file.
    A subsequent critic iteration needs new labels/thresholds and a new file or
    explicit overwrite=True. This function does not invent rewards or labels.
    """
    rows, seen = [], set()
    for record in records:
        row = _record(record)
        key = (row["episode_index"], row["frame_index"])
        if key in seen:
            raise AdvantageLabelError(f"duplicate label key {key}")
        seen.add(key)
        rows.append(json.dumps(row, ensure_ascii=False))
    if not rows:
        raise AdvantageLabelError("cannot write an empty label dataset")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")


class AdvantageLabelSource:
    """Read sidecar JSONL or existing Parquet columns using original frame IDs.

    source=none preserves legacy samples. source=sidecar uses a file relative to
    each dataset root, so mixed datasets may safely reuse the same episode IDs.
    source=parquet reads the current trajectory's row BEFORE transforms. Labels
    refer to the observation/action-chunk anchor, not the last future action.
    """

    def __init__(self, dataset_path, config=None):
        config = {} if config is None else config
        self.dataset_path = Path(dataset_path)
        self.source = config.get("source", "none")
        if self.source not in ("none", "sidecar", "parquet"):
            raise AdvantageLabelError("advantage_labels.source must be none, sidecar or parquet")
        self.column = config.get("column", "advantage_indicator")
        self.intervention_column = config.get("intervention_column", "is_intervention")
        self.labels = {}
        if self.source == "sidecar":
            relative = Path(config.get("path", "meta/advantage_labels.jsonl"))
            if relative.is_absolute() or ".." in relative.parts:
                raise AdvantageLabelError("sidecar path must be relative to each dataset root, without '..'")
            self.path = self.dataset_path / relative
            try:
                with self.path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        try:
                            record = _record(json.loads(line))
                            key = (record["episode_index"], record["frame_index"])
                            if key in self.labels:
                                raise AdvantageLabelError(f"duplicate label key {key}")
                            self.labels[key] = record
                        except (ValueError, TypeError) as exc:
                            raise AdvantageLabelError(f"{self.path}:{line_number}: {exc}") from exc
            except OSError as exc:
                raise AdvantageLabelError(f"Cannot read advantage labels: {self.path}: {exc}") from exc
            if not self.labels:
                raise AdvantageLabelError(f"No advantage labels in {self.path}")

    def get(self, episode_index, frame_index, trajectory_data=None):
        episode = _index(episode_index, "episode_index")
        frame = _index(frame_index, "frame_index")
        if self.source == "none":
            return {}
        if self.source == "sidecar":
            try:
                record = self.labels[(episode, frame)]
            except KeyError as exc:
                raise AdvantageLabelError(
                    f"Missing advantage label in {self.path}: episode_index={episode}, frame_index={frame}"
                ) from exc
        else:
            if trajectory_data is None or not 0 <= frame < len(trajectory_data):
                raise AdvantageLabelError("Parquet labels require the current complete trajectory")
            if self.column not in trajectory_data:
                raise AdvantageLabelError(f"Missing Parquet advantage column {self.column!r} in {self.dataset_path}")
            row = trajectory_data.iloc[frame]
            # LeRobot frame IDs must match the row offsets used to select actions.
            for key, expected in (("episode_index", episode), ("frame_index", frame)):
                if key in row and _index(row[key], key) != expected:
                    raise AdvantageLabelError(f"Parquet {key} does not match the selected observation")
            record = _record({
                "episode_index": episode, "frame_index": frame,
                "advantage_indicator": row[self.column],
                "is_intervention": row[self.intervention_column] if self.intervention_column in row else False,
            })
        return {
            "advantage_indicator": record["advantage_indicator"],
            "is_intervention": record["is_intervention"],
            "episode_index": episode,
            "frame_index": frame,
            "dataset_name": self.dataset_path.name,
        }
