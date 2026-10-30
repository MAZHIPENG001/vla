"""Validated RECAP label export for inspection; VLA training never reads labels."""
import json
from numbers import Integral
from pathlib import Path

import numpy as np


class AdvantageLabelError(ValueError):
    """Invalid label export record."""


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
