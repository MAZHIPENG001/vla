"""LIBERO-specific observation and action conventions (no simulator imports)."""

from typing import Literal

import numpy as np

GripperEncoding = Literal["binary_open", "binary_close", "signed_close"]


def gripper_to_libero(value, encoding: GripperEncoding = "binary_open") -> np.ndarray:
    """Convert a denormalized prediction to PandaGripper: -1 open, +1 close.

    binary_open: 0 close / 1 open (released StarVLA checkpoint).
    binary_close: 0 open / 1 close.
    signed_close: -1 open / +1 close (raw LIBERO demonstrations).
    Statistics alone cannot identify the semantics of a [0, 1] encoding.
    """
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != 1 or not np.isfinite(arr).all():
        raise ValueError(f"Expected one finite gripper value, got {arr}")
    v = float(arr[0])
    if encoding == "binary_open":
        command = -1.0 if v > 0.5 else 1.0
    elif encoding == "binary_close":
        command = 1.0 if v > 0.5 else -1.0
    elif encoding == "signed_close":
        command = float(np.sign(v))  # Native zero command holds the gripper.
    else:
        raise ValueError(f"Unknown gripper encoding: {encoding}")
    return np.array([command], dtype=np.float32)


def libero_images(obs: dict) -> list[np.ndarray]:
    """Training order: primary, wrist. RLDS conversion rotates both by 180°."""
    images = [obs["agentview_image"], obs["robot0_eye_in_hand_image"]]
    return [np.ascontiguousarray(im[::-1, ::-1]) for im in images]


def libero_state(obs: dict) -> np.ndarray:
    """Current EEF position, axis-angle orientation, and two finger positions."""
    quat = np.asarray(obs["robot0_eef_quat"])
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    rotation = np.zeros(3) if den == 0 else quat[:3] * (2.0 * np.arccos(w) / den)
    return np.concatenate((obs["robot0_eef_pos"], rotation, obs["robot0_gripper_qpos"]))
