# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""StarVLA transport adapter: observations in, denormalized action chunks out."""

import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


class ModelClient:
    def __init__(self, host: str, port: int, unnorm_key: str | None = None) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.metadata = self.client.get_server_metadata()
        self.action_chunk_size = int(self.metadata["action_chunk_size"])
        if self.action_chunk_size <= 0:
            raise ValueError(f"Invalid server action_chunk_size: {self.action_chunk_size}")
        self.include_state = self.metadata.get("include_state", True)
        self.unnorm_key = unnorm_key

    def infer(self, example: dict) -> np.ndarray:
        """Return one finite (T, 7) action chunk; the evaluator owns execution."""
        example = dict(example)
        if not self.include_state:
            example.pop("state", None)
        elif "state" in example:
            # Training packs state as float16 before OFT token discretization.
            example["state"] = np.asarray(example["state"], dtype=np.float16)
        # Match the RGB PIL resize in training's _pack_sample (default BICUBIC).
        example["image"] = [
            np.asarray(Image.fromarray(np.asarray(img)).resize((224, 224), Image.Resampling.BICUBIC))
            for img in example["image"]
        ]
        response = self.client.predict_action({"examples": [example], "unnorm_key": self.unnorm_key})
        if response.get("ok") is False or response.get("status") == "error":
            raise RuntimeError(f"Policy inference failed: {response.get('error', response)}")
        if "actions" not in response.get("data", {}):
            raise ValueError("Policy response is missing data.actions")
        actions = np.asarray(response["data"]["actions"], dtype=np.float32)
        if (
            actions.ndim != 3 or actions.shape[0] != 1
            or actions.shape[1] == 0 or actions.shape[2] != 7
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"Expected finite LIBERO actions (1, T>0, 7), got {actions.shape}")
        return actions[0]

    def close(self) -> None:
        self.client.close()
