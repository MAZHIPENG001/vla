"""Configured RECAP critic and advantage labeling, following QwenOFT's API style."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from vla.model.modules.recap.advantage import AdvantageModule
from vla.model.modules.recap.value_model import QwenValueModel

@dataclass
class QwenRecapDefaultConfig:
    """Defaults under recap; the policy's framework section is independent."""

    name: str = "QwenRecap"
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "Qwen/Qwen3-VL-2B-Instruct",  # HF ID or complete local directory.
        "attn_implementation": "sdpa",
    })
    value_model: dict = field(default_factory=lambda: {
        "num_bins": 201,
        "value_min": -1.0,
        "value_max": 0.0,
        "freeze_backbone": True,  # Default: train only the value head.
        "checkpoint": None,  # Optional QwenValueModel.state_dict(), including the head.
    })
    advantage: dict = field(default_factory=lambda: {
        "mode": "posttrain",  # posttrain | pretrain | return_to_go
        "n_steps": 50,
        "positive_fraction": None,  # None: 0.3 pretrain / 0.4 posttrain.
        "dropout_probability": 0.3,
        "prediction_batch_size": 8,  # Number of observations per critic inference.
        "thresholds": {},  # Calibrated task_id -> epsilon, required for conditions.
    })


class QwenRecap(nn.Module):
    """RECAP value component, not an action-prediction framework.

    Like QwenOFT, accepts a full config and batches of image/lang/state examples.
    forward requires a normalized scalar `return` per observation and returns a
    value_loss dictionary. predict_value needs no reward labels. No action tokens
    or advantage labels are added to the critic's inputs.
    """

    def __init__(self, config=None):
        super().__init__()
        from vla.model.framework.share_tools import _to_omegaconf
        if isinstance(config, Path):
            config = str(config)
        # Work on a copy of the full config, preserving cross-section interpolations
        # and leaving the policy's framework/datasets/trainer settings untouched.
        config = getattr(config, "_cfg", config)  # Also accept AccessTrackedConfig.
        self.config = OmegaConf.merge(
            {"recap": asdict(QwenRecapDefaultConfig()), "datasets": {"vla_data": {}}},
            _to_omegaconf(config),
        )
        self._validate_config()
        advantage = self.config.recap.advantage
        self.advantage = AdvantageModule(
            mode=advantage.mode,
            n_steps=advantage.n_steps,
            positive_fraction=advantage.positive_fraction,
            dropout_probability=advantage.dropout_probability,
            thresholds=advantage.thresholds,
        )
        cfg = self.config.recap.value_model
        if cfg.checkpoint is not None and not Path(cfg.checkpoint).is_file():
            raise FileNotFoundError(f"Value checkpoint not found: {cfg.checkpoint}")
        self.value_model = QwenValueModel(
            config=self.config,
            num_bins=cfg.num_bins,
            value_min=cfg.value_min,
            value_max=cfg.value_max,
            freeze_backbone=cfg.freeze_backbone,
        )
        # Same runtime dimension alignment as QwenOFT, never trust a YAML guess.
        cfg.hidden_size = self.value_model.value_head.in_features
        if cfg.checkpoint is not None:
            state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
            self.value_model.load_state_dict(state, strict=True)
            expected_bins = torch.linspace(cfg.value_min, cfg.value_max, cfg.num_bins)
            if not torch.allclose(self.value_model.bin_values.cpu().float(), expected_bins):
                raise ValueError("checkpoint value support does not match config")

    def _validate_config(self):
        cfg = self.config.recap
        if cfg.name != "QwenRecap":
            raise ValueError("recap.name must be QwenRecap")
        batch_size = cfg.advantage.prediction_batch_size
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("advantage.prediction_batch_size must be a positive integer")

    @property
    def qwen_vl_interface(self):
        return self.value_model.qwen_vl_interface

    def _prepare_examples(self, examples):
        from vla.model.framework.share_tools import add_discretized_state_to_instruction
        if isinstance(examples, dict):
            examples = [examples]
        if not examples:
            raise ValueError("examples must contain at least one observation")
        images, instructions = [], []
        image_size = self.config.datasets.vla_data.get("obs_image_size")
        for example in examples:
            image = to_pil_preserve(example["image"])
            views = list(image) if isinstance(image, (list, tuple)) else [image]
            if not views:
                raise ValueError("each observation must contain at least one image")
            if image_size is not None:
                # Apply the same preprocessing in both training and prediction.
                views = [view.resize(tuple(image_size)) for view in views]
            instruction = example["lang"]
            if "state" in example:
                instruction = add_discretized_state_to_instruction([instruction], [example["state"]])[0]
            images.append(views)
            instructions.append(instruction)
        return images, instructions

    def forward(self, examples: list[dict]) -> dict[str, Tensor]:
        """Train from image/lang, optional state, and normalized scalar return."""
        images, instructions = self._prepare_examples(examples)
        if isinstance(examples, dict):
            examples = [examples]
        if any("return" not in example for example in examples):
            raise ValueError("each training example requires a normalized 'return' target")
        targets = torch.as_tensor(
            [example["return"] for example in examples],
            dtype=torch.float32, device=self.value_model.bin_values.device,
        )
        output = self.value_model(images, instructions, returns=targets)
        # Per-observation losses allow distributed validation to discard padded
        # duplicate samples before averaging, including an uneven final batch.
        return {"value_loss": output.loss, "value_losses": output.losses,
                "values": output.values, "returns": targets}

    @torch.no_grad()
    def predict_value(self, examples: list[dict]) -> dict[str, Tensor]:
        """Eval-mode scalar values [observations], restoring the prior mode."""
        images, instructions = self._prepare_examples(examples)
        return {"values": self.value_model.predict_values(images, instructions)}

    @torch.no_grad()
    def predict_advantages(self, episodes: list[list[dict]], rewards: Tensor) -> dict[str, Tensor]:
        """Label complete variable-length trajectories with configured n-step/MC.

        episodes[b] contains valid observations only, including the terminal
        sample. rewards is [B, T], right padded, on the same normalized scale as
        the critic. Returns values/advantages/valid_mask [B, T] and lengths [B].
        """
        if rewards.ndim != 2 or len(episodes) != rewards.shape[0] or not episodes:
            raise ValueError("rewards must have shape [len(episodes), steps] with a nonempty batch")
        lengths = torch.tensor([len(episode) for episode in episodes], device=rewards.device)
        if ((lengths < 1) | (lengths > rewards.shape[1])).any():
            raise ValueError("each episode must contain between 1 and rewards.shape[1] observations")
        valid = torch.arange(rewards.shape[1], device=rewards.device)[None, :] < lengths[:, None]
        if not torch.isfinite(rewards[valid]).all():
            raise ValueError("valid rewards must be finite")
        examples = [example for episode in episodes for example in episode]
        batch_size = self.config.recap.advantage.prediction_batch_size
        predictions = torch.cat([
            self.predict_value(examples[start:start + batch_size])["values"]
            for start in range(0, len(examples), batch_size)
        ])
        values = torch.zeros(rewards.shape, device=predictions.device, dtype=predictions.dtype)
        valid = valid.to(values.device)
        lengths = lengths.to(values.device)
        values[valid] = predictions
        advantages = self.advantage(rewards.to(values.device), values, lengths=lengths)
        return {"values": values, "advantages": advantages, "valid_mask": valid, "lengths": lengths}

    def fit_thresholds(self, advantages, task_ids, *, valid_mask=None):
        """Calibrate on representative data and store thresholds in self.config."""
        thresholds = self.advantage.fit_thresholds(advantages, task_ids, valid_mask=valid_mask)
        self.config.recap.advantage.thresholds = thresholds
        return thresholds

    def make_condition(self, advantages, task_ids, *, valid_mask=None, intervention_mask=None, generator=None):
        """Apply saved thresholds and train/eval-dependent condition dropout."""
        return self.advantage.make_condition(
            advantages, task_ids,
            valid_mask=valid_mask, intervention_mask=intervention_mask,
            generator=generator,
        )


if __name__ == "__main__":
    import argparse
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser(description="Load and smoke-test the configured RECAP value component.")
    parser.add_argument(
        "--config_yaml",
        default=str(Path(__file__).resolve().parents[4] / "examples/LIBERO/train_files/starvla_cotrain_libero.yaml"),
        help="Shared task YAML containing framework, recap, datasets and trainer.",
    )
    parser.add_argument("--model_id", help="Override the HF model ID or local backbone directory.")
    parser.add_argument("--checkpoint", help="Optional trained QwenValueModel state_dict.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze_backbone", action="store_true", help="Only train the value head.")
    parser.add_argument("--backward", action="store_true", help="Also test loss backward; use --freeze_backbone on CPU.")
    parser.add_argument("--num_threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(0)
    cfg = OmegaConf.load(args.config_yaml)
    for key, value in (
        ("recap.qwenvl.base_vlm", args.model_id),
        ("recap.value_model.checkpoint", args.checkpoint),
    ):
        if value is not None:
            OmegaConf.update(cfg, key, value, merge=False)
    if args.freeze_backbone:
        OmegaConf.update(cfg, "recap.value_model.freeze_backbone", True)
    model = QwenRecap(cfg).to(args.device)
    print(f"\33[92m[load] backbone={model.config.recap.qwenvl.base_vlm}, "
            f"hidden_size={model.config.recap.value_model.hidden_size}, "
            f"bins={model.value_model.bin_values.numel()}, device={args.device}\33[0m]")

    # image = Image.new("RGB", (224, 224), color=(96, 128, 160))
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    batch = [
        {"image": [image], "lang": "Close the box.", "return": -0.5},
        {"image": [image], "lang": "Fold the shirt and put it on the table.",
            "return": -0.25, "state": np.zeros((1, 7), dtype=np.float32)},
    ]
    model.train()
    with torch.set_grad_enabled(args.backward):
        loss = model(batch)["value_loss"]
    assert loss.ndim == 0 and torch.isfinite(loss), "nonfinite value training loss"
    print(f"\33[93m[train] value_loss={loss.item():.6f}\33[0m")
    if args.backward:
        loss.backward()
        grad = model.value_model.value_head.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        print("[backward] finite nonzero value-head gradients")
        model.zero_grad(set_to_none=True)

    model.eval()
    prediction = model.predict_value(batch)["values"]
    assert prediction.shape == (2,) and torch.isfinite(prediction).all()
    print(f"[infer] values_shape={tuple(prediction.shape)}")
    # Two valid observations in one complete episode, normalized rewards.
    result = model.predict_advantages([batch], torch.tensor([[-0.1, 0.0]]))
    task_ids = torch.zeros_like(result["advantages"], dtype=torch.long)
    model.fit_thresholds(result["advantages"], task_ids, valid_mask=result["valid_mask"])
    condition = model.make_condition(result["advantages"], task_ids, valid_mask=result["valid_mask"])
    assert result["advantages"].shape == (1, 2)
    assert torch.isfinite(result["advantages"]).all()
    assert condition.conditioning_mask.all()
    print(f"[recap] advantages_shape={tuple(result['advantages'].shape)}, conditions={condition.to_text()}")
    print("Passed. Synthetic data only; a newly initialized value head is not a trained critic.")
