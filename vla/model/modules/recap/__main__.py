"""Real-weight smoke test: python -m vla.model.modules.recap --help."""

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from .recap import QwenRecap


def main():
    parser = argparse.ArgumentParser(description="Load and smoke-test the configured RECAP value component.")
    parser.add_argument("--config_yaml", default=str(Path(__file__).parent / "configs/qwen3_vl_2b.yaml"))
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
    print(f"[load] backbone={model.config.recap.qwenvl.base_vlm}, "
          f"hidden_size={model.config.recap.value_model.hidden_size}, "
          f"bins={model.value_model.bin_values.numel()}, device={args.device}")

    image = Image.new("RGB", (64, 64), color=(96, 128, 160))
    batch = [
        {"image": [image], "lang": "Close the box.", "return": -0.5},
        {"image": [image], "lang": "Fold the shirt and put it on the table.",
         "return": -0.25, "state": np.zeros((1, 7), dtype=np.float32)},
    ]
    model.train()
    with torch.set_grad_enabled(args.backward):
        loss = model(batch)["value_loss"]
    assert loss.ndim == 0 and torch.isfinite(loss), "nonfinite value training loss"
    print(f"[train] value_loss={loss.item():.6f}")
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


if __name__ == "__main__":
    main()
