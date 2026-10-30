"""Real-weight smoke test: python -m examples.LIBERO.test_oft_recap."""
import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from vla.model.framework.base_framework import build_framework


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", default=str(Path(__file__).parent / "train_files/starvla_oft_recap_libero.yaml"))
    parser.add_argument("--model_id", help="Optional local directory or Hub ID for the policy backbone.")
    parser.add_argument("--critic_checkpoint", help="Trained critic state_dict; task thresholds must be configured in YAML.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(7)
    cfg = OmegaConf.load(args.config_yaml)
    if args.model_id:
        cfg.framework.qwenvl.base_vlm = args.model_id
    if args.critic_checkpoint:
        cfg.recap.value_model.checkpoint = args.critic_checkpoint
    thresholds = cfg.recap.advantage.task_thresholds
    if not thresholds:
        raise ValueError("configure recap.advantage.task_thresholds before the online policy smoke test")
    task_key = next(iter(thresholds))
    model = build_framework(cfg).to(args.device)
    if args.freeze_backbone:
        model.qwen_vl_interface.requires_grad_(False)
    print(f"[load] policy={cfg.framework.name}, backbone={cfg.framework.qwenvl.base_vlm}, device={args.device}")

    image = Image.new("RGB", (64, 64), (96, 128, 160))
    horizon = model.action_horizon
    dimension = model.config.framework.action_model.action_dim
    examples = [{
        "image": [image], "lang": "Fold the shirt.", "subtask": "Grasp the sleeve.",
        "action": np.zeros((horizon, dimension), dtype=np.float32),
        "state": np.zeros((1, dimension), dtype=np.float32),
        "task_key": task_key,
        "return": (cfg.recap.value_model.value_min + cfg.recap.value_model.value_max) / 2,
        "advantage_reward": -0.05 if bootstrap else -0.1,
        "advantage_bootstrap": {"image": [image], "lang": "Fold the shirt."} if bootstrap else None,
        "is_intervention": False,
    } for bootstrap in (True, False)]
    model.train()
    with torch.set_grad_enabled(args.backward):
        output = model(examples, generator=torch.Generator().manual_seed(7))
        loss = output.get("loss", output["action_loss"])
    assert torch.isfinite(loss) and loss.ndim == 0
    print(f"[train] total_loss={loss.item():.6f}, action_loss={output['action_loss'].item():.6f}")
    if args.backward:
        loss.backward()
        gradients = [p.grad for p in model.action_model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
        model.zero_grad(set_to_none=True)
        print("[backward] finite nonzero action-head gradients")

    model.eval()
    outputs = {}
    for condition in ("positive", "negative", "unconditional"):
        actions = model.predict_action(examples[:1], advantage=condition)["normalized_actions"]
        assert actions.shape == (1, horizon, dimension) and np.isfinite(actions).all()
        outputs[condition] = actions
        print(f"[infer] {condition}: actions_shape={actions.shape}")
    assert not np.allclose(outputs["positive"], outputs["negative"])
    print("Passed. Synthetic inputs and an untrained action head validate the API, not task performance.")


if __name__ == "__main__":
    main()
