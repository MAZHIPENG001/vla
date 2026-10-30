"""On-demand RECAP labels and optional supervised critic updates during VLA training."""
import math

import torch
from torch import nn

from .advantage import AdvantageCondition
from .recap import QwenRecap


class OnlineAdvantage(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config.framework.advantage_conditioning
        self.freeze_critic = cfg.freeze_critic
        weight = cfg.value_loss_weight
        if isinstance(weight, bool) or not math.isfinite(weight) or weight < 0:
            raise ValueError("value_loss_weight must be finite and nonnegative")
        if not self.freeze_critic and weight == 0:
            raise ValueError("an unfrozen critic requires value_loss_weight > 0")
        if not config.recap.value_model.checkpoint:
            raise ValueError("online advantage requires a trained recap.value_model.checkpoint")
        # Keys identify the dataset AND task, so mixtures cannot silently remap IDs.
        self.thresholds = dict(config.recap.advantage.get("task_thresholds", {}))
        if not self.thresholds or not all(math.isfinite(v) for v in self.thresholds.values()):
            raise ValueError("online advantage requires finite recap.advantage.task_thresholds (dataset_name:task_id -> threshold)")
        self.dropout_probability = cfg.dropout_probability
        self.critic = QwenRecap(config)
        if self.freeze_critic:
            self.critic.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_critic:
            self.critic.eval()
        return self

    @staticmethod
    def _observation(example):
        # Neither future actions nor labels enter the critic encoder.
        return {key: example[key] for key in ("image", "lang", "state") if key in example}

    def forward(self, examples, *, generator=None):
        if not examples:
            raise ValueError("online advantage requires a nonempty observation batch")
        for example in examples:
            if any(key not in example for key in ("advantage_reward", "advantage_bootstrap", "task_key", "return")):
                raise ValueError("online advantage requires targets from the online VLA DataLoader")
            if example["task_key"] not in self.thresholds:
                raise ValueError(f"missing calibrated threshold for task {example['task_key']}")
        observations = [self._observation(example) for example in examples]
        # Labels always use deterministic, detached predictions, even while the
        # independent MC value loss trains the critic in this same forward pass.
        values = self.critic.predict_value(observations)["values"]
        targets = values.new_tensor([example["advantage_reward"] for example in examples])
        indices = [i for i, example in enumerate(examples) if example["advantage_bootstrap"] is not None]
        if indices:
            future = [self._observation(examples[i]["advantage_bootstrap"]) for i in indices]
            targets[indices] += self.critic.predict_value(future)["values"]
        advantages = (targets - values).detach().float().cpu()
        keys = sorted({example["task_key"] for example in examples})
        ids = {key: i for i, key in enumerate(keys)}
        condition = AdvantageCondition.compute_advantage_condition(
            advantages, torch.tensor([ids[example["task_key"]] for example in examples]),
            {ids[key]: self.thresholds[key] for key in keys},
            intervention_mask=torch.tensor([example.get("is_intervention", False) for example in examples], dtype=torch.bool),
            training=self.training, dropout_probability=self.dropout_probability, generator=generator,
        )
        output = {"condition": condition, "advantages": advantages.to(values.device)}
        if self.training and not self.freeze_critic:
            training_examples = [{**observation, "return": example["return"]}
                                 for observation, example in zip(observations, examples)]
            output["value_loss"] = self.critic(training_examples)["value_loss"]
        return output
