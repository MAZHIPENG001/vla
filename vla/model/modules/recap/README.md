# π*0.6 / RECAP 优势条件

依据仓库中的 `docs/pistar06.pdf`：第 IV-A、IV-B、V-B、V-C 节和附录 F。
本模块包含基于 `Qwen/Qwen3-VL-2B-Instruct` 的价值模型，以及优势和 VLA 条件文本的计算。
价值模型复用 `vla/model/modules/vlm/Qwen.py` 的加载、图像预处理和对话模板接口。
PDF 第 V-C 节描述的原始价值模型骨干为 **670M**；这里按需求改用 Qwen3-VL 2B。
提供可反向传播的价值训练损失，以及 vla/training 下的通用价值训练和离线标注入口；两者共用外部任务 YAML。

## 与 QwenOFT 一致的配置和批输入接口

`recap.py` 的 `QwenRecapDefaultConfig` + `QwenRecap` 参照 `QwenOFT.py` 组织：
使用独立的顶层 `recap` 节点合并默认值与 YAML，通过 `get_vlm_model` 实际加载 Qwen，
读取真实隐藏维度，并以 `examples: list[dict]` 作为训练和预测接口。
各环境的配置放在 `examples/<环境>/train_files/`，例如 `examples/LIBERO/train_files/starvla_cotrain_libero.yaml`，
RECAP 模块内不再保存独立 YAML。接口支持任务 YAML 路径、普通 dict 或 OmegaConf。
`framework` 保留给 VLA 策略；`recap` 配置独立的价值模型和优势计算。
该任务 YAML 同时包含 `framework`、`recap`、`datasets` 和 `trainer`，由策略和价值模型共同读取。
RECAP 在内部生成一份适配 Qwen 工厂的局部配置，不改写调用方的 `framework`，
也不会用策略的模型路径初始化价值模型。
这是可组合的价值/优势模块，直接用 `QwenRecap(cfg)` 创建；不注册为返回动作的
`build_framework` 策略，也不将 `value_loss` 命名为 `action_loss`。

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `recap.qwenvl.base_vlm` | `Qwen/Qwen3-VL-2B-Instruct` | Hub ID 或完整本地权重目录 |
| `recap.qwenvl.attn_implementation` | `sdpa` | 注意力实现 |
| `recap.value_model.num_bins` | `201` | 价值分类档数 |
| `recap.value_model.value_min/value_max` | `-1/0` | 归一化价值范围 |
| `recap.value_model.freeze_backbone` | `true` | 是否仅训练价值头 |
| `recap.value_model.checkpoint` | `null` | 训练好的价值模型 state_dict 路径 |
| `recap.advantage.mode` | `posttrain` | 优势估计公式 |
| `recap.advantage.n_steps` | `50` | 后训练前瞻步数 |
| `recap.advantage.positive_fraction` | `null` | 自动按阶段选择 30% / 40% |
| `recap.advantage.dropout_probability` | `0.3` | 训练时的条件丢弃概率 |
| `recap.advantage.prediction_batch_size` | `8` | 轨迹标注时每批观测数 |
| `recap.advantage.thresholds` | `{}` | 各任务已校准的阈值 |
| `datasets.vla_data.obs_image_size` | 从共享配置读取，可省略 | 训练、预测共用的图像大小，宽/高 |

同一份配置可同时创建策略和 RECAP，例如：

```yaml
framework:
  name: QwenOFT
  qwenvl:
    base_vlm: Qwen/Qwen3-VL-4B-Instruct
    attn_implementation: sdpa
  action_model:
    action_horizon: 8
    action_dim: 7

recap:
  name: QwenRecap
  qwenvl:
    base_vlm: Qwen/Qwen3-VL-2B-Instruct
    attn_implementation: sdpa
  value_model:
    num_bins: 201
  advantage:
    n_steps: 50

datasets:
  vla_data:
    obs_image_size: [224, 224]
```

```python
from omegaconf import OmegaConf
from vla.model.framework.base_framework import build_framework
from vla.model.modules.recap import QwenRecap

cfg = OmegaConf.load("examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
policy = build_framework(cfg)
recap = QwenRecap(cfg)
```

旧版 RECAP 示例中 `framework` 下的 `qwenvl/value_model/advantage` 应迁移到 `recap`，
策略原有的 `framework` 节点保留。运行时价值头维度与阈值写入 `recap.config.recap`；
`recap.config` 保留完整的共享配置副本，可以一并保存策略和 RECAP 配置。

```python
from omegaconf import OmegaConf
from vla.model.modules.recap import QwenRecap

cfg = OmegaConf.load("examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
cfg.recap.qwenvl.base_vlm = "/home/ma/vla/playground/Pretrained_models/Qwen3-VL-2B-Instruct"
recap = QwenRecap(cfg).to("cuda")

# batch 中每项包含：image（PIL/NumPy 图片或多相机图片列表）、lang（任务指令）、
# return（归一化的剩余轨迹回报）；可选 state 为 [1, state_dim] 的已归一化状态。
# 样例：{"image": [camera_image], "lang": "Close the box.", "return": -0.4}
recap.train()
loss = recap(batch)["value_loss"]
loss.backward()
# 实际训练循环还需 optimizer.step() 和梯度清零。

recap.eval()
values = recap.predict_value(batch)["values"]  # [观测数]，预测无需 return 标签

# episodes 为 list[list[example]]，每条仅放有效观测，包含终止样本。
# rewards 为 [轨迹数, 最大步数] 的归一化奖励，右侧可以 padding。
result = recap.predict_advantages(episodes, rewards)
# task_ids: 与 result["advantages"] 同形状、同设备的整数任务编号。
# 在代表性校准数据上拟合一次；不要在每个策略 minibatch 上重新拟合。
recap.fit_thresholds(result["advantages"], task_ids, valid_mask=result["valid_mask"])
condition = recap.make_condition(
    result["advantages"], task_ids, valid_mask=result["valid_mask"],
    intervention_mask=intervention_mask,  # 同形状 bool，人类纠正的位置
)
texts = condition.to_text()
```

`recap.train()` / `recap.eval()` 控制条件 dropout；预测价值始终临时使用 eval 模式。
`fit_thresholds` 写入 `recap.config.recap.advantage.thresholds`，可通过
`OmegaConf.save(recap.config, "calibrated.yaml")` 保存后复用。
用 `torch.save(recap.value_model.state_dict(), "critic.pt")` 保存完整价值模型，
然后将 `recap.value_model.checkpoint` 设为此文件路径即可严格恢复权重；
档数和价值范围需与训练时一致。

### 实际权重加载测试

在仓库根目录先将项目可编辑安装到运行环境（每个环境只需执行一次），再按文件路径启动：

```bash
uv pip install --python .venv/bin/python -e .
uv run vla/model/modules/recap/recap.py
```

可编辑安装让运行环境能找到 `vla`、`deployment` 等项目包；入口使用绝对导入，
无需在代码中修改 `sys.path`。修改源码后无需重新安装。

在仓库根目录运行下列命令。该入口真实加载本地 2B 权重和处理器，测试两个不同长度指令、
可选状态、价值损失、反向梯度、价值预测、轨迹优势和条件文本生成：

```bash
HF_HUB_OFFLINE=1 uv run vla/model/modules/recap/recap.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --model_id /home/ma/vla/playground/Pretrained_models/Qwen3-VL-2B-Instruct \
  --device cpu --freeze_backbone --backward
```

CPU 测试冻结骨干并对价值头反向传播；小规模 Qwen 单元测试另外覆盖骨干梯度。
省略 `--config_yaml` 时，入口默认加载上述 LIBERO 任务配置。
如需在 GPU 验证完整反向传播，在 YAML 中设置 `recap.value_model.freeze_backbone: false`，
再使用 `--device cuda --backward`。
`--checkpoint /path/to/critic.pt` 可测试加载训练后的价值模型。
不传 `--model_id` 时使用 YAML 指定的 Hub ID；加载远程模型时不要设置 `HF_HUB_OFFLINE=1`。
测试使用合成图片和标签，只验证代码链路，不代表新价值头已经学会任务价值。

## Qwen3-VL 2B 价值模型

`QwenValueModel` 默认加载模型 ID `Qwen/Qwen3-VL-2B-Instruct`，也接受本地完整权重目录。
模型定义可参考 [Qwen 官方模型页](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)。
它对当前观测的多相机图片和任务指令编码，取最后一个有效 token 的隐藏状态，
通过新建的线性层输出 201 档价值 logits，再求期望得到标量价值。
特征接口跳过词表 logits 的计算，默认关闭 KV cache。

**新建价值头是随机初始化的，必须先用任务轨迹回报训练，才能用于可靠的优势标注。**
默认冻结 Qwen 主干、仅训练价值头；将 `recap.value_model.freeze_backbone` 设为 `false` 可联合训练。
价值监督始终是归一化的剩余回报 `R_t = Σ_{k=t}^{L-1}r_k`，与优势模式中的
`pretrain` / `return_to_go` 选择无关。将连续回报量化到最近 bin 后计算交叉熵。
超出价值范围的标签会报错，需检查任务归一化或显式设置 `value_min` / `value_max`。

```python
import torch
from vla.model.modules.recap import AdvantageCondition, QwenValueModel

critic = QwenValueModel().to("cuda")
# 本地权重下载完整后也可使用：
# critic = QwenValueModel(
#     model_id="/home/ma/vla/playground/Pretrained_models/Qwen3-VL-2B-Instruct"
# ).to("cuda")

# batch_images: list[list[PIL.Image]]，每项是当前观测的多相机图片。
# instructions: list[str]，每个观测对应的任务指令。
# rewards: [B, T]，已经按与价值模型一致的尺度归一化。
# lengths: [B]，包含终止奖励的有效轨迹长度。
valid = torch.arange(rewards.shape[1], device=rewards.device)[None, :] < lengths[:, None]
masked_rewards = rewards.masked_fill(~valid, 0)
return_to_go = masked_rewards.flip(1).cumsum(1).flip(1)
# batch_images / instructions 只包含 valid 为 True 的观测，按行优先排列。

critic.train()
optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-5)
optimizer.zero_grad()
output = critic(batch_images, instructions, returns=return_to_go[valid])
output.loss.backward()
optimizer.step()
# 上面展示单步接口；实际需要在轨迹数据集上迭代训练并验证价值模型。
torch.save(critic.state_dict(), "critic.pt")  # 包含 Qwen、价值头和 bin 支撑点

# 训练充分后，用该模型标注策略训练样本；大批量观测可分批预测再拼接。
critic.eval()
predictions = critic.predict_values(batch_images, instructions)
values = torch.zeros(rewards.shape, device=predictions.device)
values[valid.to(values.device)] = predictions
advantages = AdvantageCondition.compute_advantages(rewards.to(values.device), values, lengths=lengths)
```

恢复模型时用相同的初始化参数创建 `QwenValueModel`，再调用
`critic.load_state_dict(torch.load("critic.pt", map_location="cpu", weights_only=True))`。
`predict_values` 自动关闭梯度并临时切换到评估模式，不会把价值模型梯度传入策略训练。
它的输出可接入下方阈值估计和条件计算函数。

## 计算约定

- 分布价值：`AdvantageCondition.values_from_logits` 计算 `Σ softmax(logits)[b] * bin_value[b]`。
  默认值域为 `[-1, 0]` 的等距支撑点；论文使用 201 个 bin，也可传入模型实际使用的支撑点。
- 奖励：非终止步为 `-1`，成功终止为 `0`，失败终止为 `-C_fail`。
  `failure_penalty` 必须显式指定；论文未给出统一数值。
- 后训练：`A_t = Σ_{k=t}^{min(t+N,L)-1} r_k + V_{t+N} - V_t`，默认 `N=50`。
  当 `t+N >= L` 时 bootstrap 为零。这里 `L` 是包含终止奖励的样本数。
- 预训练：`mode="pretrain"` 按附录 F 的印刷公式使用整条轨迹回报 `Σ_{k=0}^{L-1}r_k - V_t`。
  附录同时称其为 `N=T` 的估计，与从当前时刻起求和的常见定义存在歧义。
  若采用剩余回报，应显式选择 `mode="return_to_go"`：`Σ_{k=t}^{L-1}r_k - V_t`。
- 条件：`I_t = (A_t > ε_task)`，严格大于；人工纠正对应的动作强制为正。
  训练时默认以 30% 的概率删除整个条件，保留原始标签，避免将无条件误当作负条件。
- 阈值：按附录 F 的目标正样本比例估计优势的分位数。预训练默认 30%，后训练默认 40%，
  部分折衣任务可设 10%。例如正样本比例 30% 对应 **优势的 70% 分位数**。
  第 V-D 节也提到价值预测的 30% 分位数，本实现采用附录 F 的优势正样本比例约定。
  相同优势值可能导致实际比例偏离目标，不通过随机打破平局更改严格比较规则。

`rewards` 和 `values` 的形状均为 `[batch, steps]`，每行必须是一条完整轨迹，
最后一个有效样本携带终止奖励。`lengths` 指定每行有效样本数，右侧 padding 被忽略。
不应把跨轨迹拼接数据或需要末端 bootstrap 的截断片段作为完整轨迹传入。
所有标签计算均停止梯度，半精度输入至少以 float32 进行累计。

奖励和价值必须使用相同尺度。`AdvantageCondition.episode_rewards(normalization=...)` 支持每条轨迹对应任务的
归一化除数；应与价值模型训练时保持一致。函数不会裁剪回报，失败惩罚叠加步数后可能
超出 `[-1, 0]`；应按实际训练约定选择归一化参数和价值支撑点，不能仅裁剪某一侧。

## 使用示例

`advantage.py` 提供独立的 `AdvantageModule(nn.Module)`，无需加载 Qwen 即可使用。
`forward` 计算优势，`fit_thresholds` 标定并保存任务阈值，`make_condition` 生成条件。
模块没有可训练参数，所有优势和标签计算均停止梯度；`train()` / `eval()` 控制条件 dropout。

```python
from vla.model.modules.recap import AdvantageModule

advantage = AdvantageModule(mode="posttrain", n_steps=50, dropout_probability=0.3)
scores = advantage(rewards, values, lengths=lengths)  # [B, T]
thresholds = advantage.fit_thresholds(scores, task_ids, valid_mask=valid_mask)
advantage.eval()
condition = advantage.make_condition(scores, task_ids, valid_mask=valid_mask)
texts = condition.to_text()

# 单独使用时，将 thresholds 随实验配置保存，恢复时传回构造函数。
restored = AdvantageModule(thresholds=thresholds).eval()
```

`QwenRecap` 通过 `recap.advantage` 组合该模块，其原有 `predict_advantages`、
`fit_thresholds` 和 `make_condition` 接口继续可用。通过 `recap.fit_thresholds` 标定时，
阈值同时写入共享配置，供 YAML 保存和恢复；离线标注入口也使用此模块计算优势。
数值计算和校验函数均封装在 `AdvantageCondition` 中，文件不再提供独立函数。
通过类方法调用，无需先创建条件对象；`compute_advantage_condition` 返回新的
`AdvantageCondition`，保留 `indicator`、`conditioning_mask` 和 `to_text()` 接口：

```python
import torch
from vla.model.modules.recap import AdvantageCondition

lengths = torch.tensor([4, 3])
success = torch.tensor([True, False])
rewards = AdvantageCondition.episode_rewards(
    lengths, success, max_steps=4, failure_penalty=5, normalization=10,
)
# 实际使用时替换为价值模型对各时刻观测及任务指令的预测。
value_logits = torch.zeros(2, 4, 201)
values = AdvantageCondition.values_from_logits(value_logits)
advantages = AdvantageCondition.compute_advantages(rewards, values, lengths=lengths)  # N=50
valid = torch.arange(4)[None, :] < lengths[:, None]
task_ids = torch.tensor([0, 1])[:, None].expand_as(advantages)

# 示例为简洁起见使用当前数据；实际应先在独立的代表性校准数据上估计并保存阈值。
# 预训练使用示范数据（论文抽样 10k 点），后训练使用当前迭代的评估 rollout。
thresholds = AdvantageCondition.estimate_task_thresholds(
    advantages, task_ids, stage="posttrain", valid_mask=valid,
)
condition = AdvantageCondition.compute_advantage_condition(
    advantages, task_ids, thresholds, valid_mask=valid,
    intervention_mask=torch.zeros_like(valid),
    training=True, generator=torch.Generator().manual_seed(42),
)
print(condition.indicator)           # bool [2, 4]，原始正/负标签
print(condition.conditioning_mask)   # bool [2, 4]，条件是否保留
print(condition.to_text())           # 展平为 8 个字符串，丢弃或 padding 为 ""
```

将非空条件文本插入**子任务文本之后、动作 token 之前**，以使条件只影响动作预测。
`valid_mask` 也应继续用于训练损失屏蔽；空条件字符串本身不会屏蔽 padding 的损失。
部署时通常直接使用 `Advantage: positive`，无需在线计算价值或优势。
示范 SFT 阶段若需按论文将全部示范标为正，可直接构造正条件。

## 验证

在仓库根目录运行：

```bash
.venv/bin/python -m unittest discover -s vla/model/modules/recap/tests -v
```

当前 `test_advantage_module.py` 覆盖三种模式的手算公式、终止失败奖励、padding、
半精度输入的 float32 累计和梯度隔离，以及任务分位数、严格阈值、人工纠正和条件 dropout。
组件测试使用价值模型替身，验证按批预测、变长轨迹回填、父模块 train/eval 状态传播
和阈值配置的 YAML 保存/恢复，无需下载或加载 Qwen 权重。

## Advantage 条件策略：QwenOFTRecap

下一步策略已实现于 `vla/model/framework/VLM4A/QwenOFTRecap.py`，注册名称为
`QwenOFTRecap`，继承原 `QwenOFT` 的动作查询、MLP 动作头、L1 loss 和预测接口。
任务配置位于 `examples/LIBERO/train_files/starvla_oft_recap_libero.yaml`：
`framework` 配置策略，`recap` 配置独立价值模型，仍共用一份任务 YAML。
开启优势条件后，策略在每个训练 batch 中使用 critic 即时计算条件，并按配置冻结或训练 critic。
关闭优势条件则进行普通 OFT 训练。详见
[VLA 在线计算与冻结选项](../../../../docs/value_training.md#vla-在线计算与冻结选项)。

策略输入顺序为：图像 + 完整任务提示/状态 → 可选子任务文本 → Advantage 条件 → 动作查询。
条件只影响后面的动作查询；CoT_prompt 模板先完成展开，条件和动作查询再追加。
这里保留 OFT 的 L1 动作回归，不包含 flow matching、子任务生成训练或 CFG 动作混合。

训练样本保留原来的 `image/lang/state/action` 字段；共享 DataLoader 自动增加当前帧的
`return`、`task_key`、`is_intervention`、`advantage_reward` 和 `advantage_bootstrap`。
bootstrap 观测只包含图像、指令及可选状态，不包含动作或优势标签。
`subtask` 为可选文本，若有则放在 Advantage 之前。

```python
from omegaconf import OmegaConf
from vla.dataloader import build_dataloader
from vla.model.framework.base_framework import build_framework

cfg = OmegaConf.load("examples/LIBERO/train_files/starvla_oft_recap_libero.yaml")
# 先配置已训练的 recap.value_model.checkpoint 和 recap.advantage.task_thresholds。
loader = build_dataloader(cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
policy = build_framework(cfg).to("cuda")
policy.train()
output = policy(next(iter(loader)))
output.get("loss", output["action_loss"]).backward()

policy.eval()
actions = policy.predict_action(observations)["normalized_actions"]  # 默认正优势条件
negative = policy.predict_action(observations, advantage="negative")["normalized_actions"]
unconditional = policy.predict_action(observations, advantage="unconditional")["normalized_actions"]
```

在线逻辑直接集成在 `QwenRecap` 内：策略创建 `QwenRecap(config, online=True)`，
调用 `recap(examples, online=True)` 获取 `condition`、`advantages` 和可选的 `value_loss`。
普通 `recap(examples)` 仍执行价值监督训练，`predict_value` 仍返回预测价值。

`framework.advantage_conditioning` 包含：

- `enabled: true`：在线计算优势；设为 false 时不添加条件、不加载 critic。
- `freeze_critic: true`：冻结整个 critic；false 时增加 MC value loss。
- `value_loss_weight: 1.0`：解除冻结时总损失中 value loss 的权重。
- `dropout_probability: 0.3`：训练时以 30% 概率删去条件文本。
- `inference_condition: positive`：部署默认正优势，可选 negative/unconditional。

训练不读取预存正负标签，不接受调用方传入 `advantage_condition` 或固定 `advantage`。
旧配置中的 `source` 和 `sft_positive` 已移除，需要删除。
人工接管与 dropout 均由在线模块处理一次；评估模式关闭 dropout。

真实权重测试（先在 YAML 中配置校准后的 task_thresholds；部署策略尺寸由 framework 独立指定）：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m examples.LIBERO.test_oft_recap \
  --model_id /home/ma/vla/playground/Pretrained_models/Qwen3-VL-2B-Instruct \
  --critic_checkpoint /path/to/recap_value/final/critic.pt \
  --freeze_backbone --backward
```

该测试真实加载策略和 critic，使用合成回报目标验证在线条件、动作头反向传播和三种推理条件。
随机动作头尚未经过机器人任务训练；测试成功不代表策略表现有所提升。

## 数据加载器与在线优势

已有完整轨迹可直接复用，不需要逐帧标签文件。先配置奖励来源、训练过的 critic 和按任务校准的阈值，
再由共享 DataLoader 读取当前观测、奖励目标与必要的后续观测，策略在模型设备即时计算优势。
`posttrain` 的 bootstrap 到达轨迹末端时为零；任务阈值由
`recap.advantage.task_thresholds` 配置，不能用每个随机 batch 的分位数替代。

Dataset 已移除 sidecar/Parquet 优势标签读取；`datasets.vla_data.advantage_labels` 不再使用。
`label_recap_advantages` 仅作为独立检查与阈值校准工具，导出文件不参与 VLA 训练。

## 完整价值训练与离线标注入口

通用入口位于 `vla/training`，通过必填的 `--config_yaml` 选择环境，
价值训练、离线标注和策略训练共用传入的任务 YAML。
`examples/<环境>/train_files/` 保存环境配置；下文以
`examples/LIBERO/train_files/starvla_oft_recap_libero.yaml` 为例。
旧的 `python -m examples.LIBERO...` 入口已迁移，使用下面的新模块路径：

- `python -m vla.training.train_recap_value`：完整轨迹 → 剩余回报 → 价值训练／验证 → checkpoint。
- `python -m vla.training.label_recap_advantages`：训练好的 checkpoint → 连续优势 → 任务阈值 → 数据集标签。
- 数据加载共用 `vla/dataloader/build_dataloader` 和原有 LeRobot Dataset；
  `vla/dataloader/recap_targets.py` 只计算标量 Reward/Return，CLI 和 checkpoint 工具在 `vla/training/recap_utils.py`。

当前数据适配器支持仓库已注册的 LeRobot 数据集，通过
`datasets.vla_data.data_root_dir` 和 `data_mix` 选择数据，通过 `recap.data`
指定任务、结果和接管字段。其他环境使用同一套训练／标注代码和各自的配置。
其他数据格式仍需实现完整轨迹读取适配，不能仅靠更换 YAML 自动支持；
这里提供的是离线轨迹训练流程，不包含各仿真环境的在线交互与采集逻辑。

训练入口复用 Accelerate，支持 CPU、单 GPU、torchrun DDP 和项目现有的 DeepSpeed 配置。
默认 bf16，默认冻结 Qwen 主干、仅训练价值头；策略网络不参与这一步。
标签导出使用单进程。完整命令与续训说明见 [训练指南](../../../../docs/value_training.md)。

### 1. 准备轨迹结果和奖励配置

默认在每个 LeRobot 数据集根目录放 `meta/recap_episodes.jsonl`，每条轨迹恰好一行：

```jsonl
{"episode_index": 0, "success": true, "complete": true}
{"episode_index": 1, "success": false, "complete": true}
```

这里的 complete 表示轨迹已经真正结束。最后一条存储的 observation 被视为终止样本；
成功终止奖励为 0，失败终止奖励为负的 failure_penalty，其他时刻为 -1。
如果原数据最后一行仍然是终止前 observation，需要先对齐终止语义／补齐终止观测。
截断轨迹不能直接套用终止 Monte Carlo 标签。只有图像和动作、没有任务结果时，
脚本不会猜测成功与否。

也支持以下显式选择：

- `recap.data.success_source: column`：读取每条轨迹最后一行的
  `recap.data.success_column`，默认 `success`，必须是布尔值。
- `recap.data.success_source: all_success`：仅适用于你已确认全部完整且成功的演示数据。
- `recap.data.truncated_column`：可指定截断标志列，发现 true 就拒绝处理。
- `recap.data.intervention_column`：可指定逐帧布尔人工接管列，对应标签强制为 positive；
  未指定时视为无接管，指定后缺列会报错。

任务由 `recap.data.task_column`（默认 task_index）识别，同一轨迹必须只有一个任务。
内部任务键为 `数据集名称:任务ID`，避免不同数据集的局部 task_index 冲突。
frame_index 必须从 0 连续递增；脚本按原始轨迹读取所有帧，不使用策略的随机混合采样或删帧结果。
图像／状态／文本复用原有加载与归一化，变换置为 eval；带未来观测的窗口会被拒绝。
原 Dataset 保留 VLA 的 action 等字段，并按配置附加帧信息和回报；
价值模型只将图像、语言和可选状态编码为输入，动作、成功标志和未来回报不进入主干。
`return_frame_info`、`return_value_targets`、`sampling_mode` 由配置控制，
普通 VLA 的默认字段与随机采样保持原样；Value 入口选择 `profiles.value` 配置。

共享 YAML 的两个奖励参数必须先填写：

```yaml
recap:
  reward:
    failure_penalty: 100   # 仅为配置示例，不是论文统一参数
    normalization: 1000   # 仅当全部回报都落在 [-1, 0] 时才适用
    task_overrides: {}
```

也可设置 `recap.reward.source: column` 和 `recap.reward.column: next.reward`，
读取已有逐帧奖励（与当前 observation 对齐）。此模式无需成功清单或 failure_penalty，
仍要求完整轨迹、固定 normalization 和匹配的价值支持范围。

所有奖励先除以 normalization，然后从轨迹尾部累计得到每帧 return。
对长度 L 的失败轨迹，初始回报是 `-(L - 1 + failure_penalty) / normalization`。
尺度在同一任务内固定，训练与标注一致；超出价值支持范围会报错，不会静默截断。
不同任务可在 `task_overrides` 下按 `"数据集名称:任务ID"`
设置各自的 failure_penalty、normalization。默认 null 是必填占位，不能直接开始训练。

### 2. 检查元数据并训练

从仓库根目录执行。以下命令假设已经在共享 YAML 填好奖励配置和路径：

```bash
NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python -m vla.training.train_recap_value \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml --check_data

NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python -m vla.training.train_recap_value \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml --device cuda
```

`--check_data` 读取轨迹表并验证结果、索引和回报，不加载 Qwen，也不逐帧解码视频。
底层 LeRobot 读取需要 Parquet 引擎（如 pyarrow）和配置的视频后端（本例 pyav）。
缺少数据、结果清单或依赖时必须先补齐。

训练参数都在 `recap.training`，独立于策略的 `trainer`：
batch_size、max_steps（优化器更新次数）、gradient_accumulation_steps、
learning_rate、validation_fraction、eval_interval、save_interval 等。
每个任务按完整轨迹划分验证集，避免相邻帧泄漏；只有一条轨迹的任务保留在训练集。
如果整个数据集都无法划出验证轨迹，会报错；可明确设置 validation_fraction: 0 关闭验证。
训练均匀采样所有训练帧，不使用策略数据混合器的采样权重。

默认产物：

```text
playground/Checkpoints/recap_value/
  config.yaml                  # 完整训练配置
  split.json
  metrics.jsonl                 # 训练/验证 CE、MAE、MSE、预测 Value 均值与真实 Return 均值
  step_0001000/...              # 周期保存
  final/
    critic.pt                  # QwenValueModel.state_dict，含主干、价值头、bins
    training_state/            # Accelerate 模型、优化器、每 rank RNG 和采样器状态
    progress.json              # 优化器 step、epoch、已消费 batch 数、恢复约定
    metadata.json              # 奖励尺度、输入约定和训练划分
    config.yaml                # 完整共享配置，checkpoint 指向此 critic.pt
```

新训练的输出目录已存在时拒绝覆盖。`--checkpoint` 只加载权重，是新优化器和新步数的热启动。
`--resume_from_checkpoint /path/to/step_0001000` 则恢复完整训练状态和数据位置，
可在相同运行目录或新目录继续。恢复要求相同数据、划分、随机种子、训练超参数和 world size；
`max_steps` 是目标总更新次数，可增加。已有 final 会保留，新终态保存为 final_step_XXXXXXX。
checkpoint 先写隐藏临时目录，全部 rank 保存完成后再原子发布。

### 3. 检查优势并校准任务阈值

这是可选的独立工具；任务 YAML 和共享训练默认配置均不包含 `labeling` 节点。
需要导出检查结果时，在命令行单独指定目录：

```bash
NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python -m vla.training.label_recap_advantages \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml --device cuda \
  --set recap.labeling.output_dir=/tmp/recap_inspection
```

未指定 checkpoint 时自动选择 `recap.training.output_dir` 中步数最大的完整 checkpoint；
也可用 `--checkpoint /path/to/critic.pt` 选择中间 checkpoint。
必须同时保留同目录下的 metadata.json，标注前会检查训练步数和奖励／输入约定。
单独下载的 Instruct 权重不能作为已训练的 critic 标注数据。

脚本按 prediction_batch_size 分批解码图像和计算价值，整条轨迹只缓存数值数组。
默认使用 posttrain 50 步优势；阈值在全部标注轨迹的有效帧上按任务统一标定，
不是每个 batch 单独标定。离线标签不做条件 dropout；策略训练阶段再做 dropout。

每个数据集写入 `recap.labeling.label_path`（默认 `meta/advantage_labels.jsonl`）供检查，
这些文件不会被 VLA DataLoader 读取。
`recap.labeling.output_dir` 默认 `playground/Checkpoints/recap_labels`，保存：

- `scores.jsonl`：dataset_name、episode_index、frame_index、task_key、reward、value、return、advantage、标签及接管标记。
- `metadata.json`：checkpoint、优化器步数、任务 ID 映射、阈值、实际正样本比例和标签路径。
- `config.yaml`：保留 framework 的完整共享配置，并记录 critic 路径和标定阈值。

已有标签默认拒绝覆盖。新一轮标注需要新的 labeling.output_dir；
只有明确传入 `--overwrite_labels` 才会替换旧标签。每个文件先写临时文件再替换。
校准结束后，将导出配置中的 `recap.value_model.checkpoint` 和
`recap.advantage.task_thresholds` 写回任务 YAML，再启动 VLA 在线训练。

所有配置都可通过 `--set KEY=VALUE ...` 临时覆盖；奖励覆盖必须在训练和标注两步保持一致。
推荐把最终取值写回共享 YAML，减少手工不一致。

### 验证范围

`test_value_pipeline.py` 使用真实 Parquet 轨迹表与小型编码器夹具，运行实际价值头、
Accelerate 训练和 checkpoint 代码，验证变长回报、冻结主干、梯度累积、跨 epoch 精确续训、
已有奖励列读取及校准导出。`test_online_policy.py` 覆盖在线动作训练、冻结梯度和旧标签不参与训练。`distributed_value_smoke.py` 验证双进程保存／恢复，
并将分布式验证结果与逐样本计算对比，检查补齐重复样本的排除。
这些测试不代表真实 Qwen 多 GPU／DeepSpeed 或真实视频解码已完成验证。

```bash
NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover \
  -s vla/model/modules/recap/tests -v
```
