# π*0.6 / RECAP 优势条件

依据仓库中的 `docs/pistar06.pdf`：第 IV-A、IV-B、V-B、V-C 节和附录 F。
本模块包含基于 `Qwen/Qwen3-VL-2B-Instruct` 的价值模型，以及优势和 VLA 条件文本的计算。
价值模型复用 `vla/model/modules/vlm/Qwen.py` 的加载、图像预处理和对话模板接口。
PDF 第 V-C 节描述的原始价值模型骨干为 **670M**；这里按需求改用 Qwen3-VL 2B。
提供可反向传播的价值训练损失，但没有自动修改现有策略训练器或启动训练任务。

## Qwen3-VL 2B 价值模型

`QwenValueModel` 默认加载模型 ID `Qwen/Qwen3-VL-2B-Instruct`，也接受本地完整权重目录。
模型定义可参考 [Qwen 官方模型页](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)。
它对当前观测的多相机图片和任务指令编码，取最后一个有效 token 的隐藏状态，
通过新建的线性层输出 201 档价值 logits，再求期望得到标量价值。
特征接口跳过词表 logits 的计算，默认关闭 KV cache。

**新建价值头是随机初始化的，必须先用任务轨迹回报训练，才能用于可靠的优势标注。**
默认联合训练 Qwen 和价值头；`freeze_backbone=True` 可仅训练价值头。
价值监督始终是归一化的剩余回报 `R_t = Σ_{k=t}^{L-1}r_k`，与优势模式中的
`pretrain` / `return_to_go` 选择无关。将连续回报量化到最近 bin 后计算交叉熵。
超出价值范围的标签会报错，需检查任务归一化或显式设置 `value_min` / `value_max`。

```python
import torch
from vla.model.modules.recap import QwenValueModel, compute_advantages

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
advantages = compute_advantages(rewards.to(values.device), values, lengths=lengths)
```

恢复模型时用相同的初始化参数创建 `QwenValueModel`，再调用
`critic.load_state_dict(torch.load("critic.pt", map_location="cpu", weights_only=True))`。
`predict_values` 自动关闭梯度并临时切换到评估模式，不会把价值模型梯度传入策略训练。
它的输出可接入下方阈值估计和条件计算函数。

## 计算约定

- 分布价值：`values_from_logits` 计算 `Σ softmax(logits)[b] * bin_value[b]`。
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

奖励和价值必须使用相同尺度。`episode_rewards(normalization=...)` 支持每条轨迹对应任务的
归一化除数；应与价值模型训练时保持一致。函数不会裁剪回报，失败惩罚叠加步数后可能
超出 `[-1, 0]`；应按实际训练约定选择归一化参数和价值支撑点，不能仅裁剪某一侧。

## 使用示例

```python
import torch
from vla.model.modules.recap import (
    compute_advantage_condition,
    compute_advantages,
    episode_rewards,
    estimate_task_thresholds,
    values_from_logits,
)

lengths = torch.tensor([4, 3])
success = torch.tensor([True, False])
rewards = episode_rewards(
    lengths, success, max_steps=4, failure_penalty=5, normalization=10,
)
# 实际使用时替换为价值模型对各时刻观测及任务指令的预测。
value_logits = torch.zeros(2, 4, 201)
values = values_from_logits(value_logits)
advantages = compute_advantages(rewards, values, lengths=lengths)  # N=50
valid = torch.arange(4)[None, :] < lengths[:, None]
task_ids = torch.tensor([0, 1])[:, None].expand_as(advantages)

# 示例为简洁起见使用当前数据；实际应先在独立的代表性校准数据上估计并保存阈值。
# 预训练使用示范数据（论文抽样 10k 点），后训练使用当前迭代的评估 rollout。
thresholds = estimate_task_thresholds(
    advantages, task_ids, stage="posttrain", valid_mask=valid,
)
condition = compute_advantage_condition(
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

测试包含手算公式、不同长度轨迹的逐步参考实现、终止失败惩罚、padding、任务分位数、
严格阈值、人工纠正、条件 dropout 和混合精度累计；CUDA 可用时额外验证设备一致性。
价值模型测试使用随机初始化的小规模真实 Qwen3-VL，验证多模态前向、视觉/语言骨干
和价值头梯度、冻结模式、左右 padding、价值监督离散化与 checkpoint 恢复，无需下载 2B 权重。
