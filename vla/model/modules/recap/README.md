# π*0.6 / RECAP：价值模型与优势条件策略

本模块参考仓库中的 [π*0.6 论文](../../../../docs/pistar06.pdf)（IV-A、IV-B、V-B、V-C 节及附录 F），
实现 Qwen 价值模型、优势估计和优势条件文本，并接入 `QwenOFTRecap` 策略。
价值骨干采用 `Qwen/Qwen3-VL-2B-Instruct`，复用项目 Qwen 的加载、图像预处理、对话模板和特征接口。
策略沿用 OFT 的连续动作 MLP 与 L1 loss；当前实现不包含论文的 flow matching、子任务生成训练或 CFG 动作混合。

完整使用流程：

1. 用完整轨迹的归一化剩余回报独立训练 critic，得到 `critic.pt`。
2. 在代表性数据上校准每个任务的优势阈值，写入共享任务 YAML。
3. 加载并冻结 critic，在 VLA 训练的每个 batch 中即时计算优势条件，只优化动作损失。
   部署默认固定使用 `Advantage: positive`，不执行 critic。

这里的“在线”指训练时重算**已录制轨迹**的优势，不包含环境交互或 rollout 采集。
离线标签导出用于检查和校准，VLA 训练不读取这些标签。

本文包含：

- [配置与代码入口](#配置与代码入口)
- [价值模型与计算约定](#价值模型与计算约定)
- [数据集与奖励准备](#数据集与奖励准备)
- [独立价值训练与恢复](#独立价值训练与恢复)
- [优势检查与阈值校准](#优势检查与阈值校准)
- [VLA 在线训练与部署](#vla-在线训练与部署)
- [Python 接口](#python-接口)
- [验证入口](#验证入口)

## 配置与代码入口

策略、价值训练和阈值校准共用环境对应的任务 YAML。本文使用
[`examples/LIBERO/train_files/starvla_oft_recap_libero.yaml`](../../../../examples/LIBERO/train_files/starvla_oft_recap_libero.yaml)。
其他环境可在 `examples/<环境>/train_files/` 提供自己的配置，数据格式仍需适配已有 Dataset 工厂。

| 配置节点 | 用途 |
| --- | --- |
| `framework` | VLA 策略骨干、动作头和 `advantage_conditioning` |
| `recap.qwenvl` / `recap.value_model` | 独立 critic 骨干、价值头和权重 |
| `recap.data` / `recap.reward` | 轨迹结果、任务标识和奖励尺度 |
| `recap.advantage` | 优势公式、预测批大小和校准阈值 |
| `recap.training` | 独立价值训练参数 |
| `datasets.vla_data` | 共用数据路径、混合数据集与加载参数 |
| `trainer` | VLA 策略训练参数，与 `recap.training` 独立 |

`QwenRecap` 接受任务 YAML 路径、普通 dict 或 OmegaConf，合并 `QwenRecapDefaultConfig`，
保留完整配置副本。价值模型内部适配 Qwen 工厂需要的配置，不改写调用方的 `framework`，
也不会用策略的权重路径初始化 critic。价值头输入维度从实际骨干读取。
`QwenRecap` 直接实例化；`build_framework` 用于创建返回动作的 `QwenOFTRecap` 策略。

| 代码 | 作用 |
| --- | --- |
| [`recap.py`](recap.py) | `QwenRecap` 配置、价值训练／预测、在线优势与条件生成 |
| [`value_model.py`](value_model.py) | `QwenValueModel`、价值分类头与 MC return 损失 |
| [`advantage.py`](advantage.py) | `AdvantageModule`、`AdvantageCondition` 及张量计算 |
| [`QwenOFTRecap.py`](../../framework/VLM4A/QwenOFTRecap.py) | 优势条件策略，注册名 `QwenOFTRecap` |
| [`vla/dataloader/__init__.py`](../../../dataloader/__init__.py) | 策略与价值流程共用的 `build_dataloader` |
| [`recap_targets.py`](../../../dataloader/recap_targets.py) | 完整轨迹验证、标量 Reward/Return 和 bootstrap 索引 |
| [`train_recap_value.py`](../../../training/train_recap_value.py) | 独立价值训练、验证与恢复 |
| [`label_recap_advantages.py`](../../../training/label_recap_advantages.py) | 优势检查、标签导出与阈值校准 |
| [`recap_utils.py`](../../../training/recap_utils.py) | 两个 CLI 的配置、checkpoint 与来源校验 |

以下是 `QwenRecapDefaultConfig` 的主要默认值，实际以任务 YAML 的覆盖结果为准：

| 配置项（省略 `recap.`） | 默认值 | 说明 |
| --- | --- | --- |
| `qwenvl.base_vlm` | `Qwen/Qwen3-VL-2B-Instruct` | Hub ID 或完整本地权重目录 |
| `qwenvl.attn_implementation` | `sdpa` | 注意力实现 |
| `value_model.num_bins` | `201` | 价值分类档数 |
| `value_model.value_min/value_max` | `-1/0` | 价值支持范围 |
| `value_model.freeze_backbone` | `true` | 独立训练时仅更新价值头 |
| `value_model.checkpoint` | `null` | 完整 `QwenValueModel.state_dict()` 路径 |
| `advantage.mode` | `posttrain` | `posttrain` / `pretrain` / `return_to_go` |
| `advantage.n_steps` | `50` | posttrain 前瞻步数 |
| `advantage.positive_fraction` | `null` | 按模式选择目标正样本比例 |
| `advantage.dropout_probability` | `0.3` | 独立优势接口的条件丢弃概率 |
| `advantage.prediction_batch_size` | `8` | 价值预测每批观测数 |
| `advantage.thresholds` | `{}` | Python 张量接口使用的整数任务 ID → 阈值 |
| `advantage.task_thresholds` | `{}` | 在线 VLA 使用的稳定任务键 → 阈值 |

`task_thresholds` 的键为 `数据集目录名:原始 task_index`，避免混合数据集重排整数任务 ID 后错配。
`fit_thresholds` 保存的是 `thresholds`；校准 CLI 会同时生成 `task_thresholds`。
在线策略的 dropout 由 `framework.advantage_conditioning.dropout_probability` 控制。
`datasets.vla_data.obs_image_size` 可省略；设置后 `QwenRecap` 的训练和预测都会按 `[宽, 高]` 调整图像。

## 价值模型与计算约定

critic 编码当前时刻的多相机图像、任务指令和可选状态，取最后一个有效 token 的隐藏状态，
通过新建线性头输出价值 logits。特征接口跳过词表 logits，默认关闭 KV cache。
标量预测为分类分布的期望：

```text
V(o_t) = sum_b softmax(logits_t)[b] * bin_value[b]
R_t    = sum_{k=t}^{L-1} r_k
value_loss = CE(logits_t, nearest_bin(R_t))
```

默认使用 `[-1, 0]` 上的 201 个等距支撑点。**新建价值头随机初始化，Instruct 权重本身不是训练好的 critic。**
独立价值训练始终以剩余回报 `R_t` 监督，不随优势模式改变；超出支持范围或非有限标签会报错，不做裁剪。
默认冻结整个 Qwen 主干（视觉和语言部分），仅更新价值头；设置 `freeze_backbone: false` 可连同主干一起训练。
`recap.training.mixed_precision: 'no'` 使用 float32 骨干，默认 `bf16` 使用 bfloat16 骨干。

优势模式如下，`L` 为包含终止奖励的有效轨迹长度：

| 模式 | 公式 |
| --- | --- |
| `posttrain` | `A_t = sum(r[t:min(t+N,L)]) + V(o[t+N]) - V(o[t])`，默认 `N=50` |
| `pretrain` | `A_t = sum(r[0:L]) - V(o[t])` |
| `return_to_go` | `A_t = sum(r[t:L]) - V(o[t])` |

`posttrain` 在 `t+N >= L` 时令 bootstrap 为 0；终止样本自身的 `V(o_t)` 仍由 critic 预测。
`pretrain` 按附录 F 的印刷公式使用整条轨迹回报，其与文中 `N=T` 描述存在歧义；
需要从当前时刻累计的 MC 优势时，显式使用 `return_to_go`。

条件采用严格比较 `I_t = (A_t > ε_task)`，人工接管帧强制为 positive。
阈值按有效优势的 `1 - positive_fraction` 分位数估计：`posttrain` 默认目标 40% 正样本，
`pretrain` 和 `return_to_go` 默认 30%；可显式设为 10% 等比例。
相同优势值会使实际比例偏离目标；人工接管也可能提高正样本比例。
本实现采用附录 F 的优势正样本比例约定。

训练时以默认 30% 概率移除整个条件文本，保留原始正负标签；这也适用于人工接管帧。
`eval()` 关闭条件 dropout。优势与标签计算均停止梯度，半精度奖励至少以 float32 累计。

张量接口的 `rewards/values` 形状均为 `[B, T]`，每行是一条完整轨迹，
`lengths` 指定包含终止样本的有效长度，右侧 padding 被忽略。
不能将跨轨迹拼接数据或需要末端 bootstrap 的截断片段当作完整轨迹传入。
奖励、价值和 critic 训练回报必须使用相同归一化尺度。

## 数据集与奖励准备

### 共用 Dataset 与 Value profile

两类训练复用 `build_dataloader → get_vla_dataset → LeRobotMixtureDataset` 和单数据集的 `get_sample`，
图像解码、变换、state/action 打包沿用原有路径。`recap_targets.py` 只缓存标量轨迹目标，
不提供独立 Dataset 或观测解码器。

普通 VLA 数据配置可以按需开启：

```yaml
datasets:
  vla_data:
    return_frame_info: true      # 附加 dataset_name/episode_index/frame_index/episode_length
    return_value_targets: false  # 开启后附加 reward/return/task_key/is_intervention
```

两个开关默认均为 false。单独开启 `return_frame_info` 不需要结果清单，也不改变随机混合采样。
开启 `return_value_targets` 需要 `value_target_config`；也可在随机 VLA 样本中返回回报监督。

共享 YAML 的 `profiles.value` 为价值流程覆盖加载配置：

```yaml
datasets:
  vla_data:
    profiles:
      value:
        return_frame_info: true
        return_value_targets: true
        sampling_mode: sequential
        transform_mode: eval
        delete_pause_frame: false
        value_target_config:
          data: ${recap.data}
          reward: ${recap.reward}
          value_model:
            value_min: ${recap.value_model.value_min}
            value_max: ${recap.value_model.value_max}
        per_device_batch_size: ${recap.training.batch_size}
        num_workers: ${recap.training.num_workers}
        shuffle: true
        pin_memory: false
        save_statistics: false
```

Value 入口调用 `build_dataloader(cfg, dataset_py=..., profile="value")`，覆盖不修改原配置；
两个 CLI 也会补充缺省的 Value profile。普通 VLA 不指定 profile，继续使用原有随机混合采样。
`sequential` 使 Dataset 索引与原始帧一一对应，枚举所有帧，不使用策略删帧索引缓存；
训练仍可在 DataLoader 层 shuffle，读取失败直接报错，不用其他随机帧替代。
图像按需解码，变换使用 eval 模式。价值训练／验证按完整 Episode、按任务分组划分，
使用标准 `torch.utils.data.Subset`，避免相邻帧泄漏。各训练帧均匀采样，不按策略混合器的权重抽样。

### 完整轨迹要求

每条轨迹的 `frame_index` 必须从 0 连续递增，且只包含一个由 `recap.data.task_column`
标识的任务（默认 `task_index`）。最后一个存储帧必须对应终止样本；如果实际仍为终止前观测，
需先对齐数据的终止语义或补齐终止观测。不同长度 Episode 独立累计回报。

`recap.data.truncated_column` 可指定布尔截断列，出现 true 就拒绝计算终止 MC 回报。
输入的图像、状态和语言配置不能包含未来帧。原 Dataset 仍可返回动作字段，
但 critic 只编码图像、语言和可选状态，不输入动作、成功标志、未来回报或优势标签。
`recap.data.intervention_column` 可指定逐帧布尔接管列，未指定时全部视为无接管，指定后缺列会报错。

### 奖励来源与归一化

`recap.reward.source` 支持：

| 来源 | 奖励及所需配置 |
| --- | --- |
| `episode`（默认） | 非终止帧 -1；成功终止帧 0；失败终止帧 `-failure_penalty`。要求正的 `failure_penalty` 和 `normalization` |
| `column` | 读取 `recap.reward.column`，默认 `next.reward`。要求奖励与当前帧对齐及正的 `normalization`；不需要成功清单或失败惩罚 |

`episode` 模式通过 `recap.data.success_source` 选择结果来源：

- `manifest`：读取各数据集下的 `recap.data.outcome_path`，默认 `meta/recap_episodes.jsonl`。
- `column`：读取每条轨迹最后一行的 `recap.data.success_column`（默认 `success`），值必须为布尔类型。
- `all_success`：显式声明所有轨迹均为已确认完整且成功的示范，无需结果文件。

只有 `reward.source: episode` 且 `success_source: manifest` 才读取结果清单。
清单中每条轨迹恰好一行，`complete` 必须为 true；未知、缺失或重复轨迹会报错：

```jsonl
{"episode_index": 0, "success": true, "complete": true}
{"episode_index": 1, "success": false, "complete": true}
```

奖励统一除以固定的 `normalization`，然后从轨迹尾部累计 Return，不能按 Episode 长度临时缩放。
`recap.reward.task_overrides` 支持 `"数据集名:task_index"` 对应的任务专用 `failure_penalty/normalization`。
长度为 `L` 的失败轨迹初始回报为 `-(L - 1 + failure_penalty) / normalization`。
训练、校准和在线策略必须保持尺度一致；Return 超出 `value_min/value_max` 时应调整固定尺度或支持范围。

CLI 缺省的 `failure_penalty/normalization` 都是 null，需要自行填写。
本文的 LIBERO YAML 已设 `success_source: all_success`、`failure_penalty: 500` 和 `normalization: 999`；
这些是该配置的取值，不是论文统一参数，仍需确认适用于实际数据。

## 独立价值训练与恢复

### 配置与启动

从仓库根目录执行，先准备项目依赖并在运行环境中进行可编辑安装：

```bash
uv pip install --python .venv/bin/python -e .
```

安装后可按模块路径运行入口，无需修改 `sys.path`；修改源码无需重新安装。
先在共享 YAML 中设置模型路径、`datasets.vla_data.data_root_dir/data_mix`、奖励配置和训练参数。
独立价值训练使用自己的循环，分布式、梯度累积和状态保存交给 Accelerate。

| `recap.training` 参数 | CLI 缺省值 | 含义 |
| --- | --- | --- |
| `output_dir` | `playground/Checkpoints/recap_value` | 输出目录 |
| `batch_size` / `num_workers` | `2` / `0` | 每 rank micro batch 与加载进程数 |
| `gradient_accumulation_steps` | `4` | 梯度累积次数 |
| `max_steps` | `10000` | 优化器更新总次数 |
| `learning_rate` / `weight_decay` | `1e-5` / `0.01` | AdamW 参数，学习率保持常量 |
| `max_grad_norm` | `1.0` | 梯度裁剪 |
| `validation_fraction` | `0.1` | 按任务划分的验证 Episode 比例 |
| `eval_interval` / `save_interval` | `500` / `1000` | 验证／保存的更新步间隔 |
| `logging_steps` | `10` | 控制台打印间隔，指标文件每步写入 |
| `gradient_checkpointing` | `true` | 训练主干时启用梯度检查点 |
| `mixed_precision` | `bf16` | 支持 `bf16` 或 `'no'`（float32） |
| `resume_from_checkpoint` | `null` | 完整训练状态所在目录 |

任务 YAML 可覆盖这些默认值：当前 LIBERO 示例使用 `batch_size: 16`、`logging_steps: 1`，
输出目录为小写的 **`playground/checkpoints/recap_value`**，下文训练产物路径均与它一致。
全局名义 batch 为 `batch_size × gradient_accumulation_steps × world_size`。
梯度累积可跨 DataLoader epoch，不因 epoch 尾批提前更新。
每任务仅一条轨迹时保留在训练集；若所有任务均无法划出独立验证轨迹则报错，
可显式设置 `validation_fraction: 0` 关闭验证。

```bash
# 检查轨迹元数据、结果和 Return，不加载 Qwen、不逐帧解码视频
NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python -m vla.training.train_recap_value \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml --check_data

# 单卡训练
NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python -m vla.training.train_recap_value \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml --device cuda

# 短训练，使用独立输出目录
NO_ALBUMENTATIONS_UPDATE=1 uv run --no-sync python -m vla.training.train_recap_value \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml --device cuda \
  --set recap.training.max_steps=200 \
        recap.training.batch_size=2 \
        recap.training.logging_steps=1 \
        recap.training.save_interval=50 \
        recap.training.output_dir=playground/checkpoints/recap_value_smoke
```

CPU 训练使用 `--device cpu --set 'recap.training.mixed_precision="no"'`，
其中内层双引号用于让 OmegaConf 将 `no` 解析为字符串。
底层数据读取需要 Parquet 引擎（如 pyarrow）和配置的视频解码后端（LIBERO 示例使用 pyav）。
`--check_data` 不能替代真实图像解码检查。

支持 torchrun DDP 和已有 DeepSpeed 配置；以下命令在已激活项目运行环境时执行：

```bash
# 两卡 DDP，GPU 通过 CUDA_VISIBLE_DEVICES 选择
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m vla.training.train_recap_value \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml

# Accelerate + DeepSpeed ZeRO-2
accelerate launch --config_file vla/config/deepseeds/zero2.yaml --num_processes 2 \
  -m vla.training.train_recap_value \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml
```

也可使用 `vla/config/deepseeds/deepspeed_zero3.yaml`。入口统一从训练 YAML 设置 batch、累积和 clipping，
启用 ZeRO-3 完整权重聚合。所有 rank 必须参与 checkpoint 保存，包含冻结主干的完整 critic 才能导出。
多机运行按 Accelerate 的启动配置设置，各 rank 使用共享输出路径。
两个 CLI 均支持 `--model_id` 覆盖 critic 骨干、`--checkpoint` 指定权重和 `--set KEY=VALUE ...` 临时覆盖配置；
这些参数不会替换 `framework` 下的策略骨干。

### 产物与断点续训

```text
playground/checkpoints/recap_value/
  config.yaml
  split.json
  metrics.jsonl
  step_0001000/
    ...                       # 与 final 的文件结构一致
  final/
    critic.pt                 # 完整 Qwen 主干、Value Head、float32 bins
    training_state/           # Accelerate/DeepSpeed 模型、优化器、各 rank RNG、采样器状态
    progress.json             # step、epoch、已消费 batch 数、恢复约定
    metadata.json             # 完成步数、奖励／输入约定、Episode 划分
    config.yaml               # 完整共享配置，checkpoint 指向此 critic.pt
```

根目录 `split.json` 保存训练／验证划分。
`metrics.jsonl` 每步记录 `train_*` 指标，验证时增加 `validation_*`：
`value_loss`、`value_mae`、`value_mse`、`predicted_value`（预测均值）、`true_return`（目标均值）。
训练指标聚合全部 rank；验证用 `gather_for_metrics` 排除多卡 batch 补齐产生的重复样本。

checkpoint 先写隐藏的 `.incomplete` 临时目录，全部 rank 保存完成后原子发布。
已有 checkpoint 不覆盖；新训练遇到已有输出目录会报错，显式恢复可继续写入划分一致的原目录。

```bash
.venv/bin/python -m vla.training.train_recap_value \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml \
  --resume_from_checkpoint playground/checkpoints/recap_value/step_0001000 \
  --set recap.training.max_steps=20000
```

多卡恢复使用原启动命令并附加相同恢复参数。
恢复会重建确定性的 Episode 划分与采样顺序，跳过已消费 batch，再恢复模型、优化器和各 rank 随机状态。
要求原有数据、数据配置、随机种子、world size、batch、累积步数、冻结策略、精度及优化超参数一致；
可增加目标总步数 `max_steps`、修改日志／保存频率或使用新的输出目录。
从已有 `final` 继续时保留旧终态，新终态写到 `final_step_XXXXXXX/`。

`--checkpoint /path/to/critic.pt` 是热启动，只加载权重，重新创建优化器和步数。
训练 CLI 的热启动及校准 CLI 都要求同目录的 `metadata.json`，并校验奖励尺度、价值支持范围、
骨干与观测约定；单独的 Instruct 权重不能用于替代已训练 critic。

## 优势检查与阈值校准

训练策略必须提供明确的任务阈值。可自行在代表性数据上校准，也可使用
`label_recap_advantages` 完成检查和校准。该工具单进程按 Episode 分批解码／预测，
内存中只保留各轨迹的数值数组，不保留整条轨迹图像。
任务 YAML 和共享默认配置不包含 `labeling` 节点，导出选项按调用需求指定：

```bash
NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python -m vla.training.label_recap_advantages \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml \
  --checkpoint playground/checkpoints/recap_value/final/critic.pt --device cuda \
  --set recap.labeling.output_dir=/tmp/recap_inspection
```

权重选择顺序为：CLI `--checkpoint` 覆盖 YAML 的 `recap.value_model.checkpoint`；
两者都未指定时，选择 `recap.training.output_dir` 下 `final*` 或 `step_*` 中 metadata 记录步数最大的 checkpoint。
请保留权重旁的 `metadata.json` 供来源及配置校验。

工具支持三种优势模式，轨迹末端 bootstrap 为 0。
阈值在当前校准数据的全部有效帧上按任务统一拟合，不能用每个随机 batch 的分位数替代。
校准数据应代表相应阶段的数据分布；工具不会自动挑选代表性子集。
离线标签不做 dropout，人工接管帧强制 positive。

| 产物 | 内容 |
| --- | --- |
| 各数据集的 `meta/advantage_labels.jsonl` | 逐帧正负标签与接管标记，仅供检查；可用 `recap.labeling.label_path` 修改相对路径 |
| `labeling.output_dir/scores.jsonl` | dataset、episode、frame、task_key、reward、value、return、advantage、标签与接管标记 |
| `labeling.output_dir/metadata.json` | critic 来源、训练步数、任务映射、阈值、实际正样本比例与标签路径 |
| `labeling.output_dir/config.yaml` | 完整共享配置及 critic 路径、两类任务阈值 |

`recap.labeling.output_dir` 缺省为 `playground/Checkpoints/recap_labels`。
导出目录必须尚不存在；已有标签默认拒绝覆盖，显式传入 `--overwrite_labels` 才会原子替换逐个标签文件。
新一轮导出需选择新的 output_dir。

校准完成后，将导出配置中的 **`recap.value_model.checkpoint`** 和
**`recap.advantage.task_thresholds`** 写回用于策略训练的任务 YAML。
训练、校准及策略阶段的奖励和观测约定必须保持一致，建议将最终参数保存到共享 YAML。
VLA 不读取 `advantage_labels.jsonl` 或 Parquet 中的预存优势标签。

## VLA 在线训练与部署

### 策略配置与训练

```yaml
framework:
  name: QwenOFTRecap
  advantage_conditioning:
    enabled: true
    freeze_critic: true
    dropout_probability: 0.3
    inference_condition: positive
recap:
  value_model:
    checkpoint: /path/to/recap_value/final/critic.pt
    freeze_backbone: true
  advantage:
    mode: posttrain
    n_steps: 50
    task_thresholds:
      "your_dataset_name:0": 0.02  # 仅为格式示例，必须换成实际校准阈值
```

这是共享任务 YAML 的配置片段，还需保留策略／critic 模型路径、`datasets`、`recap.data/reward` 和训练配置。
`enabled: false` 关闭优势条件，不加载 critic，执行普通 OFT 训练；
`enabled: true` 时必须提供训练好的 critic、有限且覆盖所用任务的阈值，以及 `freeze_critic: true`。
`freeze_backbone` 只控制独立价值训练时是否更新 Qwen 主干，策略训练始终冻结整个 critic 并保持其 eval。

未指定 Value profile 的策略 DataLoader 会自动开启回报目标和在线优势字段，沿用原有随机混合采样。
训练样本保留 `image/lang/state/action`，并附加 `reward`、`return`、`task_key`、`is_intervention`、
`advantage_reward` 和 `advantage_bootstrap`。
worker 准备归一化奖励目标和必要的后续观测；critic 在模型设备计算。
bootstrap 观测只含图像、指令和可选状态，单独求值，不把未来动作或优势条件输入 critic。

策略内部创建 `QwenRecap(config, online=True)`，调用 `recap(examples, online=True)` 获取
`condition` 和 `advantages`，不产生 `value_loss`。提示输入顺序为：

```text
图像 + 完整任务提示／状态 → 可选 Subtask 文本 → Advantage 条件 → 动作查询
```

CoT_prompt 先展开，条件和动作查询后追加。条件只影响后面的动作查询；
`subtask` 可选，若提供则必须是字符串。
训练时正负标签经阈值比较后转为条件文本，按配置概率移除条件，学习无条件动作预测。
策略的优化目标为 **`loss = action_loss`**，仍使用 OFT 的 L1 动作回归。
优势不作为样本 loss 权重，也不直接加入损失；动作损失不沿此路径反传至 critic。
日志增加 `advantage_mean`，critic 随策略 state_dict 保存。

配置好路径、阈值和 WandB 等运行参数后，可使用现有策略训练器，例如：

```bash
accelerate launch --config_file vla/config/deepseeds/deepspeed_zero2.yaml --num_processes 2 \
  vla/training/train_starvla.py \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml
```

`examples/LIBERO/train_files/run_libero_train.sh` 也可复用，但其中会覆盖 framework 名称和配置路径，
需将 `Framework_name` 改为 `QwenOFTRecap`、`config_yaml` 指向 RECAP 任务配置，并调整机器相关参数。
收集下一轮数据后，先重新训练 critic、重新校准阈值，再启动下一轮策略训练。

### 部署与配置兼容

`predict_action` 默认使用 `Advantage: positive`，可传 `advantage="negative"` 或 `"unconditional"`。
`enabled: false` 则不附加条件。动作推理不执行 critic，但以在线配置创建策略实例时仍会构建和加载它；
加载完整策略 checkpoint 不会取消这一初始化要求。

旧配置需做以下迁移：

- 早期放在 `framework` 下的 RECAP `qwenvl/value_model/advantage` 移至 `recap`，保留策略自己的 `framework`。
- 删除 `framework.advantage_conditioning.source`、`sft_positive`、`value_loss_weight`，
  以及 `datasets.vla_data.advantage_labels` 和不再使用的 `trainer.learning_rate.online_advantage`。
- 开启条件时使用 `freeze_critic: true`，不再支持策略与 critic 的联合优化。

策略训练不接受调用方指定 `advantage_condition` 或固定 `advantage`；条件由内部在线计算。
策略 checkpoint 的 critic 权重路径为 `online_advantage.value_model.*`，加载时兼容旧的
`online_advantage.critic.value_model.*`。独立 `critic.pt` 格式仍为 `QwenValueModel.state_dict()`。

## Python 接口

### 价值监督与预测

`QwenRecap` 使用 `examples: list[dict]` 接口，每项包含多相机 `image`、`lang`，
可选已归一化的 `state`（如 `[1, state_dim]`）。训练要求标量 `return`，预测无需标签。
下面展示单次更新；真实使用需在轨迹数据上迭代训练和验证：

```python
import torch
from omegaconf import OmegaConf
from PIL import Image
from vla.model.modules.recap import QwenRecap

cfg = OmegaConf.load("examples/LIBERO/train_files/starvla_oft_recap_libero.yaml")
recap = QwenRecap(cfg).to("cuda")
batch = [{
    "image": [Image.new("RGB", (224, 224))],
    "lang": "Close the box.",
    "return": -0.4,
}]
optimizer = torch.optim.AdamW(
    [p for p in recap.parameters() if p.requires_grad], lr=1e-5,
)
recap.train()
optimizer.zero_grad(set_to_none=True)
output = recap(batch)
output["value_loss"].backward()
optimizer.step()

recap.eval()
values = recap.predict_value(batch)["values"]  # [观测数]
torch.save(recap.value_model.state_dict(), "critic.pt")
OmegaConf.save(recap.config, "recap_config.yaml")
```

普通 `forward` 返回 `value_loss`、逐样本 `value_losses`、`values` 和 `returns`。
`predict_value` 临时使用 eval 并关闭梯度，结束后恢复先前模式。
恢复时保持相同骨干、档数和支持范围，将 `recap.value_model.checkpoint` 设为权重路径。
此处手工保存只有权重，不包含训练 CLI 要求的 metadata 或完整续训状态；
需要后续 CLI 热启动、校准和断点续训时，应使用训练入口产出的 checkpoint。

底层 `QwenValueModel` 也可使用，但需提供含 `recap.qwenvl` 的共享 `config`，
并显式传入所需的 `num_bins/value_min/value_max/freeze_backbone`；不要依赖无参初始化。
它接收 `images: list[list[PIL.Image]]`、`instructions: list[str]` 和可选 `[B]` 的 `returns`，
返回 `ValueModelOutput(logits, values, loss, losses)`。状态转为文本和图像尺寸处理由 `QwenRecap` 包装层负责。

### 完整轨迹优势与独立张量接口

已有训练好的 `recap`、完整 `episodes: list[list[example]]` 和同尺度的 `[B, T]` 奖励时：

```python
# 每条 episode 仅包含有效观测，最后一项为终止样本；rewards 右侧允许 padding。
result = recap.predict_advantages(episodes, rewards)
# 返回 values/advantages/valid_mask: [B, T]，lengths: [B]。
# task_ids 为同形状、同设备的整数任务编号。
recap.fit_thresholds(result["advantages"], task_ids, valid_mask=result["valid_mask"])
condition = recap.make_condition(
    result["advantages"], task_ids, valid_mask=result["valid_mask"],
    intervention_mask=intervention_mask,
)
texts = condition.to_text()
OmegaConf.save(recap.config, "calibrated.yaml")
```

`fit_thresholds` 应在代表性校准数据上执行并保存，不在每个策略 minibatch 中重新拟合。
`recap.train()/eval()` 控制条件 dropout，阈值保存到 `recap.config.recap.advantage.thresholds`。
若用于在线 VLA，还需将整数任务 ID 映射为稳定任务键，写入 `task_thresholds`；校准 CLI 已完成此转换。

只需要张量计算时，`AdvantageModule` 不加载 Qwen、没有可训练参数，提供
`forward`、`fit_thresholds` 和 `make_condition`。以下示例可独立运行：

```python
import torch
from vla.model.modules.recap import AdvantageCondition, AdvantageModule

lengths = torch.tensor([4, 3])
rewards = AdvantageCondition.episode_rewards(
    lengths, torch.tensor([True, False]),
    max_steps=4, failure_penalty=5, normalization=10,
)
# 仅用于演示，实际应替换为训练好的 critic 预测。
values = AdvantageCondition.values_from_logits(torch.zeros(2, 4, 201))
valid = torch.arange(4)[None, :] < lengths[:, None]
task_ids = torch.tensor([0, 1])[:, None].expand_as(values)
advantage = AdvantageModule(mode="posttrain", n_steps=50, dropout_probability=0.3)
scores = advantage(rewards, values, lengths=lengths)
# 此处用当前数据演示；真实训练应独立选择代表性校准数据。
thresholds = advantage.fit_thresholds(scores, task_ids, valid_mask=valid)
advantage.eval()
condition = advantage.make_condition(scores, task_ids, valid_mask=valid)
print(condition.indicator)          # bool [B, T]，原始正负标签
print(condition.conditioning_mask)  # bool [B, T]，条件是否保留
print(condition.to_text())          # 按行展平，padding 或被丢弃条件为 ""
restored = AdvantageModule(thresholds=thresholds).eval()
```

相同数值与校验逻辑也可通过 `AdvantageCondition` 类方法调用：`values_from_logits`、
`episode_rewards`、`compute_advantages`、`estimate_task_thresholds`、`compute_advantage_condition`。
后者返回包含 `indicator`、`conditioning_mask` 和 `to_text()` 的条件对象。
`valid_mask` 仍需用于损失屏蔽；空条件文本本身不会屏蔽 padding 损失。

### 策略调用

以下假设任务 YAML 已配置训练好的 critic 和校准阈值，`observations` 为部署观测列表：

```python
from omegaconf import OmegaConf
from vla.dataloader import build_dataloader
from vla.model.framework.base_framework import build_framework

cfg = OmegaConf.load("examples/LIBERO/train_files/starvla_oft_recap_libero.yaml")
loader = build_dataloader(cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
policy = build_framework(cfg).to("cuda")
policy.train()
output = policy(next(iter(loader)))
output["loss"].backward()
# 实际训练还需 optimizer.step() 和梯度清零。

policy.eval()
actions = policy.predict_action(observations)["normalized_actions"]
negative = policy.predict_action(observations, advantage="negative")["normalized_actions"]
unconditional = policy.predict_action(observations, advantage="unconditional")["normalized_actions"]
```

## 验证入口

当前仓库可使用真实权重 smoke test 检查接口链路。它们使用合成输入，
不代表 critic 已学会任务价值或策略表现有所提升。
以下离线命令要求本地模型目录完整；远程加载时去掉 `HF_HUB_OFFLINE=1`。

```bash
# critic：加载、价值损失、价值头反向梯度、预测、轨迹优势与条件文本
HF_HUB_OFFLINE=1 uv run --no-sync python -m vla.model.modules.recap.recap \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml \
  --model_id playground/Pretrained_models/Qwen3-VL-2B-Instruct \
  --device cpu --freeze_backbone --backward

# 策略：需先在 YAML 配好 critic 骨干路径和实际 task_thresholds
HF_HUB_OFFLINE=1 uv run --no-sync python -m examples.LIBERO.test_oft_recap \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml \
  --model_id playground/Pretrained_models/Qwen3-VL-2B-Instruct \
  --critic_checkpoint playground/checkpoints/recap_value/final/critic.pt \
  --device cuda --freeze_backbone --backward
```

critic smoke test 可通过 `--checkpoint /path/to/critic.pt` 验证权重恢复。
上述示例冻结主干，只检查价值头／动作头梯度；critic 主干反向可在 YAML 设置
`recap.value_model.freeze_backbone: false`，去掉 `--freeze_backbone` 并使用 `--device cuda --backward`。
策略测试中的 `--model_id` 只覆盖策略骨干，critic 仍由 `recap.qwenvl` 配置。

上述 smoke test 不覆盖真实数据视频解码、多 GPU 或 DeepSpeed CUDA 执行；这些需用实际训练环境单独验证。
