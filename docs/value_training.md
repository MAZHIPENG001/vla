# Value Model 独立训练与 Advantage 标签生成

两个入口共用任务 YAML 的 `datasets.vla_data` 和 `recap`，复用项目 LeRobot 读取器、
统一的 `build_dataloader → get_vla_dataset → LeRobotMixtureDataset`、Qwen 特征接口及 Accelerate。策略训练器与 action loss 耦合，
因此独立价值训练使用自己的循环，分布式和状态保存继续交给 Accelerate。
示例配置为 `examples/LIBERO/train_files/starvla_oft_recap_libero.yaml`。

## 与 VLA 共用 Dataset

训练与标签入口都调用现有 `vla.dataloader.build_dataloader`，不再使用独立的
`ReturnDataset` 或 `build_episode_store`。单数据集和混合数据集共用 `get_sample`，
图像读取、变换、state/action 打包都使用原有路径。

在普通 VLA 数据配置中可按需开启：

```yaml
datasets:
  vla_data:
    return_frame_info: true     # 默认 false，附加 dataset_name/episode_index/frame_index/episode_length
    return_value_targets: false # 开启后附加 reward/return/task_key/is_intervention
```

只开启 `return_frame_info` 不需要奖励或结果清单，也不改变原有随机混合采样。
`return_value_targets: true` 需要 `value_target_config`，其中 `data`、`reward`、`value_model`
分别引用 `recap.data`、`recap.reward`、价值范围配置；也可在随机 VLA 样本中返回回报监督。
标量回报计算保留在 `recap_targets.py`，它只处理轨迹表，不提供观察解码或 Dataset 实现。

共享 YAML 的 `datasets.vla_data.profiles.value` 为 Value 流程配置这些开关和加载参数：

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
        advantage_labels: {source: none}
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

原 VLA 调用 `build_dataloader(cfg, dataset_py=...)` 保持原有配置；Value 调用同一入口，
增加 `profile="value"` 选择配置覆盖。覆盖不修改原配置，也不会影响策略后续读取 sidecar 标签。
`sequential` 让 Dataset 索引与原始帧一一对应，训练可再通过 DataLoader shuffle；
它不会使用策略删帧索引缓存，读取失败会直接报错，以保留轨迹/帧的对应关系。
训练/验证使用标准 `torch.utils.data.Subset` 按完整 Episode 划分，再交给同一个加载入口。

## 数据与回报

原始 Episode 按顺序完整读取，保留每帧，不做策略混合采样和暂停帧过滤。
图像按需解码，复用原有变换的 eval 模式。每条轨迹独立累计
`Return[t] = sum(reward[t:episode_length])`，不同长度的 Episode 不会互相串接。
训练/验证按完整 Episode、按任务分组划分，防止相邻帧泄漏。
截断轨迹需通过 `recap.data.truncated_column` 标识，当前流程拒绝对它计算终止 MC 回报。
最后一个存储帧必须对应终止样本，输入只包含当前/过去图像、指令和可选状态。

奖励支持两种来源：

- `recap.reward.source: episode`（默认）：非终止帧 -1，成功终止帧 0，失败终止帧
  `-failure_penalty`。必须设置正的 `failure_penalty` 和 `normalization`。
  成功信息来自 `recap.data.success_source`：`manifest`、`column` 或显式 `all_success`。
- `recap.reward.source: column`：读取 `recap.reward.column`（默认 `next.reward`）。
  奖励必须与当前帧对齐，此模式不需要成功清单或失败惩罚，仍要求正的 `normalization`。

奖励统一除以固定 normalization，不能按 Episode 长度临时缩放。
`recap.reward.task_overrides` 支持 `数据集名:task_index` 的任务专用尺度。
Return 超出 `recap.value_model.value_min/value_max` 会报错，不会裁剪。
默认模型是 201 档分类价值头，使用量化 Return 的 CE 监督，预测值为分类分布的期望。

默认成功清单位于每个数据集的 `meta/recap_episodes.jsonl`：

```jsonl
{"episode_index": 0, "success": true, "complete": true}
{"episode_index": 1, "success": false, "complete": true}
```

## 配置与启动

先设置模型、数据路径及奖励尺度。默认冻结整个 Qwen backbone（包括视觉和语言部分），
仅 Value Head 参与优化；`freeze_backbone: false` 可开启联合训练。
训练参数独立于策略的 `trainer`：

```yaml
recap:
  value_model:
    freeze_backbone: true
  training:
    batch_size: 2                    # 每 rank 的 micro batch
    gradient_accumulation_steps: 4
    learning_rate: 1.0e-5
    max_steps: 10000                 # 优化器更新总次数
    mixed_precision: bf16           # bf16 或 'no'（float32）
    resume_from_checkpoint: null
```

全局名义 batch = batch_size × gradient_accumulation_steps × world_size。
梯度累积允许跨 DataLoader epoch，避免尾批触发额外更新。学习率保持常量。

```bash
# 仅检查轨迹、结果和 Return，无需加载 Qwen
python -m vla.training.train_recap_value \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml --check_data

# 单卡
python -m vla.training.train_recap_value \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml --device cuda

TRANSFORMERS_VERBOSITY=error \
CUDA_VISIBLE_DEVICES=5 uv run --no-sync python \
  -m vla.training.train_recap_value \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml \
  --device cuda \
  --set recap.training.max_steps=200 \
        recap.training.batch_size=32 \
        recap.training.logging_steps=1 \
        recap.training.output_dir=playground/Checkpoints/recap_value_smoke \
        recap.training.save_interval=50

# 两卡 DDP；GPU 选择通过 CUDA_VISIBLE_DEVICES 控制
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m vla.training.train_recap_value \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml

# 复用项目 DeepSpeed ZeRO-2 配置，按机器修改进程数
accelerate launch --config_file vla/config/deepseeds/zero2.yaml --num_processes 2 \
  -m vla.training.train_recap_value \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml
```

也可以使用已有的 `vla/config/deepseeds/deepspeed_zero3.yaml`。
入口统一使用训练 YAML 的 batch、accumulation 和 clipping，启用 ZeRO-3 完整权重聚合，
保证导出的 critic 包含冻结主干。所有 rank 必须参与 checkpoint 保存。
单机与多机按 Accelerate 的启动配置运行，所有 rank 使用共享输出路径。

## Checkpoint 与恢复

输出根目录保存 `config.yaml`、`split.json`、`metrics.jsonl`。
周期目录 `step_0001000/` 和终态目录 `final/` 包含：

| 文件/目录 | 内容 |
| --- | --- |
| `critic.pt` | 完整 Qwen backbone、Value Head、float32 bins，可直接严格加载 |
| `training_state/` | Accelerate/DeepSpeed 模型、优化器、各 rank RNG、采样器状态 |
| `progress.json` | optimizer step、epoch、epoch 内已消费 batch、数据/配置恢复约定 |
| `metadata.json` | 步数、Reward/Return 和输入约定、Episode 划分 |
| `config.yaml` | 完整配置及导出权重路径 |

每次先写隐藏临时目录，成功后原子发布。中断的 `.incomplete` 目录不会被当作有效 checkpoint。
已有训练目录默认不覆盖；显式恢复时可继续写入原目录。已有 checkpoint 也不覆盖。

```bash
python -m vla.training.train_recap_value \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml \
  --resume_from_checkpoint playground/Checkpoints/recap_value/step_0001000 \
  --set recap.training.max_steps=20000
```

多卡恢复时使用原来的启动命令并附加相同恢复参数。
恢复重建确定性的 Episode 划分与采样顺序，跳过已消费的 batch，再加载优化器和各 rank 随机状态。
要求原有数据、随机种子、world size、batch、累积步数、冻结策略、精度与优化超参数不变；
可增加 max_steps、调整日志/保存频率或使用新的 output_dir。
若希望更换这些条件，用 `--checkpoint /path/to/critic.pt` 热启动，它重新创建优化器和步数。
从已经完成的 final 继续时，保留旧 final，新终态保存为 `final_step_XXXXXXX/`。

`metrics.jsonl` 和控制台输出训练、验证的 `value_loss`、`value_mae`、`value_mse`、
`predicted_value`（预测均值）、`true_return`（目标均值）。验证仅统计有效样本，
排除多卡 batch 补齐产生的重复帧；训练指标聚合全部 rank。
`validation_fraction: 0` 显式关闭验证；默认要求至少一个任务能划出独立验证 Episode。

## Advantage 导出

```bash
python -m vla.training.label_recap_advantages \
  --config_yaml examples/LIBERO/train_files/starvla_oft_recap_libero.yaml \
  --checkpoint playground/Checkpoints/recap_value/final/critic.pt --device cuda
```

标签导出用单进程逐 Episode 分批预测，不保留整条轨迹的图像。
未指定权重时选择训练目录中步数最大的完整 checkpoint。
支持已有 posttrain n-step、pretrain episode-total、return_to_go 三种公式，
跨 Episode 末端使用吸收态 value=0。每个任务在全部有效帧上统一校准阈值。
离线标签不做 dropout，人工接管帧强制 positive。

每个数据集保存 `meta/advantage_labels.jsonl`，由现有策略 DataLoader 直接读取。
`recap.labeling.output_dir` 保存逐帧 reward/value/return/advantage 的 `scores.jsonl`、
阈值和 checkpoint 来源的 `metadata.json`、完整配置。已有标签需要显式 `--overwrite_labels`。
标签目录已存在时请使用新的 labeling.output_dir。训练与标签导出的奖励及观测约定必须相同。

## 验证

```bash
NO_ALBUMENTATIONS_UPDATE=1 python -m unittest discover -s vla/model/modules/recap/tests -v

NO_ALBUMENTATIONS_UPDATE=1 torchrun --standalone --nproc_per_node=2 \
  -m vla.model.modules.recap.tests.distributed_value_smoke \
  --output_dir /tmp/value-ddp-check
```

单元测试使用真实 Parquet 表和小型编码器夹具，通过原有 Dataset 工厂和混合 Dataset，
运行实际的样本打包入口、Value Head、训练、checkpoint 和标签读取代码；覆盖默认 VLA 字段、
可选帧信息、随机采样返回目标、多数据集逐帧覆盖、worker 加载及读取失败不重采样。
分布式测试使用两进程 CPU/Gloo，检查完整保存、精确恢复与验证去重。
这些测试不覆盖真实 Qwen 的多 GPU 或 DeepSpeed CUDA 执行。
