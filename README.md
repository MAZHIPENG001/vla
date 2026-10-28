# VLA

VLA 模型训练与 LeRobot 数据处理工具。使用 **uv** 管理 Python、虚拟环境和依赖版本。

## 在其他电脑快速安装

### 1. 系统准备

完整模型环境面向 **Linux x86_64（glibc >= 2.28）+ Python 3.11.15 + PyTorch 2.11.0 / CUDA 12.8**。
Windows 请在满足 GPU/CUDA 条件的 WSL2 Linux 中安装；macOS、ARM 和其他 CUDA 版本
不在这份锁文件的支持范围内。仅使用数据浏览器时见下方的轻量安装。

安装前准备好：

- Git 和 C++ 编译工具（Ubuntu/Debian 可执行 `sudo apt-get install git build-essential`）。
- 兼容 CUDA 12.8 的 NVIDIA 驱动，用 `nvidia-smi` 检查 GPU 是否可见。
- CUDA 12.8 Toolkit（含 `nvcc`），用于构建 PyTorch3D 和可选的 FlashAttention。
  用 `nvcc --version` 检查；若安装在自定义位置，设置 `CUDA_HOME` 并将其 `bin` 加入 `PATH`。
  `nvidia-smi` 中显示的 CUDA 版本代表驱动支持能力，不代表已经安装 Toolkit。

安装 uv（本项目要求 uv >= 0.11.7）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

### 2. 克隆并安装

将下方 URL 换成自己的 GitHub 仓库地址。所有后续命令都在仓库根目录执行：

```bash
git clone <你的 GitHub 仓库 URL> vla
cd vla

# 自动安装指定 Python、创建 .venv、安装当前项目和锁定的依赖
uv sync --locked
```

不必手动创建环境或逐个安装 torch。PyTorch 的 CUDA 12.8 下载源已在 `pyproject.toml`
中指定；PyTorch3D 固定到当前环境使用的 Git 提交，由 uv 自动拉取并构建。
第一次下载 CUDA 包和编译扩展会比较久，后续安装会复用 uv 缓存。
构建时内存不足可改用 `MAX_JOBS=2 uv sync --locked`。

根据用途选择安装命令：

| 用途 | 命令 |
| --- | --- |
| 模型与数据加载基础环境 | `uv sync --locked` |
| 训练环境（额外安装 DeepSpeed、PEFT） | `uv sync --locked --extra train` |
| 训练并启用 FlashAttention 2 | `uv sync --locked --extra train --extra flash-attn` |
| 调试（额外安装 debugpy） | `uv sync --locked --extra train --group dev` |
| 仅数据浏览/质检，不安装模型及 CUDA 包 | `uv sync --locked --only-group data` |

`train` 是运行本仓库 Accelerate/DeepSpeed 训练入口的必要选项。
基础 Qwen 接口默认使用 SDPA；使用 `train_starvln.py` 或 `monkey_patch.py` 时还需要
`flash-attn`。这些选项也记录在锁文件中。之后同步环境时请继续带上所需的选项，
因为 `uv sync` 会移除当前选择之外的依赖；不要混用 `uv pip install` 来维护完整项目环境。

### 3. 验证和日常使用

完整环境安装后，先做不下载模型的导入检查：

```bash
uv run --locked python -c "import vla, torch, torchvision, pytorch3d.transforms; from vla.dataloader.gr00t_lerobot.schema import DatasetMetadata; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU:', torch.cuda.is_available())"
```

训练环境可另外检查 `uv run --locked python -c "import deepspeed, peft"`。
导入检查不代表已经验证模型训练；模型权重、数据集和 GPU 仍需单独准备。

运行程序时推荐 `uv run --locked`，无需激活虚拟环境，也无需设置写死路径的 `PYTHONPATH`：

```bash
uv run --locked python -m data_process view --path /path/to/dataset --no-browser
uv run --locked python -m vla.training.train_recap_value --help

# 也可以激活后直接使用 python / accelerate
source .venv/bin/activate
```

仅数据环境请使用 `uv run --no-sync python -m data_process view --path /path/to/dataset --no-browser`，
避免默认的 `uv run` 补装模型依赖。详见 [数据工具说明](data_process/README.md)。

### 4. 提交到 GitHub 和更新依赖

必须将以下文件与源码一起提交：

- `pyproject.toml`：直接依赖、可选依赖、构建方式和 PyTorch/PyTorch3D 来源。
- `uv.lock`：解析后的确切版本、下载来源与校验信息，让其他电脑安装相同的 Python 包版本。
- `.python-version`：固定 Python 3.11.15。
- `.gitignore`、`README.md`：忽略规则和安装说明。

`.venv/`、缓存、模型权重、数据集、训练输出和个人密钥不应提交；这些本地产物已加入忽略规则。
项目依赖以 `pyproject.toml` 和 `uv.lock` 为准，不再维护根目录 `requirements.txt`。
`data_process/requirements.txt` 与 Robotwin 的 requirements 是独立用途的安装清单，
不用于还原完整 VLA 环境；Robotwin 仿真器也需要按其说明单独安装。

新电脑首次安装执行 `uv sync --locked`（训练时添加上表中的选项）；以后更新代码：

```bash
git pull
uv sync --locked --extra train  # 按实际用途选择与首次安装一致的选项
```

维护者添加依赖时使用 `uv add 包名`，可选训练依赖使用 `uv add --optional train 包名`。
手动修改 `pyproject.toml` 后执行 `uv lock`，检查变更后一起提交配置和锁文件。
`--locked` 遇到配置与锁文件不一致会报错，应由维护者更新锁文件，避免每台电脑自行解析不同版本。
锁文件固定的是 Python 包；系统驱动、CUDA Toolkit、编译器和 GPU 架构仍由每台电脑提供。
uv 的索引与扩展构建配置参考 [官方 PyTorch 集成说明](https://docs.astral.sh/uv/guides/integration/pytorch/)
及 [构建依赖配置](https://docs.astral.sh/uv/concepts/projects/config/#augmenting-build-dependencies)。

## 模型和数据的存储位置

默认直接在项目内保存，`playground/` 已被 Git 忽略：

```bash
mkdir -p playground/Pretrained_models playground/Datasets
```

需要使用其他磁盘时，可自行将 `playground` 指向该磁盘目录；不要删除已有模型或数据。
换电脑后应检查示例 YAML/训练脚本中的模型、数据集、输出路径，以及 NCCL 网卡和 WandB 配置，
将 `/root/gpufree-data` 等旧机器路径改为新机器上的位置。

## WandB

```bash
uv run --locked wandb login
# 不上传训练日志时，在训练前设置：
export WANDB_MODE=disabled
```

## 模型验证
```bash
uv run --locked python vla/model/modules/vlm/Qwen.py

uv run --locked python vla/model/framework/VLM4A/QwenPI.py
uv run --locked python vla/model/framework/VLM4A/QwenGR00T.py
```

## 示例数据集下载 -- LIBERO
```bash
# 如网络需要，可在当前终端设置 Hugging Face 镜像：
# export HF_ENDPOINT=https://hf-mirror.com
export DATA_ROOT="$PWD/playground/Datasets/LEROBOT_LIBERO_DATA"
mkdir -p "$DATA_ROOT"

uv run --locked hf download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --repo-type dataset --local-dir "$DATA_ROOT"/libero_spatial_no_noops_1.0.0_lerobot
uv run --locked hf download IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot  --repo-type dataset --local-dir "$DATA_ROOT"/libero_object_no_noops_1.0.0_lerobot
uv run --locked hf download IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot    --repo-type dataset --local-dir "$DATA_ROOT"/libero_goal_no_noops_1.0.0_lerobot
uv run --locked hf download IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot      --repo-type dataset --local-dir "$DATA_ROOT"/libero_10_no_noops_1.0.0_lerobot

# Copy modality.json to each subset
for d in "$DATA_ROOT"/*/; do
  cp examples/LIBERO/train_files/modality.json "$d/meta/"
done
```

## 数据集验证
```bash
uv run --locked python vla/dataloader/lerobot_datasets.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml
```

## Hugging Face 模型下载
```bash
uv run --locked hf download Qwen/Qwen3-VL-4B-Instruct --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct
```
## 开始训练

先修改脚本中的模型、数据、输出目录、NCCL 网卡与 WandB 配置，再执行：

```bash
uv run --locked --extra train bash examples/LIBERO/train_files/run_libero_train.sh
```
