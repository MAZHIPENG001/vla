# VLA

VLA 模型训练与 LeRobot 数据处理工具。使用 **uv** 管理 Python、虚拟环境和依赖版本。

## 在其他电脑快速安装

### 1. 系统准备

项目依赖面向 **Linux x86_64（glibc >= 2.28）+ Python 3.11.15**。
**torch、torchvision、torchaudio 由你单独安装**，不纳入项目锁文件，也不固定 CUDA 下载源。
当前开发环境使用 PyTorch 2.11.0 / CUDA 12.8；其他组合需自行确认与模型代码及扩展的兼容性。
Windows 请使用满足 GPU/CUDA 条件的 WSL2 Linux；macOS、ARM 不在这份锁文件的支持范围内。
仅使用数据浏览器时见下方的轻量安装。

安装前准备好：

- Git 和 C++ 编译工具（Ubuntu/Debian 可执行 `sudo apt-get install git build-essential`）。
- 与你选择的 PyTorch/CUDA 版本兼容的 NVIDIA 驱动，用 `nvidia-smi` 检查 GPU 是否可见。
- 与手动安装的 PyTorch 匹配的 CUDA Toolkit（含 `nvcc`），用于构建 PyTorch3D 和可选的 FlashAttention。
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

# 创建项目环境，然后将你选择的 PyTorch 安装到这个环境
uv venv --python 3.11.15

# 在此执行你自己的 torch / torchvision / torchaudio 安装命令
# 使用 uv pip install --python .venv/bin/python ...，确保目标是本项目的 .venv

# 确认 PyTorch 已安装，再安装项目其余依赖
.venv/bin/python -c "import torch, torchvision; print(torch.__version__, torch.version.cuda)"
uv sync --locked --inexact
```

PyTorch 必须装在项目的 `.venv` 中；装在系统 Python 或另一个 Conda 环境中不会生效。
`pyproject.toml` 的 `exclude-dependencies` 会阻止 Accelerate、PEFT 等间接依赖自动安装 torch。
PyTorch 自带的 NVIDIA/CUDA 包及 Triton 也不由本项目锁定。

**每次同步完整环境都使用 `--inexact`**，保留你手动安装的 PyTorch 及其依赖；
直接执行默认的 `uv sync` 可能清理这些不在锁文件中的包。
PyTorch3D 保留固定 Git 提交；它和可选的 DeepSpeed、FlashAttention 使用当前环境中的 torch
进行非隔离构建，不会另行下载构建用的 torch。构建工具由项目依赖提供。
第一次编译扩展会比较久；内存不足可改用 `MAX_JOBS=2 uv sync --locked --inexact`。
更换 torch 后，应重新构建已安装的扩展，例如为同步命令添加 `--reinstall-package pytorch3d`；
已安装 DeepSpeed 或 FlashAttention 时也为对应包添加该选项。

根据用途选择安装命令：

| 用途 | 命令 |
| --- | --- |
| 模型与数据加载基础环境 | `uv sync --locked --inexact` |
| 训练环境（额外安装 DeepSpeed、PEFT） | `uv sync --locked --inexact --extra train` |
| 训练并启用 FlashAttention 2 | `uv sync --locked --inexact --extra train --extra flash-attn` |
| 调试（额外安装 debugpy） | `uv sync --locked --inexact --extra train --group dev` |
| 仅数据浏览/质检，不安装模型及 CUDA 包 | `uv sync --locked --only-group data` |

`train` 是运行本仓库 Accelerate/DeepSpeed 训练入口的必要选项。
基础 Qwen 接口默认使用 SDPA；使用 `train_starvln.py` 或 `monkey_patch.py` 时还需要
`flash-attn`。这些选项也记录在锁文件中。之后同步环境时请继续带上所需的选项。
`--inexact` 会保留额外安装的包，不会自动卸载取消选择的可选组件；
项目其余依赖通过 uv 管理，PyTorch 系列由你单独维护。

### 3. 验证和日常使用

完整环境安装后，先做不下载模型的导入检查：

```bash
uv run --no-sync python -c "import vla, torch, torchvision, pytorch3d.transforms; from vla.dataloader.gr00t_lerobot.schema import DatasetMetadata; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU:', torch.cuda.is_available())"
```

训练环境可另外检查 `uv run --no-sync python -c "import deepspeed, peft"`。
导入检查不代表已经验证模型训练；模型权重、数据集和 GPU 仍需单独准备。

完成同步后，运行程序使用 `uv run --no-sync`，直接使用已准备好的环境；
更新代码或依赖后先重新执行带 `--inexact` 的同步命令。无需激活环境或设置写死路径的 `PYTHONPATH`：

```bash
uv run --no-sync python -m data_process view --path /path/to/dataset --no-browser
uv run --no-sync python -m vla.training.train_recap_value --help

# 也可以激活后直接使用 python / accelerate
source .venv/bin/activate
```

仅数据环境同样使用 `uv run --no-sync`，避免运行时补装模型依赖。
轻量同步命令会清理该环境中的模型依赖；若要保留已有完整环境，请添加 `--inexact`。
详见 [数据工具说明](data_process/README.md)。

### 4. 提交到 GitHub 和更新依赖

必须将以下文件与源码一起提交：

- `pyproject.toml`：直接依赖、可选依赖、构建方式和 PyTorch3D 来源；PyTorch 系列被明确排除。
- `uv.lock`：解析后的确切版本、下载来源与校验信息，让其他电脑安装相同的项目依赖版本（不含手动安装的 PyTorch 系列及其依赖）。
- `.python-version`：固定 Python 3.11.15。
- `.gitignore`、`README.md`：忽略规则和安装说明。

`.venv/`、缓存、模型权重、数据集、训练输出和个人密钥不应提交；这些本地产物已加入忽略规则。
项目依赖以 `pyproject.toml` 和 `uv.lock` 为准，不再维护根目录 `requirements.txt`。
`data_process/requirements.txt` 与 Robotwin 的 requirements 是独立用途的安装清单，
不用于还原完整 VLA 环境；Robotwin 仿真器也需要按其说明单独安装。

新电脑首次安装执行 `uv sync --locked --inexact`（训练时添加上表中的选项）；以后更新代码：

```bash
git pull
uv sync --locked --inexact --extra train  # 按实际用途选择与首次安装一致的选项
```

维护者添加依赖时使用 `uv add --no-sync 包名`，可选训练依赖使用 `uv add --no-sync --optional train 包名`，
然后执行 `uv sync --locked --inexact` 并带上所需选项，保留单独安装的 PyTorch。
手动修改 `pyproject.toml` 后执行 `uv lock`，检查变更后一起提交配置和锁文件。
`--locked` 遇到配置与锁文件不一致会报错，应由维护者更新锁文件，避免每台电脑自行解析不同版本。
锁文件固定的是项目其余 Python 包。PyTorch 及其 CUDA 依赖、系统驱动、CUDA Toolkit、编译器
和 GPU 架构需由每台电脑自行准备；换电脑时记录自己的 PyTorch 安装命令。
uv 配置参考 [依赖排除](https://docs.astral.sh/uv/reference/settings/#exclude-dependencies)
及 [非隔离构建](https://docs.astral.sh/uv/concepts/projects/config/#disabling-build-isolation)。

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
uv run --no-sync wandb login
# 不上传训练日志时，在训练前设置：
export WANDB_MODE=disabled
```

## 模型验证
```bash
uv run --no-sync python vla/model/modules/vlm/Qwen.py

uv run --no-sync python vla/model/framework/VLM4A/QwenPI.py
uv run --no-sync python vla/model/framework/VLM4A/QwenGR00T.py
```

## 示例数据集下载 -- LIBERO
```bash
# 如网络需要，可在当前终端设置 Hugging Face 镜像：
# export HF_ENDPOINT=https://hf-mirror.com
export DATA_ROOT="$PWD/playground/Datasets/LEROBOT_LIBERO_DATA"
mkdir -p "$DATA_ROOT"

uv run --no-sync hf download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --repo-type dataset --local-dir "$DATA_ROOT"/libero_spatial_no_noops_1.0.0_lerobot
uv run --no-sync hf download IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot  --repo-type dataset --local-dir "$DATA_ROOT"/libero_object_no_noops_1.0.0_lerobot
uv run --no-sync hf download IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot    --repo-type dataset --local-dir "$DATA_ROOT"/libero_goal_no_noops_1.0.0_lerobot
uv run --no-sync hf download IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot      --repo-type dataset --local-dir "$DATA_ROOT"/libero_10_no_noops_1.0.0_lerobot

# Copy modality.json to each subset
for d in "$DATA_ROOT"/*/; do
  cp examples/LIBERO/train_files/modality.json "$d/meta/"
done
```

## 数据集验证
```bash
uv run --no-sync python vla/dataloader/lerobot_datasets.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml
```

## Hugging Face 模型下载
```bash
uv run --no-sync hf download Qwen/Qwen3-VL-4B-Instruct --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct
```
## 开始训练

先修改脚本中的模型、数据、输出目录、NCCL 网卡与 WandB 配置，再执行：

```bash
uv run --no-sync bash examples/LIBERO/train_files/run_libero_train.sh
```
