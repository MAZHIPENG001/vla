# VLA

## 安装环境（uv）

安装好 [uv](https://docs.astral.sh/uv/getting-started/installation/)、Git、C++ 编译器和匹配的 CUDA Toolkit 后，克隆仓库并执行：

```bash
cd vla
uv venv --python 3.11
source .venv/bin/activate
# pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install setuptools==81.0.0 wheel ninja packaging
uv pip install -r requirements.txt
uv pip install flash-attn --no-build-isolation
uv pip install -e .
```

适用于 Linux x86_64；当前开发环境为 PyTorch 2.11.0 / CUDA 12.8。
先将 PyTorch 装进上面激活的 `.venv`，再安装其余依赖。`requirements.txt` 不指定 PyTorch 版本或 CUDA 源；
其他依赖仍可能要求兼容的 torch 版本。PyTorch3D、DeepSpeed 和 FlashAttention 使用已安装的 torch 编译。
仅使用默认 SDPA 时可跳过 FlashAttention 安装；`train_starvln.py` 仍需要它。

以后进入项目只需 `source .venv/bin/activate`，直接使用 `python`、`accelerate`、`hf` 等命令。
依赖在 `requirements.txt` 中维护，不需要 `uv lock` 或 `uv sync`。
提交源码、`requirements.txt`、`pyproject.toml` 和 `.python-version`；`.venv`、模型、数据集和训练输出不提交。
仅使用数据浏览/质检工具时，见 [数据工具说明](data_process/README.md)。

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
wandb login
# 不上传训练日志时，在训练前设置：
export WANDB_MODE=disabled
```

## 环境验证:模型
### 预训练大语言模型下载
```
echo 'export MODELSCOPE_TOKEN="你的Token"' >> ~/.bashrc
source ~/.bashrc
cd ./playground/Pretrained_models
modelscope download \
    --model MaZp001/vla \
    --local_dir . \
    --include \
        "Qwen3-VL-4B-Instruct/*" \
    --token "$MODELSCOPE_TOKEN"
```
```bash
# VLM
python vla/model/modules/vlm/Qwen.py
# VLA
python vla/model/framework/VLM4A/QwenPI.py
python vla/model/framework/VLM4A/QwenGR00T.py
```
## 数据集验证
### 示例数据集下载 -- LIBERO
```bash
# 如网络需要，可在当前终端设置 Hugging Face 镜像：
# export HF_ENDPOINT=https://hf-mirror.com
export DATA_ROOT="$PWD/playground/Datasets/LEROBOT_LIBERO_DATA"
mkdir -p "$DATA_ROOT"

hf download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --repo-type dataset --local-dir "$DATA_ROOT"/libero_spatial_no_noops_1.0.0_lerobot
hf download IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot  --repo-type dataset --local-dir "$DATA_ROOT"/libero_object_no_noops_1.0.0_lerobot
hf download IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot    --repo-type dataset --local-dir "$DATA_ROOT"/libero_goal_no_noops_1.0.0_lerobot
hf download IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot      --repo-type dataset --local-dir "$DATA_ROOT"/libero_10_no_noops_1.0.0_lerobot
# or
modelscope download --dataset MaZp001/vla_data --local_dir "$DATA_ROOT"

# Copy modality.json to each subset
for d in "$DATA_ROOT"/*/; do
  cp examples/LIBERO/train_files/modality.json "$d/meta/"
done
```
```bash
python vla/dataloader/lerobot_datasets.py --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml
```

## 模型训练

先修改脚本中的模型、数据、输出目录、NCCL 网卡与 WandB 配置，再执行：

```bash
bash examples/LIBERO/train_files/run_libero_train.sh
```

Value Model 独立训练、断点续训和优势阈值校准见 [RECAP 训练指南](vla/model/modules/recap/README.md)。
默认冻结 Qwen 主干，仅训练 Value Head；支持 bf16、单卡、DDP 和现有 DeepSpeed 配置。

## 模型测评
### Step 0: Set up the LIBERO environment
Create a separate conda environment for the LIBERO simulator:

```bash
conda create -n libero python=3.10 -y
conda activate libero
pip install mujoco==3.2.3
# Clone and install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..

# Install eval dependencies
pip install tyro matplotlib mediapy websockets msgpack numpy==1.24.4
```
### Step 1: Download a checkpoint
```bash
cd ./playground/Pretrained_models
modelscope download \
    --model MaZp001/vla \
    --local_dir . \
    --include \
        "StarVLA/Qwen3-VL-OFT-LIBERO-4in1/*" \
    --token "$MODELSCOPE_TOKEN"
```
### Step 2: Start the policy server (Terminal 1 — starVLA env)
Edit the checkpoint path in `examples/LIBERO/eval_files/run_policy_server.sh`:
```bash
CKPT=playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
TRANSFORMERS_VERBOSITY=error bash examples/LIBERO/eval_files/run_policy_server.sh
```
### Step 3: Run evaluation (Terminal 2 — LIBERO env)
Edit the paths in examples/LIBERO/eval_files/eval_libero.sh, then:
```bash
# export LIBERO_HOME=/home/ma/LIBERO
export LIBERO_HOME=/mnt/data/3dgs/mzp/LIBERO
export LIBERO_PYTHON="$PWD/.venv/bin/python"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
bash examples/LIBERO/eval_files/eval_libero.sh
```