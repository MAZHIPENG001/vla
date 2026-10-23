# Install

克隆项目后，在仓库根目录安装完整依赖。当前依赖固定为 Python 3.11、Linux x86_64
和 PyTorch CUDA 12.8，需要兼容的 NVIDIA 驱动。PyTorch3D 从固定的 Git 提交构建，
请先安装 Git、C++ 编译器及 CUDA 12.8 开发工具链（包含 `nvcc`）。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
uv venv --python 3.11
uv pip install -r requirements.txt
source .venv/bin/activate
```

`requirements.txt` 中的 `-e .` 会从当前仓库安装 `vla`，不依赖用户名或克隆路径；
项目源码不依赖另行安装的本地 `lerobot` 仓库。PyTorch3D 的 PyTorch 构建依赖通过
`pyproject.toml` 的 [uv 配置](https://docs.astral.sh/uv/reference/settings/#extra-build-dependencies)
自动安装，因此安装命令需要在仓库根目录执行。

按需安装额外的训练加速组件：

```bash
uv pip install ninja packaging
uv pip install flash-attn deepspeed --no-build-isolation
```
# Wandb
```bash
https://wandb.ai/authorize?ref=models
```
# 激活环境
```bash
source ~/vla/.venv/bin/activate
export PYTHONPATH=~/vla:$PYTHONPATH
```
# 模型存储位置--软连接
```bash
mkdir -p /root/gpufree-data/playground

rm -rf /root/vla/playground
ln -s /root/gpufree-data/playground /root/vla/playground
ls -ld /root/vla/playground
readlink -f /root/vla/playground
```

# 模型验证
```bash
uv run vla/model/modules/vlm/Qwen.py

uv run vla/model/framework/VLM4A/QwenPI.py
uv run vla/model/framework/VLM4A/QwenGR00T.py
```

# 示例数据集下载 -- LIBERO
```bash
export DEST=/root/gpufree-data/playground/Datasets
bash examples/LIBERO/data_preparation.sh
# or 
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc 
source ~/.bashrc 

hf download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --repo-type dataset --local-dir /root/gpufree-data/playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot
hf download IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot  --repo-type dataset --local-dir /root/gpufree-data/playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot
hf download IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot    --repo-type dataset --local-dir /root/gpufree-data/playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot
hf download IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot      --repo-type dataset --local-dir /root/gpufree-data/playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot

# Copy modality.json to each subset
for d in /root/gpufree-data/playground/Datasets/LEROBOT_LIBERO_DATA/*/; do
  cp examples/LIBERO/train_files/modality.json "$d/meta/"
done
```

# 数据集验证
```bash
python vla/dataloader/lerobot_datasets.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml
```

# huggingface 模型下载
```bash
cd /root/gpufree-data
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct
```
# 开始训练
```bash
bash examples/LIBERO/train_files/run_libero_train.sh
```
