# Install
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
echo 'export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple' >> ~/.bashrc
source ~/.bashrc
uv sync
uv pip install ninja packaging
uv pip install flash-attn --no-build-isolation
uv pip install pandas
uv pip install numpydantic
uv pip install opencv-python
uv pip install opencv-python-headless
uv pip install albumentations
uv pip install pydantic
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install qwen-vl-utils[decord]==0.0.8
uv pip install ninja fvcore iopath
uv pip install huggingface-hub==0.35.3
uv pip install pyarrow
uv pip install deepspeed
uv pip install wandb

uv pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git"
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
