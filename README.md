# Install
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
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

uv pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git"
```
# 激活环境
```bash
cd vla
source .venv/bin/activate
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

# 数据集验证
```bash
python vla/dataloader/lerobot_datasets.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml
```

# huggingface 模型下载
```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct
```