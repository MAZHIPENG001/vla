# Install
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
uv sync
uv pip install ninja packaging
uv pip install flash-attn --no-build-isolation
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install qwen-vl-utils[decord]==0.0.8
```
# 环境
```bash
cd vla
source .venv/bin/activate
```

# 模型存储位置--软连接
```bash
mkdir -p /root/gpufree-data/playground/Pretrained_models

ln -s /root/gpufree-data/playground/Pretrained_models \
      /root/vla/playground/Pretrained_models

ls -lh /root/vla/playground
readlink -f /root/vla/playground/Pretrained_models
```

# 验证
```bash
uv run vla/model/modules/vlm/Qwen.py

uv run vla/model/framework/VLM4A/QwenPI.py
uv run vla/model/framework/VLM4A/QwenGR00T.py
```