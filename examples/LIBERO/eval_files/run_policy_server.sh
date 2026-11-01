#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/gpu_utils.sh"
configure_libero_gpu "$@"

VLA_DIR="${VLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VLA_PYTHON="${VLA_PYTHON:-python}"
CKPT="${CKPT:-${VLA_DIR}/playground/Pretrained_models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt}"
PORT="${PORT:-6694}"
USE_BF16="${USE_BF16:-1}"

cd "${VLA_DIR}"
export PYTHONPATH="${VLA_DIR}:${PYTHONPATH:-}"

CMD=(
  "${VLA_PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT}"
  --port "${PORT}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

echo "Policy server: GPU_ID=${GPU_ID}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, port=${PORT}"
"${CMD[@]}"
