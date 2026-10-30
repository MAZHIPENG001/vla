#!/usr/bin/env bash
set -euo pipefail

VLA_DIR="${VLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VLA_PYTHON="${VLA_PYTHON:-python}"
CKPT="${CKPT:-${VLA_DIR}/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6694}"
USE_BF16="${USE_BF16:-1}"
INCLUDE_STATE="${INCLUDE_STATE:-auto}"

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

case "${INCLUDE_STATE}" in
  auto) ;;
  1|true|True) CMD+=(--include-state) ;;
  0|false|False) CMD+=(--no-include-state) ;;
  *) echo "INCLUDE_STATE must be auto, true, or false" >&2; exit 1 ;;
esac

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
