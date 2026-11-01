#!/usr/bin/env bash
set -euo pipefail

VLA="${VLA:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
CKPT="${CKPT:-${VLA}/playground/Pretrained_models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-2}"
# Keep simulation running while inference waits with zero arm deltas.
REALTIME="${REALTIME:-true}"
# Simulation control frequency and replay FPS.
CONTROL_FREQ="${CONTROL_FREQ:-20}"
# H comes from server metadata; stride and delay adapt to measured control cycles.
RTC_ENABLED="${RTC_ENABLED:-true}"
RTC_STRIDE="${RTC_STRIDE:-auto}"
RTC_DELAY="${RTC_DELAY:-auto}"
RTC_BETA="${RTC_BETA:-0.5}"
RTC_INPAINTING="${RTC_INPAINTING:-true}"
RTC_GUIDANCE_BETA="${RTC_GUIDANCE_BETA:-5.0}"
RTC_TIMEOUT="${RTC_TIMEOUT:-10.0}"

case "${REALTIME}" in
  true|1) REALTIME_ARGS=(--args.realtime) ;;
  false|0) REALTIME_ARGS=(--args.no-realtime) ;;
  *)
    echo "REALTIME must be true, false, 1, or 0." >&2
    exit 1
    ;;
esac
case "${RTC_ENABLED}" in
  true|1) RTC_ARGS=(--args.rtc-enabled) ;;
  false|0) RTC_ARGS=(--args.no-rtc-enabled) ;;
  *)
    echo "RTC_ENABLED must be true, false, 1, or 0." >&2
    exit 1
    ;;
esac
case "${RTC_INPAINTING}" in
  true|1) RTC_ARGS+=(--args.rtc-inpainting) ;;
  false|0) RTC_ARGS+=(--args.no-rtc-inpainting) ;;
  *)
    echo "RTC_INPAINTING must be true, false, 1, or 0." >&2
    exit 1
    ;;
esac
if [[ "${RTC_ENABLED}" =~ ^(true|1)$ && "${REALTIME}" =~ ^(false|0)$ ]]; then
  echo "RTC_ENABLED requires REALTIME=true so inference can run alongside control." >&2
  exit 1
fi
if [[ ! "${CONTROL_FREQ}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONTROL_FREQ must be a positive integer (Hz)." >&2
  exit 1
fi
if [[ "${RTC_STRIDE}" != auto && ! "${RTC_STRIDE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RTC_STRIDE must be auto or a positive integer (control cycles)." >&2
  exit 1
fi
if [[ "${RTC_DELAY}" != auto && ! "${RTC_DELAY}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "RTC_DELAY must be auto or a non-negative integer (control cycles)." >&2
  exit 1
fi

MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-egl}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-egl}"

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required."
  echo "Example: LIBERO_HOME=/path/to/LIBERO LIBERO_PYTHON=/path/to/python bash $0"
  exit 1
fi

cd "${VLA}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${VLA}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"

FOLDER_NAME="$(echo "${CKPT}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')"
MODEL_ROOT="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
VIDEO_OUT_PATH="${MODEL_ROOT}/results/${TASK_SUITE_NAME}/${FOLDER_NAME}"

echo "LIBERO evaluation: realtime=${REALTIME}, control_freq=${CONTROL_FREQ} Hz, rtc=${RTC_ENABLED}, stride=${RTC_STRIDE}, delay=${RTC_DELAY}, inpainting=${RTC_INPAINTING}"
"${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE_NAME}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.video-out-path "${VIDEO_OUT_PATH}" \
  --args.control-freq "${CONTROL_FREQ}" \
  --args.rtc-stride "${RTC_STRIDE}" \
  --args.rtc-delay "${RTC_DELAY}" \
  --args.rtc-beta "${RTC_BETA}" \
  --args.rtc-guidance-beta "${RTC_GUIDANCE_BETA}" \
  --args.rtc-timeout "${RTC_TIMEOUT}" \
  "${RTC_ARGS[@]}" \
  "${REALTIME_ARGS[@]}"
