#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 checkpoint task_suite gpu_id port" >&2
  exit 2
fi

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:?LIBERO_HOME is required}"
LIBERO_python="${LIBERO_python:-python}"
starVLA_python="${starVLA_python:-python}"
your_ckpt=$1
task_suite_name=$2
gpu_id=$3
base_port=$4
num_trials_per_task="${NUM_TRIALS_PER_TASK:-50}"
replan_interval="${REPLAN_INTERVAL:-8}"
gripper_encoding="${GRIPPER_ENCODING:-binary_open}"
include_state="${INCLUDE_STATE:-auto}"

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

server_cmd=("${starVLA_python}" deployment/model_server/server_policy.py
  --ckpt_path "$your_ckpt" --port "$base_port" --use_bf16)
case "${include_state}" in
  auto) ;;
  1|true|True) server_cmd+=(--include-state) ;;
  0|false|False) server_cmd+=(--no-include-state) ;;
  *) echo "INCLUDE_STATE must be auto, true, or false" >&2; exit 1 ;;
esac
CUDA_VISIBLE_DEVICES="$gpu_id" "${server_cmd[@]}" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

model_root="${your_ckpt%/checkpoints/*}"
model_name="${model_root##*/}"
folder_name="${model_name}_checkpoints_${your_ckpt##*/}"
run_name="${folder_name}_replan_${replan_interval}_${gripper_encoding}"
video_out_path="${model_root}/videos/${task_suite_name}/${folder_name}/replan_${replan_interval}_${gripper_encoding}"
log_path="${model_root}/logs/${task_suite_name}"
mkdir -p "$video_out_path" "$log_path"

"${LIBERO_python}" ./examples/LIBERO/eval_files/eval_libero.py \
  --args.host 127.0.0.1 \
  --args.port "$base_port" \
  --args.task-suite-name "$task_suite_name" \
  --args.num-trials-per-task "$num_trials_per_task" \
  --args.replan-interval "$replan_interval" \
  --args.gripper-encoding "$gripper_encoding" \
  --args.video-out-path "$video_out_path" \
  2>&1 | tee "${log_path}/${run_name}.log"

echo "Evaluation completed. Videos: ${video_out_path}; log: ${log_path}/${run_name}.log"
