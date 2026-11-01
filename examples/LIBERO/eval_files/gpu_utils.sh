#!/usr/bin/env bash

# Shared by the LIBERO inference and simulation launchers. CUDA renumbers the
# selected card to cuda:0, while robosuite's EGL renderer needs the original ID.
configure_libero_gpu() {
  local selected_gpu
  if [[ -v GPU_ID ]]; then
    selected_gpu="${GPU_ID}"
  else
    selected_gpu="${CUDA_VISIBLE_DEVICES-0}"
    selected_gpu="${selected_gpu%%,*}"
  fi

  while (( $# > 0 )); do
    case "$1" in
      --gpu-id)
        if (( $# < 2 )); then
          echo "--gpu-id requires a non-negative GPU index." >&2
          return 1
        fi
        selected_gpu="$2"
        shift 2
        ;;
      --gpu-id=*)
        selected_gpu="${1#*=}"
        shift
        ;;
      -h|--help)
        echo "Usage: bash $0 [--gpu-id ID]"
        echo "Select one GPU by index (for example: --gpu-id 2 or GPU_ID=2)."
        echo "Priority: --gpu-id > GPU_ID > first CUDA_VISIBLE_DEVICES index > 0."
        exit 0
        ;;
      *)
        echo "Unknown argument: $1. Use --help for usage." >&2
        return 1
        ;;
    esac
  done

  if [[ ! "${selected_gpu}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "GPU_ID / --gpu-id must be a single non-negative GPU index; got '${selected_gpu}'." >&2
    return 1
  fi

  export GPU_ID="${selected_gpu}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
}
