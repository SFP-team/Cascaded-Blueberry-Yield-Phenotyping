#!/usr/bin/env bash
# Train the six supervised target-plant segmentation models on the fixed
# train/validation split.
#
# Usage: ./run_fixed_split.sh {smoke|train}
#   smoke  one forward+backward check per model (no training run is created)
#   train  train (or resume) all six models; reruns skip completed models
#
# Environment:
#   CONFIG  config file            (default: configs/fixed_split.yaml)
#   PYTHON  python interpreter     (default: python)
#   GPU_A   first CUDA device id   (default: 0)
#   GPU_B   second CUDA device id  (default: 1); if equal to GPU_A the models
#           run one after another instead of in pairs
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/fixed_split.yaml}"
PYTHON="${PYTHON:-python}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
MODE="${1:-train}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
# Reduce CUDA allocator fragmentation for full-resolution BiRefNet-HR.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

train_one() {
  local model="$1"
  local gpu="$2"
  echo "Starting ${model} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m plant_benchmark.cli train \
    --config "${CONFIG}" --model "${model}" --protocol finetuned --fold 0
}

train_center() {
  local gpu="$1"
  echo "Running Center BiRefNet accumulated-backward + validation smoke"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_center_birefnet.py \
    --config "${CONFIG}" smoke --fold 0
  echo "Starting Center BiRefNet on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_center_birefnet.py \
    --config "${CONFIG}" train --fold 0 \
    --epochs 15 \
    --patience 4 \
    --learning-rate 1e-5 \
    --gradient-accumulation 4 \
    --max-gradient-norm 1.0 \
    --side-negative-weight 2.0
}

# Run two jobs, in parallel on GPU_A/GPU_B, or sequentially on one GPU.
run_pair() {
  local first="$1"
  local second="$2"
  if [[ "${GPU_A}" == "${GPU_B}" ]]; then
    ${first} "${GPU_A}"
    ${second} "${GPU_A}"
    return
  fi
  ${first} "${GPU_A}" &
  local pid_a=$!
  ${second} "${GPU_B}" &
  local pid_b=$!
  local status=0
  wait "${pid_a}" || status=$?
  wait "${pid_b}" || status=$?
  return "${status}"
}

train_sam21() { train_one sam21 "$1"; }
train_birefnet() { train_one birefnet_hr "$1"; }
train_segformer() { train_one segformer_b2 "$1"; }
train_deeplab() { train_one deeplabv3plus_r50 "$1"; }
train_yolo() { train_one yolo26 "$1"; }

"${PYTHON}" -m plant_benchmark.cli doctor --config "${CONFIG}"

case "${MODE}" in
  smoke)
    for model in sam21 birefnet_hr segformer_b2 deeplabv3plus_r50 yolo26; do
      CUDA_VISIBLE_DEVICES="${GPU_A}" "${PYTHON}" -m plant_benchmark.cli smoke \
        --config "${CONFIG}" --model "${model}" --backward
    done
    ;;
  train)
    run_pair train_sam21 train_birefnet
    run_pair train_segformer train_deeplab
    # Center BiRefNet starts from the trained BiRefNet-HR best.pt.
    run_pair train_yolo train_center
    ;;
  *)
    echo "Usage: $0 {smoke|train}" >&2
    exit 2
    ;;
esac

echo "Completed fixed-split mode=${MODE}"
