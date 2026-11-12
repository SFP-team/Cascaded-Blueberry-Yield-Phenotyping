#!/usr/bin/env bash
# Predict the held-out test set with the six trained models (thresholds chosen
# on the validation split only), stage the SAM3 zero-shot masks, and compute
# per-image metrics and summaries.
#
# Usage: ./run_test_evaluation.sh
#
# Required environment:
#   TEST_MANIFEST    benchmark_manifest.csv from scripts/build_test_manifest.py
#   SAM3_OUTPUT      output root of segmentation.sam3_annotation.run_recursive
#                    run on <test root>/inputs/raw (uncorrected masks)
#   SAM3_CHECKPOINT  SAM3 checkpoint used for that run
# Optional environment:
#   CONFIG           training config        (default: configs/fixed_split.yaml)
#   EVAL_ROOT        output folder          (default: outputs/test_evaluation)
#   PYTHON           python interpreter     (default: python)
#   GPU              CUDA device id         (default: 0)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/fixed_split.yaml}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_DIR}/outputs/test_evaluation}"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
: "${TEST_MANIFEST:?Set TEST_MANIFEST to the test benchmark_manifest.csv}"
: "${SAM3_OUTPUT:?Set SAM3_OUTPUT to the SAM3 run_recursive output root}"
: "${SAM3_CHECKPOINT:?Set SAM3_CHECKPOINT to the SAM3 checkpoint}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

config_path() {
  "${PYTHON}" -c 'import sys; from plant_benchmark.config import load_config; print(load_config(sys.argv[1])["paths"][sys.argv[2]])' "${CONFIG}" "$1"
}
TRAIN_MANIFEST="$(config_path manifest)"
RUNS_ROOT="$(config_path runs)"
TEST_IMAGES="$("${PYTHON}" -c 'import sys, pandas; print(len(pandas.read_csv(sys.argv[1])))' "${TEST_MANIFEST}")"

"${PYTHON}" scripts/prepare_test_inference_config.py \
  --train-val-manifest "${TRAIN_MANIFEST}" \
  --test-manifest "${TEST_MANIFEST}" \
  --base-config "${CONFIG}" \
  --artifacts-root "${EVAL_ROOT}/inference_artifacts" \
  --output "${EVAL_ROOT}/inference"
INFERENCE_CONFIG="${EVAL_ROOT}/inference/test_inference_config.yaml"

# YOLO26 reads upright JPEGs from its own export of the derived manifest.
"${PYTHON}" -m plant_benchmark.cli prepare-yolo --config "${INFERENCE_CONFIG}"

for model in sam21 birefnet_hr segformer_b2 deeplabv3plus_r50 yolo26; do
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m plant_benchmark.cli predict \
    --config "${INFERENCE_CONFIG}" --model "${model}" --protocol finetuned --fold 0
done
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/train_center_birefnet.py \
  --config "${INFERENCE_CONFIG}" predict --fold 0

"${PYTHON}" scripts/stage_sam3_predictions.py \
  --manifest "${TEST_MANIFEST}" \
  --sam3-output "${SAM3_OUTPUT}" \
  --checkpoint "${SAM3_CHECKPOINT}" \
  --benchmark-root "${EVAL_ROOT}/sam3"

"${PYTHON}" scripts/stage_test_predictions.py \
  --manifest "${TEST_MANIFEST}" \
  --runs-root "${RUNS_ROOT}" \
  --sam3-root "${EVAL_ROOT}/sam3/predictions/sam3_original" \
  --output "${EVAL_ROOT}/staged"

"${PYTHON}" scripts/evaluate_predictions.py \
  --manifest "${TEST_MANIFEST}" \
  --predictions-root "${EVAL_ROOT}/staged" \
  --output "${EVAL_ROOT}/results/full_test" \
  --subset-name full_test \
  --expected-images "${TEST_IMAGES}"

echo "Results: ${EVAL_ROOT}/results/full_test"
