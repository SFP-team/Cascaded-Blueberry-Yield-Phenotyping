#!/bin/bash
# macOS launcher: double-click to open the browser mask-review tool.
# It starts the local app on 127.0.0.1 and opens the page; close this window
# (or press Ctrl-C) to stop it.
#
# Settings (environment variables, optional):
#   SAM3_PYTHON      Python interpreter of the environment with sam3 installed
#                    (default: python3 on PATH)
#   SAM3_REVIEW_DIR  output folder of `segment` that contains masks/
#                    (default: sam3_results, relative to the repository root)
# A folder can also be given as the first argument when run from a terminal.

cd "$(dirname "$0")/../.." || exit 1   # repository root, so `segmentation` is importable

PY="${SAM3_PYTHON:-python3}"
OUT="${1:-${SAM3_REVIEW_DIR:-sam3_results}}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PY (set SAM3_PYTHON)."
  echo "Press any key to close."; read -n 1; exit 1
fi
if [ ! -d "$OUT/masks" ]; then
  echo "No $OUT/masks found. Run the segment step first or set SAM3_REVIEW_DIR."
  echo "Press any key to close."; read -n 1; exit 1
fi

# Smart-add loads the SAM 3 weights on first use (from the Hugging Face cache if
# no local checkpoint is configured). On Apple silicon, unsupported MPS
# operations fall back to the CPU.
export HF_HUB_DISABLE_XET=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "Opening Mask Review for: $OUT"
echo "Keep this window open while you review. Close it (or Ctrl-C) when done."
exec "$PY" -m segmentation.sam3_annotation.review_app "$OUT"
