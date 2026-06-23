#!/bin/bash
# Double-click this file to open the Mask Review tool in your browser.
# No terminal commands needed — it starts the local app and opens the page,
# then cleans up when you close this window (or press Ctrl-C).
#
# It reviews the most recently produced output folder by default. To review a
# specific one, drag-drop is not supported here — instead run from a terminal:
#     (from the Segmentation/ folder) ../.venv-sam3/bin/python -m SAM3_new.review_app <output_dir>

cd "$(dirname "$0")" || exit 1   # = Segmentation/  (so SAM3_new is importable, venv is one level up)

PY="../.venv-sam3/bin/python"
if [ ! -x "$PY" ]; then
  echo "Could not find $PY — keep this launcher inside the Segmentation/ folder."
  echo "Press any key to close."; read -n 1; exit 1
fi

# Segmentation results (masks/) live in Segmentation/results/.
OUT="results"
if [ ! -d "$OUT/masks" ]; then
  echo "No results/masks found. Run the segmentation pipeline first."
  echo "Press any key to close."; read -n 1; exit 1
fi

# Smart-add (SAM click-to-add) loads the SAM 3 weights on first use; these let it
# read the already-cached weights. (No HF token is stored here — it is only needed
# if the weights aren't cached yet, in which case set HF_TOKEN in your shell first.)
export HF_HUB_DISABLE_XET=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "Opening Mask Review for: $OUT"
echo "Keep this window open while you review. Close it (or Ctrl-C) when done."
exec "$PY" -m SAM3_new.review_app "$OUT"
