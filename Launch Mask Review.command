#!/bin/bash
# Double-click this file to open the Mask Review tool in your browser.
# No terminal commands needed — it starts the local app and opens the page,
# then cleans up when you close this window (or press Ctrl-C).
#
# It reviews the most recently produced output folder by default. To review a
# specific one, drag-drop is not supported here — instead run from a terminal:
#     .venv-sam3/bin/python -m SAM3_new.review_app <output_dir>

cd "$(dirname "$0")" || exit 1

PY=".venv-sam3/bin/python"
if [ ! -x "$PY" ]; then
  echo "Could not find $PY — run this from the project folder."
  echo "Press any key to close."; read -n 1; exit 1
fi

# Pick the output dir: prefer output_Testimages, else the newest output*/ that
# actually contains a masks/ folder.
OUT="output_Testimages"
if [ ! -d "$OUT/masks" ]; then
  OUT=""
  for d in $(ls -dt output*/ 2>/dev/null); do
    if [ -d "${d}masks" ]; then OUT="${d%/}"; break; fi
  done
fi
if [ -z "$OUT" ]; then
  echo "No output folder with masks/ found. Run the segmentation first."
  echo "Press any key to close."; read -n 1; exit 1
fi

echo "Opening Mask Review for: $OUT"
echo "Keep this window open while you review. Close it (or Ctrl-C) when done."
exec "$PY" -m SAM3_new.review_app "$OUT"
