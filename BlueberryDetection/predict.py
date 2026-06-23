"""Run the trained blueberry detector on image(s).

Optionally attribute detections to a SAM3 center-plant mask — i.e. count the ripe / unripe
berries that belong to *this* plant (detections whose box center falls inside the mask). That
closes the loop with Phase 1 (segmentation) + Phase 2 (architecture): per-plant berry load.

Examples:
  # detect on a folder, save annotated images under runs/predict/
  .venv-yolo/bin/python predict.py --weights runs/blueberry_yolo26s/weights/best.pt \
      --source ../Testimages

  # detect on one image and attribute to its plant mask
  .venv-yolo/bin/python predict.py --weights runs/blueberry_yolo26s/weights/best.pt \
      --source ../Testimages/W24120501.jpg \
      --mask ../output_Testimages/masks/W24120501.mask.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def find_best() -> str | None:
    cands = sorted(HERE.glob("runs/**/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return str(cands[-1]) if cands else None


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect ripe/unripe blueberries with a trained YOLO model.")
    ap.add_argument("--weights", default=None, help="best.pt (default: newest under runs/)")
    ap.add_argument("--source", required=True, help="image, folder, or video")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--mask", default=None, help="optional SAM3 plant mask .npy for per-plant attribution")
    args = ap.parse_args()

    weights = args.weights or find_best()
    if not weights:
        raise SystemExit("No weights found — train first or pass --weights.")
    device = args.device or pick_device()

    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.predict(source=args.source, conf=args.conf, imgsz=args.imgsz, device=device,
                            save=True, project=str(HERE / "runs"), name="predict")
    names = model.names

    # overall class tally
    totals: dict[str, int] = {}
    for r in results:
        for b in r.boxes:
            n = names[int(b.cls)]
            totals[n] = totals.get(n, 0) + 1
    print("\nDetections by class (all images):", totals or "(none)")

    if args.mask:
        mask = np.load(args.mask).astype(bool)
        for r in results:
            inside, per_class = 0, {}
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx]:
                    inside += 1
                    n = names[int(b.cls)]
                    per_class[n] = per_class.get(n, 0) + 1
            print(f"{Path(r.path).name}: {inside} berries inside plant mask -> {per_class}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
