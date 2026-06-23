"""SAHI-tiled detection on full-frame rover images with the fb YOLO26 model.

Why: a berry in a 5184x3456 rover photo is only a few pixels at imgsz 1280, so whole-image
inference misses most. SAHI slices each image into overlapping tiles, runs the detector per tile
(berries keep their pixels), then merges — typically 2-4x more true detections on small objects.

Visualization: COLOR-CODED BOXES ONLY (no per-box text, which was unreadable when dense) plus a
compact legend with live counts in the top-left:
    blue   = berry-immature
    red    = berry-mature
    yellow = flower
Also writes a per-image count CSV.

Usage:
  .venv-yolo/bin/python predict_sahi.py --weights weights/fb_yolo26m_best.pt --source ../Testimages
  .venv-yolo/bin/python predict_sahi.py --source ../Testimages --slice 768 --overlap 0.25
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent

# BGR colors per class. Keyed by both naming schemes (trial1 used hyphen names
# 'berry-immature'; the tiled merged model uses underscore names 'immature_berry').
COLORS = {
    "immature_berry": (255, 0, 0), "berry-immature": (255, 0, 0),  # blue
    "mature_berry":   (0, 0, 255), "berry-mature":   (0, 0, 255),  # red
    "flower":         (0, 255, 255),                               # yellow
}
DEFAULT_COLOR = (0, 200, 0)
# canonical display order for the legend/counts (immature -> mature -> flower)
CLASS_ORDER = ["immature_berry", "mature_berry", "flower"]


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _draw_legend(img, counts):
    h, w = img.shape[:2]
    fs = max(0.8, min(h, w) / 1400.0)
    ft = max(2, round(fs * 2))
    labels = [f"{n}  ({counts.get(n, 0)})" for n in CLASS_ORDER]
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)[0] for t in labels]
    text_w = max(s[0] for s in sizes)
    text_h = max(s[1] for s in sizes)
    pad = int(text_h * 0.9)
    gap = int(text_h * 0.7)
    line_h = text_h + gap
    sq = text_h
    box_w = pad + sq + pad // 2 + text_w + pad
    box_h = pad + line_h * len(CLASS_ORDER)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    y = pad + text_h
    for name, label in zip(CLASS_ORDER, labels):
        cv2.rectangle(img, (pad, y - text_h), (pad + sq, y), COLORS[name], -1)
        cv2.putText(img, label, (pad + sq + pad // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), ft, cv2.LINE_AA)
        y += line_h


def _draw(img, preds):
    h, w = img.shape[:2]
    t = max(2, round(min(h, w) / 600.0))  # box thickness scales with image size
    counts: dict[str, int] = {}
    for p in preds:
        name = p.category.name
        counts[name] = counts.get(name, 0) + 1
        x1, y1, x2, y2 = (int(v) for v in p.bbox.to_xyxy())
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS.get(name, DEFAULT_COLOR), t)
    _draw_legend(img, counts)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="SAHI-tiled fb detection with color-coded boxes.")
    ap.add_argument("--weights", default=str(HERE / "weights/fb_yolo26m_best.pt"))
    ap.add_argument("--source", required=True, help="image or folder")
    ap.add_argument("--slice", type=int, default=1024, help="tile size (smaller = better tiny-object recall, slower)")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(HERE / "runs/predict_sahi"))
    # By default merge overlapping detections ACROSS classes so one object can't be
    # counted twice (e.g. the same berry boxed as both immature and flower). Pass
    # --keep-class-dupes to revert to SAHI's class-aware merge.
    ap.add_argument("--keep-class-dupes", action="store_true",
                    help="disable class-agnostic merge (allow one object to get one box per class)")
    ap.add_argument("--merge-iou", type=float, default=0.5,
                    help="IoS match threshold for the tile-merge dedup")
    args = ap.parse_args()

    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    device = args.device or pick_device()
    try:
        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=args.weights,
            confidence_threshold=args.conf, device=device, image_size=args.slice)
    except Exception:
        model = AutoDetectionModel.from_pretrained(
            model_type="yolov8", model_path=args.weights,
            confidence_threshold=args.conf, device=device, image_size=args.slice)

    src = Path(args.source)
    imgs = sorted(p for p in (src.iterdir() if src.is_dir() else [src])
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, ip in enumerate(imgs, 1):
        res = get_sliced_prediction(
            str(ip), model,
            slice_height=args.slice, slice_width=args.slice,
            overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
            postprocess_match_threshold=args.merge_iou,
            postprocess_class_agnostic=not args.keep_class_dupes,
            verbose=0)
        img = cv2.imread(str(ip))
        counts = _draw(img, res.object_prediction_list)
        cv2.imwrite(str(out / ip.name), img)
        row = {"image": ip.name, **{c: counts.get(c, 0) for c in CLASS_ORDER}}
        rows.append(row)
        print(f"[{i}/{len(imgs)}] {ip.name}: "
              + ", ".join(f"{counts.get(c, 0)} {c}" for c in CLASS_ORDER))

    with (out / "counts.csv").open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["image"] + CLASS_ORDER)
        wtr.writeheader()
        wtr.writerows(rows)
    totals = {c: sum(r[c] for r in rows) for c in CLASS_ORDER}
    print("\nTOTAL:", totals)
    print("Saved annotated images + counts.csv to:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
