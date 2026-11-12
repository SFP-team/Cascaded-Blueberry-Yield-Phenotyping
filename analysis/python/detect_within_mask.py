#!/usr/bin/env python3
"""Run the berry/flower detector on raw and mask-restricted test images.

For every image and every condition, pixels outside the condition's plant mask are set to
black and the detector (YOLO26-L through SAHI sliced inference) runs on that masked RGB image.
The raw condition runs on the unmasked image. Each detection is also attributed to the Manual
Mask: it is on-target when its box center lies inside the manual mask.

Outputs (under --output):
  detection/predictions/<condition>/<stem>.json   all detections at confidence >= 0.25
  detection/overlays/<condition>/<stem>.jpg       audit overlays (unless --no-overlays)
  detection_protocol.json, detection_run_summary.json

Runs are resumable: completed JSON files are skipped unless --overwrite is given.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from common import (
    CLASS_NAMES,
    CONDITIONS,
    DEFAULT_WEIGHTS,
    REFERENCE_CONDITION,
    add_input_arguments,
    load_manifest,
    load_mask,
    load_rgb,
    mask_path,
    parse_mask_dirs,
    resolve_from_repo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_arguments(parser)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Detector weights.")
    # SAHI delegates device parsing to Ultralytics. Pin a physical GPU with CUDA_VISIBLE_DEVICES.
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help="Comma-separated condition keys (default: all).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-overlays", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def box_mask_fraction(mask: np.ndarray, xyxy: list[float]) -> float:
    h, w = mask.shape
    x1, y1, x2, y2 = xyxy
    xa = max(0, min(w, int(math.floor(x1))))
    xb = max(0, min(w, int(math.ceil(x2))))
    ya = max(0, min(h, int(math.floor(y1))))
    yb = max(0, min(h, int(math.ceil(y2))))
    if xb <= xa or yb <= ya:
        return 0.0
    return float(mask[ya:yb, xa:xb].mean())


def point_in_mask(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = mask.shape
    xi = max(0, min(w - 1, int(round(x))))
    yi = max(0, min(h - 1, int(round(y))))
    return bool(mask[yi, xi])


def object_to_record(obj, reference_mask: np.ndarray, input_mask: np.ndarray | None) -> dict:
    xyxy = [float(v) for v in obj.bbox.to_xyxy()]
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    category_id = int(obj.category.id)
    return {
        "category_id": category_id,
        "category_name": CLASS_NAMES.get(category_id, str(obj.category.name)),
        "confidence": float(obj.score.value),
        "xyxy": xyxy,
        "center_xy": [cx, cy],
        "box_area_px": max(0.0, x2 - x1) * max(0.0, y2 - y1),
        "center_in_manual_reference": point_in_mask(reference_mask, cx, cy),
        "manual_reference_box_coverage": box_mask_fraction(reference_mask, xyxy),
        "center_in_input_mask": (
            point_in_mask(input_mask, cx, cy) if input_mask is not None else None
        ),
        "input_mask_box_coverage": (
            box_mask_fraction(input_mask, xyxy) if input_mask is not None else None
        ),
    }


def make_overlay(
    rgb: np.ndarray,
    reference_mask: np.ndarray,
    input_mask: np.ndarray | None,
    detections: list[dict],
    condition: str,
    output_path: Path,
) -> None:
    """Draw the manual mask (green), the input mask (cyan) and the detection boxes."""
    h, w = rgb.shape[:2]
    scale = min(1.0, 1600.0 / max(h, w))
    sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    base = Image.fromarray(rgb).resize((sw, sh), Image.Resampling.LANCZOS)
    arr = np.asarray(base).copy()

    def contour(mask: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        small = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(arr, cnts, -1, color, thickness)

    contour(reference_mask, (0, 255, 0), 2)
    if input_mask is not None and condition != REFERENCE_CONDITION:
        contour(input_mask, (0, 220, 255), 2)

    canvas = Image.fromarray(arr)
    draw = ImageDraw.Draw(canvas)
    palette = {
        0: (255, 190, 0),
        1: (180, 40, 255),
        2: (255, 255, 255),
    }
    for det in detections:
        x1, y1, x2, y2 = [v * scale for v in det["xyxy"]]
        color = palette.get(det["category_id"], (255, 255, 0))
        if not det["center_in_manual_reference"]:
            draw.rectangle((x1 - 2, y1 - 2, x2 + 2, y2 + 2), outline=(255, 40, 40), width=4)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    title = (
        f"{CONDITIONS[condition]} | detections={len(detections)} | "
        f"on-target={sum(d['center_in_manual_reference'] for d in detections)}"
    )
    draw.rectangle((0, 0, min(sw, 820), 25), fill=(0, 0, 0))
    draw.text((6, 6), title, fill=(255, 255, 255), font=ImageFont.load_default())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90, optimize=True)


def valid_cached_json(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
        return payload.get("status") == "complete" and isinstance(payload.get("detections"), list)
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    output_root: Path = args.output
    output_root.mkdir(parents=True, exist_ok=True)
    mask_dirs = parse_mask_dirs(args.masks_root, args.mask_dir)
    weights = resolve_from_repo(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Detector weights not found: {weights}")
    requested = [x.strip() for x in args.conditions.split(",") if x.strip()]
    unknown = [x for x in requested if x not in CONDITIONS]
    if unknown:
        raise SystemExit(f"Unknown conditions: {unknown}")

    rows = load_manifest(args.manifest)[args.start_index :]
    if args.limit is not None:
        rows = rows[: args.limit]

    detector_device = args.device
    respect_visible_device = os.environ.get(
        "BLUEBERRY_RESPECT_CUDA_VISIBLE_DEVICES", ""
    ).strip().lower() in {"1", "true", "yes"}
    if respect_visible_device:
        # SAHI 0.12.x rewrites CUDA_VISIBLE_DEVICES inside select_device() and then
        # returns the pre-remapping ordinal. For a process already pinned by the
        # launcher, preserve that isolation and use visible ordinal zero.
        import torch
        import sahi.models.base as sahi_base

        detector_device = torch.device("cuda:0")
        sahi_base.select_device = lambda _device=None: detector_device

    protocol = {
        "created_utc": utc_now(),
        "weights": str(weights),
        "classes": CLASS_NAMES,
        "conditions": {k: CONDITIONS[k] for k in requested},
        "inference": {
            "framework": "Ultralytics YOLO through SAHI",
            "slice_height": 640,
            "slice_width": 640,
            "image_size": 640,
            "overlap_height_ratio": 0.2,
            "overlap_width_ratio": 0.2,
            "confidence_threshold": 0.25,
            "max_det": 3000,
            "perform_standard_pred": False,
            "postprocess_type": "GREEDYNMM",
            "postprocess_match_metric": "IOS",
            "postprocess_match_threshold": 0.5,
            "postprocess_class_agnostic": True,
            "batch_size": args.batch_size,
            "exif_transpose": True,
            "masked_background_rgb": [0, 0, 0],
            "respect_cuda_visible_devices": respect_visible_device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "reference_attribution": (
            "A detection is on-target when its bounding-box center lies inside the Manual Mask. "
            "This is an attribution proxy, not object-detection precision."
        ),
    }
    (output_root / "detection_protocol.json").write_text(json.dumps(protocol, indent=2))

    print(
        f"[{utc_now()}] Loading detector from {weights} on {detector_device} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})",
        flush=True,
    )
    detector = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights),
        confidence_threshold=0.25,
        image_size=640,
        device=detector_device,
    )
    try:
        detector.model.overrides["max_det"] = 3000
    except Exception:
        pass

    completed = skipped = failed = 0
    failures: list[dict] = []
    total = len(rows) * len(requested)
    for row in rows:
        stem = row["stem"]
        rgb = load_rgb(Path(row["image_path"]))
        reference_path = mask_path(mask_dirs, stem, REFERENCE_CONDITION)
        reference = load_mask(reference_path, rgb.shape[:2])
        for condition in requested:
            pred_path = output_root / "detection/predictions" / condition / f"{stem}.json"
            overlay_path = output_root / "detection/overlays" / condition / f"{stem}.jpg"
            cache_complete = valid_cached_json(pred_path) and (
                args.no_overlays or overlay_path.exists()
            )
            if not args.overwrite and cache_complete:
                skipped += 1
                continue
            started = time.monotonic()
            try:
                mpath = mask_path(mask_dirs, stem, condition)
                input_mask = load_mask(mpath, rgb.shape[:2]) if mpath is not None else None
                inference_rgb = rgb if input_mask is None else np.where(input_mask[..., None], rgb, 0).astype(np.uint8)
                result = get_sliced_prediction(
                    inference_rgb,
                    detector,
                    slice_height=640,
                    slice_width=640,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    perform_standard_pred=False,
                    postprocess_type="GREEDYNMM",
                    postprocess_match_metric="IOS",
                    postprocess_match_threshold=0.5,
                    postprocess_class_agnostic=True,
                    verbose=0,
                    progress_bar=False,
                    batch_size=args.batch_size,
                )
                detections = [
                    object_to_record(obj, reference, input_mask)
                    for obj in result.object_prediction_list
                ]
                payload = {
                    "status": "complete",
                    "created_utc": utc_now(),
                    "stem": stem,
                    "condition": condition,
                    "condition_display": CONDITIONS[condition],
                    "image_path": row["image_path"],
                    "input_mask_path": str(mpath) if mpath else None,
                    "manual_reference_path": str(reference_path),
                    "image_height": int(rgb.shape[0]),
                    "image_width": int(rgb.shape[1]),
                    "input_mask_fraction": float(input_mask.mean()) if input_mask is not None else None,
                    "manual_reference_fraction": float(reference.mean()),
                    "runtime_seconds": time.monotonic() - started,
                    "detections": detections,
                }
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                pred_path.write_text(json.dumps(payload, indent=2))
                if not args.no_overlays:
                    make_overlay(rgb, reference, input_mask, detections, condition, overlay_path)
                completed += 1
                done = completed + skipped + failed
                print(
                    f"[{utc_now()}] {done}/{total} {stem} {condition}: "
                    f"n={len(detections)} runtime={payload['runtime_seconds']:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                failed += 1
                failures.append(
                    {
                        "stem": stem,
                        "condition": condition,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"ERROR {stem} {condition}: {exc!r}", file=sys.stderr, flush=True)

    run_summary = {
        "finished_utc": utc_now(),
        "requested_outputs": total,
        "completed_this_run": completed,
        "skipped_cached": skipped,
        "failed": failed,
        "failures": failures,
    }
    (output_root / "detection_run_summary.json").write_text(json.dumps(run_summary, indent=2))
    print(json.dumps(run_summary, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
