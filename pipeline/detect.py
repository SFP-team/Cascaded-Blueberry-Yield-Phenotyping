"""YOLO26-L + SAHI. Count a box when conf >= 0.35 and the center is in the mask."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = {
    0: "immature_berry",
    1: "mature_berry",
    2: "flowers",
}
HEADLINE_THRESHOLD = 0.35


def point_in_mask(mask: np.ndarray, x: float, y: float) -> bool:
    height, width = mask.shape
    xi = max(0, min(width - 1, int(round(x))))
    yi = max(0, min(height - 1, int(round(y))))
    return bool(mask[yi, xi])


def object_to_record(obj, mask: np.ndarray) -> dict[str, Any]:
    xyxy = [float(value) for value in obj.bbox.to_xyxy()]
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    category_id = int(obj.category.id)
    return {
        "category_id": category_id,
        "category_name": CLASS_NAMES.get(category_id, str(obj.category.name)),
        "confidence": float(obj.score.value),
        "xyxy": xyxy,
        "center_xy": [cx, cy],
        "center_in_mask": point_in_mask(mask, cx, cy),
    }


def counts_from_detections(detections: list[dict], threshold: float) -> dict[str, int]:
    kept = [
        item
        for item in detections
        if float(item["confidence"]) >= threshold and bool(item.get("center_in_mask"))
    ]
    unripe = sum(int(item["category_id"]) == 0 for item in kept)
    ripe = sum(int(item["category_id"]) == 1 for item in kept)
    flowers = sum(int(item["category_id"]) == 2 for item in kept)
    return {
        "unripe": unripe,
        "ripe": ripe,
        "flowers": flowers,
        "total_berries": unripe + ripe,
        "total_detections": unripe + ripe + flowers,
    }


def write_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    detections: list[dict],
    destination: Path,
    threshold: float = HEADLINE_THRESHOLD,
) -> None:
    height, width = rgb.shape[:2]
    scale = min(1.0, 1600.0 / max(height, width))
    sw, sh = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    base = Image.fromarray(rgb).resize((sw, sh), Image.Resampling.LANCZOS)
    arr = np.asarray(base).copy()
    small = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(arr, contours, -1, (0, 220, 80), 2)
    canvas = Image.fromarray(arr)
    draw = ImageDraw.Draw(canvas)
    palette = {0: (255, 190, 0), 1: (180, 40, 255), 2: (255, 255, 255)}
    shown = [
        item
        for item in detections
        if float(item["confidence"]) >= threshold and bool(item.get("center_in_mask"))
    ]
    for item in shown:
        x1, y1, x2, y2 = [value * scale for value in item["xyxy"]]
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=palette.get(int(item["category_id"]), (255, 255, 0)),
            width=3,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=88)


def load_detector(weights: Path, device: str, infer_confidence: float, image_size: int, max_det: int):
    import torch
    import sahi.models.base as sahi_base
    from sahi import AutoDetectionModel

    detector_device = torch.device(device)
    if detector_device.type == "cuda":
        # Keep the model on the requested GPU (cuda:N) when SAHI selects the device itself.
        sahi_base.select_device = lambda _device=None: detector_device
    detector = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights),
        confidence_threshold=infer_confidence,
        image_size=image_size,
        device=detector_device,
    )
    try:
        detector.model.overrides["max_det"] = max_det
    except Exception:
        pass
    return detector


def run_sliced(
    rgb: np.ndarray,
    detector,
    mask: np.ndarray,
    overlap: float,
    slice_size: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    from sahi.predict import get_sliced_prediction

    result = get_sliced_prediction(
        rgb,
        detector,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        perform_standard_pred=False,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
        postprocess_class_agnostic=True,
        verbose=0,
        progress_bar=False,
        batch_size=batch_size,
    )
    return [object_to_record(obj, mask) for obj in result.object_prediction_list]


def prediction_path(output: Path, stem: str) -> Path:
    return output / "predictions" / f"{stem}.json"


def overlay_path(output: Path, stem: str) -> Path:
    return output / "overlays" / f"{stem}.jpg"


def already_complete(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    return payload.get("status") == "complete" and isinstance(payload.get("detections"), list)
