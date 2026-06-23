"""Persistence helpers for SAM3_new segmentation outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MASKS_SUBDIR = "masks"
NEIGHBOR_SUBDIR = "neighbor_masks"


@dataclass
class PersistedResult:
    image_name: str
    source_path: str
    height: int
    width: int
    mask: np.ndarray
    no_plant: bool
    mask_confidence: float
    seed_xy: tuple[int, int]
    pred_iou: float = 0.0
    agreement: float = 0.0
    rotate: int | None = None
    prompt_used: str | None = None
    fallback_reason: str | None = None


def save_result(result, source_path: Path, out_dir: Path, rotate: int | None = None, save_neighbor_masks: bool = True) -> Path:
    masks_dir = out_dir / MASKS_SUBDIR
    masks_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.image_name).stem
    mask_path = masks_dir / f"{stem}.mask.npy"
    np.save(mask_path, np.asarray(result.mask, dtype=bool))

    if save_neighbor_masks and getattr(result, "neighbor_masks", None):
        neigh_dir = out_dir / NEIGHBOR_SUBDIR
        neigh_dir.mkdir(parents=True, exist_ok=True)
        for i, mask in enumerate(result.neighbor_masks):
            np.save(neigh_dir / f"{stem}.neighbor_{i}.npy", np.asarray(mask, dtype=bool))

    meta = {
        "image_name": result.image_name,
        "source_path": str(Path(source_path).resolve()),
        "height": int(result.height),
        "width": int(result.width),
        "no_plant": bool(result.no_plant),
        "mask_confidence": float(result.mask_confidence),
        "seed_xy": [int(result.seed_xy[0]), int(result.seed_xy[1])],
        "pred_iou": float(getattr(result, "pred_iou", 0.0)),
        "agreement": float(getattr(result, "agreement", 0.0)),
        "rotate": rotate,
        "mask_file": mask_path.name,
        "prompt_used": getattr(result, "prompt_used", None),
        "fallback_reason": getattr(result, "fallback_reason", None),
    }
    (masks_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2))
    return mask_path

