"""Backend-agnostic mask operations for SAM3_new."""

from __future__ import annotations

import cv2
import numpy as np


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def multimask_agreement(masks: np.ndarray) -> float:
    n = len(masks)
    if n < 2:
        return 1.0
    ious = [mask_iou(masks[i], masks[j]) for i in range(n) for j in range(i + 1, n)]
    return float(max(ious)) if ious else 1.0


def postprocess_mask(mask: np.ndarray, seed_xy: tuple[int, int], min_component_area: int,
                     dilation_margin: int, keep_seed_component: bool = True,
                     seed_component_close_frac: float = 0.0) -> np.ndarray:
    """Clean the full mask, then optionally dilate.

    When `keep_seed_component` (default), keep ONLY the connected component containing
    the stem seed — this drops the disconnected background/neighbor plants the concept
    detector included. Runs on the un-eroded mask (unlike the core) so a sparse
    see-through canopy isn't split; an optional close bridges small gaps first.
    """
    if not mask.any():
        return mask.astype(bool)
    if keep_seed_component:
        out = _keep_seed_component(mask, seed_xy, seed_component_close_frac)
    else:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        out = np.zeros_like(mask, dtype=bool)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
                out |= labels == i
    if dilation_margin > 0 and out.any():
        k = 2 * dilation_margin + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.dilate(out.astype(np.uint8), kernel).astype(bool)
    return out


def _seed_component(binary: np.ndarray, seed_xy: tuple[int, int]) -> np.ndarray:
    """Keep only the connected component containing the seed (nearest if seed off-mask)."""
    if not binary.any():
        return binary.astype(bool)
    h, w = binary.shape
    n, labels = cv2.connectedComponents(binary.astype(np.uint8))
    x, y = seed_xy
    x, y = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
    seed_label = labels[y, x]
    if seed_label == 0:
        ys, xs = np.nonzero(binary)
        nearest = np.argmin((xs - x) ** 2 + (ys - y) ** 2)
        seed_label = labels[ys[nearest], xs[nearest]]
    return labels == seed_label


def _keep_seed_component(mask: np.ndarray, seed_xy: tuple[int, int], close_frac: float) -> np.ndarray:
    work = mask.astype(np.uint8)
    close = int(close_frac * min(mask.shape))
    if close > 0:
        k = 2 * close + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
    return _seed_component(work.astype(bool), seed_xy)


def extract_core(mask: np.ndarray, seed_xy: tuple[int, int], erosion_frac: float) -> np.ndarray:
    h, w = mask.shape
    k = max(3, int(erosion_frac * min(h, w)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    if not eroded.any():
        return eroded
    return _seed_component(eroded, seed_xy)

