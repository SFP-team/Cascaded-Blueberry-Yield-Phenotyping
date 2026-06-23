"""Stem/base anchoring and prompt construction for SAM3_new."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Sam3NewConfig


@dataclass
class StemSeed:
    points: np.ndarray
    labels: np.ndarray
    center_box_xyxy: np.ndarray | None
    prompt_boxes: list[tuple[np.ndarray, bool]]
    ground_line_y: int | None
    confidence: float
    is_fallback: bool

    @property
    def foreground_xy(self) -> tuple[int, int]:
        fg = self.points[self.labels == 1]
        return int(fg[0][0]), int(fg[0][1])


def excess_green(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[:, :, i].astype(np.float32) for i in range(3))
    s = r + g + b + 1e-6
    return (2 * g - r - b) / s


def vegetation_mask(rgb: np.ndarray, cfg: Sam3NewConfig) -> np.ndarray:
    return excess_green(rgb) > 0.06


def _central_row_profile(rgb: np.ndarray, cfg: Sam3NewConfig):
    h, w = rgb.shape[:2]
    veg = vegetation_mask(rgb, cfg)
    cw = max(1, int(0.30 * w))
    x0 = w // 2 - cw // 2
    central = veg[:, x0 : x0 + cw]
    return x0, central, central.mean(axis=1)


def estimate_ground_line(rgb: np.ndarray, cfg: Sam3NewConfig) -> int | None:
    _, _, row_frac = _central_row_profile(rgb, cfg)
    rows = np.where(row_frac > 0.15)[0]
    return int(rows.max()) if len(rows) else None


def _center_box_xyxy(seed_x: int, ground_y: int | None, rgb: np.ndarray, cfg: Sam3NewConfig) -> np.ndarray:
    h, w = rgb.shape[:2]
    half = int(cfg.plant_box_width_frac * w / 2)
    x_lo = max(0, seed_x - half)
    x_hi = min(w, seed_x + half)
    veg = vegetation_mask(rgb, cfg).astype(np.uint8)
    k = max(3, int(0.03 * min(h, w)) | 1)
    closed = cv2.morphologyEx(
        veg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    ys = np.nonzero(closed.any(axis=1))[0]
    top = int(ys.min()) if len(ys) else 0
    bottom = int(ground_y) if ground_y is not None else (int(ys.max()) if len(ys) else h - 1)
    return np.array([x_lo, top, x_hi, min(h, bottom + 1)], dtype=np.float32)


def _box_to_cxcywh(box_xyxy: np.ndarray, h: int, w: int) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    cx = ((x0 + x1) / 2) / w
    cy = ((y0 + y1) / 2) / h
    bw = max(1.0, x1 - x0) / w
    bh = max(1.0, y1 - y0) / h
    return np.array([cx, cy, bw, bh], dtype=np.float32)


def _build_seed(seed_x, seed_y, center_box, ground_y, cfg, h, w, confidence, is_fallback) -> StemSeed:
    points = [[seed_x, seed_y]]
    labels = [1]

    if cfg.use_negative_points:
        points.append([w // 2, int(0.04 * h)])
        labels.append(0)
        mulch_y = max(seed_y + int(0.22 * h), (ground_y + int(0.06 * h)) if ground_y is not None else 0)
        points.append([w // 2, int(min(h - 1, mulch_y))])
        labels.append(0)

    prompt_boxes: list[tuple[np.ndarray, bool]] = []
    if center_box is not None:
        prompt_boxes.append((_box_to_cxcywh(center_box, h, w), True))
    if cfg.use_negative_boxes and center_box is not None:
        mid_y = int((center_box[1] + center_box[3]) / 2)
        neg_half = int(cfg.negative_box_frac * w / 2)
        for sx in (int(center_box[0]) - neg_half, int(center_box[2]) + neg_half):
            if 0 <= sx < w:
                nx0 = max(0, sx - neg_half // 2)
                nx1 = min(w, sx + neg_half // 2)
                ny0 = max(0, mid_y - int(0.10 * h))
                ny1 = min(h, mid_y + int(0.10 * h))
                prompt_boxes.append((_box_to_cxcywh(np.array([nx0, ny0, nx1, ny1], dtype=np.float32), h, w), False))

    return StemSeed(
        points=np.array(points, dtype=np.float32),
        labels=np.array(labels, dtype=np.int32),
        center_box_xyxy=center_box,
        prompt_boxes=prompt_boxes,
        ground_line_y=ground_y,
        confidence=confidence,
        is_fallback=is_fallback,
    )


def find_stem_seed(rgb: np.ndarray, cfg: Sam3NewConfig) -> StemSeed:
    h, w = rgb.shape[:2]
    x0, central, row_frac = _central_row_profile(rgb, cfg)
    ground_y = estimate_ground_line(rgb, cfg)

    total = row_frac.sum()
    if total <= 0:
        seed_x, seed_y = w // 2, int(0.55 * h)
        return _build_seed(seed_x, seed_y, None, None, cfg, h, w, confidence=0.5, is_fallback=True)

    cumulative = np.cumsum(row_frac) / total
    stem_y = int(np.clip(np.searchsorted(cumulative, 0.70), 0, h - 1))
    band = max(1, h // 40)
    band_veg = central[max(0, stem_y - band) : stem_y + band + 1]
    cols = np.nonzero(band_veg.any(axis=0))[0]
    seed_x = int(x0 + cols.mean()) if len(cols) else w // 2

    center_box = _center_box_xyxy(seed_x, ground_y, rgb, cfg)
    return _build_seed(seed_x, stem_y, center_box, ground_y, cfg, h, w, confidence=1.0, is_fallback=False)

