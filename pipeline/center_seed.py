"""Stem-base seed point of the target (centre) plant, from vegetation colour alone.

The target plant stands in the middle of each photograph. Inside the central 30 % of the
image width, rows with Excess Green above a fixed threshold are treated as vegetation. The seed
row is where 70 % of that central vegetation lies above it; the seed column is the mean
vegetation column in a thin band around that row. Center BiRefNet receives this point as its
fourth-channel prior.

This is the base-seed part of the SAM3 annotation tool's stem anchoring
(``find_stem_seed(...).foreground_xy``); the SAM3 prompt points and boxes are not needed here.
"""

from __future__ import annotations

import numpy as np

VEGETATION_EXG = 0.06
CENTRAL_WIDTH_FRAC = 0.30
STEM_ROW_QUANTILE = 0.70
FALLBACK_ROW_FRAC = 0.55


def excess_green(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[:, :, i].astype(np.float32) for i in range(3))
    s = r + g + b + 1e-6
    return (2 * g - r - b) / s


def find_stem_seed(rgb: np.ndarray) -> tuple[int, int, bool]:
    """Return ``(x, y, is_fallback)`` of the stem-base seed in pixel coordinates."""
    h, w = rgb.shape[:2]
    veg = excess_green(rgb) > VEGETATION_EXG
    cw = max(1, int(CENTRAL_WIDTH_FRAC * w))
    x0 = w // 2 - cw // 2
    central = veg[:, x0 : x0 + cw]
    row_frac = central.mean(axis=1)

    total = row_frac.sum()
    if total <= 0:
        return w // 2, int(FALLBACK_ROW_FRAC * h), True

    cumulative = np.cumsum(row_frac) / total
    stem_y = int(np.clip(np.searchsorted(cumulative, STEM_ROW_QUANTILE), 0, h - 1))
    band = max(1, h // 40)
    band_veg = central[max(0, stem_y - band) : stem_y + band + 1]
    cols = np.nonzero(band_veg.any(axis=0))[0]
    seed_x = int(x0 + cols.mean()) if len(cols) else w // 2
    return seed_x, stem_y, False
