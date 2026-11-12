"""Image-derived plant-architecture traits from a binary target-plant mask.

Pure ``mask -> traits`` (no torch, no model weights). ``analyze_plant`` takes the binary
target-plant mask, the aligned upright RGB photograph and a small metadata dict, and returns
one row of traits.

Everything is computed only from the target-plant mask (and the photo pixels under it), so
neighbouring and background plants are already excluded.

Trait groups:
  * size, shape and vertical/horizontal silhouette profile (bush habit);
  * porosity, a solid-of-revolution volume surrogate and a box-counting boundary dimension;
  * colour indices, HSV colour-coverage fractions (coverage proxies, not object counts) and
    grey-level texture.
Lengths are reported in pixels; pass ``--scale CM_PER_PX`` to also emit cm columns. All
shape ratios and colour traits are scale-free and need no calibration.

Command line (folder of ``masks/<stem>.mask.npy`` + ``masks/<stem>.meta.json``)::

    python traits/architecture.py <output_dir> [--scale CM_PER_PX] [--no-figures]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.feature import graycomatrix, graycoprops

log = logging.getLogger("traits.architecture")

EPS = 1e-6

# Colour bins on OpenCV HSV (H:0-179, S:0-255, V:0-255). Heuristic and tunable; these are
# *coverage proxies*, not classifications. Fractions are independent and may overlap slightly.
_HSV_BINS = {
    "frac_green": lambda h, s, v: (h >= 36) & (h <= 89) & (s >= 40) & (v >= 40),
    "frac_yellow": lambda h, s, v: (h >= 20) & (h < 36) & (s >= 50) & (v >= 70),
    "frac_brown": lambda h, s, v: (h >= 5) & (h < 25) & (s >= 30) & (v >= 25) & (v < 120),
    "frac_red": lambda h, s, v: ((h <= 9) | (h >= 165)) & (s >= 60) & (v >= 40),
    "frac_blue_fruit": lambda h, s, v: (h >= 100) & (h <= 140) & (s >= 35) & (v >= 18),
    "frac_flower": lambda h, s, v: ((s < 35) & (v > 165)) | ((h >= 140) & (h < 165) & (s >= 30) & (v > 120)),
}


# --------------------------------------------------------------------------------------------
# Trait groups
# --------------------------------------------------------------------------------------------

def size_traits(region, mask: np.ndarray, base_xy) -> dict:
    """Extent in pixels. Canopy height = silhouette top-to-bottom (bbox vertical extent)."""
    minr, minc, maxr, maxc = region.bbox
    area = int(region.area)
    # Fraction of plant area below the stem-base anchor: a low-skirt / drooping-cane indicator.
    base_y = int(base_xy[1])
    skirt_below_base = round(float(mask[base_y:].sum()) / (area or 1), 4) if 0 <= base_y < mask.shape[0] else 0.0
    return {
        "area_px": area,
        "canopy_height_px": int(maxr - minr),
        "canopy_width_px": int(maxc - minc),
        # Four-direction Crofton perimeter is less sensitive to pixel-grid direction than
        # chain-code/contour length.
        "perimeter_px": round(float(region.perimeter_crofton), 2),
        "equiv_diameter_px": round(math.sqrt(4.0 * area / math.pi), 2),
        "feret_max_px": round(float(region.feret_diameter_max), 2),
        "skirt_below_base_frac": skirt_below_base,
    }


def shape_traits(region, mask: np.ndarray, cnt: np.ndarray) -> dict:
    """Scale-free shape descriptors (bush habit / compactness)."""
    area = float(region.area)
    minr, minc, maxr, maxc = region.bbox
    bbox_area = float((maxr - minr) * (maxc - minc)) or 1.0
    perim = cv2.arcLength(cnt, True) or EPS
    hull = cv2.convexHull(cnt)
    hull_perim = cv2.arcLength(hull, True) or EPS
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    rect_area = (rw * rh) or 1.0
    major = float(region.axis_major_length) or EPS
    minor = float(region.axis_minor_length)

    # log-scaled Hu moments (sign-preserving) for a compact rotation/scale-invariant signature
    hu = region.moments_hu
    hu_log = {f"hu{i + 1}": round(float(-np.sign(h) * np.log10(abs(h) + EPS)), 4) for i, h in enumerate(hu)}

    out = {
        "solidity": round(area / (float(region.area_convex) or 1.0), 4),
        "convex_hull_area_px": int(region.area_convex),
        "extent": round(area / bbox_area, 4),
        "aspect_ratio_hw": round((maxr - minr) / ((maxc - minc) or EPS), 4),
        "eccentricity": round(float(region.eccentricity), 4),
        "circularity": round(4.0 * math.pi * area / (perim * perim), 4),
        "convexity": round(min(hull_perim / perim, 1.0), 4),
        "rectangularity": round(area / rect_area, 4),
        "elongation": round(1.0 - (minor / major), 4),
        "major_axis_px": round(major, 2),
        "minor_axis_px": round(minor, 2),
        "orientation_deg": round(math.degrees(float(region.orientation)), 2),
    }
    out.update(hu_log)
    return out


def profile_traits(mask: np.ndarray, region, base_xy) -> dict:
    """Vertical/horizontal silhouette profiles: the core upright-vs-spreading signals."""
    w = mask.sum(axis=1).astype(np.float64)          # width per row
    rows = np.flatnonzero(w > 0)
    top_row, bot_row = int(rows[0]), int(rows[-1])
    wv = w[top_row:bot_row + 1]
    n = len(wv)
    band = max(1, n // 5)
    width_top = float(wv[:band].mean())
    width_base = float(wv[-band:].mean())
    width_mid = float(wv[2 * n // 5:3 * n // 5].mean()) if n >= 5 else float(wv.mean())
    argmax_from_top = int(np.argmax(wv))

    cy, cx = region.centroid
    base_x, base_y = float(base_xy[0]), float(base_xy[1])
    minr, minc, maxr, maxc = region.bbox
    bbox_w = (maxc - minc) or EPS
    plant_h = (bot_row - top_row) or EPS

    mid_row = (top_row + bot_row) // 2
    area_above = int(mask[:mid_row].sum())
    area_below = int(mask[mid_row:].sum())

    bx = int(round(base_x))
    bx = min(max(bx, 0), mask.shape[1] - 1)
    area_left = int(mask[:, :bx].sum())
    area_right = int(mask[:, bx:].sum())
    total = (area_left + area_right) or 1

    # 2D solid-of-revolution volume surrogate (px^3): revolve each row's width about a vertical axis.
    revolution_volume = float(np.sum(math.pi * (wv / 2.0) ** 2))

    return {
        "width_top_px": round(width_top, 1),
        "width_mid_px": round(width_mid, 1),
        "width_base_px": round(width_base, 1),
        "top_base_width_ratio": round(width_top / (width_base or EPS), 4),
        "widest_height_frac": round(1.0 - argmax_from_top / (n or 1), 4),  # 0=base, 1=top
        "lateral_lean": round((cx - base_x) / bbox_w, 4),                  # signed fraction of width
        "centroid_height_frac": round((bot_row - cy) / plant_h, 4),       # 0=bottom, 1=top
        "top_heaviness": round(area_above / (area_below or 1), 4),
        "lr_balance": round((area_right - area_left) / total, 4),         # signed; 0 = symmetric
        "asymmetry": round(abs(area_right - area_left) / total, 4),
        "revolution_volume_px3": round(revolution_volume, 1),
    }


def porosity_traits(mask: np.ndarray, region) -> dict:
    """Gaps / openness proxies (outline-level, not true interior light penetration)."""
    filled = ndimage.binary_fill_holes(mask)
    area = int(region.area)
    area_filled = int(filled.sum())
    holes = filled & ~mask
    lbl, _ = ndimage.label(holes)
    if lbl.max():
        sizes = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, lbl.max() + 1))
        n_holes = int((sizes >= 25).sum())
    else:
        n_holes = 0
    return {
        "holes_area_px": area_filled - area,
        "holes_area_frac": round((area_filled - area) / (area_filled or 1), 4),
        "n_holes": n_holes,
        "gap_fraction": round(1.0 - area / (float(region.area_convex) or 1.0), 4),  # = 1 - solidity
    }


def boundary_fractal_traits(region) -> dict:
    """Multi-scale box-counting dimension of the largest component's one-pixel boundary.

    Counts occupied boxes across dyadic scales and reports the log-log fit quality. Multiple
    half-box grid offsets are tested at each scale; the minimum occupied-box count approximates
    grid-position-independent covering.
    """
    component = region.image.astype(bool)
    boundary = component & ~ndimage.binary_erosion(component)
    min_dim = min(boundary.shape)
    max_size = max(0, min_dim // 4)
    sizes: list[int] = []
    size = 2
    while size <= max_size:
        sizes.append(size)
        size *= 2
    if len(sizes) < 3 or boundary.sum() < 16:
        return {"fractal_dimension_boxcount": None, "fractal_dimension_r2": None}

    counts: list[float] = []
    used_sizes: list[int] = []
    for box_size in sizes:
        offsets = (0, box_size // 2)
        offset_counts: list[int] = []
        for row_offset in offsets:
            for col_offset in offsets:
                sample = boundary[row_offset:, col_offset:]
                if not sample.any():
                    continue
                rows = int(math.ceil(sample.shape[0] / box_size) * box_size)
                cols = int(math.ceil(sample.shape[1] / box_size) * box_size)
                padded = np.zeros((rows, cols), dtype=bool)
                padded[: sample.shape[0], : sample.shape[1]] = sample
                blocks = padded.reshape(
                    rows // box_size,
                    box_size,
                    cols // box_size,
                    box_size,
                )
                offset_counts.append(int(blocks.any(axis=(1, 3)).sum()))
        if offset_counts and min(offset_counts) > 0:
            used_sizes.append(box_size)
            counts.append(float(min(offset_counts)))

    if len(counts) < 3:
        return {"fractal_dimension_boxcount": None, "fractal_dimension_r2": None}
    x = np.log(1.0 / np.asarray(used_sizes, dtype=np.float64))
    y = np.log(np.asarray(counts, dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(np.square(y - fitted).sum())
    total = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - residual / total if total > EPS else 1.0
    return {
        "fractal_dimension_boxcount": round(float(slope), 4),
        "fractal_dimension_r2": round(float(r2), 4),
    }


def color_traits(rgb_crop: np.ndarray, m: np.ndarray) -> dict:
    """Colour indices and HSV colour-coverage fractions over masked (eroded) pixels."""
    px = rgb_crop[m].astype(np.float64)
    if px.shape[0] == 0:
        return {}
    R, G, B = px[:, 0], px[:, 1], px[:, 2]
    s = R + G + B + EPS
    r, g, b = R / s, G / s, B / s

    exg = 2 * g - r - b
    gcc = G / s

    def mp(a):  # mean + 10/50/90 percentiles helper for headline indices
        p = np.percentile(a, [10, 50, 90])
        return round(float(a.mean()), 4), round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)

    exg_m, exg_p10, exg_p50, exg_p90 = mp(exg)
    gcc_m, gcc_p10, gcc_p50, gcc_p90 = mp(gcc)

    out = {
        "exg_mean": exg_m, "exg_p10": exg_p10, "exg_p50": exg_p50, "exg_p90": exg_p90,
        "exr_mean": round(float((1.4 * r - g).mean()), 4),
        "exgr_mean": round(float((exg - (1.4 * r - g)).mean()), 4),
        "gli_mean": round(float(((2 * G - R - B) / (2 * G + R + B + EPS)).mean()), 4),
        # VARI denominator can pass through zero -> compute on normalised channels and clip to [-1,1]
        "vari_mean": round(float(np.clip((g - r) / (g + r - b + EPS), -1.0, 1.0).mean()), 4),
        "ngrdi_mean": round(float(((G - R) / (G + R + EPS)).mean()), 4),
        "mgrvi_mean": round(float(((G ** 2 - R ** 2) / (G ** 2 + R ** 2 + EPS)).mean()), 4),
        "rgbvi_mean": round(float(((G ** 2 - B * R) / (G ** 2 + B * R + EPS)).mean()), 4),
        # TGI on 0-1 reflectance-scaled channels (keeps magnitude interpretable)
        "tgi_mean": round(float((-0.5 * (190 * (R - G) - 120 * (R - B)) / 255.0).mean()), 4),
        "gcc_mean": gcc_m, "gcc_p10": gcc_p10, "gcc_p50": gcc_p50, "gcc_p90": gcc_p90,
    }

    # HSV colour fractions + circular hue stats; Lab a*/b* means.
    hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2HSV)
    H = hsv[..., 0][m].astype(np.float64)
    S = hsv[..., 1][m].astype(np.float64)
    V = hsv[..., 2][m].astype(np.float64)
    npx = H.shape[0]
    for name, fn in _HSV_BINS.items():
        out[name] = round(float(fn(H, S, V).sum()) / npx, 4)

    out.update(
        {
            "hsv_saturation_mean": round(float(S.mean()), 2),
            "hsv_saturation_std": round(float(S.std()), 2),
            "hsv_value_mean": round(float(V.mean()), 2),
            "hsv_value_std": round(float(V.std()), 2),
        }
    )

    ang = H / 180.0 * 2 * math.pi
    csum, ssum = np.cos(ang).mean(), np.sin(ang).mean()
    rlen = min(max(math.hypot(csum, ssum), EPS), 1.0)  # mean resultant length in (0, 1]
    out["hue_circ_mean_deg"] = round((math.degrees(math.atan2(ssum, csum)) % 360) / 2, 2)  # back to 0-179
    out["hue_circ_std"] = round(math.sqrt(-2 * math.log(rlen)), 4)

    # Preferred hue summary: circular statistics weighted by saturation, so nearly achromatic
    # pixels do not contribute arbitrary hue angles. Mean remains on OpenCV's 0..179 hue scale;
    # spread is the normalized circular standard deviation in radians.
    weights = S / 255.0
    weight_sum = float(weights.sum())
    if weight_sum > EPS:
        weighted_cos = float((weights * np.cos(ang)).sum())
        weighted_sin = float((weights * np.sin(ang)).sum())
        weighted_rlen = min(
            max(math.hypot(weighted_cos, weighted_sin) / weight_sum, EPS),
            1.0,
        )
        out["hue_sat_weighted_mean_deg"] = round(
            (math.degrees(math.atan2(weighted_sin, weighted_cos)) % 360) / 2,
            2,
        )
        out["hue_sat_weighted_std"] = round(math.sqrt(-2 * math.log(weighted_rlen)), 4)

    lab = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2Lab)
    out["lab_a_mean"] = round(float(lab[..., 1][m].astype(np.float64).mean()) - 128, 2)
    out["lab_b_mean"] = round(float(lab[..., 2][m].astype(np.float64).mean()) - 128, 2)
    return out


def legacy_color_traits(rgb_crop: np.ndarray, m: np.ndarray) -> dict:
    """Linear (non-circular) HSV summaries on the full, non-eroded supplied mask.

    Plain hue/saturation/value means and standard deviations plus yellow and brown pixel
    percentages. Hue is circular, so ``color_traits`` also reports circular, saturation-weighted
    hue statistics.
    """
    if not m.any():
        return {}
    hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0][m].astype(np.float64)
    s = hsv[..., 1][m].astype(np.float64)
    v = hsv[..., 2][m].astype(np.float64)
    yellow = (h >= 20) & (h <= 40) & (s >= 50) & (v >= 50)
    brown = (h >= 0) & (h <= 20) & (s >= 40) & (s <= 150) & (v >= 40) & (v <= 180)
    return {
        "legacy_hue_mean": round(float(h.mean()), 4),
        "legacy_hue_std": round(float(h.std()), 4),
        "legacy_saturation_mean": round(float(s.mean()), 4),
        "legacy_saturation_std": round(float(s.std()), 4),
        "legacy_value_mean": round(float(v.mean()), 4),
        "legacy_value_std": round(float(v.std()), 4),
        "legacy_yellow_pct": round(float(yellow.mean()) * 100.0, 4),
        "legacy_brown_pct": round(float(brown.mean()) * 100.0, 4),
    }


def texture_traits(rgb_crop: np.ndarray, m: np.ndarray) -> dict:
    """Canopy texture on masked grayscale (relative; illumination-sensitive)."""
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    if not m.any():
        return {}
    gvals = gray[m].astype(np.float64)
    edges = cv2.Canny(gray, 50, 150)
    out = {
        "gray_mean": round(float(gvals.mean()), 2),
        "gray_std": round(float(gvals.std()), 2),
        "edge_density": round(float((edges[m] > 0).mean()), 4),
    }

    # GLCM on a downsampled, background-excluded, fixed 32-level quantisation. Fixed bins make
    # values comparable between photographs; two distances and four angles reduce orientation bias.
    h, w = gray.shape
    f = max(1, int(math.ceil(max(h, w) / 256)))
    g_s = gray[::f, ::f]
    m_s = m[::f, ::f]
    if m_s.sum() < 16:
        return out
    levels = 32
    q = np.zeros(g_s.shape, dtype=np.uint8)
    q[m_s] = (g_s[m_s].astype(np.uint16) * levels // 256).astype(np.uint8) + 1
    glcm = graycomatrix(
        q,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels + 1,
        symmetric=True,
        normed=False,
    )
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0
    if glcm.sum() == 0:
        return out
    def prop(name: str) -> float:
        values = graycoprops(glcm, name)
        finite_values = values[np.isfinite(values)]
        return round(float(finite_values.mean()), 4) if finite_values.size else 0.0

    out["glcm_contrast"] = prop("contrast")
    out["glcm_dissimilarity"] = prop("dissimilarity")
    out["glcm_homogeneity"] = prop("homogeneity")
    out["glcm_energy"] = prop("energy")
    out["glcm_asm"] = prop("ASM")
    out["glcm_correlation"] = prop("correlation")
    p = glcm.astype(np.float64)
    p /= p.sum(axis=(0, 1), keepdims=True) + EPS
    out["glcm_entropy"] = round(float((-(p * np.log2(p + EPS)).sum(axis=(0, 1))).mean()), 4)
    return out


# --------------------------------------------------------------------------------------------
# Per-plant driver
# --------------------------------------------------------------------------------------------

def _largest_region(mask: np.ndarray):
    lbl = measure.label(mask, connectivity=2)
    props = measure.regionprops(lbl)
    if not props:
        return None
    return max(props, key=lambda r: r.area)


def analyze_plant(mask: np.ndarray, rgb: np.ndarray | None, meta: dict, scale_cm_per_px: float | None = None) -> dict:
    """Compute the full trait row for one plant. ``rgb`` may be None (geometry-only)."""
    stem = Path(meta["image_name"]).stem
    base_xy = tuple(meta.get("seed_xy", [mask.shape[1] // 2, mask.shape[0] - 1]))
    row: dict = {
        "stem": stem,
        "no_plant": bool(meta.get("no_plant", False)),
        "manually_edited": bool(meta.get("manually_edited", False)),
        "mask_confidence": round(float(meta.get("mask_confidence", 0.0)), 4),
        "img_height": int(meta.get("height", mask.shape[0])),
        "img_width": int(meta.get("width", mask.shape[1])),
    }
    if not mask.any():
        row["no_plant"] = True
        return row

    region = _largest_region(mask)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)

    row.update(size_traits(region, mask, base_xy))
    row.update(shape_traits(region, mask, cnt))
    row.update(profile_traits(mask, region, base_xy))
    row.update(porosity_traits(mask, region))
    row.update(boundary_fractal_traits(region))

    if rgb is not None and rgb.shape[:2] == mask.shape:
        minr, minc, maxr, maxc = region.bbox
        rgb_crop = np.ascontiguousarray(rgb[minr:maxr, minc:maxc])
        m_crop = mask[minr:maxr, minc:maxc]
        # erode to keep colour stats off the soil/sky edge bleed
        k = max(1, int(round(0.01 * min(m_crop.shape))))
        m_eroded = cv2.erode(m_crop.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        if m_eroded.sum() < 50:
            m_eroded = m_crop
        row.update(color_traits(rgb_crop, m_eroded))
        row.update(legacy_color_traits(rgb_crop, m_crop))
        row.update(texture_traits(rgb_crop, m_eroded))

    if scale_cm_per_px:
        sc = scale_cm_per_px
        row["canopy_height_cm"] = round(row["canopy_height_px"] * sc, 2)
        row["canopy_width_cm"] = round(row["canopy_width_px"] * sc, 2)
        row["area_cm2"] = round(row["area_px"] * sc * sc, 2)
    return row


# --------------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------------

def make_figure(stem: str, rgb: np.ndarray | None, mask: np.ndarray, region, base_xy, traits: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    H, W = mask.shape
    f = max(1, int(round(max(H, W) / 1100)))
    mask_s = mask[::f, ::f]
    minr, minc, maxr, maxc = region.bbox
    cy, cx = region.centroid

    fig, axes = plt.subplots(1, 3, figsize=(16, 7), gridspec_kw={"width_ratios": [2, 1, 1]})

    ax = axes[0]
    if rgb is not None and rgb.shape[:2] == mask.shape:
        ax.imshow(rgb[::f, ::f])
    else:
        ax.imshow(mask_s, cmap="Greens")
    ax.contour(mask_s, levels=[0.5], colors="lime", linewidths=1.2)
    cnts, _ = cv2.findContours(mask_s.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea)).reshape(-1, 2)
        ax.plot(np.append(hull[:, 0], hull[0, 0]), np.append(hull[:, 1], hull[0, 1]), "y--", lw=1)
    ax.add_patch(Rectangle((minc / f, minr / f), (maxc - minc) / f, (maxr - minr) / f,
                           fill=False, edgecolor="cyan", lw=1.2))
    ax.plot(cx / f, cy / f, "r+", ms=14, mew=2, label="centroid")
    ax.plot(base_xy[0] / f, base_xy[1] / f, "m*", ms=14, label="stem base")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(stem)
    ax.axis("off")

    # width-vs-height profile
    w = mask.sum(axis=1).astype(float)
    rows = np.flatnonzero(w > 0)
    ax = axes[1]
    ax.plot(w[rows[0]:rows[-1] + 1], np.arange(rows[0], rows[-1] + 1), "g-")
    ax.invert_yaxis()
    ax.set_title("width profile w(y)")
    ax.set_xlabel("width (px)")
    ax.set_ylabel("row (y)")

    # colour fractions
    ax = axes[2]
    frac_keys = ["frac_green", "frac_yellow", "frac_brown", "frac_red", "frac_blue_fruit", "frac_flower"]
    colors = ["#2ca02c", "#ffd92f", "#8c564b", "#d62728", "#1f77b4", "#e377c2"]
    vals = [traits.get(k, 0.0) for k in frac_keys]
    ax.bar([k.replace("frac_", "") for k in frac_keys], vals, color=colors)
    ax.set_title("canopy colour fractions")
    ax.set_ylim(0, max(0.05, max(vals) * 1.2))
    ax.tick_params(axis="x", rotation=45)

    sub = (f"habit H/W={traits.get('aspect_ratio_hw','?')}  solidity={traits.get('solidity','?')}  "
           f"ExG={traits.get('exg_mean','?')}  green={traits.get('frac_green','?')}")
    fig.suptitle(sub, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def _load_rgb(meta: dict, mask_shape) -> np.ndarray | None:
    """Load the upright RGB photograph from meta['source_path']."""
    try:
        from PIL import Image, ImageOps
        src = Path(meta["source_path"])
        if not src.is_file():
            log.warning("RGB missing: %s", src)
            return None
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode != "RGB":
                im = im.convert("RGB")
            rgb = np.asarray(im)
        # Optional rotate from meta (degrees CCW): 90/180/270
        rot = meta.get("rotate")
        if rot in (90, 180, 270):
            k = {90: 1, 180: 2, 270: 3}[int(rot)]
            rgb = np.rot90(rgb, k)
        if rgb.shape[:2] != mask_shape:
            # Small size mismatch: resize the photo onto the mask grid.
            log.warning(
                "RGB shape %s != mask %s for %s; colour/texture may be wrong",
                rgb.shape[:2],
                mask_shape,
                meta.get("image_name"),
            )
            if abs(rgb.shape[0] - mask_shape[0]) < 4 and abs(rgb.shape[1] - mask_shape[1]) < 4:
                rgb = cv2.resize(rgb, (mask_shape[1], mask_shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                return None
        return rgb
    except Exception as e:  # pragma: no cover - missing source image
        log.warning("Could not load RGB for %s: %s", meta.get("image_name"), e)
        return None


def run(output_dir: Path, scale_cm_per_px: float | None, figures: bool, limit: int | None,
        out_dir: Path | None = None) -> int:
    masks_dir = output_dir / "masks"
    if not masks_dir.is_dir():
        log.error("No masks/ directory under %s", output_dir)
        return 1
    arch_dir = out_dir if out_dir is not None else output_dir / "architecture"
    fig_dir = arch_dir / "figures"
    meta_paths = sorted(masks_dir.glob("*.meta.json"))
    if limit:
        meta_paths = meta_paths[:limit]
    if not meta_paths:
        log.error("No *.meta.json found under %s", masks_dir)
        return 1

    rows: list[dict] = []
    for i, mp in enumerate(meta_paths, 1):
        meta = json.loads(mp.read_text())
        stem = mp.name[: -len(".meta.json")]
        mask_path = masks_dir / f"{stem}.mask.npy"
        if not mask_path.exists():
            log.warning("[%d/%d] %s: mask missing", i, len(meta_paths), stem)
            continue
        mask = np.load(mask_path).astype(bool)
        rgb = _load_rgb(meta, mask.shape)
        log.info("[%d/%d] %s", i, len(meta_paths), stem)
        row = analyze_plant(mask, rgb, meta, scale_cm_per_px)
        rows.append(row)
        if figures and mask.any():
            region = _largest_region(mask)
            base_xy = tuple(meta.get("seed_xy", [mask.shape[1] // 2, mask.shape[0] - 1]))
            try:
                make_figure(stem, rgb, mask, region, base_xy, row, fig_dir / f"{stem}.png")
            except Exception as e:  # pragma: no cover
                log.warning("figure failed for %s: %s", stem, e)

    arch_dir.mkdir(parents=True, exist_ok=True)
    # union of keys, first-seen order, so optional colour/cm columns are preserved
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    csv_path = arch_dir / "architecture_report.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path = arch_dir / "architecture_report.json"
    json_path.write_text(json.dumps(rows, indent=2))

    print()
    print(f"Plants analysed:  {len(rows)}")
    print(f"Traits per plant: {len(fields)}")
    print(f"Report (CSV):     {csv_path}")
    print(f"Report (JSON):    {json_path}")
    if figures:
        print(f"Figures:          {fig_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="architecture.py",
                                description="Plant-architecture traits from a folder of masks.")
    p.add_argument("output_dir", help="Directory containing masks/<stem>.mask.npy and "
                                       "masks/<stem>.meta.json")
    p.add_argument("--out-dir", default=None, dest="out_dir",
                   help="Where to write architecture_report.* + figures/ "
                        "(default: <output_dir>/architecture)")
    p.add_argument("--scale", type=float, default=None, dest="scale_cm_per_px",
                   help="cm per pixel; if given, also emit cm/cm2 columns.")
    p.add_argument("--no-figures", action="store_true", help="Skip per-plant figures (faster).")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N plants.")
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    return run(Path(args.output_dir), args.scale_cm_per_px, not args.no_figures, args.limit,
               Path(args.out_dir) if args.out_dir else None)


if __name__ == "__main__":
    import sys
    sys.exit(main())
