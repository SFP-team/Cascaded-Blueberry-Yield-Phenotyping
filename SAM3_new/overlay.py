"""Overlay rendering for SAM3_new outputs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .pipeline import PlantResult

_MASK_COLOR = np.array([0, 200, 0], dtype=np.float32)
_CORE_COLOR = np.array([255, 140, 0], dtype=np.float32)
_SEED_COLOR = (255, 255, 0)
_NEG_COLOR = (255, 0, 255)
_BOX_COLOR = (255, 128, 0)
_NEIGHBOR_COLOR = np.array([0, 180, 255], dtype=np.float32)


def render_overlay(rgb: np.ndarray, result: PlantResult, out_path: Path) -> None:
    vis = rgb.astype(np.float32).copy()
    if result.mask.any():
        vis[result.mask] = vis[result.mask] * 0.5 + _MASK_COLOR * 0.5
    if result.core.any():
        vis[result.core] = vis[result.core] * 0.45 + _CORE_COLOR * 0.55
    for mask in result.neighbor_masks:
        if mask is not None and np.asarray(mask).any():
            vis[np.asarray(mask, dtype=bool)] = vis[np.asarray(mask, dtype=bool)] * 0.75 + _NEIGHBOR_COLOR * 0.25
    vis = vis.astype(np.uint8)

    if result.box is not None:
        b = np.asarray(result.box, dtype=int)
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), _BOX_COLOR, 4)

    if result.seed_points is not None:
        for (x, y), lab in zip(result.seed_points.astype(int), result.seed_labels):
            if lab == 1:
                cv2.circle(vis, (x, y), 20, _SEED_COLOR, -1)
                cv2.circle(vis, (x, y), 20, (0, 0, 0), 3)
            else:
                cv2.drawMarker(vis, (x, y), _NEG_COLOR, cv2.MARKER_TILTED_CROSS, 34, 5)

    label = "NO CENTER PLANT" if result.no_plant else (
        f"conf={result.mask_confidence:.2f} iou={result.pred_iou:.2f} "
        f"agree={result.agreement:.2f} cand={result.sam3_candidate_count}"
    )
    color = (0, 0, 255) if result.no_plant else (255, 255, 255)
    cv2.putText(vis, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 7)
    cv2.putText(vis, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    if result.fallback_reason:
        cv2.putText(vis, result.fallback_reason, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 6)
        cv2.putText(vis, result.fallback_reason, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def render_mask_preview(rgb: np.ndarray, result: PlantResult, out_path: Path) -> None:
    """Write a high-contrast mask-on-image preview for quick visual inspection."""
    vis = rgb.astype(np.float32).copy()
    mask = np.asarray(result.mask, dtype=bool)
    if mask.any():
        fill = np.array([0, 255, 80], dtype=np.float32)
        vis[mask] = vis[mask] * 0.35 + fill * 0.65
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 7)
        cv2.drawContours(vis, contours, -1, (0, 140, 0), 3)

    if result.core.any():
        core = np.asarray(result.core, dtype=bool)
        core_fill = np.array([255, 180, 0], dtype=np.float32)
        vis[core] = vis[core] * 0.60 + core_fill * 0.40

    label = "NO CENTER PLANT" if result.no_plant else f"CENTER MASK  conf={result.mask_confidence:.2f}"
    vis = vis.astype(np.uint8)
    cv2.putText(vis, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 8)
    cv2.putText(vis, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def render_binary_mask(mask: np.ndarray, out_path: Path) -> None:
    """Write a plain black/white PNG of the mask."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.asarray(mask, dtype=np.uint8) * 255)
