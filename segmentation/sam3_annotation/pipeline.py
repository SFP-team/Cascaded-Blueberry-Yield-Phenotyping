"""Batch orchestration for the SAM3 annotation tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Sam3NewConfig
from .io_utils import load_upright
from .mask_ops import extract_core, mask_iou, multimask_agreement, postprocess_mask
from .segmenter import Sam3HybridSegmenter, SegmentationResult
from .stem_anchor import find_stem_seed


@dataclass
class PlantResult:
    image_name: str
    height: int
    width: int
    mask: np.ndarray
    core: np.ndarray
    seed_xy: tuple[int, int]
    seed_is_fallback: bool
    no_plant: bool
    pred_iou: float
    agreement: float
    mask_confidence: float
    box: np.ndarray | None = None
    seed_points: np.ndarray | None = None
    seed_labels: np.ndarray | None = None
    confidence_map: np.ndarray | None = None
    neighbor_masks: list[np.ndarray] = field(default_factory=list)
    prompt_used: str | None = None
    sam3_candidate_count: int = 0
    sam3_selected_score: float = 0.0
    sam3_text_iou: float | None = None
    sam3_refined_iou: float | None = None
    neighbor_candidate_count: int = 0
    fallback_reason: str | None = None
    center_selection_score: float = 0.0

    @property
    def mask_area(self) -> int:
        return int(self.mask.sum())

    @property
    def core_area(self) -> int:
        return int(self.core.sum())


def _to_plant_result(seg: SegmentationResult) -> PlantResult:
    return PlantResult(
        image_name=seg.image_name,
        height=seg.height,
        width=seg.width,
        mask=seg.mask,
        core=seg.core,
        seed_xy=seg.seed_xy,
        seed_is_fallback=seg.seed_is_fallback,
        no_plant=seg.no_plant,
        pred_iou=seg.pred_iou,
        agreement=seg.agreement,
        mask_confidence=seg.mask_confidence,
        box=seg.box,
        seed_points=seg.seed_points,
        seed_labels=seg.seed_labels,
        confidence_map=seg.confidence_map,
        neighbor_masks=seg.neighbor_masks,
        prompt_used=seg.prompt_used,
        sam3_candidate_count=seg.sam3_candidate_count,
        sam3_selected_score=seg.sam3_selected_score,
        sam3_text_iou=seg.sam3_text_iou,
        sam3_refined_iou=seg.sam3_refined_iou,
        neighbor_candidate_count=seg.neighbor_candidate_count,
        fallback_reason=seg.fallback_reason,
        center_selection_score=seg.center_selection_score,
    )


def process_image(path: Path, segmenter: Sam3HybridSegmenter, cfg: Sam3NewConfig) -> PlantResult:
    rgb = load_upright(path, cfg.rotate)
    seg = segmenter.segment(rgb, image_name=path.name)
    return _to_plant_result(seg)


def confidence_map(mask: np.ndarray, core: np.ndarray, scalar: float) -> np.ndarray:
    import cv2

    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    dmax = dist.max() or 1.0
    cmap = (dist / dmax).astype(np.float32) * scalar
    cmap[core] = scalar
    return cmap

