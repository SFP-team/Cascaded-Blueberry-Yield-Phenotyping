"""Report writers for the SAM3 annotation tool."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

from .pipeline import PlantResult

FIELDS = [
    "image",
    "width",
    "height",
    "no_plant",
    "seed_x",
    "seed_y",
    "seed_is_fallback",
    "mask_area",
    "mask_area_frac",
    "core_area",
    "pred_iou",
    "agreement",
    "mask_confidence",
]

DIAGNOSTIC_FIELDS = [
    "image",
    "prompt_used",
    "sam3_candidate_count",
    "sam3_selected_score",
    "sam3_text_iou",
    "sam3_refined_iou",
    "neighbor_candidate_count",
    "fallback_reason",
    "center_selection_score",
]


def result_row(r: PlantResult) -> dict:
    area = r.width * r.height or 1
    return {
        "image": r.image_name,
        "width": r.width,
        "height": r.height,
        "no_plant": r.no_plant,
        "seed_x": r.seed_xy[0],
        "seed_y": r.seed_xy[1],
        "seed_is_fallback": r.seed_is_fallback,
        "mask_area": r.mask_area,
        "mask_area_frac": round(r.mask_area / area, 5),
        "core_area": r.core_area,
        "pred_iou": round(r.pred_iou, 4),
        "agreement": round(r.agreement, 4),
        "mask_confidence": round(r.mask_confidence, 4),
    }


def diagnostic_row(r: PlantResult) -> dict:
    return {
        "image": r.image_name,
        "prompt_used": r.prompt_used,
        "sam3_candidate_count": r.sam3_candidate_count,
        "sam3_selected_score": round(r.sam3_selected_score, 4),
        "sam3_text_iou": None if r.sam3_text_iou is None else round(float(r.sam3_text_iou), 4),
        "sam3_refined_iou": None if r.sam3_refined_iou is None else round(float(r.sam3_refined_iou), 4),
        "neighbor_candidate_count": r.neighbor_candidate_count,
        "fallback_reason": r.fallback_reason,
        "center_selection_score": round(r.center_selection_score, 4),
    }


def write_reports(rows: list[PlantResult], output_dir: Path, fmt: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    seg_rows = [result_row(r) for r in rows]
    diag_rows = [diagnostic_row(r) for r in rows]

    if fmt in ("csv", "both"):
        path = output_dir / "segmentation_report.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(seg_rows)
        written.append(path)

        dpath = output_dir / "sam3_diagnostics.csv"
        with dpath.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_FIELDS)
            writer.writeheader()
            writer.writerows(diag_rows)
        written.append(dpath)

    if fmt in ("json", "both"):
        path = output_dir / "segmentation_report.json"
        path.write_text(json.dumps(seg_rows, indent=2))
        written.append(path)

        dpath = output_dir / "sam3_diagnostics.json"
        dpath.write_text(json.dumps(diag_rows, indent=2))
        written.append(dpath)

    return written

