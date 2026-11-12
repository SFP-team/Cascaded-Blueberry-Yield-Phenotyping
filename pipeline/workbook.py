"""Write the four-sheet Excel: Per_image, Summary, Protocol, Plant_Architecture."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from traits.columns import TRAIT_COLUMNS

DETECTION_COLUMNS = [
    "image_name",
    "stem",
    "week",
    "year_class",
    "unripe",
    "ripe",
    "flowers",
    "total_berries",
    "total_detections",
    "image_height",
    "image_width",
    "mask_foreground_pixels",
    "runtime_seconds",
    "status",
    "image_path",
    "mask_path",
]

ARCHITECTURE_ID_COLUMNS = [
    "image_name",
    "stem",
    "week",
    "year_class",
    "status",
    "architecture_runtime_seconds",
    "image_path",
    "mask_path",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_workbook(
    destination: Path,
    detection_rows: list[dict[str, Any]],
    architecture_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    detections = pd.DataFrame(detection_rows)
    for column in DETECTION_COLUMNS:
        if column not in detections.columns:
            detections[column] = ""
    if detections.empty:
        detections = pd.DataFrame(columns=DETECTION_COLUMNS)
    else:
        detections = detections[DETECTION_COLUMNS]
        detections = detections.sort_values(
            ["week", "year_class", "image_name"], kind="stable"
        )
    numeric = ["unripe", "ripe", "flowers", "total_berries", "total_detections"]
    for column in numeric:
        detections[column] = pd.to_numeric(detections[column], errors="coerce").fillna(0).astype(int)

    summary = pd.DataFrame(
        [
            {
                "images": int(len(detections)),
                "unripe": int(detections["unripe"].sum()) if len(detections) else 0,
                "ripe": int(detections["ripe"].sum()) if len(detections) else 0,
                "flowers": int(detections["flowers"].sum()) if len(detections) else 0,
                "total_berries": int(detections["total_berries"].sum()) if len(detections) else 0,
                "total_detections": int(detections["total_detections"].sum()) if len(detections) else 0,
                "threshold": protocol.get("headline_threshold", 0.35),
                "filtering_rule": protocol.get("filtering_rule", ""),
                "detector": protocol.get("detector", "YOLO26-L"),
                "created_at": utc_now(),
            }
        ]
    )

    notes = pd.DataFrame(
        [
            {
                "key": key,
                "value": value if isinstance(value, (str, int, float, bool)) or value is None else str(value),
            }
            for key, value in protocol.items()
        ]
    )

    architecture = pd.DataFrame(architecture_rows)
    arch_columns = ARCHITECTURE_ID_COLUMNS + [header for header, _, _ in TRAIT_COLUMNS]
    for column in arch_columns:
        if column not in architecture.columns:
            architecture[column] = ""
    if architecture.empty:
        architecture = pd.DataFrame(columns=arch_columns)
    else:
        architecture = architecture[arch_columns]
        architecture = architecture.sort_values(
            ["week", "year_class", "image_name"], kind="stable"
        )

    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        detections.to_excel(writer, sheet_name="Per_image", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        notes.to_excel(writer, sheet_name="Protocol", index=False)
        architecture.to_excel(writer, sheet_name="Plant_Architecture", index=False)
    return destination
