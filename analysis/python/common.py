"""Shared settings and input helpers for the downstream mask-condition analysis.

Inputs
------
A manifest CSV lists the test images, one row per image:

    stem,image_path[,age_group][,canopy_architecture][,seed_x][,seed_y][,metadata_path]

``image_path`` and ``metadata_path`` may be absolute or relative to the manifest file.
Masks for each condition are binary PNGs named ``<stem>.mask.png`` (nonzero = plant) in
``<masks-root>/<condition key>/``. ``--mask-dir KEY=DIR`` overrides the folder of one condition.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "blueberry_yolo26l.pt"

# Ordered: raw baseline, manual reference mask, then the seven automatic masks.
# Display names match the ``Model`` column used by the R harvest-prediction script.
CONDITIONS = OrderedDict(
    [
        ("raw_unmasked", "Raw image"),
        ("manual_mask", "Manual Mask"),
        ("sam3", "SAM3"),
        ("sam21", "SAM2.1"),
        ("birefnet_hr", "BiRefNet-HR"),
        ("center_birefnet", "Center BiRefNet"),
        ("segformer_b2", "SegFormer-B2"),
        ("deeplabv3plus", "DeepLabV3+"),
        ("yolo26_seg", "YOLO26-Seg"),
    ]
)
REFERENCE_CONDITION = "manual_mask"
MASK_CONDITIONS = tuple(k for k in CONDITIONS if k != "raw_unmasked")
MODEL_MASK_CONDITIONS = tuple(k for k in CONDITIONS if k not in {"raw_unmasked", REFERENCE_CONDITION})

CLASS_NAMES = {
    0: "immature_berry",
    1: "mature_berry",
    2: "flowers",
}

# Optional manifest columns used to stratify the comparison tables.
GROUP_COLUMNS = ("age_group", "canopy_architecture")


def add_input_arguments(parser: argparse.ArgumentParser, *, masks: bool = True) -> None:
    parser.add_argument("--manifest", type=Path, required=True, help="Test-image manifest CSV.")
    if masks:
        parser.add_argument(
            "--masks-root",
            type=Path,
            required=True,
            help="Folder with one subfolder of <stem>.mask.png files per condition key.",
        )
        parser.add_argument(
            "--mask-dir",
            action="append",
            default=[],
            metavar="KEY=DIR",
            help="Override the mask folder of one condition (repeatable).",
        )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs"),
        help="Output root (default: ./analysis_outputs).",
    )


def parse_mask_dirs(masks_root: Path, overrides: list[str]) -> dict[str, Path]:
    folders = {key: masks_root / key for key in MASK_CONDITIONS}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or key not in MASK_CONDITIONS:
            raise SystemExit(f"--mask-dir expects KEY=DIR with KEY in {MASK_CONDITIONS}: {item}")
        folders[key] = Path(value)
    return folders


def load_manifest(path: Path) -> list[dict[str, str]]:
    base = path.resolve().parent
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Empty manifest: {path}")
    missing = {"stem", "image_path"} - set(rows[0])
    if missing:
        raise SystemExit(f"Manifest {path} lacks columns: {sorted(missing)}")
    stems = [row["stem"] for row in rows]
    if len(set(stems)) != len(stems):
        raise SystemExit(f"Manifest {path} has duplicate stems")
    for row in rows:
        for column in ("image_path", "metadata_path"):
            value = (row.get(column) or "").strip()
            if value and not Path(value).is_absolute():
                row[column] = str(base / value)
        for column in GROUP_COLUMNS:
            row[column] = (row.get(column) or "").strip() or "Unspecified"
    return rows


def mask_path(mask_dirs: dict[str, Path], stem: str, condition: str) -> Path | None:
    """Return the mask of a condition, or None for the raw (unmasked) baseline."""
    if condition == "raw_unmasked":
        return None
    return mask_dirs[condition] / f"{stem}.mask.png"


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def load_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 0
    if mask.shape != expected_shape:
        raise ValueError(
            f"Mask/image dimensions differ: mask={mask.shape}, image={expected_shape}, path={path}"
        )
    return mask


def resolve_from_repo(path: Path) -> Path:
    """Resolve a relative default (for example weights/...) against the repository root."""
    return path if path.is_absolute() or path.exists() else REPO_ROOT / path
