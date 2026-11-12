"""Find field photographs and derive the Excel identity columns."""

from __future__ import annotations

import re
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SKIP_DIR_NAMES = {
    "masks",
    "mask_overlays",
    "overlays",
    "predictions",
    "tables",
    "weights",
    "vendor",
    "pipeline",
    "output",
    "__pycache__",
    ".git",
    # Derived-image folders from earlier canopy analyses that can sit next to the raw photos.
    "final_canopy_mask",
    "detections",
    "canopy_metric_visualization",
    "HSV_mask",
    "leaf_segmentation",
    "distance_transform",
    "silhouette_analysis",
}

STEM_YEAR = re.compile(r"^(W\d{2})")
AGE_FOLDER = re.compile(r"^Age_\d+_Year_(W\d{2})$", re.IGNORECASE)
YEAR_FOLDER = re.compile(r"^(20\d{2})$")
GENOTYPE_FOLDER = re.compile(r"^\d{2}[.-]\d")
SKIP_REL_PARTS = {"phenome-raw", "phenome", "raw", "inputs"}


def year_class_from_stem(stem: str) -> str:
    match = STEM_YEAR.match(stem)
    return match.group(1) if match else ""


def relative_parts(image_path: Path, source: Path) -> tuple[str, ...]:
    # absolute(), not resolve(): a symlinked photo keeps its place under the input folder.
    try:
        return image_path.absolute().relative_to(source.absolute()).parts
    except ValueError:
        return image_path.parts


def week_and_year(image_path: Path, source: Path) -> tuple[str, str]:
    parts = relative_parts(image_path, source)
    week = ""
    year = ""
    for part in parts:
        age = AGE_FOLDER.match(part)
        if age:
            week = part
            year = age.group(1)
            break
        if part.startswith("Age_"):
            week = part
    if not year:
        for part in parts:
            if YEAR_FOLDER.match(part):
                year = part
                break
    if not year:
        year = year_class_from_stem(image_path.stem)
    if not week:
        after_raw = False
        for part in parts[:-1]:
            lowered = part.lower()
            if lowered == "phenome-raw":
                after_raw = True
                continue
            if not after_raw:
                continue
            if YEAR_FOLDER.match(part) or GENOTYPE_FOLDER.match(part):
                continue
            if lowered in SKIP_REL_PARTS:
                continue
            week = part
            break
    if not week:
        for part in parts[:-1]:
            lowered = part.lower()
            if lowered in SKIP_REL_PARTS:
                continue
            if YEAR_FOLDER.match(part) or GENOTYPE_FOLDER.match(part):
                continue
            week = part
            break
    if not year:
        for part in image_path.resolve().parts:
            if YEAR_FOLDER.match(part):
                year = part
    return week, year


def unique_stem(image_path: Path, source: Path, seen: set[str]) -> str:
    stem = image_path.stem
    if stem not in seen:
        return stem
    parts = relative_parts(image_path, source)
    qualified = "__".join(Path(*parts).with_suffix("").parts)
    if qualified not in seen:
        return qualified
    raise ValueError(f"Duplicate image id {qualified!r}: {image_path}")


def iter_images(source: Path) -> list[Path]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Not an image: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)

    found: list[Path] = []
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.name.startswith("."):
            continue
        if path.name.endswith(".mask.png"):
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        found.append(path)
    return sorted(found)


def discover_records(source: Path) -> list[dict[str, str]]:
    source = source.expanduser().resolve()
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for image_path in iter_images(source):
        stem = unique_stem(image_path, source, seen)
        seen.add(stem)
        week, year = week_and_year(image_path, source)
        rows.append(
            {
                "stem": stem,
                "image_name": image_path.name,
                "week": week,
                "year_class": year,
                "image_path": str(image_path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No images under {source}")
    return rows
