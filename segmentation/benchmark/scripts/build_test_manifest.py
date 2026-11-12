#!/usr/bin/env python3
"""Assemble the held-out test images and Manual Masks into age groups.

Input is a CSV with one row per test image and the columns

  stem, image_path, mask_path, metadata_path

where ``mask_path`` is the binary Manual Mask PNG (same size as the upright
image) and ``metadata_path`` is the SAM3 annotation tool's ``<stem>.meta.json``
containing ``seed_xy`` (the image-only stem seed used as the SAM2.1 point and
the Center BiRefNet location prior). Relative paths are resolved against the
CSV's folder.

The age group is taken from the first three characters of the stem
(W24 = 1 year, W23 = 2 years, W22 = 3 years, other prefixes = unspecified).
Images and masks are hard-linked (or copied) into

  <output>/inputs/raw/<category>/<image file>
  <output>/references/manually_corrected/<category>/<stem>.mask.png

and ``<output>/benchmark_manifest.csv`` plus ``reference_lock.json`` are written.
SAM3 zero-shot masks are produced later by running the SAM3 annotation tool on
``<output>/inputs/raw``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


AGE_GROUPS = {
    "W24": ("Age_1_Year_W24", 1),
    "W23": ("Age_2_Years_W23", 2),
    "W22": ("Age_3_Years_W22", 3),
}
UNSPECIFIED = "Age_Unspecified_W20_W21"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-images",
        type=int,
        help="Fail unless the CSV contains exactly this many unique stems.",
    )
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.stat().st_ino == destination.stat().st_ino:
            return "hardlink"
        if digest(source) == digest(destination):
            return "existing-identical"
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    args = parse_args()
    source_csv = args.source_csv.resolve()
    output = args.output.resolve()

    def resolve(value: object) -> Path:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (source_csv.parent / path).resolve()

    source = pd.read_csv(source_csv)
    required = {"stem", "image_path", "mask_path", "metadata_path"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"{source_csv} is missing columns: {missing}")
    if source.stem.duplicated().any():
        raise RuntimeError("Test stems must be unique")
    if args.expected_images is not None and len(source) != args.expected_images:
        raise RuntimeError(
            f"Expected {args.expected_images} test images, found {len(source)}"
        )

    rows = []
    storage_counts: dict[str, int] = {}
    for row in source.sort_values("stem").itertuples(index=False):
        prefix = str(row.stem)[:3]
        category, age_years = AGE_GROUPS.get(prefix, (UNSPECIFIED, None))
        image_path = resolve(row.image_path)
        mask_path = resolve(row.mask_path)
        metadata_path = resolve(row.metadata_path)
        if not all(path.is_file() for path in (image_path, mask_path, metadata_path)):
            raise FileNotFoundError(f"Missing source artifact for {row.stem}")
        metadata = json.loads(metadata_path.read_text())
        seed = metadata.get("seed_xy")
        if not seed or len(seed) != 2:
            raise ValueError(f"{metadata_path}: missing seed_xy")
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened)
            width, height = image.size
        with Image.open(mask_path) as opened:
            mask_size = opened.size
        if mask_size != (width, height):
            raise ValueError(f"{row.stem}: image/mask size mismatch")

        locked_image = output / "inputs" / "raw" / category / image_path.name
        locked_mask = (
            output
            / "references"
            / "manually_corrected"
            / category
            / f"{row.stem}.mask.png"
        )
        for method in (
            hardlink_or_copy(image_path, locked_image),
            hardlink_or_copy(mask_path, locked_mask),
        ):
            storage_counts[method] = storage_counts.get(method, 0) + 1

        rows.append(
            {
                "stem": str(row.stem),
                "category": category,
                "age_group": category,
                "age_years": age_years,
                "year_prefix": prefix,
                "image_path": str(locked_image),
                "reference_path": str(locked_mask),
                "width": width,
                "height": height,
                "image_sha256": digest(image_path),
                "reference_sha256": digest(mask_path),
                "metadata_path": str(metadata_path),
                "metadata_sha256": digest(metadata_path),
                "seed_x": float(seed[0]),
                "seed_y": float(seed[1]),
                "reference_provenance": "SAM3-initialized mask, manually corrected",
            }
        )

    frame = pd.DataFrame(rows).sort_values(["category", "stem"]).reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "benchmark_manifest.csv"
    frame.to_csv(manifest, index=False)
    lock = {
        "source_csv": str(source_csv),
        "output": str(output),
        "images": len(frame),
        "age_groups": {k: int(v) for k, v in frame.groupby("category").size().items()},
        "storage_methods": storage_counts,
        "manifest_sha256": digest(manifest),
        "incorporation_bias_warning": (
            "Reference masks were initialized with SAM3 and then manually corrected; "
            "this may favor SAM3 in comparisons."
        ),
    }
    (output / "reference_lock.json").write_text(json.dumps(lock, indent=2))
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
