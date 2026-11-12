#!/usr/bin/env python3
"""Export every staged test mask with a full-resolution image overlay.

Reads the test manifest and the staged predictions
(``<predictions-root>/predictions/<condition>/<category>/<stem>.mask.png``) and
writes ``<output>/<condition>/masks/<stem>.mask.png`` (hard-linked when
possible) and ``<output>/<condition>/overlays/<stem>.overlay.jpg`` for the seven
conditions and the Manual Mask (``reference``). Overlays are rendered on the
EXIF-corrected upright RGB image: green is the mask foreground and yellow its
one-pixel boundary. ``export_manifest.csv`` maps every file to its source.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


EXPERIMENTS = [
    ("sam3_original", "SAM3"),
    ("sam21_finetuned", "SAM2.1"),
    ("birefnet_hr_finetuned", "BiRefNet-HR"),
    ("birefnet_hr_center_balanced", "Center BiRefNet"),
    ("segformer_b2_finetuned", "SegFormer-B2"),
    ("deeplabv3plus_r50_finetuned", "DeepLabV3+"),
    ("yolo26_finetuned", "YOLO26-Seg"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Test manifest CSV.")
    parser.add_argument(
        "--predictions-root",
        type=Path,
        required=True,
        help="Staged root written by stage_test_predictions.py.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Export at most this many images (for a bounded check).",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Do not export the Manual Mask and its overlay.",
    )
    return parser.parse_args()


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def valid_jpeg(path: Path, size: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            return image.size == size and image.format == "JPEG"
    except OSError:
        return False


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(mask)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def render_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    destination: Path,
    quality: int,
) -> None:
    height, width = mask.shape
    if image_rgb.shape[:2] != (height, width):
        raise ValueError(
            f"Image/mask shape mismatch: {image_rgb.shape[:2]} versus {(height, width)}"
        )
    if valid_jpeg(destination, (width, height)):
        return

    overlay = image_rgb.copy()
    foreground = mask.astype(bool)
    overlay[foreground, 0] = (overlay[foreground, 0].astype(np.float32) * 0.42).astype(
        np.uint8
    )
    overlay[foreground, 1] = np.clip(
        overlay[foreground, 1].astype(np.float32) * 0.42 + 148,
        0,
        255,
    ).astype(np.uint8)
    overlay[foreground, 2] = (overlay[foreground, 2].astype(np.float32) * 0.42).astype(
        np.uint8
    )
    overlay[mask_boundary(foreground)] = (255, 210, 0)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.jpg")
    Image.fromarray(overlay, mode="RGB").save(
        temporary,
        format="JPEG",
        quality=quality,
        subsampling=1,
    )
    temporary.replace(destination)


def read_mask(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask_image = ImageOps.exif_transpose(image).convert("L")
        if mask_image.size != expected_size:
            raise ValueError(
                f"{path}: mask size {mask_image.size}, expected {expected_size}"
            )
        return np.asarray(mask_image) > 127


def export_one_image(task: dict[str, Any]) -> list[dict[str, Any]]:
    stem = str(task["stem"])
    category = str(task["category"])
    image_path = Path(task["image_path"])
    output_root = Path(task["output_root"])
    predictions_root = Path(task["predictions_root"])
    jpeg_quality = int(task["jpeg_quality"])

    with Image.open(image_path) as source_image:
        upright = ImageOps.exif_transpose(source_image).convert("RGB")
        image_rgb = np.asarray(upright)
        expected_size = upright.size

    records: list[dict[str, Any]] = []
    entries = list(EXPERIMENTS)
    if bool(task["include_reference"]):
        entries.append(("reference", "Manual Mask"))

    for key, label in entries:
        if key == "reference":
            source_mask = Path(task["reference_path"])
        else:
            source_mask = predictions_root / key / category / f"{stem}.mask.png"
        if not source_mask.is_file():
            raise FileNotFoundError(source_mask)

        mask_destination = output_root / key / "masks" / f"{stem}.mask.png"
        overlay_destination = output_root / key / "overlays" / f"{stem}.overlay.jpg"
        hardlink_or_copy(source_mask, mask_destination)
        mask = read_mask(source_mask, expected_size)
        render_overlay(image_rgb, mask, overlay_destination, jpeg_quality)
        records.append(
            {
                "stem": stem,
                "category": category,
                "experiment_key": key,
                "experiment": label,
                "image_path": str(image_path),
                "source_mask_path": str(source_mask),
                "mask_path": str(mask_destination),
                "overlay_path": str(overlay_destination),
                "width": expected_size[0],
                "height": expected_size[1],
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    frame = pd.read_csv(args.manifest).sort_values("stem").reset_index(drop=True)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        frame = frame.head(args.limit)
    predictions_root = args.predictions_root
    if (predictions_root / "predictions").is_dir():
        predictions_root = predictions_root / "predictions"
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    task_common = {
        "output_root": str(output_root),
        "predictions_root": str(predictions_root),
        "jpeg_quality": args.jpeg_quality,
        "include_reference": not args.skip_reference,
    }
    tasks = [
        {
            **task_common,
            "stem": row.stem,
            "category": row.category,
            "image_path": row.image_path,
            "reference_path": row.reference_path,
        }
        for row in frame.itertuples(index=False)
    ]

    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(export_one_image, task) for task in tasks]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Exporting masks and overlays",
            unit="image",
        ):
            records.extend(future.result())

    export_frame = pd.DataFrame(records).sort_values(["experiment_key", "stem"])
    export_frame.to_csv(output_root / "export_manifest.csv", index=False)
    metadata = {
        "images": int(len(frame)),
        "experiments": int(len(EXPERIMENTS)),
        "reference_included": not args.skip_reference,
        "mask_files": int(len(export_frame)),
        "overlay_files": int(len(export_frame)),
        "overlay": {
            "foreground_color": "green",
            "boundary_color": "yellow",
            "jpeg_quality": args.jpeg_quality,
            "full_resolution": True,
        },
    }
    (output_root / "export_summary.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(output_root)


if __name__ == "__main__":
    main()
