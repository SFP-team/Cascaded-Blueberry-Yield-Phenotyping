#!/usr/bin/env python3
"""Render the age-group segmentation comparison figure (one row per age group).

Each row shows one test image with the Manual Mask and the seven model masks as
green overlays with a yellow boundary; model panels are labelled with that
image's Dice and IoU from evaluate_predictions.py.

Inputs: the test manifest, the staged predictions root (stage_test_predictions.py)
and ``tables/per_image_metrics.csv`` (evaluate_predictions.py). Writes
``figures/age_segmentation_models_grid.{png,pdf}``, the selected masks and
full-resolution overlays, and ``selected_per_image_metrics.csv``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


DEFAULT_ROWS = [
    "1 year:W2420071",
    "2 years:W23010504W13",
    "3 years:W22141310",
]

COLUMNS = [
    ("reference", "Manual\nreference"),
    ("sam3_original", "SAM3"),
    ("sam21_finetuned", "SAM2.1"),
    ("birefnet_hr_finetuned", "BiRefNet"),
    ("birefnet_hr_center_balanced", "Center-Aware"),
    ("segformer_b2_finetuned", "SegFormer-B2"),
    ("deeplabv3plus_r50_finetuned", "DeepLabV3+"),
    ("yolo26_finetuned", "YOLO26"),
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
        help="Staged root containing predictions/<condition>/<category>/<stem>.mask.png.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="tables/per_image_metrics.csv written by evaluate_predictions.py.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--row",
        action="append",
        dest="rows",
        metavar="LABEL:STEM",
        help="Figure row as 'row label:test stem' (repeatable, in display order). "
        "Default: the three images shown in the paper.",
    )
    return parser.parse_args()


def load_upright(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert("RGB"))


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as source:
        mask = np.asarray(source.convert("L")) > 127
    if mask.shape != shape:
        raise RuntimeError(f"{path} has shape {mask.shape}; expected {shape}")
    return mask


def render_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    view = rgb.copy()
    green = np.array([0, 232, 64], dtype=np.float32)
    foreground = view[mask].astype(np.float32)
    view[mask] = np.clip(0.58 * foreground + 0.42 * green, 0, 255).astype(np.uint8)
    eroded = cv2.erode(
        mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1
    ).astype(bool)
    view[mask & ~eroded] = np.array([255, 214, 0], dtype=np.uint8)
    return view


def thumbnail(rgb: np.ndarray) -> np.ndarray:
    image = Image.fromarray(rgb)
    image.thumbnail((760, 1140), Image.Resampling.LANCZOS)
    return np.asarray(image)


def source_mask(root: Path, row: dict[str, object], key: str) -> Path:
    if key == "reference":
        return Path(str(row["reference_path"]))
    return root / "predictions" / key / str(row["category"]) / f"{row['stem']}.mask.png"


def main() -> None:
    args = parse_args()
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest).set_index("stem")
    rows: list[dict[str, object]] = []
    for item in args.rows or DEFAULT_ROWS:
        label, _, stem = item.partition(":")
        if not stem:
            raise SystemExit(f"--row must be 'LABEL:STEM', got {item!r}")
        if stem not in manifest.index:
            raise SystemExit(f"{stem} is not in {args.manifest}")
        record = manifest.loc[stem]
        rows.append(
            {
                "age": label,
                "stem": stem,
                "category": record["category"],
                "image": Path(str(record["image_path"])),
                "reference_path": record["reference_path"],
            }
        )

    model_keys = [key for key, _ in COLUMNS if key != "reference"]
    metrics = pd.read_csv(args.metrics)
    selected_metrics = metrics[
        metrics["stem"].isin([str(row["stem"]) for row in rows])
        & metrics["model_key"].isin(model_keys)
    ].copy()
    expected = len(rows) * len(model_keys)
    if len(selected_metrics) != expected:
        raise RuntimeError(
            f"Expected {expected} selected metric rows, found {len(selected_metrics)}"
        )
    selected_metrics.to_csv(output_root / "selected_per_image_metrics.csv", index=False)

    panels: dict[tuple[str, str], np.ndarray] = {}
    for row in rows:
        stem = str(row["stem"])
        category = str(row["category"])
        rgb = load_upright(Path(row["image"]))
        for key, _ in COLUMNS:
            mask_path = source_mask(args.predictions_root, row, key)
            mask = load_mask(mask_path, rgb.shape[:2])
            overlay = render_overlay(rgb, mask)
            panels[(stem, key)] = thumbnail(overlay)

            overlay_path = output_root / "overlays" / key / category / f"{stem}.overlay.jpg"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlay).save(
                overlay_path, quality=94, subsampling=0, optimize=True
            )
            selected_mask = output_root / "masks" / key / category / f"{stem}.mask.png"
            selected_mask.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask_path, selected_mask)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titleweight": "semibold",
        }
    )
    figure, axes = plt.subplots(
        len(rows),
        len(COLUMNS),
        figsize=(22, 3.5 * len(rows)),
        facecolor="white",
        gridspec_kw={"wspace": 0.16, "hspace": 0.28},
        squeeze=False,
    )
    metric_index = selected_metrics.set_index(["stem", "model_key"])
    for row_index, row in enumerate(rows):
        stem = str(row["stem"])
        for column_index, (key, label) in enumerate(COLUMNS):
            axis = axes[row_index, column_index]
            axis.imshow(panels[(stem, key)])
            axis.axis("off")
            if row_index == 0:
                axis.set_title(
                    label,
                    fontsize=16,
                    fontweight="bold",
                    color="#1f2937",
                    pad=14,
                    linespacing=0.92,
                )
            if key != "reference":
                record = metric_index.loc[(stem, key)]
                axis.text(
                    0.5,
                    -0.045,
                    f"Dice {record['dice']:.3f}  |  IoU {record['iou']:.3f}",
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=11.5,
                    fontweight="medium",
                    color="#374151",
                    clip_on=False,
                )
        axes[row_index, 0].text(
            -0.16,
            0.5,
            str(row["age"]),
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=16,
            fontweight="bold",
            color="#1f2937",
        )

    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "age_segmentation_models_grid.png"
    pdf = figures / "age_segmentation_models_grid.pdf"
    figure.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.965,
        bottom=0.035,
        hspace=0.28,
        wspace=0.16,
    )
    figure.savefig(
        png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1
    )
    figure.savefig(
        pdf, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1
    )
    plt.close(figure)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
