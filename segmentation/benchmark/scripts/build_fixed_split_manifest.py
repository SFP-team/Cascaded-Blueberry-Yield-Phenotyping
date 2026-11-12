#!/usr/bin/env python3
"""Build the fixed image-level train/validation manifest.

Records are discovered recursively under the dataset root and paired by
acquisition session, cohort and original file name:

  <dataset>/<session>/<cohort>/.../<stem>.jpg                  RGB photograph
  <dataset>/<session>/<cohort>/.../mask_pngs/<stem>.mask.png   Manual Mask
  <dataset>/<session>/<cohort>/.../<stem>.meta.json            SAM3 tool metadata
  <dataset>/<session>/<cohort>/.../<stem>.mask.npy             (next to meta.json)

This is the output tree of ``segmentation.sam3_annotation`` after manual review
(raw/, results/mask_pngs/, results/masks/). Cohort folder names start with
``W20``. Every accepted record must be marked ``manually_edited`` by the review
tool. No biological-plant grouping is applied: every photograph is an
independent row. About ``--validation-fraction`` of the photographs in each
session/cohort group are assigned to ``val`` with a seeded shuffle.

Prompt points (``seed_x``, ``seed_y``) come from the metadata ``seed_xy``; when
that is absent they are derived from the image alone with the SAM3 tool's stem
anchor. A point that falls outside the Manual Mask is moved to the nearest mask
pixel because SAM2.1 needs a positive point on the target. The original point
and source are kept (``seed_x_original`` etc.) and Center BiRefNet uses only
those reference-independent coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


BENCHMARK = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
MASK_SUFFIX = ".mask.png"
META_SUFFIX = ".meta.json"
SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fixed_split.yaml"),
        help="Training config; paths.dataset is the source root and paths.manifest "
        "the output CSV (relative to segmentation/benchmark unless absolute).",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        help="Optional test manifest (stem, image_sha256). Training rows that are "
        "byte-identical to, or share a file name with, a test image are flagged.",
    )
    parser.add_argument(
        "--quarantine",
        action="append",
        default=[],
        metavar="STEM",
        help="Original file stem to exclude (repeatable), e.g. a record whose "
        "review metadata is incomplete.",
    )
    parser.add_argument(
        "--expected-records",
        type=int,
        help="Fail unless exactly this many image/mask/metadata records are paired.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_frame_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def original_stem(path: Path, suffix: str) -> str:
    if not path.name.endswith(suffix):
        raise ValueError(path)
    return path.name[: -len(suffix)]


def record_key(root: Path, path: Path, stem: str) -> tuple[str, str, str]:
    relative = path.relative_to(root)
    if len(relative.parts) < 3:
        raise ValueError(f"Unexpected source layout: {path}")
    session, cohort = relative.parts[0], relative.parts[1]
    if not cohort.startswith("W20"):
        raise ValueError(f"Unexpected cohort in {path}: {cohort}")
    return session, cohort, stem


def discover(root: Path) -> tuple[dict[tuple[str, str, str], Path], ...]:
    images: dict[tuple[str, str, str], Path] = {}
    masks: dict[tuple[str, str, str], Path] = {}
    metadata: dict[tuple[str, str, str], Path] = {}

    for path in root.rglob("*.jpg"):
        key = record_key(root, path, path.stem)
        if key in images:
            raise ValueError(f"Duplicate raw record key {key}: {images[key]} and {path}")
        images[key] = path
    for path in root.rglob(f"*{MASK_SUFFIX}"):
        if path.parent.name != "mask_pngs":
            continue
        stem = original_stem(path, MASK_SUFFIX)
        key = record_key(root, path, stem)
        if key in masks:
            raise ValueError(f"Duplicate final-mask key {key}: {masks[key]} and {path}")
        masks[key] = path
    for path in root.rglob(f"*{META_SUFFIX}"):
        stem = original_stem(path, META_SUFFIX)
        key = record_key(root, path, stem)
        if key in metadata:
            raise ValueError(f"Duplicate metadata key {key}: {metadata[key]} and {path}")
        metadata[key] = path

    image_keys, mask_keys, metadata_keys = set(images), set(masks), set(metadata)
    if image_keys != mask_keys or image_keys != metadata_keys:
        raise ValueError(
            "Pairing failure: "
            f"raw_only={sorted(image_keys - mask_keys)[:10]}, "
            f"mask_only={sorted(mask_keys - image_keys)[:10]}, "
            f"missing_meta={sorted(image_keys - metadata_keys)[:10]}, "
            f"extra_meta={sorted(metadata_keys - image_keys)[:10]}"
        )
    return images, masks, metadata


def safe_stem(session: str, cohort: str, stem: str) -> str:
    session_slug = SAFE.sub("_", session).strip("_")
    cohort_slug = SAFE.sub("_", cohort).strip("_")
    return f"{session_slug}__{cohort_slug}__{stem}"


def load_test_hashes(path: Path) -> tuple[set[str], set[str]]:
    frame = pd.read_csv(path)
    required = {"stem", "image_sha256"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} lacks {sorted(required - set(frame.columns))}")
    return set(frame["image_sha256"].astype(str)), set(frame["stem"].astype(str))


def load_image_seed_finder():
    """Load the SAM3 annotation tool's image-only stem anchor."""

    sys.path.insert(0, str(REPO_ROOT))
    from segmentation.sam3_annotation.config import Sam3NewConfig
    from segmentation.sam3_annotation.stem_anchor import find_stem_seed

    return find_stem_seed, Sam3NewConfig()


def assign_splits(frame: pd.DataFrame, fraction: float, seed: int) -> pd.Series:
    if not 0.0 < fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5")
    rng = np.random.default_rng(seed)
    assignments = pd.Series("train", index=frame.index, dtype="object")
    # Image-level stratification only. Session/cohort keeps acquisition periods
    # and plant-year prefixes represented without grouping plant identities.
    for _, group in frame.groupby(["session", "cohort"], sort=True):
        indexes = group.index.to_numpy(copy=True)
        rng.shuffle(indexes)
        count = int(round(len(indexes) * fraction))
        if len(indexes) >= 2:
            count = min(max(count, 1), len(indexes) - 1)
        else:
            count = 0
        assignments.loc[indexes[:count]] = "val"
    return assignments


def nearest_foreground(mask: np.ndarray, x: int, y: int) -> tuple[int, int, float]:
    """Return the globally nearest foreground pixel without a full coordinate list."""

    height, width = mask.shape
    radius = 16
    best: tuple[int, int, float] | None = None
    while radius < max(width, height) * 2:
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        ys, xs = np.nonzero(mask[y0:y1, x0:x1])
        if len(xs):
            global_x = xs.astype(np.int64) + x0
            global_y = ys.astype(np.int64) + y0
            squared = (global_x - x) ** 2 + (global_y - y) ** 2
            index = int(np.argmin(squared))
            distance = math.sqrt(float(squared[index]))
            best = (int(global_x[index]), int(global_y[index]), distance)
            # Anything outside this square has Euclidean distance > radius.
            if radius >= distance:
                return best
        if x0 == 0 and x1 == width and y0 == 0 and y1 == height:
            break
        radius *= 2
    if best is None:
        raise RuntimeError("Cannot repair a seed against an empty mask")
    return best


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(BENCHMARK / "src"))
    from plant_benchmark.config import load_config

    config_path = args.config if args.config.is_absolute() else BENCHMARK / args.config
    cfg = load_config(config_path)
    source = Path(cfg["paths"]["dataset"]).resolve()
    output = Path(cfg["paths"]["manifest"])
    images, masks, metadata_paths = discover(source)
    if args.expected_records is not None and len(images) != args.expected_records:
        raise RuntimeError(
            f"Expected {args.expected_records} paired records, found {len(images)}"
        )
    if args.test_manifest is not None:
        test_hashes, test_stems = load_test_hashes(args.test_manifest)
    else:
        test_hashes, test_stems = set(), set()
    find_stem_seed, stem_config = load_image_seed_finder()
    quarantine = set(args.quarantine)

    rows: list[dict[str, Any]] = []
    quarantined: list[dict[str, str]] = []
    seen_unique_stems: set[str] = set()
    for session, cohort, stem in sorted(images):
        image_path = images[(session, cohort, stem)]
        mask_path = masks[(session, cohort, stem)]
        metadata_path = metadata_paths[(session, cohort, stem)]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stem in quarantine:
            quarantined.append(
                {
                    "session": session,
                    "cohort": cohort,
                    "original_stem": stem,
                    "reason": "excluded with --quarantine",
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "metadata_path": str(metadata_path.resolve()),
                }
            )
            continue
        if metadata.get("manually_edited") is not True:
            raise RuntimeError(
                f"Accepted record is not explicitly manually edited: {metadata_path}"
            )

        with Image.open(image_path) as raw:
            stored_width, stored_height = raw.size
            upright = ImageOps.exif_transpose(raw)
            upright.load()
            width, height = upright.size
        with Image.open(mask_path) as mask_image:
            mask_width, mask_height = mask_image.size
            mask_array = np.asarray(mask_image.convert("L"))
        if (mask_width, mask_height) != (width, height):
            raise RuntimeError(
                f"Dimension mismatch for {image_path}: upright={(width, height)}, "
                f"mask={(mask_width, mask_height)}"
            )

        if np.any((mask_array != 0) & (mask_array != 255)):
            raise RuntimeError(f"Nonbinary final mask: {mask_path}")
        actual_area = int(np.count_nonzero(mask_array))
        if actual_area <= 0 or actual_area >= mask_array.size:
            raise RuntimeError(f"Empty/full final mask: {mask_path}")
        edited_area = metadata.get("edited_mask_area")
        if edited_area is None:
            raise RuntimeError(f"Accepted record lacks edited_mask_area: {metadata_path}")
        foreground_pixels = int(edited_area)
        if foreground_pixels != actual_area:
            raise RuntimeError(
                f"Edited-area mismatch for {metadata_path}: "
                f"metadata={foreground_pixels}, pixels={actual_area}"
            )

        seed = metadata.get("seed_xy")
        if isinstance(seed, (list, tuple)) and len(seed) >= 2:
            seed_x, seed_y = float(seed[0]), float(seed[1])
            seed_source = "metadata_seed_xy"
            seed_confidence = None
            seed_is_fallback = False
        else:
            rgb = np.asarray(upright.convert("RGB"))
            derived = find_stem_seed(rgb, stem_config)
            seed_x, seed_y = map(float, derived.foreground_xy)
            seed_source = "sam3_image_only_stem_anchor"
            seed_confidence = float(derived.confidence)
            seed_is_fallback = bool(derived.is_fallback)
        if not (0.0 <= seed_x < width and 0.0 <= seed_y < height):
            raise RuntimeError(
                f"Out-of-bounds seed for {image_path}: {(seed_x, seed_y)}"
            )
        seed_ix = int(np.clip(round(seed_x), 0, width - 1))
        seed_iy = int(np.clip(round(seed_y), 0, height - 1))
        foreground = mask_array > 0
        seed_on_reference = bool(foreground[seed_iy, seed_ix])

        # Positive-prompt repair for SAM2.1; the original point is preserved.
        repair: dict[str, Any] = {
            "seed_x_original": np.nan,
            "seed_y_original": np.nan,
            "seed_source_original": "",
            "seed_repaired": False,
            "seed_repair_distance_px": 0.0,
        }
        if not seed_on_reference:
            new_x, new_y, distance = nearest_foreground(foreground, seed_ix, seed_iy)
            repair = {
                "seed_x_original": seed_x,
                "seed_y_original": seed_y,
                "seed_source_original": seed_source,
                "seed_repaired": True,
                "seed_repair_distance_px": distance,
            }
            seed_x, seed_y = float(new_x), float(new_y)
            seed_source = f"{seed_source}_reference_checked_nearest_foreground"

        mask_npy_path = metadata_path.parent / f"{stem}.mask.npy"
        if not mask_npy_path.is_file():
            raise FileNotFoundError(mask_npy_path)

        image_hash = sha256(image_path)
        unique_stem = safe_stem(session, cohort, stem)
        if unique_stem in seen_unique_stems:
            raise RuntimeError(f"Output stem collision: {unique_stem}")
        seen_unique_stems.add(unique_stem)
        rows.append(
            {
                "stem": unique_stem,
                "original_stem": stem,
                "session": session,
                "cohort": cohort,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "mask_npy_path": str(mask_npy_path.resolve()),
                "metadata_path": str(metadata_path.resolve()),
                "width": width,
                "height": height,
                "stored_width": stored_width,
                "stored_height": stored_height,
                "foreground_pixels": foreground_pixels,
                "foreground_fraction": foreground_pixels / (width * height),
                "seed_x": seed_x,
                "seed_y": seed_y,
                "seed_source": seed_source,
                "seed_confidence": seed_confidence,
                "seed_is_fallback": seed_is_fallback,
                "seed_on_reference": True,
                "image_sha256": image_hash,
                "mask_sha256": sha256(mask_path),
                "manually_edited": True,
                "overlaps_test_exact_image": image_hash in test_hashes,
                "overlaps_test_stem": stem in test_stems,
                **repair,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No accepted records under {source}")
    missing_quarantine = quarantine - {row["original_stem"] for row in quarantined}
    if missing_quarantine:
        raise RuntimeError(f"Quarantine stems not found: {sorted(missing_quarantine)}")
    if frame["image_sha256"].duplicated().any():
        raise RuntimeError("Internal exact image duplicates found unexpectedly")
    frame["area_bin"] = pd.qcut(
        frame["foreground_fraction"], q=5, labels=False, duplicates="drop"
    ).astype(int)
    frame["split"] = assign_splits(frame, args.validation_fraction, args.seed)
    frame["fold"] = -1
    repair_columns = [
        "seed_x_original",
        "seed_y_original",
        "seed_source_original",
        "seed_repaired",
        "seed_repair_distance_px",
    ]
    frame = frame[[c for c in frame.columns if c not in repair_columns] + repair_columns]
    frame = frame.sort_values("stem").reset_index(drop=True)

    atomic_frame_csv(frame, output)
    atomic_frame_csv(pd.DataFrame(quarantined), output.parent / "quarantine.csv")

    repaired = frame.loc[frame["seed_repaired"]]
    distances = repaired["seed_repair_distance_px"].astype(float).tolist()
    summary = {
        "source_records": len(images),
        "accepted_records": len(frame),
        "quarantined_records": len(quarantined),
        "quarantined_stems": sorted(row["original_stem"] for row in quarantined),
        "split_counts": {k: int(v) for k, v in frame["split"].value_counts().items()},
        "session_counts": {k: int(v) for k, v in frame["session"].value_counts().items()},
        "cohort_counts": {k: int(v) for k, v in frame["cohort"].value_counts().items()},
        "seed_source_counts": {
            k: int(v) for k, v in frame["seed_source"].value_counts().items()
        },
        "positive_prompt_repair": {
            "method": "nearest foreground pixel in the Manual Mask",
            "reason": "SAM2.1 requires a positive point on the target plant",
            "repaired_count": int(len(repaired)),
            "counts_by_original_source": {
                k: int(v)
                for k, v in repaired["seed_source_original"].value_counts().items()
            },
            "distance_px_max": max(distances) if distances else 0.0,
            "original_coordinates_preserved": True,
        },
        "image_only_seed_fallback_count": int(
            frame.loc[
                frame["seed_source"].str.startswith("sam3_image_only_stem_anchor"),
                "seed_is_fallback",
            ].sum()
        ),
        "test_manifest": str(args.test_manifest) if args.test_manifest else None,
        "test_exact_image_overlap_count": int(frame["overlaps_test_exact_image"].sum()),
        "test_stem_overlap_count": int(frame["overlaps_test_stem"].sum()),
        "image_level_split": True,
        "plant_grouping": False,
        "random_seed": args.seed,
        "validation_fraction_requested": args.validation_fraction,
        "manifest_sha256": sha256(output),
        "training_config_sha256": sha256(config_path),
    }
    atomic_text(json.dumps(summary, indent=2), output.parent / "manifest_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
