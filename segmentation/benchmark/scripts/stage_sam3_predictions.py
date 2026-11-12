#!/usr/bin/env python3
"""Validate and stage SAM3 zero-shot masks for the test evaluation.

Run the SAM3 annotation tool on the test images first (no manual correction):

  python -m segmentation.sam3_annotation.run_recursive \\
      <test_root>/inputs/raw <sam3_output> <sam3.pt> cuda

``run_recursive`` mirrors the category folders, so this script reads
``<sam3_output>/<category>/mask_pngs/<stem>.mask.png`` and
``<sam3_output>/<category>/masks/<stem>.meta.json`` and copies each mask to
``<benchmark-root>/predictions/sam3_original/<category>/<stem>.mask.png``.
The SAM3 prompt defaults are asserted so that only the protocol described in
the paper is staged. Empty masks are kept and scored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = REPO_ROOT / "segmentation" / "sam3_annotation"
EXPECTED_CHECKPOINT_SHA256 = (
    "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
)
EXPECTED_PROMPTS = {"blueberry bush", "blueberry plant", "plant", None}
EXPECTED_DEFAULTS = {
    "text_prompt": "blueberry bush",
    "fallback_prompts": ("blueberry plant", "plant"),
    "sam3_confidence_thresh": 0.35,
    "use_spine_points": True,
    "spine_points": 3,
    "use_negative_points": True,
    "use_negative_boxes": True,
    "keep_seed_component": True,
    "seed_component_close_frac": 0.015,
    "prune_side_entries": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--sam3-output",
        type=Path,
        required=True,
        help="Output root written by segmentation.sam3_annotation.run_recursive.",
    )
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="SAM3 checkpoint used for the run (hashed for provenance).",
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=EXPECTED_CHECKPOINT_SHA256,
        help="SHA-256 of the SAM3 checkpoint used in the paper; pass an empty "
        "string to skip this check.",
    )
    parser.add_argument(
        "--expected-images",
        type=int,
        help="Fail unless the manifest has exactly this many unique images.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    sys.path.insert(0, str(REPO_ROOT))
    from segmentation.sam3_annotation.config import Sam3NewConfig

    config = Sam3NewConfig(
        device="cuda",
        sam3_checkpoint=checkpoint,
        text_prompt="blueberry bush",
        fallback_prompts=("blueberry plant", "plant"),
        sam3_confidence_thresh=0.35,
    )
    effective_defaults = {
        name: getattr(config, name) for name in EXPECTED_DEFAULTS
    }
    if effective_defaults != EXPECTED_DEFAULTS:
        raise RuntimeError(
            "SAM3 runtime defaults changed: "
            f"expected {EXPECTED_DEFAULTS}, found {effective_defaults}"
        )
    checkpoint_hash = sha256(checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_hash != args.expected_checkpoint_sha256:
        raise RuntimeError(
            f"Unexpected SAM3 checkpoint hash {checkpoint_hash}; "
            f"expected {args.expected_checkpoint_sha256}"
        )

    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    unique_stems = len({row["stem"] for row in rows})
    if unique_stems != len(rows):
        raise RuntimeError("Manifest stems are not unique")
    if args.expected_images is not None and len(rows) != args.expected_images:
        raise RuntimeError(
            f"Expected {args.expected_images} manifest rows, found {len(rows)}"
        )

    prompts: dict[str, int] = {}
    empty_masks: list[str] = []
    invalid: list[str] = []
    for row in rows:
        category = row["category"]
        stem = row["stem"]
        native_root = args.sam3_output / category
        mask_path = native_root / "mask_pngs" / f"{stem}.mask.png"
        meta_path = native_root / "masks" / f"{stem}.meta.json"
        if not mask_path.is_file() or not meta_path.is_file():
            invalid.append(f"{stem}: missing SAM3 mask or metadata")
            continue
        try:
            with Image.open(mask_path) as image:
                image.load()
                expected_size = (int(row["width"]), int(row["height"]))
                if image.size != expected_size:
                    invalid.append(
                        f"{stem}: size {image.size}, expected {expected_size}"
                    )
                    continue
                binary = np.asarray(image.convert("L")) > 127
            if not binary.any():
                empty_masks.append(stem)
            metadata = json.loads(meta_path.read_text())
            source_path = Path(str(metadata.get("source_path", ""))).resolve()
            if source_path != Path(row["image_path"]).resolve():
                invalid.append(
                    f"{stem}: SAM3 source {source_path} does not match manifest "
                    f"image {Path(row['image_path']).resolve()}"
                )
                continue
            prompt = metadata.get("prompt_used")
            if prompt not in EXPECTED_PROMPTS:
                invalid.append(f"{stem}: unexpected prompt {prompt!r}")
                continue
            prompt_label = "stem_only" if prompt is None else str(prompt)
            prompts[prompt_label] = prompts.get(prompt_label, 0) + 1
            staged_mask = (
                args.benchmark_root
                / "predictions"
                / "sam3_original"
                / category
                / f"{stem}.mask.png"
            )
            hardlink_or_copy(mask_path, staged_mask)
            metadata.update(
                {
                    "condition": "sam3_original",
                    "native_mask": str(mask_path.resolve()),
                    "native_metadata": str(meta_path.resolve()),
                    "reference_read_during_inference": False,
                }
            )
            staged_meta = (
                args.benchmark_root
                / "diagnostics"
                / "sam3_original"
                / category
                / f"{stem}.json"
            )
            staged_meta.parent.mkdir(parents=True, exist_ok=True)
            staged_meta.write_text(json.dumps(metadata, indent=2, default=str))
        except Exception as exc:  # noqa: BLE001
            invalid.append(f"{stem}: {type(exc).__name__}: {exc}")

    validation = {
        # An empty mask is a legitimate model failure and must remain in the
        # benchmark so it receives the appropriate zero-overlap/error scores.
        # Only malformed, missing, mismatched, or provenance-invalid artifacts
        # block staging.
        "status": "pass" if not invalid else "fail",
        "images": len(rows),
        "valid_masks": len(rows) - len(invalid),
        "valid_nonempty_masks": len(rows) - len(invalid) - len(empty_masks),
        "empty_masks": empty_masks,
        "invalid": invalid,
        "prompt_distribution": prompts,
        "code_tree_sha256": code_tree_sha256(CODE_ROOT),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "effective_defaults": effective_defaults,
        "reference_read_during_inference": False,
        "stored_seed_read_during_inference": False,
        "stored_prompt_read_during_inference": False,
    }
    diagnostics = args.benchmark_root / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / "sam3_validation.json").write_text(
        json.dumps(validation, indent=2)
    )
    print(json.dumps(validation, indent=2))
    if validation["status"] != "pass":
        raise RuntimeError("SAM3 staging validation failed")


if __name__ == "__main__":
    main()
