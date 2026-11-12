#!/usr/bin/env python3
"""Stage the seven test-set mask sets in one layout for evaluation.

Trained-model masks are read from ``<runs-root>/<model>/finetuned/fixed_split/
predictions/test`` (Center BiRefNet: the ``balanced`` operating point) and SAM3
masks from ``--sam3-root``. Every mask is checked against the manifest image
size and hard-linked (or copied) to
``<output>/predictions/<condition>/<category>/<stem>.mask.png``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from PIL import Image


SOURCES = OrderedDict(
    [
        ("sam3_original", None),
        ("sam21_finetuned", "sam21/finetuned/fixed_split/predictions/test"),
        (
            "birefnet_hr_finetuned",
            "birefnet_hr/finetuned/fixed_split/predictions/test",
        ),
        (
            "birefnet_hr_center_balanced",
            "birefnet_hr_centered/finetuned/fixed_split/predictions/balanced/test",
        ),
        (
            "segformer_b2_finetuned",
            "segformer_b2/finetuned/fixed_split/predictions/test",
        ),
        (
            "deeplabv3plus_r50_finetuned",
            "deeplabv3plus_r50/finetuned/fixed_split/predictions/test",
        ),
        ("yolo26_finetuned", "yolo26/finetuned/fixed_split/predictions/test"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--sam3-root",
        type=Path,
        required=True,
        help="Directory containing SAM3 masks, flat or category/stem.mask.png.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def locate(root: Path, category: str, stem: str) -> Path:
    candidates = (
        root / category / f"{stem}.mask.png",
        root / f"{stem}.mask.png",
        root / category / f"{stem}.png",
        root / f"{stem}.png",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No mask for {stem} under {root}")


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise RuntimeError(f"Staged prediction must not be a symlink: {destination}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        if destination.exists():
            destination.unlink()
        try:
            os.link(source, destination)
            return
        except OSError:
            # SMB/CIFS mounts can reject the utime call
            # performed by copy2(). Copy bytes only, fsync, verify content,
            # then atomically rename the destination-local temporary file.
            shutil.copyfile(source, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            if file_sha256(source) != file_sha256(temporary):
                raise RuntimeError(
                    f"Staged prediction copy failed SHA-256 verification: {source}"
                )
            temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    if frame["stem"].nunique() != len(frame):
        raise ValueError("Test manifest stems must be unique")
    counts: dict[str, int] = {}
    for key, relative in SOURCES.items():
        source_root = args.sam3_root if relative is None else args.runs_root / relative
        count = 0
        for row in frame.itertuples(index=False):
            source = locate(source_root, str(row.category), str(row.stem))
            with Image.open(source) as image:
                if image.size != (int(row.width), int(row.height)):
                    raise ValueError(
                        f"{source}: size={image.size}, expected="
                        f"{(int(row.width), int(row.height))}"
                    )
            destination = (
                args.output
                / "predictions"
                / key
                / str(row.category)
                / f"{row.stem}.mask.png"
            )
            hardlink_or_copy(source, destination)
            count += 1
        counts[key] = count
    expected = len(frame)
    if any(count != expected for count in counts.values()):
        raise RuntimeError(f"Incomplete staging: {counts}")
    payload = {
        "status": "pass",
        "images": expected,
        "models": len(SOURCES),
        "prediction_counts": counts,
        "runs_root": str(args.runs_root.resolve()),
        "sam3_root": str(args.sam3_root.resolve()),
        "output": str(args.output.resolve()),
    }
    diagnostics = args.output / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / "fixed_split_staging.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
