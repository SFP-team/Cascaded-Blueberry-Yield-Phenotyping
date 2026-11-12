"""Image loading and directory enumeration for the SAM3 annotation tool."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    images: list[Path] = []
    for path in sorted(input_dir.iterdir()):
        if path.is_dir() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            log.warning("Skipping non-image file: %s", path.name)
            continue
        images.append(path)
    if not images:
        log.warning("No images found in %s", input_dir)
    return images


def load_upright(path: Path, rotate: int | None = None) -> np.ndarray:
    try:
        img = Image.open(path)
    except (UnidentifiedImageError, OSError) as e:
        raise OSError(f"Unreadable image {path}: {e}") from e

    if rotate is None:
        img = ImageOps.exif_transpose(img)
    elif rotate % 360 != 0:
        if rotate % 90 != 0:
            raise ValueError(f"rotate must be a multiple of 90, got {rotate}")
        img = img.rotate(-rotate, expand=True)

    return np.asarray(img.convert("RGB"))

