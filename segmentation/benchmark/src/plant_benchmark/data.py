from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _atomic_materialize(source: Path, destination: Path) -> None:
    """Copy a staged dataset file atomically without requiring symlink support."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    path = Path(cfg["paths"]["manifest"])
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest is missing: {path}. Build it with "
            "scripts/build_fixed_split_manifest.py before training; it is never "
            "created implicitly from the source tree."
        )
    frame = pd.read_csv(path)
    if cfg.get("data", {}).get("split_mode") == "fixed":
        validate_fixed_manifest(frame, path)
    return frame


def validate_fixed_manifest(frame: pd.DataFrame, path: Path | None = None) -> None:
    """Validate the externally frozen image-level train/validation manifest.

    ``stem`` is deliberately required to be unique even when the same biological
    plant was photographed in several weeks. The manifest builder should encode
    the acquisition/session in ``stem`` so predictions and checkpoints cannot
    silently overwrite one another.
    """
    location = str(path) if path is not None else "fixed-split manifest"
    required = {
        "stem",
        "image_path",
        "mask_path",
        "width",
        "height",
        "seed_x",
        "seed_y",
        "manually_edited",
        "split",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{location} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{location} is empty")
    duplicate_stems = frame.loc[frame["stem"].duplicated(keep=False), "stem"].tolist()
    if duplicate_stems:
        raise ValueError(
            f"{location} contains duplicate unique stems: {duplicate_stems[:10]}. "
            "Prefix repeated plant names with their acquisition/session."
        )
    normalized = frame["split"].astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - {"train", "val", "test"})
    if unknown:
        raise ValueError(f"{location} has unsupported split values: {unknown}")
    required_splits = {"train", "val"}
    absent = sorted(required_splits - set(normalized))
    if absent:
        raise ValueError(f"{location} has no records for required splits: {absent}")
    if frame[["image_path", "mask_path"]].isna().any().any():
        raise ValueError(f"{location} contains blank image_path or mask_path values")
    for column in ("width", "height", "seed_x", "seed_y"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{location} contains non-finite {column} values")
    widths = pd.to_numeric(frame["width"])
    heights = pd.to_numeric(frame["height"])
    seed_x = pd.to_numeric(frame["seed_x"])
    seed_y = pd.to_numeric(frame["seed_y"])
    if (widths <= 0).any() or (heights <= 0).any():
        raise ValueError(f"{location} contains non-positive image dimensions")
    if ((seed_x < 0) | (seed_x >= widths) | (seed_y < 0) | (seed_y >= heights)).any():
        raise ValueError(f"{location} contains out-of-bounds prompt seeds")
    blank_paths = frame[["image_path", "mask_path"]].astype(str).apply(
        lambda column: column.str.strip().eq("")
    )
    if blank_paths.any().any():
        raise ValueError(f"{location} contains blank image_path or mask_path values")


def yolo_root(cfg: dict[str, Any]) -> Path:
    artifacts = cfg.get("paths", {}).get("artifacts")
    root = Path(artifacts) if artifacts else Path(cfg["_project_dir"]) / "artifacts"
    return root / "yolo26"


def yolo_partition_root(cfg: dict[str, Any], fold: int) -> Path:
    if cfg.get("data", {}).get("split_mode") == "fixed":
        return yolo_root(cfg) / "fixed_split"
    return yolo_root(cfg) / f"fold_{fold}"


def _valid_sha256(value: Any) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value.strip()):
        return value.strip().lower()
    return None


def _atomic_pil_save(
    image: Image.Image, destination: Path, image_format: str, **kwargs: Any
) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        image.save(temporary, format=image_format, **kwargs)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_yolo26_dataset(cfg: dict[str, Any], frame: pd.DataFrame) -> Path:
    """Export upright RGB images and dense 0/1 class maps for YOLO26."""
    root = yolo_root(cfg)
    images_all = root / "images_all"
    masks_all = root / "masks_all"
    images_all.mkdir(parents=True, exist_ok=True)
    masks_all.mkdir(parents=True, exist_ok=True)

    index_path = root / "export_index.json"
    try:
        previous_index = json.loads(index_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        previous_index = {}
    if not isinstance(previous_index, dict):
        previous_index = {}
    current_index: dict[str, Any] = {}

    for row in frame.itertuples(index=False):
        image_output = images_all / f"{row.stem}.jpg"
        mask_output = masks_all / f"{row.stem}.png"
        expected_size = (int(row.width), int(row.height))
        image_source_hash = sha256(Path(row.image_path))
        recorded_image_hash = _valid_sha256(getattr(row, "image_sha256", None))
        if recorded_image_hash is not None and recorded_image_hash != image_source_hash:
            raise RuntimeError(
                f"Source image changed after manifest freeze: {row.image_path}"
            )
        mask_source_hash = sha256(Path(row.mask_path))
        recorded_mask_hash = _valid_sha256(getattr(row, "mask_sha256", None))
        if recorded_mask_hash is not None and recorded_mask_hash != mask_source_hash:
            raise RuntimeError(
                f"Source mask changed after manifest freeze: {row.mask_path}"
            )
        signature = {
            "image_sha256": image_source_hash,
            "mask_sha256": mask_source_hash,
            "width": expected_size[0],
            "height": expected_size[1],
        }
        cache_matches = previous_index.get(str(row.stem)) == signature
        valid_image = False
        if cache_matches and image_output.exists():
            try:
                with Image.open(image_output) as existing:
                    valid_image = existing.size == expected_size
            except OSError:
                valid_image = False
        if not valid_image:
            image = load_upright_image(row.image_path)
            _atomic_pil_save(
                image,
                image_output,
                "JPEG",
                quality=95,
                subsampling=1,
                optimize=True,
            )
        valid_mask = False
        if cache_matches and mask_output.exists():
            try:
                with Image.open(mask_output) as existing:
                    values = np.asarray(existing.convert("L"))
                    valid_mask = existing.size == expected_size and set(
                        np.unique(values).tolist()
                    ).issubset({0, 1})
            except OSError:
                valid_mask = False
        if not valid_mask:
            # 0 = background and 1 = the designated center plant.
            _atomic_pil_save(
                Image.fromarray(load_mask(row.mask_path).astype(np.uint8), mode="L"),
                mask_output,
                "PNG",
                optimize=True,
            )
        current_index[str(row.stem)] = signature

    _atomic_json_save(current_index, index_path)

    n_folds = int(cfg["data"].get("folds", 1))
    if cfg.get("data", {}).get("split_mode") == "fixed":
        partitions = [(0, root / "fixed_split")]
        split_names = tuple(
            split
            for split in ("train", "val", "test")
            if split in set(frame["split"].astype(str).str.strip().str.lower())
        )
    else:
        partitions = [(fold, root / f"fold_{fold}") for fold in range(n_folds)]
        split_names = ("train", "val", "test")

    for fold, fold_root in partitions:
        for split in split_names:
            selected = split_frame(frame, fold, split, n_folds)
            image_dir = fold_root / "images" / split
            mask_dir = fold_root / "masks" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            expected_names = {f"{stem}.jpg" for stem in selected["stem"]}
            for directory in (image_dir, mask_dir):
                for existing in directory.iterdir():
                    corresponding = (
                        f"{existing.stem}.jpg"
                        if directory == image_dir
                        else f"{existing.stem}.jpg"
                    )
                    if corresponding not in expected_names:
                        existing.unlink()
            for stem in selected["stem"]:
                for source, destination in (
                    (images_all / f"{stem}.jpg", image_dir / f"{stem}.jpg"),
                    (masks_all / f"{stem}.png", mask_dir / f"{stem}.png"),
                ):
                    if destination.is_symlink():
                        destination.unlink()
                    if destination.exists():
                        source_stat = source.stat()
                        destination_stat = destination.stat()
                        if (
                            source_stat.st_size == destination_stat.st_size
                            and sha256(source) == sha256(destination)
                        ):
                            continue
                    _atomic_materialize(source, destination)
        dataset_yaml: dict[str, Any] = {
            "path": str(fold_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "masks_dir": "masks",
            "names": {0: "background", 1: "center_plant"},
        }
        if "test" in split_names:
            dataset_yaml["test"] = "images/test"
        with (fold_root / "dataset.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dataset_yaml, handle, sort_keys=False)

    smoke_root = root / "smoke"
    for split, stems in (
        ("train", frame["stem"].head(2)),
        ("val", frame["stem"].iloc[2:3]),
    ):
        for kind, source_dir, suffix in (
            ("images", images_all, ".jpg"),
            ("masks", masks_all, ".png"),
        ):
            destination_dir = smoke_root / kind / split
            destination_dir.mkdir(parents=True, exist_ok=True)
            for existing in destination_dir.iterdir():
                existing.unlink()
            for stem in stems:
                _atomic_materialize(
                    source_dir / f"{stem}{suffix}",
                    destination_dir / f"{stem}{suffix}",
                )
    with (smoke_root / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "path": str(smoke_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "masks_dir": "masks",
                "names": {0: "background", 1: "center_plant"},
            },
            handle,
            sort_keys=False,
        )
    return root


def split_frame(frame: pd.DataFrame, fold: int, split: str, n_folds: int) -> pd.DataFrame:
    if "split" in frame.columns:
        normalized = frame["split"].astype(str).str.strip().str.lower()
        return frame[normalized == split.lower()].reset_index(drop=True)
    val_fold = (fold + 1) % n_folds
    if split == "test":
        result = frame[frame["fold"] == fold]
    elif split == "val":
        result = frame[frame["fold"] == val_fold]
    elif split == "train":
        result = frame[~frame["fold"].isin([fold, val_fold])]
    else:
        raise ValueError(f"Unknown split: {split}")
    return result.reset_index(drop=True)


def load_upright_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_mask(path: str | Path) -> np.ndarray:
    mask_path = Path(path)
    if mask_path.suffix.lower() == ".npy":
        return np.asarray(np.load(mask_path), dtype=bool)
    with Image.open(mask_path) as image:
        return np.asarray(image.convert("L")) > 0


def pil_mask(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


class PlantDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        height: int,
        width: int,
        augment: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.height = int(height)
        self.width = int(width)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = load_upright_image(row.image_path)
        mask = pil_mask(load_mask(row.mask_path))
        image = TF.resize(
            image, [self.height, self.width], interpolation=InterpolationMode.BILINEAR
        )
        mask = TF.resize(
            mask, [self.height, self.width], interpolation=InterpolationMode.NEAREST
        )

        if self.augment:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            angle = random.uniform(-8.0, 8.0)
            scale = random.uniform(0.92, 1.08)
            image = TF.affine(
                image,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )
            if random.random() < 0.8:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
                image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = (TF.pil_to_tensor(mask).float() / 255.0 > 0.5).float()
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "stem": str(row.stem),
            "image_path": str(row.image_path),
            "mask_path": str(row.mask_path),
            "seed_x": float(row.seed_x),
            "seed_y": float(row.seed_y),
            "orig_width": int(row.width),
            "orig_height": int(row.height),
        }


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
