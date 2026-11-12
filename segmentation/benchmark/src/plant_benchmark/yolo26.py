from __future__ import annotations

import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import run_dir
from .data import (
    prepare_yolo26_dataset,
    read_manifest,
    split_frame,
    yolo_partition_root,
    yolo_root,
)
from .engine import (
    ensure_run_fingerprint,
    experiment_fingerprint,
    file_sha256,
    mark_training_complete,
    resume_enabled,
    save_binary,
    training_is_complete,
)


def _yolo_training_outcome(output: Path, configured_epochs: int) -> tuple[int, float | None]:
    """Read the actual final epoch and best fitness from Ultralytics results.

    These values are provenance only; the immutable native/copy hashes remain
    the authoritative checkpoint identity.
    """

    results = output / "ultralytics" / "results.csv"
    if not results.is_file():
        return int(configured_epochs), None
    with results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return int(configured_epochs), None
    raw_epochs: list[int] = []
    fitness: list[float] = []
    for index, row in enumerate(rows, 1):
        raw_epoch = row.get("epoch")
        try:
            raw_epochs.append(int(float(raw_epoch)) if raw_epoch is not None else index)
        except (TypeError, ValueError):
            raw_epochs.append(index)
        for key in ("fitness", "metrics/fitness"):
            raw_value = row.get(key)
            try:
                value = float(raw_value) if raw_value is not None else float("nan")
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                fitness.append(value)
                break
    # Ultralytics releases differ: some results.csv files are zero-based while
    # the semantic trainer used here writes epochs starting at 1. Normalize to
    # a human-readable one-based completed-epoch count without adding an extra
    # epoch to already one-based results.
    epoch_offset = 1 if raw_epochs and min(raw_epochs) == 0 else 0
    final_epoch = max(raw_epochs) + epoch_offset
    return final_epoch, (max(fitness) if fitness else None)


def _atomic_verified_checkpoint_copy(source: Path, destination: Path) -> str:
    """Copy checkpoint bytes atomically without unsupported CIFS metadata writes.

    The run root may be an SMB/CIFS network share. ``shutil.copy2`` copies the
    file and then calls ``utime`` to preserve metadata, which such mounts can
    reject with ``EPERM``. Checkpoint identity is defined by its bytes,
    so copy only the contents, fsync, verify SHA-256, and atomically rename.
    Existing identical destinations are reused; mismatched ones fail closed.
    """

    source = source.resolve()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"Checkpoint source must be a real file: {source}")
    source_hash = file_sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise RuntimeError(f"Checkpoint destination must not be a symlink: {destination}")
    if destination.exists():
        destination_hash = file_sha256(destination)
        if destination_hash != source_hash:
            raise RuntimeError(
                f"Refusing to replace mismatched checkpoint: {destination}; "
                f"source={source_hash}, destination={destination_hash}"
            )
        return source_hash

    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary_hash = file_sha256(temporary)
        if temporary_hash != source_hash:
            raise RuntimeError(
                f"Checkpoint copy failed SHA-256 verification: {temporary}; "
                f"expected {source_hash}, found {temporary_hash}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return source_hash


def finalize_yolo26_training(cfg: dict[str, Any], fold: int) -> dict[str, Any]:
    """Finalize an existing native YOLO26 run without resuming training."""

    model_cfg = cfg["models"]["yolo26"]
    output = run_dir(cfg, "yolo26", "finetuned", fold)
    fingerprint = experiment_fingerprint(cfg)
    ensure_run_fingerprint(output, fingerprint)
    native_best = output / "ultralytics" / "weights" / "best.pt"
    native_last = output / "ultralytics" / "weights" / "last.pt"
    if not native_best.is_file():
        raise FileNotFoundError(f"Ultralytics did not create {native_best}")
    copied_best = output / "best.pt"
    native_hash = _atomic_verified_checkpoint_copy(native_best, copied_best)
    copied_hash = file_sha256(copied_best)
    if copied_hash != native_hash:
        raise RuntimeError(
            f"YOLO26 copied/native checkpoint mismatch: {copied_best}"
        )
    final_epoch, best_fitness = _yolo_training_outcome(
        output, int(model_cfg["epochs"])
    )
    extra = {
        "native_best": str(native_best),
        "native_last": str(native_last),
        "best_checkpoint_sha256": copied_hash,
        "native_best_sha256": native_hash,
        "actual_final_epoch": final_epoch,
        "best_val_loss_semantics": "not reported by Ultralytics semantic trainer",
        "best_fitness": best_fitness,
        "checkpoint_copy_semantics": (
            "atomic content-only copy with SHA-256 verification; filesystem "
            "metadata is intentionally not preserved on CIFS/SMB"
        ),
    }
    mark_training_complete(
        output,
        "yolo26",
        final_epoch,
        # Ultralytics does not report the BCE/Dice validation loss used by the
        # dense models. Preserve a finite compatibility value and explicitly
        # record that it is not a measured validation loss.
        0.0,
        fingerprint,
        extra=extra,
    )
    return {
        "output": str(output),
        "best": str(copied_best),
        "best_sha256": copied_hash,
        "final_epoch": final_epoch,
        "best_fitness": best_fitness,
    }


def train_yolo26(cfg: dict[str, Any], fold: int) -> None:
    from ultralytics import YOLO

    frame = read_manifest(cfg)
    prepare_yolo26_dataset(cfg, frame)
    model_cfg = cfg["models"]["yolo26"]
    output = run_dir(cfg, "yolo26", "finetuned", fold)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = experiment_fingerprint(cfg)
    if cfg.get("data", {}).get("split_mode") == "fixed" or resume_enabled(cfg):
        ensure_run_fingerprint(output, fingerprint)
    if resume_enabled(cfg) and training_is_complete(output, fingerprint):
        print("YOLO26: fixed-split training already complete; skipping")
        return
    native_last = output / "ultralytics" / "weights" / "last.pt"
    if resume_enabled(cfg) and native_last.exists():
        print(f"YOLO26: resuming native trainer from {native_last}")
        model = YOLO(str(native_last))
        model.train(resume=True, device=0)
    else:
        model = YOLO(model_cfg["pretrained_id"])
        model.train(
            data=str(yolo_partition_root(cfg, fold) / "dataset.yaml"),
            task="semantic",
            epochs=int(model_cfg["epochs"]),
            patience=int(model_cfg["patience"]),
            imgsz=int(model_cfg["image_size"]),
            batch=int(model_cfg["batch_size"]),
            optimizer=str(model_cfg["optimizer"]),
            lr0=float(model_cfg["learning_rate"]),
            cos_lr=True,
            device=0,
            workers=int(cfg["data"]["num_workers"]),
            project=str(output),
            name="ultralytics",
            exist_ok=True,
            seed=int(cfg["seed"]) + fold,
            deterministic=True,
            amp=bool(cfg["training"]["amp"]),
            cache=False,
            plots=True,
            degrees=8.0,
            scale=0.08,
            translate=0.0,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            fliplr=0.5,
            flipud=0.0,
            hsv_h=0.015,
            hsv_s=0.15,
            hsv_v=0.15,
        )
    finalize_yolo26_training(cfg, fold)


def _semantic_array(result: Any) -> np.ndarray:
    semantic_mask = getattr(result, "semantic_mask", None)
    if semantic_mask is None:
        raise RuntimeError("YOLO26 did not return a semantic_mask")
    value = semantic_mask.data
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    value = np.asarray(value)
    while value.ndim > 2:
        value = value[0]
    return value


def predict_yolo26(cfg: dict[str, Any], fold: int) -> None:
    from ultralytics import YOLO

    output = run_dir(cfg, "yolo26", "finetuned", fold)
    checkpoint = output / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"{checkpoint} is missing. Train YOLO26 first.")
    model = YOLO(str(checkpoint))
    model_cfg = cfg["models"]["yolo26"]
    frame = read_manifest(cfg)
    n_folds = int(cfg["data"]["folds"])
    for split in ("val", "test"):
        selected = split_frame(frame, fold, split, n_folds)
        prediction_dir = output / "predictions" / split
        prediction_dir.mkdir(parents=True, exist_ok=True)
        for row in tqdm(
            selected.itertuples(index=False),
            total=len(selected),
            desc=f"YOLO26 {split}",
        ):
            source = (
                yolo_partition_root(cfg, fold)
                / "images"
                / split
                / f"{row.stem}.jpg"
            )
            results = model.predict(
                source=str(source),
                task="semantic",
                imgsz=int(model_cfg["image_size"]),
                device=0,
                verbose=False,
            )
            class_map = _semantic_array(results[0])
            plant = (class_map == 1).astype(np.float32)
            save_binary(
                plant,
                0.5,
                prediction_dir / f"{row.stem}.mask.png",
                int(row.height),
                int(row.width),
            )
    with (output / "threshold.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected": None,
                "reason": "YOLO26 semantic output is a hard class map; class 1 is center_plant.",
            },
            handle,
            indent=2,
        )


def smoke_yolo26(cfg: dict[str, Any], backward: bool) -> tuple[tuple[int, ...], float]:
    from ultralytics import YOLO

    frame = read_manifest(cfg)
    prepare_yolo26_dataset(cfg, frame)
    model_cfg = cfg["models"]["yolo26"]
    model = YOLO(model_cfg["pretrained_id"])
    if backward:
        smoke_output = Path(cfg["paths"]["runs"]) / "_smoke" / "yolo26"
        model.train(
            data=str(yolo_root(cfg) / "smoke" / "dataset.yaml"),
            task="semantic",
            epochs=1,
            imgsz=int(model_cfg["image_size"]),
            batch=2,
            optimizer=str(model_cfg["optimizer"]),
            lr0=float(model_cfg["learning_rate"]),
            cos_lr=True,
            device=0,
            workers=0,
            project=str(smoke_output),
            name="train_step",
            exist_ok=True,
            amp=True,
            plots=False,
            val=False,
            degrees=8.0,
            scale=0.08,
            translate=0.0,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            fliplr=0.5,
            flipud=0.0,
            hsv_h=0.015,
            hsv_s=0.15,
            hsv_v=0.15,
        )
        size = int(model_cfg["image_size"])
        return (2, size, size), 0.0
    first = frame.iloc[0]
    results = model.predict(
        source=str(yolo_root(cfg) / "images_all" / f"{first.stem}.jpg"),
        task="semantic",
        imgsz=int(model_cfg["image_size"]),
        device=0,
        verbose=False,
    )
    value = _semantic_array(results[0])
    return tuple(value.shape), 0.0
