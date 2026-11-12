from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from .config import run_dir
from .data import (
    PlantDataset,
    load_mask,
    load_upright_image,
    pil_mask,
    read_manifest,
    split_frame,
    worker_seed,
)
from .models import create_dense_model, create_sam21, freeze_batch_norm, set_train_scope


def resume_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("training", {}).get("resume", False))


def completion_path(output: Path) -> Path:
    return output / "training_complete.json"


def _fingerprint_value(value: Any) -> Any:
    """Return a stable, JSON-serializable view of configuration data."""

    if isinstance(value, dict):
        return {
            str(key): _fingerprint_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def experiment_fingerprint(cfg: dict[str, Any]) -> str:
    """Bind a run to both its effective config and frozen manifest bytes."""

    manifest = Path(cfg["paths"]["manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(f"Cannot fingerprint missing manifest: {manifest}")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            _fingerprint_value(cfg),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(b"\0manifest\0")
    with manifest.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_finite_state_dict(state: dict[str, Any], source: Path | str) -> None:
    """Reject a checkpoint containing any non-finite floating-point tensor."""

    for name, value in state.items():
        if isinstance(value, torch.Tensor) and (
            value.is_floating_point() or value.is_complex()
        ):
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"Non-finite tensor {name!r} in checkpoint: {source}")


def guarded_amp_step(
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[torch.nn.Parameter],
    max_gradient_norm: float,
) -> dict[str, float | bool]:
    """Clip unscaled gradients and safely consume an AMP overflow.

    ``GradScaler.unscale_`` records non-finite gradients internally. Calling
    ``step`` afterward is important even on overflow: the scaler then skips
    ``optimizer.step`` rather than applying corrupt gradients, and ``update``
    reduces the scale for the next batch. Gradients are always cleared.
    """

    trainable = list(parameters)
    scale_before = float(scaler.get_scale())
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable,
        max_norm=float(max_gradient_norm),
        error_if_nonfinite=False,
    )
    finite = bool(torch.isfinite(gradient_norm))
    if finite or scaler.is_enabled():
        # When enabled, GradScaler skips the underlying optimizer step if
        # unscale_ recorded inf/NaN. When disabled, step only on finite grads.
        scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    if not finite and scaler.is_enabled():
        if not scale_after < scale_before:
            raise FloatingPointError(
                "GradScaler detected non-finite gradients but did not reduce "
                f"its scale: before={scale_before}, after={scale_after}"
            )
        if scale_after < 1.0:
            raise FloatingPointError(
                "GradScaler scale fell below the audited safety floor after an "
                f"overflow: before={scale_before}, after={scale_after}"
            )
    optimizer.zero_grad(set_to_none=True)
    return {
        "finite": finite,
        "gradient_norm": float(gradient_norm),
        "scale_before": scale_before,
        "scale_after": scale_after,
    }


def atomic_json_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_run_fingerprint(output: Path, fingerprint: str) -> None:
    """Create or validate the immutable config/manifest binding for a run."""

    path = output / "experiment_fingerprint.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unreadable run fingerprint: {path}") from exc
        if payload.get("experiment_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Run fingerprint mismatch at {output}. Use a new run directory "
                "instead of mixing a changed config or manifest with old checkpoints."
            )
        return
    existing_state = [
        output / "best.pt",
        output / "last.pt",
        completion_path(output),
        output / "ultralytics" / "weights" / "last.pt",
    ]
    if any(candidate.exists() for candidate in existing_state):
        raise RuntimeError(
            f"Unbound checkpoint state already exists at {output}; refusing an "
            "unsafe resume. Move it aside or use a new run directory."
        )
    atomic_json_save(
        {
            "experiment_fingerprint": fingerprint,
            "created_unix_time": time.time(),
        },
        path,
    )


def assert_checkpoint_fingerprint(
    checkpoint: dict[str, Any], fingerprint: str, checkpoint_path: Path
) -> None:
    if checkpoint.get("experiment_fingerprint") != fingerprint:
        raise RuntimeError(
            f"Checkpoint fingerprint mismatch or missing fingerprint: {checkpoint_path}"
        )


def training_is_complete(output: Path, fingerprint: str | None = None) -> bool:
    path = completion_path(output)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable completion marker: {path}") from exc
    if fingerprint is not None and payload.get("experiment_fingerprint") != fingerprint:
        raise RuntimeError(f"Completion marker fingerprint mismatch: {path}")
    if not (output / "best.pt").is_file():
        raise RuntimeError(f"Completion marker exists but best checkpoint is missing: {output}")
    return True


def mark_training_complete(
    output: Path,
    model_name: str,
    final_epoch: int,
    best_val_loss: float,
    fingerprint: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model": model_name,
        "final_epoch": int(final_epoch),
        "best_val_loss": float(best_val_loss),
        "completed_unix_time": time.time(),
    }
    if fingerprint is not None:
        payload["experiment_fingerprint"] = fingerprint
    if extra:
        payload.update(extra)
    atomic_json_save(payload, completion_path(output))


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid()
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def combined_loss(logits: torch.Tensor, target: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss(logits, target)
    training = cfg["training"]
    return training["loss_bce_weight"] * bce + training["loss_dice_weight"] * dice


def dense_loaders(
    cfg: dict[str, Any], fold: int, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    frame = read_manifest(cfg)
    n_folds = int(cfg["data"]["folds"])
    height, width = int(cfg["data"]["height"]), int(cfg["data"]["width"])
    train = PlantDataset(
        split_frame(frame, fold, "train", n_folds), height, width, augment=True
    )
    val = PlantDataset(
        split_frame(frame, fold, "val", n_folds), height, width, augment=False
    )
    common = {
        "num_workers": int(cfg["data"]["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_seed,
        "persistent_workers": int(cfg["data"]["num_workers"]) > 0,
    }
    generator = torch.Generator().manual_seed(int(cfg["seed"]) + fold)
    # DeepLabV3+'s ASPP pooling branch applies BatchNorm to a 1x1 feature
    # map. Avoid a final singleton batch when an odd-sized split is trained
    # with batch_size > 1; BatchNorm cannot estimate statistics from it.
    drop_singleton_batch = batch_size > 1 and len(train) % batch_size == 1
    return (
        DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            drop_last=drop_singleton_batch,
            **common,
        ),
        DataLoader(val, batch_size=1, shuffle=False, **common),
    )


@torch.no_grad()
def validate_dense(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict[str, Any],
    target_device: torch.device,
) -> tuple[float, float]:
    model.eval()
    losses, dices = [], []
    for batch in loader:
        image = batch["image"].to(target_device, non_blocking=True)
        target = batch["mask"].to(target_device, non_blocking=True)
        with torch.autocast(
            device_type=target_device.type,
            dtype=torch.float16,
            enabled=bool(cfg["training"]["amp"]) and target_device.type == "cuda",
        ):
            logits = model(image)
            loss = combined_loss(logits, target, cfg)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Dense validation produced a non-finite loss")
        losses.append(float(loss))
        pred = logits.sigmoid() >= 0.5
        intersection = (pred & target.bool()).sum().item()
        dices.append(
            (2 * intersection + 1.0) / (pred.sum().item() + target.sum().item() + 1.0)
        )
    return float(np.mean(losses)), float(np.mean(dices))


def train_dense(cfg: dict[str, Any], model_name: str, fold: int) -> None:
    model_cfg = cfg["models"][model_name]
    output = run_dir(cfg, model_name, "finetuned", fold)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = experiment_fingerprint(cfg)
    if cfg.get("data", {}).get("split_mode") == "fixed" or resume_enabled(cfg):
        ensure_run_fingerprint(output, fingerprint)
    if resume_enabled(cfg) and training_is_complete(output, fingerprint):
        print(f"{model_name}: fixed-split training already complete; skipping")
        return
    target_device = device()
    model = create_dense_model(model_name, cfg).to(target_device)
    if model_name == "birefnet_hr":
        # The remote checkpoint is stored in FP16. Keep FP32 master parameters
        # so GradScaler can safely unscale gradients during AMP fine-tuning.
        model.float()
        trainable = set_train_scope(model, model_cfg.get("train_scope", "all"))
        print(f"BiRefNet trainable parameters: {trainable:,}")
    batch_size = int(model_cfg["batch_size"])
    train_loader, val_loader = dense_loaders(cfg, fold, batch_size)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = int(model_cfg.get("epochs", cfg["training"]["epochs"]))
    accumulation = int(
        model_cfg.get(
            "gradient_accumulation", cfg["training"]["gradient_accumulation"]
        )
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(cfg["training"]["amp"]) and target_device.type == "cuda"
    )
    patience = int(cfg["training"]["early_stopping_patience"])
    best, stale = float("inf"), 0
    start_epoch = 1
    last_path = output / "last.pt"
    if resume_enabled(cfg) and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        assert_checkpoint_fingerprint(checkpoint, fingerprint, last_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_val_loss"])
        stale = int(checkpoint["stale_epochs"])
        print(f"{model_name}: resuming at epoch {start_epoch} from {last_path}")
        if stale >= patience or start_epoch > epochs:
            if not (output / "best.pt").exists():
                raise RuntimeError(f"{model_name}: resume state has no best checkpoint")
            mark_training_complete(
                output, model_name, start_epoch - 1, best, fingerprint
            )
            print(f"{model_name}: recovered completed training state")
            return

    history_path = output / "history.csv"
    append_history = start_epoch > 1 and history_path.exists()
    final_epoch = start_epoch - 1
    with history_path.open("a" if append_history else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "train_loss", "val_loss", "val_dice", "lr"]
        )
        if not append_history:
            writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            final_epoch = epoch
            model.train()
            if model_name == "birefnet_hr" and batch_size == 1:
                freeze_batch_norm(model)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            progress = tqdm(train_loader, desc=f"{model_name} fold={fold} epoch={epoch}")
            for step, batch in enumerate(progress, 1):
                image = batch["image"].to(target_device, non_blocking=True)
                target = batch["mask"].to(target_device, non_blocking=True)
                with torch.autocast(
                    device_type=target_device.type,
                    dtype=torch.float16,
                    enabled=scaler.is_enabled(),
                ):
                    logits = model(image)
                    loss = combined_loss(logits, target, cfg) / accumulation
                scaler.scale(loss).backward()
                if step % accumulation == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss) * accumulation)
                progress.set_postfix(loss=f"{np.mean(losses[-10:]):.4f}")
                # Do not retain the previous full-resolution graph tensors into
                # the next BiRefNet iteration on memory-constrained GPUs.
                del image, target, logits, loss
            val_loss, val_dice = validate_dense(model, val_loader, cfg, target_device)
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            handle.flush()
            if val_loss < best:
                best, stale = val_loss, 0
                atomic_torch_save(
                    {
                        "model": model.state_dict(),
                        "model_name": model_name,
                        "fold": fold,
                        "config": cfg,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "experiment_fingerprint": fingerprint,
                    },
                    output / "best.pt",
                )
            else:
                stale += 1
            scheduler.step()
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "model_name": model_name,
                    "fold": fold,
                    "config": cfg,
                    "epoch": epoch,
                    "best_val_loss": best,
                    "stale_epochs": stale,
                    "experiment_fingerprint": fingerprint,
                },
                last_path,
            )
            if stale >= patience:
                print(f"Early stopping after {epoch} epochs")
                break
    if not (output / "best.pt").exists():
        raise RuntimeError(f"{model_name}: training ended without a best checkpoint")
    mark_training_complete(output, model_name, final_epoch, best, fingerprint)


def sam_inputs(
    processor: Any,
    image: Image.Image,
    seed_x: float,
    seed_y: float,
    target_mask: np.ndarray | None = None,
) -> Any:
    return processor(
        images=image,
        segmentation_maps=pil_mask(target_mask) if target_mask is not None else None,
        input_points=[[[[float(seed_x), float(seed_y)]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    )


def model_input_tensors(inputs: Any, target_device: torch.device) -> dict[str, torch.Tensor]:
    allowed = {
        "pixel_values",
        "input_points",
        "input_labels",
        "input_boxes",
        "pixel_attention_mask",
    }
    return {
        key: value.to(target_device)
        for key, value in inputs.items()
        if key in allowed and isinstance(value, torch.Tensor)
    }


def sam_logits(model: torch.nn.Module, inputs: Any, target_device: torch.device) -> torch.Tensor:
    outputs = model(**model_input_tensors(inputs, target_device), multimask_output=False)
    logits = outputs.pred_masks
    while logits.ndim > 4:
        logits = logits[:, 0]
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    return logits


def sam_target(inputs: Any, logits: torch.Tensor, target_device: torch.device) -> torch.Tensor:
    if "labels" not in inputs:
        raise RuntimeError("SAM2 processor did not return labels for segmentation_maps")
    target = inputs["labels"].float()
    while target.ndim < 4:
        target = target.unsqueeze(1)
    if target.max() > 1:
        target = target / 255.0
    target = F.interpolate(target, logits.shape[-2:], mode="nearest")
    return (target > 0.5).float().to(target_device)


@torch.no_grad()
def validate_sam(
    model: torch.nn.Module,
    processor: Any,
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    target_device: torch.device,
) -> float:
    model.eval()
    losses = []
    for row in frame.itertuples(index=False):
        image = load_upright_image(row.image_path)
        mask = load_mask(row.mask_path)
        inputs = sam_inputs(processor, image, row.seed_x, row.seed_y, mask)
        with torch.autocast(
            device_type=target_device.type,
            dtype=torch.float16,
            enabled=bool(cfg["training"]["amp"]) and target_device.type == "cuda",
        ):
            logits = sam_logits(model, inputs, target_device)
            target = sam_target(inputs, logits, target_device)
            loss = combined_loss(logits, target, cfg)
        losses.append(float(loss))
    return float(np.mean(losses))


def train_sam(cfg: dict[str, Any], fold: int) -> None:
    output = run_dir(cfg, "sam21", "finetuned", fold)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = experiment_fingerprint(cfg)
    if cfg.get("data", {}).get("split_mode") == "fixed" or resume_enabled(cfg):
        ensure_run_fingerprint(output, fingerprint)
    if resume_enabled(cfg) and training_is_complete(output, fingerprint):
        print("sam21: fixed-split training already complete; skipping")
        return
    target_device = device()
    model, processor = create_sam21(cfg)
    model.to(target_device)
    model_cfg = cfg["models"]["sam21"]
    trainable = set_train_scope(model, model_cfg.get("train_scope", "mask_decoder"))
    print(f"SAM 2.1 trainable parameters: {trainable:,}")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(cfg["training"]["amp"]) and target_device.type == "cuda"
    )
    accumulation = int(model_cfg.get("gradient_accumulation", 4))
    epochs = int(model_cfg.get("epochs", 30))
    frame = read_manifest(cfg)
    n_folds = int(cfg["data"]["folds"])
    train_frame = split_frame(frame, fold, "train", n_folds)
    val_frame = split_frame(frame, fold, "val", n_folds)
    rng = np.random.default_rng(int(cfg["seed"]) + fold)
    best, stale = float("inf"), 0
    start_epoch = 1
    last_path = output / "last.pt"
    if resume_enabled(cfg) and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        assert_checkpoint_fingerprint(checkpoint, fingerprint, last_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_val_loss"])
        stale = int(checkpoint["stale_epochs"])
        if checkpoint.get("numpy_rng_state"):
            rng.bit_generator.state = checkpoint["numpy_rng_state"]
        print(f"sam21: resuming at epoch {start_epoch} from {last_path}")
        if stale >= int(cfg["training"]["early_stopping_patience"]) or start_epoch > epochs:
            if not (output / "best.pt").exists():
                raise RuntimeError("sam21: resume state has no best checkpoint")
            mark_training_complete(
                output, "sam21", start_epoch - 1, best, fingerprint
            )
            print("sam21: recovered completed training state")
            return

    history_path = output / "history.csv"
    append_history = start_epoch > 1 and history_path.exists()
    final_epoch = start_epoch - 1
    with history_path.open("a" if append_history else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        if not append_history:
            writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            final_epoch = epoch
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            indexes = rng.permutation(len(train_frame))
            progress = tqdm(indexes, desc=f"sam21 fold={fold} epoch={epoch}")
            for step, index in enumerate(progress, 1):
                row = train_frame.iloc[int(index)]
                image = load_upright_image(row.image_path)
                mask = load_mask(row.mask_path)
                ys, xs = np.nonzero(mask)
                selected = int(rng.integers(0, len(xs)))
                # Positive GT points are used only during training. Evaluation
                # uses the stored SAM3 seed for an identical prompt protocol.
                inputs = sam_inputs(
                    processor, image, xs[selected], ys[selected], target_mask=mask
                )
                with torch.autocast(
                    device_type=target_device.type,
                    dtype=torch.float16,
                    enabled=scaler.is_enabled(),
                ):
                    logits = sam_logits(model, inputs, target_device)
                    target = sam_target(inputs, logits, target_device)
                    loss = combined_loss(logits, target, cfg) / accumulation
                scaler.scale(loss).backward()
                if step % accumulation == 0 or step == len(indexes):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss) * accumulation)
                progress.set_postfix(loss=f"{np.mean(losses[-10:]):.4f}")
            val_loss = validate_sam(model, processor, val_frame, cfg, target_device)
            writer.writerow(
                {"epoch": epoch, "train_loss": np.mean(losses), "val_loss": val_loss}
            )
            handle.flush()
            if val_loss < best:
                best, stale = val_loss, 0
                atomic_torch_save(
                    {
                        "model": model.state_dict(),
                        "model_name": "sam21",
                        "fold": fold,
                        "config": cfg,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "experiment_fingerprint": fingerprint,
                    },
                    output / "best.pt",
                )
            else:
                stale += 1
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "model_name": "sam21",
                    "fold": fold,
                    "config": cfg,
                    "epoch": epoch,
                    "best_val_loss": best,
                    "stale_epochs": stale,
                    "numpy_rng_state": rng.bit_generator.state,
                    "experiment_fingerprint": fingerprint,
                },
                last_path,
            )
            if stale >= int(cfg["training"]["early_stopping_patience"]):
                print(f"Early stopping after {epoch} epochs")
                break
    if not (output / "best.pt").exists():
        raise RuntimeError("sam21: training ended without a best checkpoint")
    mark_training_complete(output, "sam21", final_epoch, best, fingerprint)


def train(cfg: dict[str, Any], model_name: str, protocol: str, fold: int) -> None:
    if protocol == "pretrained":
        print(f"{model_name}: pretrained protocol requires no training")
        return
    seed_everything(int(cfg["seed"]) + fold)
    if model_name == "yolo26":
        from .yolo26 import train_yolo26

        train_yolo26(cfg, fold)
    elif model_name == "sam21":
        train_sam(cfg, fold)
    elif model_name in {"birefnet_hr", "segformer_b2", "deeplabv3plus_r50"}:
        train_dense(cfg, model_name, fold)
    else:
        raise ValueError(model_name)


def threshold_counts(
    probability: np.ndarray,
    target: np.ndarray,
    thresholds: Iterable[float],
    counts: dict[float, list[int]],
) -> None:
    for threshold in thresholds:
        pred = probability >= threshold
        counts[threshold][0] += int(np.count_nonzero(pred & target))
        counts[threshold][1] += int(np.count_nonzero(pred & ~target))
        counts[threshold][2] += int(np.count_nonzero(~pred & target))


def choose_threshold(counts: dict[float, list[int]]) -> tuple[float, dict[str, float]]:
    scores = {}
    for threshold, (tp, fp, fn) in counts.items():
        scores[str(threshold)] = (2 * tp + 1.0) / (2 * tp + fp + fn + 1.0)
    best = float(max(scores, key=scores.get))
    return best, scores


def load_trained_weights(
    model: torch.nn.Module, cfg: dict[str, Any], model_name: str, protocol: str, fold: int
) -> None:
    if protocol == "pretrained":
        return
    checkpoint_path = run_dir(cfg, model_name, protocol, fold) / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} is missing. Train this experiment before prediction."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])


def resize_target(mask_path: str, height: int, width: int) -> np.ndarray:
    resized = TF.resize(
        pil_mask(load_mask(mask_path)),
        [height, width],
        interpolation=InterpolationMode.NEAREST,
    )
    return np.asarray(resized) > 127


@torch.no_grad()
def dense_probabilities(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    target_device: torch.device,
):
    dataset = PlantDataset(
        frame,
        int(cfg["data"]["height"]),
        int(cfg["data"]["width"]),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    for batch in tqdm(loader, desc="predict"):
        image = batch["image"].to(target_device, non_blocking=True)
        with torch.autocast(
            device_type=target_device.type,
            dtype=torch.float16,
            enabled=bool(cfg["training"]["amp"]) and target_device.type == "cuda",
        ):
            logits = model(image)
        probability = logits.sigmoid()[0, 0].float().cpu().numpy()
        yield batch["stem"][0], probability


def sam_postprocess_logits(processor: Any, logits: torch.Tensor, inputs: Any) -> torch.Tensor:
    original = inputs.get("original_sizes")
    if hasattr(processor, "post_process_masks"):
        result = processor.post_process_masks(
            logits.cpu(), original.cpu(), binarize=False
        )
        value = result[0] if isinstance(result, (list, tuple)) else result
        while value.ndim > 2:
            value = value[0]
        return value
    height, width = map(int, original[0].tolist())
    return F.interpolate(
        logits.float(), size=(height, width), mode="bilinear", align_corners=False
    )[0, 0].cpu()


@torch.no_grad()
def sam_probabilities(
    model: torch.nn.Module,
    processor: Any,
    frame: pd.DataFrame,
    target_device: torch.device,
):
    model.eval()
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="predict SAM2.1"):
        image = load_upright_image(row.image_path)
        inputs = sam_inputs(processor, image, row.seed_x, row.seed_y)
        outputs = model(
            **model_input_tensors(inputs, target_device), multimask_output=False
        )
        full_logits = sam_postprocess_logits(processor, outputs.pred_masks, inputs)
        yield row.stem, full_logits.sigmoid().numpy()


def save_binary(
    probability: np.ndarray,
    threshold: float,
    destination: Path,
    original_height: int,
    original_width: int,
) -> None:
    binary = Image.fromarray((probability >= threshold).astype(np.uint8) * 255, mode="L")
    if binary.size != (original_width, original_height):
        binary = binary.resize((original_width, original_height), Image.Resampling.NEAREST)
    destination.parent.mkdir(parents=True, exist_ok=True)
    binary.save(destination, optimize=True)


def predict(cfg: dict[str, Any], model_name: str, protocol: str, fold: int) -> None:
    if model_name == "yolo26":
        if protocol != "finetuned":
            raise ValueError("YOLO26 is configured only as a fine-tuned model")
        from .yolo26 import predict_yolo26

        predict_yolo26(cfg, fold)
        return
    seed_everything(int(cfg["seed"]) + fold)
    output = run_dir(cfg, model_name, protocol, fold)
    output.mkdir(parents=True, exist_ok=True)
    target_device = device()
    frame = read_manifest(cfg)
    n_folds = int(cfg["data"]["folds"])
    val_frame = split_frame(frame, fold, "val", n_folds)
    test_frame = split_frame(frame, fold, "test", n_folds)
    thresholds = [float(x) for x in cfg["evaluation"]["threshold_candidates"]]
    counts = {threshold: [0, 0, 0] for threshold in thresholds}

    if model_name == "sam21":
        model, processor = create_sam21(cfg)
        load_trained_weights(model, cfg, model_name, protocol, fold)
        model.to(target_device)

        def probability_stream(selected: pd.DataFrame):
            return sam_probabilities(model, processor, selected, target_device)

    else:
        model = create_dense_model(model_name, cfg)
        load_trained_weights(model, cfg, model_name, protocol, fold)
        model.to(target_device)

        def probability_stream(selected: pd.DataFrame):
            return dense_probabilities(model, selected, cfg, target_device)

    val_lookup = val_frame.set_index("stem")
    for stem, probability in probability_stream(val_frame):
        row = val_lookup.loc[stem]
        if probability.shape == (int(row.height), int(row.width)):
            target = load_mask(row.mask_path)
        else:
            target = resize_target(row.mask_path, *probability.shape)
        threshold_counts(probability, target, thresholds, counts)
    selected_threshold, threshold_scores = choose_threshold(counts)
    with (output / "threshold.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"selected": selected_threshold, "validation_dice": threshold_scores},
            handle,
            indent=2,
        )

    for split, selected in (("val", val_frame), ("test", test_frame)):
        lookup = selected.set_index("stem")
        for stem, probability in probability_stream(selected):
            row = lookup.loc[stem]
            save_binary(
                probability,
                selected_threshold,
                output / "predictions" / split / f"{stem}.mask.png",
                int(row.height),
                int(row.width),
            )
    print(f"Selected validation threshold: {selected_threshold:.2f}")
