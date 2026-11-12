from __future__ import annotations

import csv
import gc
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import ImageEnhance
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from .data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_mask,
    load_upright_image,
    pil_mask,
    read_manifest,
    split_frame,
    worker_seed,
)
from .engine import (
    assert_finite_state_dict,
    assert_checkpoint_fingerprint,
    atomic_torch_save,
    choose_threshold,
    completion_path,
    dice_loss,
    ensure_run_fingerprint,
    experiment_fingerprint,
    file_sha256,
    mark_training_complete,
    resume_enabled,
    save_binary,
    seed_everything,
    threshold_counts,
    training_is_complete,
)
from .config import run_dir
from .center_seed import resolve_center_seed
from .models import (
    create_dense_model,
    enable_birefnet_swin_checkpointing,
    freeze_batch_norm,
)
from .provenance import CENTER_SEED_PROTOCOL


MODEL_NAME = "birefnet_hr_centered"
BASE_MODEL_NAME = "birefnet_hr"
PROTOCOL = "finetuned"
CENTER_MEMORY_STRATEGY = "whole_swin_backbone_nonreentrant_per_invocation"
CENTER_FP16_INIT_SCALE = 1024.0
INPUT_CONVOLUTIONS = (
    "model.bb.patch_embed.proj",
    "model.decoder.ipt_blk5.conv1",
    "model.decoder.ipt_blk4.conv1",
    "model.decoder.ipt_blk3.conv1",
    "model.decoder.ipt_blk2.conv1",
    "model.decoder.ipt_blk1.conv1",
)
def center_run_dir(cfg: dict[str, Any], fold: int) -> Path:
    if cfg.get("data", {}).get("split_mode") == "fixed":
        return Path(cfg["paths"]["runs"]) / MODEL_NAME / PROTOCOL / "fixed_split"
    return Path(cfg["paths"]["runs"]) / MODEL_NAME / PROTOCOL / f"fold_{fold}"


def center_base_checkpoint_identity(
    cfg: dict[str, Any], fold: int
) -> dict[str, Any]:
    """Resolve and hash the exact BiRefNet base used by the center delta."""

    checkpoint = run_dir(cfg, BASE_MODEL_NAME, PROTOCOL, fold) / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = file_sha256(checkpoint)
    return {
        "path": str(checkpoint.resolve()),
        "sha256": checkpoint_hash,
    }


def _load_center_base_weights(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    fold: int,
) -> dict[str, Any]:
    identity = center_base_checkpoint_identity(cfg, fold)
    checkpoint_path = Path(identity["path"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("model_name") != BASE_MODEL_NAME:
        raise RuntimeError(
            f"Wrong Center BiRefNet base model in {checkpoint_path}: "
            f"{payload.get('model_name')!r}"
        )
    if int(payload.get("fold", -1)) != fold:
        raise RuntimeError(f"Center BiRefNet base fold mismatch: {checkpoint_path}")
    state = payload.get("model")
    if not isinstance(state, dict):
        raise RuntimeError(f"Center BiRefNet base has no model state: {checkpoint_path}")
    assert_finite_state_dict(state, checkpoint_path)
    model.load_state_dict(state)
    after_load_hash = file_sha256(checkpoint_path)
    if after_load_hash != identity["sha256"]:
        raise RuntimeError(
            f"Center BiRefNet base changed while loading: {checkpoint_path}"
        )
    # Persist the resolved identity on the wrapper without entering state_dict.
    model._plant_center_base_identity = identity
    return identity


def _model_base_identity(model: torch.nn.Module) -> dict[str, Any]:
    identity = getattr(model, "_plant_center_base_identity", None)
    if not isinstance(identity, dict) or not identity.get("sha256"):
        raise RuntimeError("Center BiRefNet model has no verified base identity")
    return identity


def _assert_center_checkpoint_binding(
    payload: dict[str, Any],
    base_identity: dict[str, Any],
    checkpoint_path: Path,
    fold: int,
) -> None:
    if payload.get("model_name") != MODEL_NAME:
        raise RuntimeError(f"Wrong Center BiRefNet checkpoint model: {checkpoint_path}")
    if int(payload.get("fold", -1)) != fold:
        raise RuntimeError(f"Center BiRefNet checkpoint fold mismatch: {checkpoint_path}")
    if payload.get("base_checkpoint_sha256") != base_identity["sha256"]:
        raise RuntimeError(
            "Center BiRefNet delta/base mismatch: "
            f"delta={checkpoint_path}, expected_base="
            f"{payload.get('base_checkpoint_sha256')}, actual_base="
            f"{base_identity['sha256']}"
        )
    trainable_state = payload.get("trainable_state")
    if not isinstance(trainable_state, dict) or not trainable_state:
        raise RuntimeError(
            f"Center BiRefNet checkpoint has no trainable state: {checkpoint_path}"
        )
    assert_finite_state_dict(trainable_state, checkpoint_path)
    trainable_names = payload.get("trainable_names")
    if not isinstance(trainable_names, list) or set(trainable_names) != set(
        trainable_state
    ):
        raise RuntimeError(
            f"Center BiRefNet checkpoint trainable-name mismatch: {checkpoint_path}"
        )
    expected_training_fingerprint = payload.get("experiment_fingerprint")
    if not expected_training_fingerprint:
        raise RuntimeError(
            f"Center BiRefNet checkpoint has no training fingerprint: {checkpoint_path}"
        )
    if payload.get("center_seed_protocol") != CENTER_SEED_PROTOCOL:
        raise RuntimeError(
            "Center BiRefNet checkpoint was not trained with the locked "
            f"reference-independent seed protocol: {checkpoint_path}"
        )


def _validate_center_best(
    checkpoint_path: Path,
    fingerprint: str,
    base_identity: dict[str, Any],
    fold: int,
) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert_checkpoint_fingerprint(payload, fingerprint, checkpoint_path)
    _assert_center_checkpoint_binding(payload, base_identity, checkpoint_path, fold)


def _assert_completed_center_run(
    output: Path,
    checkpoint_path: Path,
    fingerprint: str,
    base_identity: dict[str, Any],
    fold: int,
) -> None:
    marker_path = completion_path(output)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable Center BiRefNet completion: {marker_path}") from exc
    if marker.get("model") != MODEL_NAME:
        raise RuntimeError(f"Wrong model in Center completion marker: {marker_path}")
    if marker.get("experiment_fingerprint") != fingerprint:
        raise RuntimeError(f"Center completion fingerprint mismatch: {marker_path}")
    if marker.get("base_checkpoint_sha256") != base_identity["sha256"]:
        raise RuntimeError(f"Center completion/base mismatch: {marker_path}")
    if marker.get("center_seed_protocol") != CENTER_SEED_PROTOCOL:
        raise RuntimeError(f"Center completion seed protocol mismatch: {marker_path}")
    actual_best_hash = file_sha256(checkpoint_path)
    if marker.get("best_checkpoint_sha256") != actual_best_hash:
        raise RuntimeError(f"Center completion/best checkpoint mismatch: {marker_path}")
    _validate_center_best(checkpoint_path, fingerprint, base_identity, fold)


def _parent_and_attribute(
    root: torch.nn.Module, dotted_name: str
) -> tuple[torch.nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def expand_center_channel(model: torch.nn.Module) -> torch.nn.Module:
    """Expand both BiRefNet RGB input paths from three to four channels.

    The fourth-channel weights start at zero, making the expanded network
    exactly equivalent to the source checkpoint before center-aware training.
    """

    for name in INPUT_CONVOLUTIONS:
        parent, attribute = _parent_and_attribute(model, name)
        source = getattr(parent, attribute)
        if not isinstance(source, torch.nn.Conv2d):
            raise TypeError(f"{name} is {type(source)!r}, expected Conv2d")
        expected_input_channels = source.in_channels * 4 // 3
        if name == "model.bb.patch_embed.proj":
            expected_input_channels = 4
        if source.in_channels == expected_input_channels:
            continue
        if source.in_channels % 3 != 0 or source.groups != 1:
            raise ValueError(
                f"{name}: expected a convolution derived from three RGB channels, "
                f"got in_channels={source.in_channels}, groups={source.groups}"
            )
        expanded = torch.nn.Conv2d(
            in_channels=expected_input_channels,
            out_channels=source.out_channels,
            kernel_size=source.kernel_size,
            stride=source.stride,
            padding=source.padding,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.bias is not None,
            padding_mode=source.padding_mode,
        ).to(device=source.weight.device, dtype=source.weight.dtype)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, : source.in_channels].copy_(source.weight)
            if source.bias is not None:
                expanded.bias.copy_(source.bias)
        setattr(parent, attribute, expanded)
    return model


def create_center_model(
    cfg: dict[str, Any],
    fold: int,
    center_checkpoint: Path | None = None,
) -> torch.nn.Module:
    model = create_dense_model(BASE_MODEL_NAME, cfg)
    base_identity = _load_center_base_weights(model, cfg, fold)
    expand_center_channel(model)
    if center_checkpoint is not None:
        payload = torch.load(
            center_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        _assert_center_checkpoint_binding(
            payload, base_identity, center_checkpoint, fold
        )
        _, unexpected = model.load_state_dict(
            payload["trainable_state"],
            strict=False,
        )
        expected_trainable = set(payload["trainable_names"])
        stored_trainable = set(payload["trainable_state"])
        if expected_trainable != stored_trainable or unexpected:
            raise RuntimeError(
                "Center checkpoint mismatch: "
                f"missing_trainable={sorted(expected_trainable - stored_trainable)[:5]}, "
                f"unexpected={unexpected[:5]}"
            )
    return model


def set_center_train_scope(model: torch.nn.Module) -> tuple[int, list[str]]:
    """Train the decoder plus the backbone patch-input projection."""

    trainable_names: list[str] = []
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if (
            name.startswith("model.decoder.")
            or name.startswith("model.bb.patch_embed.proj.")
        ):
            parameter.requires_grad = True
            trainable_names.append(name)
    count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if count == 0:
        raise RuntimeError("Center-aware train scope selected no parameters")
    return count, trainable_names


def center_trainable_state(
    model: torch.nn.Module, trainable_names: list[str]
) -> dict[str, torch.Tensor]:
    wanted = set(trainable_names)
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in wanted
    }


def load_center_trainable_state(
    model: torch.nn.Module,
    payload: dict[str, Any],
    expected_trainable_names: list[str],
) -> None:
    expected = set(expected_trainable_names)
    stored_names = set(payload.get("trainable_names", []))
    stored_state = payload.get("trainable_state", {})
    if expected != stored_names or expected != set(stored_state):
        raise RuntimeError(
            "Center checkpoint trainable-state mismatch: "
            f"missing={sorted(expected - set(stored_state))[:5]}, "
            f"unexpected={sorted(set(stored_state) - expected)[:5]}"
        )
    _, unexpected = model.load_state_dict(stored_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected center checkpoint keys: {unexpected[:5]}")


def center_heatmap(
    seed_x: float,
    seed_y: float,
    source_width: int,
    source_height: int,
    width: int,
    height: int,
) -> torch.Tensor:
    """Create a vertical center corridor plus a localized stem-seed peak."""

    scaled_x = seed_x * width / source_width
    scaled_y = seed_y * height / source_height
    x = torch.arange(width, dtype=torch.float32)
    y = torch.arange(height, dtype=torch.float32)
    dx = (x - scaled_x) / max(width, 1)
    dy = (y - scaled_y) / max(height, 1)
    corridor = torch.exp(-0.5 * (dx / 0.22) ** 2).unsqueeze(0).expand(height, -1)
    point = torch.exp(
        -0.5
        * (
            (dx.unsqueeze(0) / 0.07) ** 2
            + (dy.unsqueeze(1) / 0.10) ** 2
        )
    )
    return torch.maximum(corridor * 0.75, point).unsqueeze(0)


class CenterPlantDataset(Dataset):
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
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        center_seed = resolve_center_seed(row)
        image = load_upright_image(row.image_path)
        mask = pil_mask(load_mask(row.mask_path))
        image = TF.resize(
            image,
            [self.height, self.width],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            [self.height, self.width],
            interpolation=InterpolationMode.NEAREST,
        )
        seed_x = center_seed.x
        seed_y = center_seed.y
        if self.augment and random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
            seed_x = float(row.width) - 1.0 - seed_x
        if self.augment and random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(0.88, 1.12)
            )
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(0.88, 1.12)
            )
            image = ImageEnhance.Color(image).enhance(
                random.uniform(0.88, 1.12)
            )
        rgb = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        prompt = center_heatmap(
            seed_x,
            seed_y,
            int(row.width),
            int(row.height),
            self.width,
            self.height,
        )
        target = (TF.pil_to_tensor(mask).float() / 255.0 > 0.5).float()
        return {
            "input": torch.cat([rgb, prompt], dim=0),
            "target": target,
            "stem": str(row.stem),
            "width": int(row.width),
            "height": int(row.height),
            "center_seed_source": center_seed.source,
            "center_seed_policy": center_seed.policy,
            "center_seed_reference_mask_read": center_seed.reference_mask_read,
        }


def center_loaders(
    cfg: dict[str, Any],
    fold: int,
) -> tuple[DataLoader, DataLoader]:
    frame = read_manifest(cfg)
    folds = int(cfg["data"]["folds"])
    height, width = int(cfg["data"]["height"]), int(cfg["data"]["width"])
    train = CenterPlantDataset(
        split_frame(frame, fold, "train", folds),
        height,
        width,
        augment=True,
    )
    val = CenterPlantDataset(
        split_frame(frame, fold, "val", folds),
        height,
        width,
        augment=False,
    )
    common = {
        "batch_size": 1,
        "num_workers": int(cfg["data"]["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_seed,
        "persistent_workers": int(cfg["data"]["num_workers"]) > 0,
    }
    generator = torch.Generator().manual_seed(int(cfg["seed"]) + 100 + fold)
    return (
        DataLoader(train, shuffle=True, generator=generator, **common),
        DataLoader(val, shuffle=False, **common),
    )


def center_aware_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    prompt: torch.Tensor,
    cfg: dict[str, Any],
    side_negative_weight: float,
) -> torch.Tensor:
    """Penalize off-center false positives without penalizing true branches.

    Only target-background pixels receive the extra weight. Consequently,
    ground-truth plant branches remain ordinary positive examples even when
    they extend far outside the center corridor.
    """

    background = (target < 0.5).float()
    outside_center = (1.0 - prompt).clamp(0.0, 1.0)
    weight = 1.0 + float(side_negative_weight) * background * outside_center
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    weighted_bce = (bce * weight).sum() / weight.sum().clamp_min(1.0)
    return (
        float(cfg["training"]["loss_bce_weight"]) * weighted_bce
        + float(cfg["training"]["loss_dice_weight"]) * dice_loss(logits, target)
    )


def _center_amp_dtype(target_device: torch.device) -> torch.dtype:
    if target_device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _release_center_batch(target_device: torch.device) -> None:
    """Return completed-batch activations before the validation forward."""

    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def validate_center(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict[str, Any],
    target_device: torch.device,
    side_negative_weight: float,
) -> tuple[float, float]:
    model.eval()
    amp_dtype = _center_amp_dtype(target_device)
    losses: list[float] = []
    dices: list[float] = []
    for batch in loader:
        inputs = batch["input"].to(target_device, non_blocking=True)
        target = batch["target"].to(target_device, non_blocking=True)
        with torch.autocast(
            device_type=target_device.type,
            dtype=amp_dtype,
            enabled=(
                bool(cfg["training"]["amp"])
                and target_device.type == "cuda"
            ),
        ):
            logits = model(inputs)
            loss = center_aware_loss(
                logits,
                target,
                inputs[:, 3:4],
                cfg,
                side_negative_weight,
            )
        if not torch.isfinite(loss).all():
            raise RuntimeError("Center validation produced non-finite loss")
        losses.append(float(loss))
        prediction = logits.sigmoid() >= 0.5
        intersection = (prediction & target.bool()).sum().item()
        dices.append(
            (2 * intersection + 1.0)
            / (prediction.sum().item() + target.sum().item() + 1.0)
        )
    return float(np.mean(losses)), float(np.mean(dices))


def train_center(
    cfg: dict[str, Any],
    fold: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    accumulation: int,
    max_gradient_norm: float,
    side_negative_weight: float,
    force: bool = False,
) -> None:
    output = center_run_dir(cfg, fold)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pt"
    fingerprint = experiment_fingerprint(cfg)
    if cfg.get("data", {}).get("split_mode") == "fixed" or resume_enabled(cfg):
        ensure_run_fingerprint(output, fingerprint)
    if force:
        completion_path(output).unlink(missing_ok=True)
        (output / "last.pt").unlink(missing_ok=True)
    base_identity = center_base_checkpoint_identity(cfg, fold)
    if (
        resume_enabled(cfg)
        and not force
        and training_is_complete(output, fingerprint)
    ):
        _assert_completed_center_run(
            output, checkpoint_path, fingerprint, base_identity, fold
        )
        print(f"{output}: center-aware training already complete; skipping")
        return
    # Without training.resume, an existing best checkpoint is never overwritten.
    if checkpoint_path.is_file() and not force and not resume_enabled(cfg):
        print(f"{checkpoint_path} exists; skipping training")
        return
    seed_everything(int(cfg["seed"]) + 100 + fold)
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_center_model(cfg, fold).float().to(target_device)
    loaded_base_identity = _model_base_identity(model)
    if loaded_base_identity["sha256"] != base_identity["sha256"]:
        raise RuntimeError("Center BiRefNet base changed while the model was loading")
    trainable_count, trainable_names = set_center_train_scope(model)
    checkpoint_stages = enable_birefnet_swin_checkpointing(model)
    amp_dtype = _center_amp_dtype(target_device)
    print(
        f"fold={fold} center-aware trainable parameters: "
        f"{trainable_count:,}; memory={CENTER_MEMORY_STRATEGY}; "
        f"checkpoint_stages={checkpoint_stages}; amp_dtype={amp_dtype}"
    )
    train_loader, val_loader = center_loaders(cfg, fold)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs)
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            bool(cfg["training"]["amp"])
            and target_device.type == "cuda"
            and amp_dtype == torch.float16
        ),
        init_scale=CENTER_FP16_INIT_SCALE,
        growth_interval=2000,
    )
    best, stale = float("inf"), 0
    start_epoch = 1
    last_path = output / "last.pt"
    if resume_enabled(cfg) and last_path.is_file() and not force:
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        assert_checkpoint_fingerprint(payload, fingerprint, last_path)
        _assert_center_checkpoint_binding(
            payload, base_identity, last_path, fold
        )
        load_center_trainable_state(model, payload, trainable_names)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload["epoch"]) + 1
        best = float(payload["best_val_loss"])
        stale = int(payload["stale_epochs"])
        print(f"center-birefnet: resuming at epoch {start_epoch} from {last_path}")
        if stale >= int(patience) or start_epoch > int(epochs):
            if not checkpoint_path.is_file():
                raise RuntimeError(
                    "Center-aware resume state has no best checkpoint"
                )
            _validate_center_best(
                checkpoint_path, fingerprint, base_identity, fold
            )
            mark_training_complete(
                output,
                MODEL_NAME,
                start_epoch - 1,
                best,
                fingerprint,
                extra={
                    "base_checkpoint_path": base_identity["path"],
                    "base_checkpoint_sha256": base_identity["sha256"],
                    "best_checkpoint_sha256": file_sha256(checkpoint_path),
                    "center_seed_protocol": CENTER_SEED_PROTOCOL,
                },
            )
            print("center-birefnet: recovered completed training state")
            return

    history_path = output / "history.csv"
    append_history = start_epoch > 1 and history_path.exists()
    final_epoch = start_epoch - 1
    with history_path.open(
        "a" if append_history else "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_dice",
                "lr",
                "skipped_batches",
            ],
        )
        if not append_history:
            writer.writeheader()
        for epoch in range(start_epoch, int(epochs) + 1):
            final_epoch = epoch
            model.train()
            freeze_batch_norm(model)
            optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            accumulated_steps = 0
            skipped_batches = 0
            progress = tqdm(
                train_loader,
                desc=f"center-birefnet fold={fold} epoch={epoch}",
                unit="image",
            )
            for step, batch in enumerate(progress, 1):
                inputs = batch["input"].to(target_device, non_blocking=True)
                target = batch["target"].to(target_device, non_blocking=True)
                with torch.autocast(
                    device_type=target_device.type,
                    dtype=amp_dtype,
                    enabled=(
                        bool(cfg["training"]["amp"])
                        and target_device.type == "cuda"
                    ),
                ):
                    logits = model(inputs)
                    loss = (
                        center_aware_loss(
                            logits,
                            target,
                            inputs[:, 3:4],
                            cfg,
                            side_negative_weight,
                        )
                        / accumulation
                    )
                if not torch.isfinite(loss).all():
                    if not scaler.is_enabled():
                        raise RuntimeError(
                            f"fold={fold} epoch={epoch} step={step}: "
                            "non-finite BF16/FP32 loss"
                        )
                    skipped_batches += 1
                    accumulated_steps = 0
                    optimizer.zero_grad(set_to_none=True)
                    print(
                        f"fold={fold} epoch={epoch} step={step}: "
                        "skipping non-finite loss"
                    )
                    continue
                scaler.scale(loss).backward()
                accumulated_steps += 1
                if accumulated_steps >= accumulation or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        max_norm=float(max_gradient_norm),
                        error_if_nonfinite=False,
                    )
                    if torch.isfinite(gradient_norm):
                        scaler.step(optimizer)
                    else:
                        if not scaler.is_enabled():
                            raise RuntimeError(
                                f"fold={fold} epoch={epoch} step={step}: "
                                "non-finite BF16/FP32 gradients"
                            )
                        skipped_batches += accumulated_steps
                        print(
                            f"fold={fold} epoch={epoch} step={step}: "
                            "skipping non-finite gradients"
                        )
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_steps = 0
                losses.append(float(loss) * accumulation)
                progress.set_postfix(loss=f"{np.mean(losses[-10:]):.4f}")
            if not losses:
                raise RuntimeError(
                    f"fold={fold} epoch={epoch} produced no finite training losses"
                )
            # The final loop variables otherwise retain the last full-resolution
            # autograd graph while validation allocates another BiRefNet forward.
            del batch, inputs, target, logits, loss
            optimizer.zero_grad(set_to_none=True)
            _release_center_batch(target_device)
            val_loss, val_dice = validate_center(
                model,
                val_loader,
                cfg,
                target_device,
                side_negative_weight,
            )
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "lr": optimizer.param_groups[0]["lr"],
                "skipped_batches": skipped_batches,
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps({"fold": fold, **row}))
            if val_loss < best:
                best, stale = val_loss, 0
                atomic_torch_save(
                    {
                        "trainable_state": center_trainable_state(
                            model, trainable_names
                        ),
                        "trainable_names": trainable_names,
                        "fold": fold,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_dice": val_dice,
                        "model_name": MODEL_NAME,
                        "center_prompt": (
                            "fourth input channel: 0.75 horizontal Gaussian "
                            "corridor (sigma=0.22W), max-combined with a stem-seed "
                            "Gaussian (sigma_x=0.07W, sigma_y=0.10H)"
                        ),
                        "side_negative_weight": float(side_negative_weight),
                        "max_gradient_norm": float(max_gradient_norm),
                        "memory_strategy": CENTER_MEMORY_STRATEGY,
                        "checkpoint_stages": checkpoint_stages,
                        "amp_dtype": str(amp_dtype),
                        "fp16_init_scale": CENTER_FP16_INIT_SCALE,
                        "experiment_fingerprint": fingerprint,
                        "base_checkpoint_path": base_identity["path"],
                        "base_checkpoint_sha256": base_identity["sha256"],
                        "center_seed_protocol": CENTER_SEED_PROTOCOL,
                    },
                    checkpoint_path,
                )
            else:
                stale += 1
            scheduler.step()
            atomic_torch_save(
                {
                    "trainable_state": center_trainable_state(model, trainable_names),
                    "trainable_names": trainable_names,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "fold": fold,
                    "epoch": epoch,
                    "best_val_loss": best,
                    "stale_epochs": stale,
                    "model_name": MODEL_NAME,
                    "side_negative_weight": float(side_negative_weight),
                    "max_gradient_norm": float(max_gradient_norm),
                    "memory_strategy": CENTER_MEMORY_STRATEGY,
                    "checkpoint_stages": checkpoint_stages,
                    "amp_dtype": str(amp_dtype),
                    "fp16_init_scale": CENTER_FP16_INIT_SCALE,
                    "experiment_fingerprint": fingerprint,
                    "base_checkpoint_path": base_identity["path"],
                    "base_checkpoint_sha256": base_identity["sha256"],
                    "center_seed_protocol": CENTER_SEED_PROTOCOL,
                },
                last_path,
            )
            if stale >= int(patience):
                print(f"fold={fold} early stopping after epoch {epoch}")
                break
    if not checkpoint_path.is_file():
        raise RuntimeError("Center-aware training ended without a best checkpoint")
    _validate_center_best(checkpoint_path, fingerprint, base_identity, fold)
    mark_training_complete(
        output,
        MODEL_NAME,
        final_epoch,
        best,
        fingerprint,
        extra={
            "base_checkpoint_path": base_identity["path"],
            "base_checkpoint_sha256": base_identity["sha256"],
            "best_checkpoint_sha256": file_sha256(checkpoint_path),
            "center_seed_protocol": CENTER_SEED_PROTOCOL,
            "memory_strategy": CENTER_MEMORY_STRATEGY,
            "amp_dtype": str(amp_dtype),
        },
    )


@torch.no_grad()
def probability_stream(
    model: torch.nn.Module,
    dataset: CenterPlantDataset,
    cfg: dict[str, Any],
    target_device: torch.device,
):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    for batch in tqdm(loader, desc="center-birefnet predict", unit="image"):
        inputs = batch["input"].to(target_device, non_blocking=True)
        with torch.autocast(
            device_type=target_device.type,
            dtype=torch.float16,
            enabled=target_device.type == "cuda",
        ):
            logits = model(inputs)
        yield (
            str(batch["stem"][0]),
            logits.sigmoid()[0, 0].float().cpu().numpy(),
            int(batch["height"][0]),
            int(batch["width"][0]),
        )


def predict_center(cfg: dict[str, Any], fold: int) -> None:
    output = center_run_dir(cfg, fold)
    checkpoint = output / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    seed_everything(int(cfg["seed"]) + 100 + fold)
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_center_model(cfg, fold, checkpoint).float().to(target_device)
    frame = read_manifest(cfg)
    folds = int(cfg["data"]["folds"])
    val_frame = split_frame(frame, fold, "val", folds)
    test_frame = split_frame(frame, fold, "test", folds)
    height, width = int(cfg["data"]["height"]), int(cfg["data"]["width"])
    val_dataset = CenterPlantDataset(val_frame, height, width, augment=False)
    candidates = [float(value) for value in cfg["evaluation"]["threshold_candidates"]]
    counts = {threshold: [0, 0, 0] for threshold in candidates}
    val_targets = {
        str(row.stem): np.asarray(
            TF.resize(
                pil_mask(load_mask(row.mask_path)),
                [height, width],
                interpolation=InterpolationMode.NEAREST,
            )
        )
        > 127
        for row in val_frame.itertuples(index=False)
    }
    for stem, probability, _, _ in probability_stream(
        model, val_dataset, cfg, target_device
    ):
        threshold_counts(
            probability,
            val_targets[stem],
            candidates,
            counts,
        )
    balanced_threshold, scores = choose_threshold(counts)
    beta = 0.8
    beta_squared = beta * beta

    def fbeta(threshold: float) -> float:
        tp, fp, fn = counts[threshold]
        return (1.0 + beta_squared) * tp / (
            (1.0 + beta_squared) * tp
            + beta_squared * fn
            + fp
            + 1e-9
        )

    center_strict_threshold = max(candidates, key=fbeta)
    (output / "threshold.json").write_text(
        json.dumps(
            {
                "selected": center_strict_threshold,
                "center_strict": {
                    "threshold": center_strict_threshold,
                    "objective": "validation F-beta",
                    "beta": beta,
                    "score": fbeta(center_strict_threshold),
                },
                "balanced": {
                    "threshold": balanced_threshold,
                    "objective": "validation Dice/F1",
                },
                "validation_dice": scores,
            },
            indent=2,
        )
    )
    for split_name, selected_frame in (
        ("val", val_frame),
        ("test", test_frame),
    ):
        dataset = CenterPlantDataset(
            selected_frame,
            height,
            width,
            augment=False,
        )
        for stem, probability, original_height, original_width in probability_stream(
            model, dataset, cfg, target_device
        ):
            save_binary(
                probability,
                balanced_threshold,
                output
                / "predictions"
                / "balanced"
                / split_name
                / f"{stem}.mask.png",
                original_height,
                original_width,
            )
            save_binary(
                probability,
                center_strict_threshold,
                output
                / "predictions"
                / "center_strict"
                / split_name
                / f"{stem}.mask.png",
                original_height,
                original_width,
            )
    print(
        f"fold={fold} thresholds: balanced={balanced_threshold:.2f}, "
        f"center_strict={center_strict_threshold:.2f}"
    )


def smoke_center(cfg: dict[str, Any], fold: int) -> None:
    seed_everything(int(cfg["seed"]) + 100 + fold)
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = create_center_model(cfg, fold).float().to(target_device)
    trainable, _ = set_center_train_scope(model)
    checkpoint_stages = enable_birefnet_swin_checkpointing(model)
    amp_dtype = _center_amp_dtype(target_device)
    train_loader, val_loader = center_loaders(cfg, fold)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-5,
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(target_device.type == "cuda" and amp_dtype == torch.float16),
        init_scale=CENTER_FP16_INIT_SCALE,
        growth_interval=2000,
    )
    model.train()
    freeze_batch_norm(model)
    optimizer.zero_grad(set_to_none=True)
    smoke_losses: list[float] = []
    for step, batch in enumerate(train_loader, 1):
        inputs = batch["input"].to(target_device)
        target = batch["target"].to(target_device)
        with torch.autocast(
            device_type=target_device.type,
            dtype=amp_dtype,
            enabled=target_device.type == "cuda",
        ):
            logits = model(inputs)
            loss = center_aware_loss(
                logits,
                target,
                inputs[:, 3:4],
                cfg,
                side_negative_weight=2.0,
            ) / 4
        if not torch.isfinite(loss).all():
            raise RuntimeError(f"Center smoke produced non-finite loss at step {step}")
        scaler.scale(loss).backward()
        smoke_losses.append(float(loss.detach()) * 4)
        if step == 4:
            break
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        max_norm=1.0,
        error_if_nonfinite=False,
    )
    if not torch.isfinite(gradient_norm):
        raise RuntimeError("Center smoke produced non-finite gradients")
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    del batch, inputs, target, logits, loss
    _release_center_batch(target_device)

    model.eval()
    val_batch = next(iter(val_loader))
    val_inputs = val_batch["input"].to(target_device)
    val_target = val_batch["target"].to(target_device)
    with torch.no_grad(), torch.autocast(
        device_type=target_device.type,
        dtype=amp_dtype,
        enabled=target_device.type == "cuda",
    ):
        val_logits = model(val_inputs)
        val_loss = center_aware_loss(
            val_logits,
            val_target,
            val_inputs[:, 3:4],
            cfg,
            side_negative_weight=2.0,
        )
    if not torch.isfinite(val_loss).all():
        raise RuntimeError("Center smoke produced non-finite validation loss")
    peak = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if target_device.type == "cuda"
        else 0.0
    )
    print(
        json.dumps(
            {
                "input_shape": list(val_inputs.shape),
                "output_shape": list(val_logits.shape),
                "train_loss": float(np.mean(smoke_losses)),
                "val_loss": float(val_loss),
                "gradient_norm": float(gradient_norm),
                "trainable_parameters": trainable,
                "peak_cuda_gib": peak,
                "memory_strategy": CENTER_MEMORY_STRATEGY,
                "checkpoint_stages": checkpoint_stages,
                "amp_dtype": str(amp_dtype),
                "fourth_channel_min": float(val_inputs[:, 3].min()),
                "fourth_channel_max": float(val_inputs[:, 3].max()),
            },
            indent=2,
        )
    )
