"""Center BiRefNet inference: stem seed + fourth-channel centre prior + fine-tuned checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .center_seed import find_stem_seed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_CONVOLUTIONS = (
    "model.bb.patch_embed.proj",
    "model.decoder.ipt_blk5.conv1",
    "model.decoder.ipt_blk4.conv1",
    "model.decoder.ipt_blk3.conv1",
    "model.decoder.ipt_blk2.conv1",
    "model.decoder.ipt_blk1.conv1",
)


def extract_logits(output: Any) -> torch.Tensor:
    if hasattr(output, "logits"):
        return extract_logits(output.logits)
    if isinstance(output, dict):
        for key in ("logits", "preds", "out", "prediction"):
            if key in output:
                return extract_logits(output[key])
    tensors: list[torch.Tensor] = []

    def collect(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)

    collect(output)
    candidates = [value for value in tensors if value.ndim >= 3]
    if not candidates:
        raise TypeError(f"Cannot extract logits from {type(output)!r}")
    value = max(
        enumerate(candidates),
        key=lambda item: (item[1].shape[-2] * item[1].shape[-1], item[0]),
    )[1]
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4:
        raise ValueError(f"Expected BCHW logits, got {tuple(value.shape)}")
    if value.shape[1] > 1:
        value = value[:, :1]
    return value


class ResizeOutput(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = extract_logits(self.model(image))
        if logits.shape[-2:] != image.shape[-2:]:
            logits = F.interpolate(
                logits, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits


def center_heatmap(
    seed_x: float,
    seed_y: float,
    source_width: int,
    source_height: int,
    width: int,
    height: int,
) -> torch.Tensor:
    scaled_x = seed_x * width / source_width
    scaled_y = seed_y * height / source_height
    x = torch.arange(width, dtype=torch.float32)
    y = torch.arange(height, dtype=torch.float32)
    dx = (x - scaled_x) / max(width, 1)
    dy = (y - scaled_y) / max(height, 1)
    corridor = torch.exp(-0.5 * (dx / 0.22) ** 2).unsqueeze(0).expand(height, -1)
    point = torch.exp(
        -0.5 * ((dx.unsqueeze(0) / 0.07) ** 2 + (dy.unsqueeze(1) / 0.10) ** 2)
    )
    return torch.maximum(corridor * 0.75, point).unsqueeze(0)


def _parent_and_attribute(model: torch.nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def expand_center_channel(model: torch.nn.Module) -> torch.nn.Module:
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
                f"{name}: expected a 3-channel convolution, "
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


def load_center_model(
    pretrained_id: str,
    base_weights: Path,
    center_weights: Path,
    fold: int,
) -> torch.nn.Module:
    from transformers import AutoModelForImageSegmentation

    model = ResizeOutput(
        AutoModelForImageSegmentation.from_pretrained(pretrained_id, trust_remote_code=True)
    )
    base_payload = torch.load(base_weights, map_location="cpu", weights_only=False)
    if base_payload.get("model_name") != "birefnet_hr":
        raise RuntimeError(f"Unexpected base checkpoint: {base_weights}")
    if int(base_payload.get("fold", -1)) != fold:
        raise RuntimeError(f"Base fold mismatch: {base_weights}")
    model.load_state_dict(base_payload["model"])
    expand_center_channel(model)

    center_payload = torch.load(center_weights, map_location="cpu", weights_only=False)
    if center_payload.get("model_name") != "birefnet_hr_centered":
        raise RuntimeError(f"Unexpected center checkpoint: {center_weights}")
    if int(center_payload.get("fold", -1)) != fold:
        raise RuntimeError(f"Center fold mismatch: {center_weights}")
    state = center_payload.get("trainable_state")
    names = center_payload.get("trainable_names")
    if not isinstance(state, dict) or not state or set(state) != set(names or []):
        raise RuntimeError(f"Invalid Center BiRefNet delta: {center_weights}")
    _, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected Center BiRefNet tensors: {unexpected[:5]}")
    return model


def read_threshold(threshold_file: Path, operating_point: str) -> float:
    payload = json.loads(threshold_file.read_text())
    if operating_point not in payload:
        raise KeyError(f"{threshold_file} has no {operating_point!r}")
    value = payload[operating_point]
    if isinstance(value, dict):
        return float(value["threshold"])
    return float(value)


def load_upright_rgb(path: Path) -> tuple[Image.Image, np.ndarray]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    return image, np.asarray(image)


def prepare_tensor(
    image: Image.Image,
    seed_x: float,
    seed_y: float,
    height: int,
    width: int,
) -> torch.Tensor:
    original_width, original_height = image.size
    resized = TF.resize(
        image,
        [height, width],
        interpolation=InterpolationMode.BILINEAR,
    )
    rgb = TF.normalize(TF.to_tensor(resized), IMAGENET_MEAN, IMAGENET_STD)
    prompt = center_heatmap(
        seed_x,
        seed_y,
        original_width,
        original_height,
        width,
        height,
    )
    return torch.cat([rgb, prompt], dim=0).unsqueeze(0)


def stem_seed_xy(rgb: np.ndarray) -> tuple[int, int, bool]:
    return find_stem_seed(rgb)


def predict_mask(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    original_size: tuple[int, int],
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        inputs = tensor.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            probability = model(inputs).sigmoid()[0, 0].float().cpu().numpy()
    low = Image.fromarray((probability >= threshold).astype(np.uint8) * 255, mode="L")
    full = np.asarray(low.resize(original_size, Image.Resampling.NEAREST)) > 127
    return full


def load_mask_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def save_mask(mask: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(destination)


def mask_overlay_path(output: Path, stem: str) -> Path:
    return output / "mask_overlays" / f"{stem}.jpg"


def write_mask_overlay(rgb: np.ndarray, mask: np.ndarray, destination: Path) -> None:
    """Photo with a green tinted plant mask. No detection boxes."""
    height, width = rgb.shape[:2]
    scale = min(1.0, 1600.0 / max(height, width))
    sw, sh = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    base = Image.fromarray(rgb).resize((sw, sh), Image.Resampling.LANCZOS)
    arr = np.asarray(base).astype(np.float32)
    small = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
    tint = np.array([0.0, 220.0, 80.0], dtype=np.float32)
    foreground = small.astype(bool)
    arr[foreground] = arr[foreground] * 0.55 + tint * 0.45
    out = np.clip(arr, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 220, 80), 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(destination, quality=88)


def save_meta(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))


def mask_paths(output: Path, stem: str) -> tuple[Path, Path]:
    masks = output / "masks"
    return masks / f"{stem}.mask.png", masks / f"{stem}.meta.json"
