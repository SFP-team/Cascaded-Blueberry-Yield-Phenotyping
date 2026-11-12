from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


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
    # BiRefNet returns a nested multi-scale supervision tree in train mode.
    # Prefer the last prediction at the greatest spatial resolution.
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


def create_dense_model(name: str, cfg: dict[str, Any]) -> nn.Module:
    model_cfg = cfg["models"][name]
    if name == "segformer_b2":
        from transformers import SegformerForSemanticSegmentation

        model = SegformerForSemanticSegmentation.from_pretrained(
            model_cfg["pretrained_id"],
            num_labels=1,
            id2label={0: "plant"},
            label2id={"plant": 0},
            ignore_mismatched_sizes=True,
        )
        return ResizeOutput(model)
    if name == "deeplabv3plus_r50":
        import segmentation_models_pytorch as smp

        model = smp.DeepLabV3Plus(
            encoder_name=model_cfg["encoder_name"],
            encoder_weights=model_cfg.get("encoder_weights"),
            in_channels=3,
            classes=1,
            activation=None,
        )
        return ResizeOutput(model)
    if name == "birefnet_hr":
        from transformers import AutoModelForImageSegmentation

        model = AutoModelForImageSegmentation.from_pretrained(
            model_cfg["pretrained_id"], trust_remote_code=True
        )
        return ResizeOutput(model)
    raise ValueError(f"{name} is not a dense model")


def create_sam21(cfg: dict[str, Any]) -> tuple[nn.Module, Any]:
    from transformers import Sam2Model, Sam2Processor

    model_id = cfg["models"]["sam21"]["pretrained_id"]
    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id)
    return model, processor


def set_train_scope(model: nn.Module, scope: str) -> int:
    if scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
        wanted = {
            "mask_decoder": ("mask_decoder",),
            "decoder": ("decoder", "refine", "head", "out_conv"),
        }.get(scope)
        if not wanted:
            raise ValueError(f"Unsupported train scope: {scope}")
        for name, parameter in model.named_parameters():
            if any(token in name.lower() for token in wanted):
                parameter.requires_grad = True
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count == 0:
        raise RuntimeError(f"Train scope {scope!r} selected no parameters")
    return count


def freeze_batch_norm(model: nn.Module) -> None:
    """Use pretrained running statistics for batch-size-one fine-tuning."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def enable_birefnet_swin_checkpointing(model: nn.Module) -> int:
    """Checkpoint each BiRefNet Swin backbone call with non-reentrant replay.

    ``create_dense_model`` wraps the remote BiRefNet as ``ResizeOutput``, so
    the expected backbone is ``model.model.bb``. BiRefNet-HR invokes this same
    backbone at two input scales in one forward. Its native block checkpoint
    path stores H/W on mutable block attributes; the second-scale forward
    overwrites those values before the first-scale backward recomputation.

    Wrap the unmodified whole-backbone forward instead. Each invocation gets
    its own non-reentrant checkpoint whose tensor input determines H/W again
    during replay. Disable the unsafe nested native block checkpoint path.
    Failing closed prevents a run from silently proceeding without the memory
    safeguard if the remote-code architecture changes.
    """

    try:
        backbone = model.model.bb
        layers = backbone.layers
    except AttributeError as exc:
        raise RuntimeError(
            "BiRefNet-HR does not expose model.model.bb.layers; cannot enable "
            "the required safe Swin backbone checkpointing"
        ) from exc
    layers = list(layers)
    if not layers:
        raise RuntimeError("BiRefNet-HR Swin backbone has no stages")
    for index, layer in enumerate(layers):
        if not hasattr(layer, "use_checkpoint"):
            raise RuntimeError(
                f"BiRefNet-HR Swin stage {index} has no use_checkpoint flag"
            )
        layer.use_checkpoint = False
    if getattr(backbone, "_plant_backbone_checkpointing", False):
        return len(layers)

    original_forward = backbone.forward

    def checkpointed_backbone_forward(self, image):
        if self.training and torch.is_grad_enabled():
            return checkpoint(
                original_forward,
                image,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        return original_forward(image)

    backbone.forward = MethodType(checkpointed_backbone_forward, backbone)
    backbone._plant_backbone_checkpointing = True
    return len(layers)
