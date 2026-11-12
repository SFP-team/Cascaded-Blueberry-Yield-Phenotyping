#!/usr/bin/env python3
"""Time automatic SAM3 center-plant mask generation on a folder of images (one CUDA GPU).

Uses the zero-shot SAM3 center-plant segmenter in ../../segmentation/sam3_annotation with its
default settings (checked below) and a single SAM3 checkpoint from Meta. Warm-up passes are
excluded from the reported timings.

Outputs (under --output):
  masks/<stem>.mask.png
  diagnostics/<stem>.json
  per_image_timing.csv   decode, SAM3 segmentation, save and complete time (ms)
  summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import segmentation.sam3_annotation.segmenter as segmenter_module  # noqa: E402
from segmentation.sam3_annotation.config import Sam3NewConfig  # noqa: E402
from segmentation.sam3_annotation.io_utils import load_upright  # noqa: E402
from segmentation.sam3_annotation.segmenter import Sam3HybridSegmenter  # noqa: E402

# Settings of the center-plant protocol; the run stops if the segmenter defaults differ.
EXPECTED_SETTINGS = {
    "text_prompt": "blueberry bush",
    "fallback_prompts": ("blueberry plant", "plant"),
    "sam3_confidence_thresh": 0.35,
    "plant_box_width_frac": 0.75,
    "negative_box_frac": 0.22,
    "use_negative_points": True,
    "use_negative_boxes": True,
    "keep_seed_component": True,
    "seed_component_close_frac": 0.0,
    "fallback_to_stem_only": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="Folder of .jpg images.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="SAM3 checkpoint (sam3.pt).")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def install_numpy_compatibility_adapter(module: object) -> None:
    """Make CUDA outputs NumPy-compatible without changing inference logic."""

    def safe_to_numpy(value, dtype=None):
        if value is None:
            array = np.asarray([])
        elif hasattr(value, "detach"):
            tensor = value.detach().cpu()
            if tensor.is_floating_point():
                tensor = tensor.float()
            array = tensor.numpy()
        else:
            array = np.asarray(value)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    module._to_numpy = safe_to_numpy


def main() -> None:
    args = parse_args()
    image_dir = args.images.resolve()
    output = args.output.resolve()
    checkpoint = args.checkpoint.resolve()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")

    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise RuntimeError(f"No .jpg images in {image_dir}")
    mask_dir = output / "masks"
    diagnostic_dir = output / "diagnostics"
    mask_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    install_numpy_compatibility_adapter(segmenter_module)
    cfg = Sam3NewConfig(device="cuda", sam3_checkpoint=checkpoint)
    for key, value in EXPECTED_SETTINGS.items():
        if getattr(cfg, key) != value:
            raise RuntimeError(
                f"SAM3 setting changed: {key}={getattr(cfg, key)!r}, expected={value!r}"
            )

    device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    segmenter = Sam3HybridSegmenter(cfg)
    torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - load_started

    warm_rgb = load_upright(images[0], cfg.rotate)
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = segmenter.segment(warm_rgb, image_name=images[0].name)
    torch.cuda.synchronize(device)
    del warm_rgb

    rows: list[dict[str, object]] = []
    batch_started = time.perf_counter()
    with torch.inference_mode():
        for index, path in enumerate(images, start=1):
            total_started = time.perf_counter()
            rgb = load_upright(path, cfg.rotate)
            decode_ms = (time.perf_counter() - total_started) * 1000.0

            torch.cuda.synchronize(device)
            segment_started = time.perf_counter()
            result = segmenter.segment(rgb, image_name=path.name)
            torch.cuda.synchronize(device)
            segmentation_ms = (time.perf_counter() - segment_started) * 1000.0

            save_started = time.perf_counter()
            if result.mask.shape != rgb.shape[:2]:
                raise RuntimeError(
                    f"Dimension-mismatched SAM3 mask for {path}: "
                    f"mask={result.mask.shape}, image={rgb.shape[:2]}"
                )
            if not result.mask.any() or result.mask.all():
                raise RuntimeError(f"Degenerate SAM3 mask for {path}")
            destination = mask_dir / f"{path.stem}.mask.png"
            temporary = destination.with_suffix(".tmp.png")
            Image.fromarray(result.mask.astype(np.uint8) * 255, mode="L").save(temporary, optimize=True)
            temporary.replace(destination)
            (diagnostic_dir / f"{path.stem}.json").write_text(
                json.dumps(
                    {
                        "image_name": path.name,
                        "seed_xy": list(result.seed_xy),
                        "prompt_used": result.prompt_used,
                        "fallback_reason": result.fallback_reason,
                        "mask_area": int(result.mask.sum()),
                        "mask_confidence": float(result.mask_confidence),
                        "sam3_selected_score": float(result.sam3_selected_score),
                        "sam3_refined_iou": result.sam3_refined_iou,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            save_ms = (time.perf_counter() - save_started) * 1000.0
            complete_ms = (time.perf_counter() - total_started) * 1000.0
            rows.append(
                {
                    "number": index,
                    "image_name": path.name,
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                    "decode_ms": decode_ms,
                    "sam3_in_memory_segmentation_ms": segmentation_ms,
                    "png_and_diagnostic_save_ms": save_ms,
                    "complete_generation_ms": complete_ms,
                    "prompt_used": result.prompt_used,
                    "seed_x": int(result.seed_xy[0]),
                    "seed_y": int(result.seed_xy[1]),
                    "fallback_reason": result.fallback_reason,
                    "mask_area": int(result.mask.sum()),
                    "mask_path": str(destination),
                }
            )
            print(
                f"[{index:02d}/{len(images)}] {path.name}: SAM3={segmentation_ms/1000:.3f} s, "
                f"complete={complete_ms/1000:.3f} s, prompt={result.prompt_used}",
                flush=True,
            )
            del rgb, result
    batch_wall_seconds = time.perf_counter() - batch_started

    csv_path = output / "per_image_timing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    complete_values = [float(row["complete_generation_ms"]) for row in rows]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "single_checkpoint_sam3_center_plant",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "image_count": len(rows),
        "input_directory": str(image_dir),
        "output_directory": str(output),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "warmup_iterations": args.warmup,
        "model_load_seconds": model_load_seconds,
        "batch_wall_seconds": batch_wall_seconds,
        "summed_complete_generation_seconds": sum(complete_values) / 1000.0,
        "decode_ms": stats([float(row["decode_ms"]) for row in rows]),
        "sam3_in_memory_segmentation_ms": stats([float(row["sam3_in_memory_segmentation_ms"]) for row in rows]),
        "png_and_diagnostic_save_ms": stats([float(row["png_and_diagnostic_save_ms"]) for row in rows]),
        "complete_generation_ms": stats(complete_values),
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "settings": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in EXPECTED_SETTINGS.items()
        },
        "per_image_csv": str(csv_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
