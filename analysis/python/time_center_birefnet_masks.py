#!/usr/bin/env python3
"""Time automatic Center BiRefNet mask generation on a folder of images (one CUDA GPU).

The model is loaded with the deployment loader in ../../pipeline/segment.py from
weights/birefnet_hr.pt, weights/center_birefnet.pt and weights/threshold.json (paths and
input size from pipeline/configs/pipeline.yaml). The center prompt is the exact image center;
no reference mask is used. Warm-up passes are excluded from the reported timings.

Outputs (under --output):
  masks/<stem>.mask.png
  per_image_timing.csv   preprocess, model inference, postprocess+save and complete time (ms)
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
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from pipeline.segment import (  # noqa: E402
    load_center_model,
    load_upright_rgb,
    prepare_tensor,
    read_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="Folder of .jpg images.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "pipeline/configs/pipeline.yaml",
        help="Pipeline config with the segmentation section.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def prepare(path: Path, height: int, width: int) -> tuple[torch.Tensor, int, int]:
    image, _ = load_upright_rgb(path)
    original_width, original_height = image.size
    tensor = prepare_tensor(image, original_width / 2.0, original_height / 2.0, height, width)
    return tensor, original_width, original_height


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    device = torch.device("cuda")
    seg_cfg = yaml.safe_load(args.config.read_text())["segmentation"]
    base_weights = repo_path(seg_cfg["base_weights"])
    center_weights = repo_path(seg_cfg["center_weights"])
    threshold_file = repo_path(seg_cfg["threshold_file"])
    target_height = int(seg_cfg["height"])
    target_width = int(seg_cfg["width"])
    threshold = read_threshold(threshold_file, seg_cfg["operating_point"])

    image_dir = args.images.resolve()
    output = args.output.resolve()
    mask_dir = output / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise RuntimeError(f"No .jpg images in {image_dir}")

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = load_center_model(
        seg_cfg["pretrained_id"], base_weights, center_weights, int(seg_cfg["fold"])
    )
    model = model.float().to(device).eval()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_start

    warm_tensor, _, _ = prepare(images[0], target_height, target_width)
    warm_tensor = warm_tensor.to(device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(warm_tensor).sigmoid()
        torch.cuda.synchronize()
    del warm_tensor

    rows: list[dict[str, float | int | str]] = []
    batch_start = time.perf_counter()
    with torch.inference_mode():
        for index, path in enumerate(images, start=1):
            total_start = time.perf_counter()
            tensor, original_width, original_height = prepare(path, target_height, target_width)
            tensor = tensor.to(device)
            torch.cuda.synchronize()
            preprocess_ms = (time.perf_counter() - total_start) * 1000.0

            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_start.record()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                probability_tensor = model(tensor).sigmoid()[0, 0]
            gpu_end.record()
            torch.cuda.synchronize()
            model_ms = float(gpu_start.elapsed_time(gpu_end))

            post_start = time.perf_counter()
            probability = probability_tensor.float().cpu().numpy()
            binary = Image.fromarray((probability >= threshold).astype(np.uint8) * 255, mode="L")
            if binary.size != (original_width, original_height):
                binary = binary.resize((original_width, original_height), Image.Resampling.NEAREST)
            destination = mask_dir / f"{path.stem}.mask.png"
            temporary = destination.with_suffix(".tmp.png")
            binary.save(temporary, optimize=True)
            temporary.replace(destination)
            postprocess_save_ms = (time.perf_counter() - post_start) * 1000.0
            total_ms = (time.perf_counter() - total_start) * 1000.0
            foreground_pixels = int(np.count_nonzero(np.asarray(binary)))
            if foreground_pixels <= 0 or foreground_pixels >= original_width * original_height:
                raise RuntimeError(f"Degenerate output mask: {destination}")
            rows.append(
                {
                    "number": index,
                    "image_name": path.name,
                    "width": original_width,
                    "height": original_height,
                    "point_x": original_width / 2.0,
                    "point_y": original_height / 2.0,
                    "preprocess_ms": preprocess_ms,
                    "model_inference_ms": model_ms,
                    "postprocess_and_save_ms": postprocess_save_ms,
                    "complete_generation_ms": total_ms,
                    "foreground_pixels": foreground_pixels,
                    "mask_path": str(destination),
                }
            )
            print(
                f"[{index:02d}/{len(images)}] {path.name}: model={model_ms:.2f} ms, "
                f"complete={total_ms:.2f} ms",
                flush=True,
            )
            del tensor, probability_tensor, probability, binary
    batch_wall_seconds = time.perf_counter() - batch_start

    csv_path = output / "per_image_timing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    complete_values = [float(row["complete_generation_ms"]) for row in rows]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "single_checkpoint_center_birefnet",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "image_count": len(rows),
        "input_directory": str(image_dir),
        "output_directory": str(output),
        "input_size": [target_width, target_height],
        "output_size": "original EXIF-upright image dimensions",
        "point_protocol": "automatic exact image center; no reference mask",
        "threshold": threshold,
        "threshold_source": str(threshold_file),
        "operating_point": seg_cfg["operating_point"],
        "warmup_iterations": args.warmup,
        "model_load_seconds": model_load_seconds,
        "batch_wall_seconds": batch_wall_seconds,
        "summed_complete_generation_seconds": sum(complete_values) / 1000.0,
        "preprocess_ms": quantiles([float(row["preprocess_ms"]) for row in rows]),
        "model_inference_ms": quantiles([float(row["model_inference_ms"]) for row in rows]),
        "postprocess_and_save_ms": quantiles([float(row["postprocess_and_save_ms"]) for row in rows]),
        "complete_generation_ms": quantiles(complete_values),
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "base_checkpoint": str(base_weights),
        "base_checkpoint_sha256": sha256(base_weights),
        "center_checkpoint": str(center_weights),
        "center_checkpoint_sha256": sha256(center_weights),
        "per_image_csv": str(csv_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
