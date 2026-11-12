#!/usr/bin/env python3
"""Train YOLO26-L on the training tiles written by prepare_tiles.py.

    python train.py [--root DATA_ROOT] [--name yolo26l] [--device 0] [--epochs 60] [--pretrained yolo26l.pt]

Recipe: COCO-pretrained yolo26l.pt fine-tuned at imgsz 640 with batch 6, seed 42 and deterministic mode, images cached
in RAM. Optimizer MuSGD with lr0 0.01, momentum 0.9 and weight decay 0.0005, 3 warm-up epochs, then a cosine schedule
(lrf 0.01) that runs to the last epoch: no early stop. Augmentation: mosaic 0.5, switched off for the last 15 epochs
(close_mosaic 15), no mixup or copy-paste, rotation up to 5 degrees, translation 0.1, scale jitter 0.3 (whole photos
are sliced at native resolution at inference, so berries are seen at their real size and large scale jitter would
shrink berries that are already small), horizontal flip 0.5, vertical flip 0.1, HSV jitter h 0.015, s 0.4, v 0.3.
max_det 3000, mixed precision.

The run is written to <root>/runs/NAME; the weights the pipeline uses are <root>/runs/NAME/weights/best.pt copied to
weights/blueberry_yolo26l.pt at the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help=f"data root (default: ${config.ROOT_ENV} or {config.DEFAULT_ROOT.relative_to(config.REPO_ROOT)})")
    parser.add_argument("--name", default="yolo26l", help="run name under <root>/runs")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--pretrained", default="yolo26l.pt",
                        help="COCO-pretrained YOLO26-L weights; Ultralytics downloads yolo26l.pt when it is not present")
    args = parser.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    root = config.data_root(args.root).resolve()
    dataset, tiles, runs = config.dataset_dir(root), config.tiles_dir(root), config.runs_dir(root)
    run_dir = runs / args.name
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; pick another --name")
    hyper = dict(
        data=str(tiles / "data.yaml"), epochs=args.epochs, patience=args.epochs + 1, imgsz=config.TILE, batch=6,
        device=args.device, workers=8, cache="ram", seed=42, deterministic=True,
        optimizer="MuSGD", lr0=0.01, momentum=0.9, weight_decay=0.0005, warmup_epochs=3.0, cos_lr=True, lrf=0.01,
        mosaic=0.5, mixup=0.0, copy_paste=0.0, close_mosaic=15, degrees=5.0, translate=0.1, scale=0.3,
        fliplr=0.5, flipud=0.1, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, max_det=config.MAX_DET, amp=True, plots=True,
        project=str(runs), name=args.name, exist_ok=False,
    )
    model = YOLO(args.pretrained)
    pretrained = Path(model.ckpt_path or args.pretrained)
    meta = {
        "pretrained": pretrained.name, "pretrained_sha256": sha256(pretrained),
        "split_sha256": sha256(dataset / "split.json"), "tile_stats": json.loads((tiles / "tile_stats.json").read_text()),
        "versions": {"ultralytics": ultralytics.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
                     "python": platform.python_version()},
        "hyperparameters": hyper,
    }
    model.train(**hyper)
    (run_dir / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print("best", run_dir / "weights" / "best.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
