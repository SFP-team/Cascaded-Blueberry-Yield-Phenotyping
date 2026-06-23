"""Train a YOLO blueberry ripe/unripe detector on the Roboflow dataset (Ultralytics).

Default model: yolo26s.pt — Ultralytics **YOLO26** (latest generation; NMS-free / end-to-end,
tuned for small-object + edge inference). Swap with --model:
    yolo26n.pt      fastest / smallest
    yolo26s.pt      balanced  (default)
    yolo26m.pt      most accurate that still trains on modest hardware
    yolo26-p2.yaml  adds a high-res P2 detection head for very small berries (trains from
                    scratch — no COCO-pretrained weights — so needs more data + compute)
    yolo11s.pt      proven YOLO11 fallback

Small, clustered berries benefit from a large --imgsz (1280 default). Drop to 960/640 for speed.
Apple Silicon trains on the MPS GPU automatically; it works but is slower than CUDA — for full
runs a CUDA GPU (or Colab) is recommended. Lower --batch if you hit memory limits.

Run (after download_data.py):
    .venv-yolo/bin/python train.py --epochs 100 --imgsz 1280
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_data_yaml() -> str | None:
    cands = sorted(HERE.glob("datasets/**/data.yaml"))
    return str(cands[0]) if cands else None


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _coerce(v: str):
    import yaml
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a YOLO blueberry ripe/unripe detector.")
    ap.add_argument("--data", default=None, help="path to data.yaml (default: autodiscover under datasets/)")
    ap.add_argument("--model", default="yolo26s.pt", help="ultralytics weights or model .yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280, help="train image size; 1280 helps small berries")
    ap.add_argument("--batch", type=int, default=8, help="lower if you hit OOM, raise on a big GPU")
    ap.add_argument("--device", default=None, help="cpu / mps / 0 (auto-detected if omitted)")
    ap.add_argument("--name", default="blueberry_yolo26s", help="run name under runs/")
    ap.add_argument("--patience", type=int, default=30, help="early-stop patience (epochs)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], metavar="K=V",
                    help="extra ultralytics train args, e.g. --extra lr0=0.005 mixup=0.1")
    args = ap.parse_args()

    data = args.data or find_data_yaml()
    if not data:
        raise SystemExit("No data.yaml under datasets/. Run download_data.py first (or pass --data).")
    device = args.device or pick_device()
    print(f"model={args.model}  data={data}  device={device}  imgsz={args.imgsz}  "
          f"batch={args.batch}  epochs={args.epochs}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    overrides = dict(
        data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
        project=str(HERE / "runs"), name=args.name, patience=args.patience, resume=args.resume,
        cos_lr=True, close_mosaic=10, plots=True,
    )
    for kv in args.extra:
        k, _, v = kv.partition("=")
        overrides[k] = _coerce(v)

    model.train(**overrides)
    best = getattr(model.trainer, "best", None)
    print("\nDone. Best weights:", best)
    print("Validate/predict with predict.py (point --weights at that best.pt).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
