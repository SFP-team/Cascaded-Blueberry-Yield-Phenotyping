#!/usr/bin/env python3
"""Train and run the four-channel center-conditioned BiRefNet-HR (Center BiRefNet).

Commands:
  smoke    bounded accumulated-backward and validation check
  train    fine-tune from the trained BiRefNet-HR best.pt; writes best.pt
  predict  choose thresholds on the validation split, then write balanced and
           center_strict masks for the val and test splits; writes threshold.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        default="configs/fixed_split.yaml",
        help="Config path, relative to segmentation/benchmark unless absolute.",
    )
    parser.add_argument("command", choices=("smoke", "train", "predict"))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--side-negative-weight", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require_fold(value: int | None) -> int:
    if value is None:
        raise SystemExit("--fold is required for this command")
    if value not in range(5):
        raise SystemExit("--fold must be between 0 and 4")
    return value


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str((project / "src").resolve()))
    from plant_benchmark.center_birefnet import (
        predict_center,
        smoke_center,
        train_center,
    )
    from plant_benchmark.config import load_config

    cfg = load_config(project / args.config)
    if args.command == "smoke":
        smoke_center(cfg, require_fold(args.fold))
    elif args.command == "train":
        train_center(
            cfg,
            require_fold(args.fold),
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            accumulation=args.gradient_accumulation,
            max_gradient_norm=args.max_gradient_norm,
            side_negative_weight=args.side_negative_weight,
            force=args.force,
        )
    elif args.command == "predict":
        predict_center(cfg, require_fold(args.fold))


if __name__ == "__main__":
    main()
