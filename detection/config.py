"""Paths and constants shared by the berry and flower detector scripts.

The data root holds the dataset and everything the scripts write. It is taken from --root, else from the
BLUEBERRY_DETECTION_ROOT environment variable, else data/detection under the repository root.

    <root>/dataset/   {train,val,test}/{images,labels} in YOLO format, and split.json
    <root>/tiles_640/ training tiles written by prepare_tiles.py: {train,val}/{images,labels}, data.yaml
    <root>/runs/      training runs written by train.py
    <root>/evals/     predictions and scores written by evaluate.py: <name>/<split>/
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_ENV = "BLUEBERRY_DETECTION_ROOT"
DEFAULT_ROOT = REPO_ROOT / "data" / "detection"
RELEASE_WEIGHTS = REPO_ROOT / "weights" / "blueberry_yolo26l.pt"  # the trained detector, as read by the pipeline


def data_root(cli_value: str | None = None) -> Path:
    return Path(cli_value or os.environ.get(ROOT_ENV) or DEFAULT_ROOT)


def dataset_dir(root: Path) -> Path:
    return root / "dataset"


def tiles_dir(root: Path) -> Path:
    return root / "tiles_640"


def runs_dir(root: Path) -> Path:
    return root / "runs"


def evals_dir(root: Path) -> Path:
    return root / "evals"


CLASSES = ("immature berry", "mature berry", "flowers")
SEED = 20261022  # picks the background tiles kept per training photo

# Conditional tiling of the training photos.
TILE = 640
OVERLAP = 0.2
MIN_VISIBLE = 0.30  # keep a clipped box when at least this share of its area is inside the tile
REL_LARGE_OBJ = 0.08  # keep an image whole when its median box short side / long side is above this
EMPTY_TILES_PER_IMAGE = 3  # training tiles without boxes kept per image

# Whole-photo inference, as in the pipeline: SAHI-style slicing and GREEDYNMM merge.
SLICE = 640
SLICE_OVERLAP = 0.2
MERGE = dict(match_metric="IOS", match_threshold=0.5, class_agnostic=True)
MAX_DET = 3000
EVAL_CONF = 0.01  # keep for AP; counting uses COUNT_CONF
COUNT_CONF = 0.35
