#!/usr/bin/env python3
"""Cut the train and val photos of the dataset into 640 px training tiles for YOLO.

    python prepare_tiles.py [--root DATA_ROOT] [--workers 16]   # writes <root>/tiles_640; refuses to overwrite it

Conditional tiling that matches the whole-photo slicing used at inference: 640 px tiles with 20% overlap at native
resolution. A photo stays whole when its long side is <= 640 or its boxes are large (median box short side / long
side > 0.08). A clipped box is kept when at least 30% of its area is inside the tile. Training keeps up to 3 tiles
without boxes per photo so the model sees background; validation keeps every tile so the per-epoch validation also
counts false detections on background. Only train and val are tiled; test is scored on whole photos (evaluate.py).
"""

from __future__ import annotations

import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

import config

STEP = int(config.TILE * (1 - config.OVERLAP))


def read_boxes(path: Path) -> np.ndarray:
    rows = [line.split()[:5] for line in path.read_text().splitlines() if line.strip()]
    return np.array(rows, dtype=float).reshape(-1, 5)


def should_tile(w: int, h: int, boxes: np.ndarray) -> bool:
    long_side = max(w, h)
    if long_side <= config.TILE:
        return False
    if len(boxes):
        median_min_side = float(np.median(np.minimum(boxes[:, 3] * w, boxes[:, 4] * h)))
        if median_min_side / long_side > config.REL_LARGE_OBJ:
            return False
    return True


def origins(size: int) -> list[int]:
    if size <= config.TILE:
        return [0]
    starts = list(range(0, size - config.TILE + 1, STEP))
    if starts[-1] != size - config.TILE:
        starts.append(size - config.TILE)
    return starts


def clip(boxes_px: np.ndarray, x0: int, y0: int, tw: int, th: int) -> list[tuple]:
    kept = []
    for c, x1, y1, x2, y2 in boxes_px:
        ox1, oy1, ox2, oy2 = max(x1, x0), max(y1, y0), min(x2, x0 + tw), min(y2, y0 + th)
        if ox2 <= ox1 or oy2 <= oy1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area <= 0 or (ox2 - ox1) * (oy2 - oy1) / area < config.MIN_VISIBLE:
            continue
        # Normalised to the 640 x 640 canvas; a tile cut from a side shorter than 640 is padded at the bottom/right.
        kept.append((int(c), ((ox1 + ox2) / 2 - x0) / config.TILE, ((oy1 + oy2) / 2 - y0) / config.TILE,
                     (ox2 - ox1) / config.TILE, (oy2 - oy1) / config.TILE))
    return kept


def write_label(path: Path, boxes) -> None:
    lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def tile_image(job: tuple[str, str, Path, Path]) -> dict:
    name, split, dataset, tiles = job
    stem = Path(name).stem
    src = dataset / split
    boxes = read_boxes(src / "labels" / f"{stem}.txt")
    out_images, out_labels = tiles / split / "images", tiles / split / "labels"
    with Image.open(src / "images" / name) as image:
        image = image.convert("RGB")
        w, h = image.size
        if not should_tile(w, h, boxes):
            image.save(out_images / f"{stem}.jpg", quality=95)
            write_label(out_labels / f"{stem}.txt", [tuple(b) for b in boxes])
            return {"name": name, "split": split, "whole": 1, "positive": 1 if len(boxes) else 0,
                    "empty_kept": 0 if len(boxes) else 1, "empty_dropped": 0, "boxes": len(boxes)}
        cx, cy, bw, bh = boxes[:, 1] * w, boxes[:, 2] * h, boxes[:, 3] * w, boxes[:, 4] * h
        boxes_px = np.stack([boxes[:, 0], cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        positive, empty = [], []
        for y0 in origins(h):
            for x0 in origins(w):
                tw, th = min(config.TILE, w - x0), min(config.TILE, h - y0)
                kept = clip(boxes_px, x0, y0, tw, th)
                (positive if kept else empty).append((x0, y0, tw, th, kept))
        rng = random.Random(f"{config.SEED}:{name}")
        rng.shuffle(empty)
        kept_empty = empty[: config.EMPTY_TILES_PER_IMAGE] if split == "train" else empty
        for x0, y0, tw, th, kept in positive + kept_empty:
            crop = image.crop((x0, y0, x0 + tw, y0 + th))
            if crop.size != (config.TILE, config.TILE):
                canvas = Image.new("RGB", (config.TILE, config.TILE))
                canvas.paste(crop, (0, 0))
                crop = canvas
            tile_stem = f"{stem}_x{x0}_y{y0}"
            crop.save(out_images / f"{tile_stem}.jpg", quality=95)
            write_label(out_labels / f"{tile_stem}.txt", kept)
    return {"name": name, "split": split, "whole": 0, "positive": len(positive), "empty_kept": len(kept_empty),
            "empty_dropped": len(empty) - len(kept_empty), "boxes": sum(len(t[4]) for t in positive)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help=f"data root (default: ${config.ROOT_ENV} or {config.DEFAULT_ROOT.relative_to(config.REPO_ROOT)})")
    parser.add_argument("--workers", type=int, default=16, help="parallel processes")
    args = parser.parse_args()

    root = config.data_root(args.root).resolve()
    dataset, tiles = config.dataset_dir(root), config.tiles_dir(root)
    if tiles.exists():
        raise SystemExit(f"{tiles} exists; remove it to re-tile")
    records = json.loads((dataset / "split.json").read_text())["images"]
    for split in ("train", "val"):
        (tiles / split / "images").mkdir(parents=True)
        (tiles / split / "labels").mkdir(parents=True)
    jobs = [(name, r["split"], dataset, tiles) for name, r in sorted(records.items()) if r["split"] in ("train", "val")]
    with Pool(args.workers) as pool:
        rows = pool.map(tile_image, jobs, chunksize=4)
    stats = {}
    for split in ("train", "val"):
        part = [r for r in rows if r["split"] == split]
        stats[split] = {key: sum(r[key] for r in part) for key in ("whole", "positive", "empty_kept", "empty_dropped", "boxes")}
        stats[split]["images"] = len(part)
        stats[split]["tiles"] = stats[split]["positive"] + stats[split]["empty_kept"]
    (tiles / "data.yaml").write_text(
        f"path: {tiles}\ntrain: train/images\nval: val/images\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(config.CLASSES))
    )
    stats["settings"] = {"tile": config.TILE, "overlap": config.OVERLAP, "min_visible": config.MIN_VISIBLE,
                         "rel_large_obj": config.REL_LARGE_OBJ, "empty_tiles_per_image": config.EMPTY_TILES_PER_IMAGE}
    (tiles / "tile_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
