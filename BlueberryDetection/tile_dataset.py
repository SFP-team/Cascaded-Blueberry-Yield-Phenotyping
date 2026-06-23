"""Conditional tiling preprocessor for the heterogeneous blueberry_merged_fb dataset.

Why: the merged set spans 187px stock photos -> 4896px rover frames, and ~98% of boxes are
"small" (median berry ~35px at native res -> ~15px once a 3024px image is squeezed to imgsz=1280).
Blindly tiling everything would shred the small stock photos; downscaling everything loses the
berries on the big frames. So we tile *conditionally*:

  - Large image (max side > --min-tile-dim): slice into <tile>x<tile> windows with overlap,
    edge-aligned last tile, and remap/clip the YOLO labels into each tile.
  - Small image (already <= --min-tile-dim): pass through unchanged.

A clipped box is kept only if enough of it survives (--min-vis area ratio) and it isn't a
degenerate sliver (--min-px). Fully-empty tiles are dropped except a --empty-frac sample, which
gives the model some hard negatives without diluting the dense positives.

Output is a NEW dataset dir (original untouched) with the same split layout + a fixed data.yaml.

    python tile_dataset.py --src <in> --dst <out>
    python tile_dataset.py --src <in> --dst <out> --tile 1024 --overlap 0.2 --limit 3 --viz
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def tile_1d(total: int, ts: int, overlap: float):
    """Return (start, size) windows covering `total`. Interior tiles are full `ts`,
    the last is shifted back to end exactly at the edge (no padding). If the dim is
    smaller than `ts`, one window of the whole dim."""
    if total <= ts:
        return [(0, total)]
    stride = max(1, int(round(ts * (1.0 - overlap))))
    starts = list(range(0, total - ts + 1, stride))
    if not starts or starts[-1] != total - ts:
        starts.append(total - ts)
    return [(s, ts) for s in starts]


def read_labels(lp: Path):
    """YOLO labels -> list of (cls, cx, cy, w, h) floats (normalized)."""
    out = []
    if not lp.exists():
        return out
    for line in lp.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            out.append((int(float(p[0])), *map(float, p[1:5])))
    return out


def remap_box(b, W, H, tx, ty, tw, th, min_vis, min_px):
    """Map one normalized box from full image into a tile. Returns (cls,cx,cy,w,h) in
    tile-normalized coords, or None if it doesn't survive the crop."""
    cls, cx, cy, bw, bh = b
    x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
    x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
    ix1, iy1 = max(x1, tx), max(y1, ty)
    ix2, iy2 = min(x2, tx + tw), min(y2, ty + th)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return None
    orig_area = max((x2 - x1) * (y2 - y1), 1e-9)
    if (iw * ih) / orig_area < min_vis:   # mostly outside the tile -> bad label
        return None
    if iw < min_px or ih < min_px:        # degenerate sliver
        return None
    ncx = ((ix1 + ix2) / 2 - tx) / tw
    ncy = ((iy1 + iy2) / 2 - ty) / th
    nw, nh = iw / tw, ih / th
    return (cls, min(max(ncx, 0), 1), min(max(ncy, 0), 1), min(nw, 1), min(nh, 1))


def write_labels(lp: Path, boxes):
    lp.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes))


def draw(img, boxes, color=(0, 0, 255)):
    h, w = img.shape[:2]
    for c, cx, cy, bw, bh in boxes:
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


def process_split(src: Path, dst: Path, split: str, args, stats, rng):
    si, sl = src / split / "images", src / split / "labels"
    di, dl = dst / split / "images", dst / split / "labels"
    di.mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in si.iterdir() if p.suffix.lower() in IMG_EXT)
    if args.limit:
        imgs = imgs[: args.limit]

    n_tiled = n_passed = n_out = n_boxes = 0
    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = read_labels(sl / f"{ip.stem}.txt")
        big = max(W, H) > args.min_tile_dim

        if not big:
            # pass-through unchanged
            shutil.copy(ip, di / ip.name)
            write_labels(dl / f"{ip.stem}.txt", boxes)
            n_passed += 1
            n_out += 1
            n_boxes += len(boxes)
            continue

        n_tiled += 1
        xs = tile_1d(W, args.tile, args.overlap)
        ys = tile_1d(H, args.tile, args.overlap)
        for r, (ty, th) in enumerate(ys):
            for c, (tx, tw) in enumerate(xs):
                tb = []
                for b in boxes:
                    m = remap_box(b, W, H, tx, ty, tw, th, args.min_vis, args.min_px)
                    if m is not None:
                        tb.append(m)
                if not tb and rng.random() > args.empty_frac:
                    continue  # drop empty tile unless sampled as a negative
                crop = img[ty:ty + th, tx:tx + tw]
                name = f"{ip.stem}__t{r}_{c}"
                cv2.imwrite(str(di / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                write_labels(dl / f"{name}.txt", tb)
                if args.viz and tb:
                    cv2.imwrite(str(dst / f"_viz_{name}.jpg"), draw(crop.copy(), tb))
                n_out += 1
                n_boxes += len(tb)

    stats[split] = dict(src_imgs=len(imgs), tiled=n_tiled, passed=n_passed,
                        out_imgs=n_out, out_boxes=n_boxes)
    print(f"  {split}: {len(imgs)} src ({n_tiled} tiled, {n_passed} whole) "
          f"-> {n_out} out imgs, {n_boxes} boxes")


def main() -> int:
    ap = argparse.ArgumentParser(description="Conditional tiling for small-object detection.")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--min-tile-dim", type=int, default=1536,
                    help="only tile images whose max side exceeds this; else pass through")
    ap.add_argument("--min-vis", type=float, default=0.3, help="min surviving area ratio of a clipped box")
    ap.add_argument("--min-px", type=float, default=4.0, help="drop boxes thinner than this after clip")
    ap.add_argument("--empty-frac", type=float, default=0.1, help="fraction of empty tiles kept as negatives")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--names", nargs="+", default=["immature_berry", "mature_berry", "flower"])
    ap.add_argument("--limit", type=int, default=0, help="process only first N imgs/split (debug)")
    ap.add_argument("--viz", action="store_true", help="also dump _viz_*.jpg with boxes drawn")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    stats = {}
    print(f"Tiling {src} -> {dst}  (tile={args.tile} overlap={args.overlap} "
          f"min_tile_dim={args.min_tile_dim})")
    for split in args.splits:
        if (src / split / "images").is_dir():
            process_split(src, dst, split, args, stats, rng)

    # data.yaml with server/abs path filled at deploy time; write relative-friendly version
    yaml = (f"path: {dst.resolve()}\n"
            f"train: train/images\nval: valid/images\ntest: test/images\n\n"
            f"nc: {len(args.names)}\nnames: {args.names}\n")
    (dst / "data.yaml").write_text(yaml)
    print("\nWrote", dst / "data.yaml")
    tot = sum(s["out_imgs"] for s in stats.values())
    print(f"TOTAL out images: {tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
