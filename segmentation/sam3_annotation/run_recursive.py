"""Recursive driver for the SAM3 annotation tool: build the segmenter once, walk a nested
input tree (for example session/cohort/raw/*.jpg), and write outputs mirroring the tree.

Usage (from the repository root):
    python -m segmentation.sam3_annotation.run_recursive <input_root> <output_root> <sam3_checkpoint> [device]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Sam3NewConfig
from .io_utils import load_upright
from .mask_io import save_result
from .overlay import render_binary_mask, render_mask_preview, render_overlay
from .pipeline import confidence_map, process_image
from .report import write_reports
from .segmenter import Sam3HybridSegmenter

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("run_recursive")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="segmentation.sam3_annotation.run_recursive",
        description="Segment the center plant in every image below input_root and mirror the folder tree.",
    )
    p.add_argument("input_root", help="Root folder searched recursively for images.")
    p.add_argument("output_root", help="Root folder for outputs (one output folder per image folder).")
    p.add_argument("sam3_checkpoint", help="Path to the SAM 3 checkpoint (sam3.pt).")
    p.add_argument("device", nargs="?", default="cuda", help="Torch device (default: cuda).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    ckpt = Path(args.sam3_checkpoint)
    device = args.device

    cfg = Sam3NewConfig(device=device, sam3_checkpoint=ckpt, output_dir=output_root)

    # Group images by their parent folder so we mirror the tree and write one
    # report per leaf folder.
    by_dir: dict[Path, list[Path]] = {}
    for p in sorted(input_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            by_dir.setdefault(p.parent, []).append(p)

    total = sum(len(v) for v in by_dir.values())
    log.info("Found %d images across %d folders under %s", total, len(by_dir), input_root)
    if total == 0:
        log.error("No images found.")
        return 1

    log.info("Loading SAM3 model once (device=%s) ...", device)
    segmenter = Sam3HybridSegmenter(cfg)

    done = 0
    processed = 0
    for src_dir in sorted(by_dir):
        rel = src_dir.relative_to(input_root)
        out_dir = output_root / rel
        overlay_dir = out_dir / "overlays"
        mask_preview_dir = out_dir / "mask_previews"
        binary_mask_dir = out_dir / "mask_pngs"
        rows = []
        for path in sorted(by_dir[src_dir]):
            done += 1
            log.info("[%d/%d] %s", done, total, path.relative_to(input_root))
            try:
                result = process_image(path, segmenter, cfg)
            except Exception as e:  # noqa: BLE001
                log.warning("Failed on %s: %s", path.name, e)
                continue
            rgb = load_upright(path, cfg.rotate)
            render_overlay(rgb, result, overlay_dir / f"{path.stem}.overlay.png")
            render_mask_preview(rgb, result, mask_preview_dir / f"{path.stem}.mask_preview.png")
            render_binary_mask(result.mask, binary_mask_dir / f"{path.stem}.mask.png")
            if result.mask_confidence is not None:
                result.confidence_map = confidence_map(result.mask, result.core, result.mask_confidence)
            save_result(result, path, out_dir, rotate=cfg.rotate, save_neighbor_masks=cfg.save_neighbor_masks)
            rows.append(result)
            processed += 1
        if rows:
            write_reports(rows, out_dir, cfg.report_format)

    log.info("DONE: processed %d/%d images across %d folders", processed, total, len(by_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
