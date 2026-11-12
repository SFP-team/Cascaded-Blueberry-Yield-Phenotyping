"""CLI for the SAM3 center-plant annotation pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Sam3NewConfig
from .io_utils import iter_images, load_upright
from .mask_io import save_result
from .overlay import render_binary_mask, render_mask_preview, render_overlay
from .pipeline import confidence_map, process_image
from .report import write_reports
from .segmenter import Sam3HybridSegmenter

log = logging.getLogger("sam3_annotation")


def _build_config(args) -> Sam3NewConfig:
    return Sam3NewConfig(
        device=args.device,
        sam3_checkpoint=Path(args.sam3_checkpoint) if getattr(args, "sam3_checkpoint", None) else None,
        rotate=args.rotate,
        text_prompt=args.text_prompt,
        fallback_prompts=tuple(args.fallback_prompts),
        sam3_confidence_thresh=args.sam3_confidence_thresh,
        plant_box_width_frac=args.plant_box_width_frac,
        negative_box_frac=args.negative_box_frac,
        save_neighbor_masks=args.save_neighbor_masks,
        report_format=args.report_format,
        output_dir=Path(args.output_dir),
    )


def cmd_segment(args) -> int:
    cfg = _build_config(args)
    input_dir = Path(args.input_dir)
    out_dir = cfg.output_dir
    overlay_dir = out_dir / "overlays"
    mask_preview_dir = out_dir / "mask_previews"
    binary_mask_dir = out_dir / "mask_pngs"

    images = iter_images(input_dir)
    if not images:
        log.error("No images to process in %s", input_dir)
        return 1

    segmenter = Sam3HybridSegmenter(cfg)
    rows = []
    no_plant = 0
    conf_sum = 0.0

    for i, path in enumerate(images, 1):
        log.info("[%d/%d] %s", i, len(images), path.name)
        try:
            result = process_image(path, segmenter, cfg)
        except Exception as e:
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
        no_plant += int(result.no_plant)
        conf_sum += result.mask_confidence

    written = write_reports(rows, out_dir, cfg.report_format)
    mean_conf = conf_sum / len(rows) if rows else 0.0

    print()
    print(f"Processed:        {len(rows)}/{len(images)} images")
    print(f"No center plant:  {no_plant}")
    print(f"Mean confidence:  {mean_conf:.3f}")
    print(f"Overlays:         {overlay_dir}")
    print(f"Mask previews:    {mask_preview_dir}")
    print(f"Binary masks:     {binary_mask_dir}")
    for w in written:
        print(f"Report:           {w}")
    if cfg.save_neighbor_masks:
        print(f"Neighbor masks:   {out_dir / 'neighbor_masks'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="segmentation.sam3_annotation", description="Standalone SAM3-first center-plant pipeline.")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("segment", help="Segment the center plant across a directory of images.")
    s.add_argument("input_dir", help="Directory of rover images.")
    s.add_argument("output_dir", help="Directory for overlays, masks and reports.")
    s.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    s.add_argument("--rotate", type=int, default=None, help="Explicit CW rotation (0/90/180/270) instead of EXIF.")
    s.add_argument("--sam3-checkpoint", default=None, help="Path to a local SAM 3 checkpoint.")
    s.add_argument("--text-prompt", default="blueberry bush", help="Primary concept prompt.")
    s.add_argument("--fallback-prompts", nargs="+", default=["blueberry plant", "plant"], help="Fallback concept prompts.")
    s.add_argument("--sam3-confidence-thresh", type=float, default=0.35, help="SAM3 confidence threshold.")
    s.add_argument("--plant-box-width-frac", type=float, default=0.75, help="Center box width as fraction of image width.")
    s.add_argument("--negative-box-frac", type=float, default=0.22, help="Sideline negative box width fraction.")
    s.add_argument("--save-neighbor-masks", action=argparse.BooleanOptionalAction, default=True, help="Persist non-selected concept masks.")
    s.add_argument("--report-format", default="both", choices=["csv", "json", "both"])
    s.set_defaults(func=cmd_segment)
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
