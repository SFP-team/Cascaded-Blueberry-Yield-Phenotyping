#!/usr/bin/env python3
"""Folder of photos in; Center BiRefNet masks, YOLO26-L detections, plant traits, one Excel out."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import yaml

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.detect import (
    already_complete,
    counts_from_detections,
    load_detector,
    overlay_path,
    prediction_path,
    run_sliced,
    write_overlay,
)
from pipeline.discover import discover_records
from pipeline.segment import (
    load_center_model,
    load_mask_png,
    load_upright_rgb,
    mask_overlay_path,
    mask_paths,
    predict_mask,
    prepare_tensor,
    read_threshold,
    save_mask,
    save_meta,
    stem_seed_xy,
    write_mask_overlay,
)
from pipeline.workbook import utc_now, write_workbook
from traits.columns import architecture_excel_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Center BiRefNet masks, plant architecture, YOLO26 counts, one Excel."
    )
    parser.add_argument("--input", required=True, help="Folder of photographs, or one image")
    parser.add_argument("--output", required=True, help="Folder for masks, detections, Excel")
    parser.add_argument(
        "--config",
        default=str(PIPELINE_DIR / "configs" / "pipeline.yaml"),
        help="Pipeline YAML. Relative weight paths in it resolve against the repository root.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--limit", type=int, help="Process only the first N images")
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute masks and detections that already exist"
    )
    parser.add_argument(
        "--no-overlays", action="store_true", help="Do not write detection overlay JPEGs"
    )
    parser.add_argument(
        "--shard-index", type=int, default=0, help="This process's shard, 0..shard-count-1"
    )
    parser.add_argument(
        "--shard-count", type=int, default=1, help="Total number of shards (one per GPU)"
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Skip segmentation; reuse masks and architecture from the existing shard Excel.",
    )
    parser.add_argument(
        "--excel-every",
        type=int,
        default=0,
        help="Write the Excel every N detect images. 0 = only at the end (default).",
    )
    parser.add_argument(
        "--operating-point",
        choices=("balanced", "center_strict"),
        help="Mask threshold key in threshold.json; overrides config segmentation.operating_point",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def resolve_weight(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def shard_records(records: list[dict], index: int, count: int) -> list[dict]:
    if count <= 1:
        return records
    if index < 0 or index >= count:
        raise ValueError(f"shard-index {index} is outside 0..{count - 1}")
    return [row for position, row in enumerate(records) if position % count == index]


def workbook_path(output: Path, shard_index: int, shard_count: int) -> Path:
    if shard_count > 1:
        return output / f"Plant_Architecture_and_Detections.shard{shard_index}.xlsx"
    return output / "Plant_Architecture_and_Detections.xlsx"


def table_paths(output: Path, shard_index: int, shard_count: int) -> tuple[Path, Path]:
    tables = output / "tables"
    if shard_count > 1:
        return (
            tables / f"detections_per_image.shard{shard_index}.csv",
            tables / f"architecture_per_image.shard{shard_index}.csv",
        )
    return tables / "detections_per_image.csv", tables / "architecture_per_image.csv"


def protocol_path(output: Path, shard_index: int, shard_count: int) -> Path:
    if shard_count > 1:
        return output / f"protocol.shard{shard_index}.json"
    return output / "protocol.json"


def load_architecture_rows(excel: Path) -> list[dict]:
    import pandas as pd

    if not excel.is_file():
        return []
    try:
        frame = pd.read_excel(excel, sheet_name="Plant_Architecture")
    except Exception:
        return []
    return frame.to_dict(orient="records")


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config))
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    seg_cfg = cfg["segmentation"]
    det_cfg = cfg["detection"]
    operating_point = args.operating_point or seg_cfg["operating_point"]
    base_weights = resolve_weight(REPO_ROOT, seg_cfg["base_weights"])
    center_weights = resolve_weight(REPO_ROOT, seg_cfg["center_weights"])
    threshold_file = resolve_weight(REPO_ROOT, seg_cfg["threshold_file"])
    yolo_weights = resolve_weight(REPO_ROOT, det_cfg["weights"])
    threshold = read_threshold(threshold_file, operating_point)
    height = int(seg_cfg["height"])
    width = int(seg_cfg["width"])
    fold = int(seg_cfg["fold"])

    if args.shard_count < 1:
        raise ValueError("shard-count must be >= 1")

    print(f"[{utc_now()}] discovering images under {source}", flush=True)
    records = discover_records(source)
    if args.limit is not None:
        records = records[: args.limit]
    records = shard_records(records, args.shard_index, args.shard_count)

    protocol = {
        "cohort": f"{len(records)} images from {source}",
        "mask_condition": "Center BiRefNet fold 0",
        "segmenter": "Center BiRefNet-HR, 4th-channel stem prior",
        "operating_point": operating_point,
        "mask_threshold": threshold,
        "base_weights": str(base_weights),
        "center_weights": str(center_weights),
        "detector": "YOLO26-L with SAHI sliced inference",
        "detection_weights": str(yolo_weights),
        "headline_threshold": float(det_cfg["count_confidence"]),
        "filtering_rule": (
            "Detection retained when confidence >= 0.35 and the bounding-box "
            "center lies inside the Center BiRefNet mask. "
            "total_berries = unripe + ripe. Flowers are excluded from berry totals."
        ),
        "n_images": len(records),
        "device": args.device,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "created_utc": utc_now(),
    }
    protocol_path(output, args.shard_index, args.shard_count).write_text(
        json.dumps(protocol, indent=2)
    )
    excel = workbook_path(output, args.shard_index, args.shard_count)
    detection_csv, architecture_csv = table_paths(output, args.shard_index, args.shard_count)

    if not records:
        print(
            f"[{utc_now()}] shard {args.shard_index}/{args.shard_count} has no images",
            flush=True,
        )
        write_workbook(excel, [], [], protocol)
        detection_csv.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        pd.DataFrame([]).to_csv(detection_csv, index=False)
        pd.DataFrame([]).to_csv(architecture_csv, index=False)
        return 0

    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    print(
        f"[{utc_now()}] images={len(records)} shard={args.shard_index}/{args.shard_count} "
        f"device={args.device} segment={operating_point}@{threshold} "
        f"detect_only={args.detect_only} excel_every={args.excel_every}",
        flush=True,
    )
    architecture_rows: list[dict] = []
    if args.detect_only:
        architecture_rows = load_architecture_rows(excel)
        print(
            f"[{utc_now()}] detect-only architecture_rows={len(architecture_rows)} from {excel}",
            flush=True,
        )
    else:
        model = load_center_model(
            seg_cfg["pretrained_id"],
            base_weights,
            center_weights,
            fold,
        ).float().to(device)
        from traits.architecture import analyze_plant

        for index, record in enumerate(records, start=1):
            stem = record["stem"]
            mask_png, meta_json = mask_paths(output, stem)
            started = time.monotonic()
            try:
                image, rgb = load_upright_rgb(Path(record["image_path"]))
                if not args.overwrite and mask_png.is_file() and meta_json.is_file():
                    mask = load_mask_png(mask_png)
                    meta = json.loads(meta_json.read_text())
                    seed_x, seed_y = meta.get("seed_xy", [rgb.shape[1] // 2, rgb.shape[0] // 2])
                    status = "cached"
                else:
                    seed_x, seed_y, fallback = stem_seed_xy(rgb)
                    tensor = prepare_tensor(image, float(seed_x), float(seed_y), height, width)
                    mask = predict_mask(model, tensor, image.size, threshold, device)
                    save_mask(mask, mask_png)
                    meta = {
                        "image_name": record["image_name"],
                        "source_path": record["image_path"],
                        "height": int(rgb.shape[0]),
                        "width": int(rgb.shape[1]),
                        "seed_xy": [int(seed_x), int(seed_y)],
                        "seed_fallback": fallback,
                        "no_plant": not bool(mask.any()),
                        "mask_confidence": 1.0,
                        "model": "birefnet_hr_centered",
                        "fold": fold,
                        "operating_point": operating_point,
                        "threshold": threshold,
                    }
                    save_meta(meta, meta_json)
                    status = "complete"
                overlay_mask = mask_overlay_path(output, stem)
                if args.overwrite or not overlay_mask.is_file():
                    write_mask_overlay(rgb, mask, overlay_mask)
                arch_started = time.monotonic()
                traits = analyze_plant(mask, rgb, meta)
                traits["foreground_fraction"] = float(mask.mean())
                architecture_rows.append(
                    architecture_excel_row(
                        record,
                        traits,
                        str(mask_png),
                        round(time.monotonic() - arch_started, 4),
                        status,
                    )
                )
                print(
                    f"[{utc_now()}] segment {index}/{len(records)} {stem} "
                    f"fg={int(mask.sum())} {status} {time.monotonic() - started:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                architecture_rows.append(
                    architecture_excel_row(record, {}, str(mask_png), 0.0, f"error: {exc!r}")
                )
                print(f"ERROR segment {stem}: {exc!r}", flush=True)
                traceback.print_exc()

        model.to("cpu")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[{utc_now()}] loading YOLO26-L", flush=True)
    detector = load_detector(
        yolo_weights,
        args.device,
        float(det_cfg["infer_confidence"]),
        int(det_cfg["slice"]),
        int(det_cfg["max_det"]),
    )
    detection_rows: list[dict] = []
    count_threshold = float(det_cfg["count_confidence"])
    for index, record in enumerate(records, start=1):
        stem = record["stem"]
        mask_png, _ = mask_paths(output, stem)
        pred_json = prediction_path(output, stem)
        started = time.monotonic()
        try:
            if not mask_png.is_file():
                raise FileNotFoundError(f"Missing mask for {stem}")
            if not args.overwrite and already_complete(pred_json):
                payload = json.loads(pred_json.read_text())
                counts = counts_from_detections(payload.get("detections", []), count_threshold)
                detection_rows.append(
                    {
                        **record,
                        **counts,
                        "image_height": payload.get("image_height"),
                        "image_width": payload.get("image_width"),
                        "mask_foreground_pixels": payload.get("mask_foreground_pixels"),
                        "runtime_seconds": payload.get("runtime_seconds"),
                        "status": "cached",
                        "mask_path": str(mask_png),
                    }
                )
                continue
            image, rgb = load_upright_rgb(Path(record["image_path"]))
            mask = load_mask_png(mask_png)
            detections = run_sliced(
                rgb,
                detector,
                mask,
                float(det_cfg["overlap"]),
                int(det_cfg["slice"]),
                int(det_cfg["batch_size"]),
            )
            counts = counts_from_detections(detections, count_threshold)
            payload = {
                "status": "complete",
                "created_utc": utc_now(),
                **record,
                "image_height": int(rgb.shape[0]),
                "image_width": int(rgb.shape[1]),
                "mask_foreground_pixels": int(mask.sum()),
                "runtime_seconds": round(time.monotonic() - started, 4),
                "headline_threshold": count_threshold,
                "counts_at_0_35": counts,
                "detections": detections,
                "mask_path": str(mask_png),
            }
            pred_json.parent.mkdir(parents=True, exist_ok=True)
            tmp = pred_json.with_suffix(".tmp.json")
            tmp.write_text(json.dumps(payload))
            tmp.replace(pred_json)
            if not args.no_overlays:
                write_overlay(rgb, mask, detections, overlay_path(output, stem), count_threshold)
            detection_rows.append(
                {
                    **record,
                    **counts,
                    "image_height": int(rgb.shape[0]),
                    "image_width": int(rgb.shape[1]),
                    "mask_foreground_pixels": int(mask.sum()),
                    "runtime_seconds": payload["runtime_seconds"],
                    "status": "complete",
                    "mask_path": str(mask_png),
                }
            )
            print(
                f"[{utc_now()}] detect {index}/{len(records)} {stem} "
                f"berries={counts['total_berries']} unripe={counts['unripe']} "
                f"ripe={counts['ripe']} flowers={counts['flowers']} "
                f"{payload['runtime_seconds']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            detection_rows.append(
                {
                    **record,
                    "unripe": 0,
                    "ripe": 0,
                    "flowers": 0,
                    "total_berries": 0,
                    "total_detections": 0,
                    "runtime_seconds": round(time.monotonic() - started, 4),
                    "status": f"error: {exc!r}",
                    "mask_path": str(mask_png),
                }
            )
            print(f"ERROR detect {stem}: {exc!r}", flush=True)
            traceback.print_exc()
        if args.excel_every > 0 and (
            index % args.excel_every == 0 or index == len(records)
        ):
            write_workbook(excel, detection_rows, architecture_rows, protocol)

    write_workbook(excel, detection_rows, architecture_rows, protocol)
    detection_csv.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(detection_rows).to_csv(detection_csv, index=False)
    if architecture_rows:
        pd.DataFrame(architecture_rows).to_csv(architecture_csv, index=False)
    elif not architecture_csv.is_file():
        pd.DataFrame([]).to_csv(architecture_csv, index=False)
    summary = {
        "finished_utc": utc_now(),
        "images": len(records),
        "excel": str(excel),
        "output": str(output),
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
