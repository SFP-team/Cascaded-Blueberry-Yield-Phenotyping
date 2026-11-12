#!/usr/bin/env python3
"""Extract image-derived plant phenotypes for every mask condition.

Traits come from ``traits.architecture.analyze_plant`` (repository folder ../../traits), applied
to the binary plant mask and the EXIF-upright RGB image. The raw (unmasked) condition is not
applicable: plant morphology needs a foreground definition.

Outputs (under --output):
  architecture/per_image/<stem>.json            traits of all mask conditions for one image
  tables/architecture_traits_per_image.csv      one row per (image, mask condition)
  tables/architecture_applicability.csv
  architecture_run_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import (
    CONDITIONS,
    MASK_CONDITIONS,
    REFERENCE_CONDITION,
    REPO_ROOT,
    add_input_arguments,
    load_manifest,
    load_mask,
    load_rgb,
    mask_path,
    parse_mask_dirs,
)

sys.path.insert(0, str(REPO_ROOT))
from traits.architecture import analyze_plant  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_arguments(parser)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=MASK_CONDITIONS,
        help="Recompute only these mask conditions while keeping other cached traits.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def valid_cache(path: Path) -> bool:
    try:
        obj = json.loads(path.read_text())
        return obj.get("status") == "complete" and len(obj.get("conditions", {})) == len(MASK_CONDITIONS)
    except Exception:
        return False


def analyze_one(
    row: dict[str, str],
    mask_dirs: dict[str, Path],
    output_root: Path,
    overwrite: bool,
    selected_conditions: tuple[str, ...],
) -> dict:
    stem = row["stem"]
    output_path = output_root / "architecture/per_image" / f"{stem}.json"
    partial_update = set(selected_conditions) != set(MASK_CONDITIONS)
    if not partial_update and not overwrite and valid_cache(output_path):
        return {"stem": stem, "status": "cached", "path": str(output_path)}

    existing: dict = {}
    if partial_update:
        if not valid_cache(output_path):
            raise ValueError(
                f"Partial condition update requires a complete existing cache: {output_path}"
            )
        existing = json.loads(output_path.read_text())

    started = time.monotonic()
    rgb = load_rgb(Path(row["image_path"]))
    base_meta: dict = {}
    metadata_path = Path(row["metadata_path"]) if row.get("metadata_path") else None
    if metadata_path and metadata_path.exists():
        try:
            base_meta = json.loads(metadata_path.read_text())
        except Exception:
            base_meta = {}
    seed_x = float(row.get("seed_x") or rgb.shape[1] / 2)
    seed_y = float(row.get("seed_y") or rgb.shape[0] - 1)
    conditions: dict[str, dict] = dict(existing.get("conditions", {}))
    for condition in selected_conditions:
        mpath = mask_path(mask_dirs, stem, condition)
        assert mpath is not None
        mask = load_mask(mpath, rgb.shape[:2])
        meta = dict(base_meta)
        meta.update(
            {
                "image_name": stem,
                "source_path": row["image_path"],
                "height": rgb.shape[0],
                "width": rgb.shape[1],
                "seed_xy": [seed_x, seed_y],
                "manually_edited": condition == REFERENCE_CONDITION,
                "no_plant": not bool(mask.any()),
                "mask_confidence": 1.0 if condition == REFERENCE_CONDITION else 0.0,
            }
        )
        traits = analyze_plant(mask, rgb, meta)
        conditions[condition] = {
            "condition_display": CONDITIONS[condition],
            "mask_path": str(mpath),
            "foreground_fraction": float(mask.mean()),
            "traits": traits,
        }
    payload = {
        "status": "complete",
        "created_utc": utc_now(),
        "stem": stem,
        "age_group": row["age_group"],
        "canopy_architecture": row["canopy_architecture"],
        "image_path": row["image_path"],
        "raw_unsegmented_architecture": {
            "status": "not_applicable",
            "reason": (
                "Plant-architecture morphology requires a foreground mask; treating the entire "
                "photograph as a plant would mix the target, neighbours, and background."
            ),
        },
        "runtime_seconds": float(existing.get("runtime_seconds", 0.0)) + time.monotonic() - started,
        "conditions": conditions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return {"stem": stem, "status": "complete", "path": str(output_path)}


def consolidate(rows: list[dict[str, str]], output_root: Path) -> Path:
    flattened: list[dict] = []
    for row in rows:
        path = output_root / "architecture/per_image" / f"{row['stem']}.json"
        if not valid_cache(path):
            continue
        obj = json.loads(path.read_text())
        for condition, item in obj["conditions"].items():
            flat = {
                "stem": obj["stem"],
                "age_group": obj["age_group"],
                "canopy_architecture": obj["canopy_architecture"],
                "condition": condition,
                "condition_display": item["condition_display"],
                "mask_path": item["mask_path"],
                "foreground_fraction": item["foreground_fraction"],
            }
            flat.update(item["traits"])
            flattened.append(flat)
    out = output_root / "tables/architecture_traits_per_image.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flattened).to_csv(out, index=False)
    return out


def main() -> int:
    args = parse_args()
    output_root: Path = args.output
    (output_root / "tables").mkdir(parents=True, exist_ok=True)
    mask_dirs = parse_mask_dirs(args.masks_root, args.mask_dir)
    selected_conditions = tuple(args.conditions or MASK_CONDITIONS)
    manifest = load_manifest(args.manifest)
    rows = manifest[args.start_index :]
    if args.limit is not None:
        rows = rows[: args.limit]

    applicability = [
        {
            "input_condition": "raw_unmasked",
            "architecture_applicable": False,
            "reason": "No foreground plant definition; whole-frame morphology is biologically invalid.",
        },
        *[
            {
                "input_condition": condition,
                "architecture_applicable": True,
                "reason": "Traits computed from the binary plant mask and aligned RGB pixels.",
            }
            for condition in MASK_CONDITIONS
        ],
    ]
    with (output_root / "tables/architecture_applicability.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=applicability[0].keys())
        writer.writeheader()
        writer.writerows(applicability)

    completed = cached = failed = 0
    failures: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(
                analyze_one, row, mask_dirs, output_root, args.overwrite, selected_conditions
            ): row
            for row in rows
        }
        for i, future in enumerate(as_completed(future_map), 1):
            row = future_map[future]
            try:
                result = future.result()
                if result["status"] == "cached":
                    cached += 1
                else:
                    completed += 1
                print(f"[{utc_now()}] {i}/{len(rows)} {row['stem']} {result['status']}", flush=True)
            except Exception as exc:
                failed += 1
                failures.append(
                    {
                        "stem": row["stem"],
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"ERROR {row['stem']}: {exc!r}", file=sys.stderr, flush=True)

    table_path = consolidate(manifest, output_root)
    summary = {
        "finished_utc": utc_now(),
        "images_requested": len(rows),
        "completed_this_run": completed,
        "cached": cached,
        "failed": failed,
        "recomputed_conditions": list(selected_conditions),
        "consolidated_table": str(table_path),
        "failures": failures,
    }
    (output_root / "architecture_run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
