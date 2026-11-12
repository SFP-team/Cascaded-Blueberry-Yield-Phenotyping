#!/usr/bin/env python3
"""Merge per-GPU shard Excel/CSV files into the four-sheet workbook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.workbook import utc_now, write_workbook


def load_rows(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    if frame.empty:
        return []
    return frame.to_dict(orient="records")


def merge_output(output: Path, shard_count: int) -> Path:
    output = output.expanduser().resolve()
    detection_rows: list[dict] = []
    architecture_rows: list[dict] = []
    protocols: list[dict] = []
    tables = output / "tables"
    for index in range(shard_count):
        detection_rows.extend(load_rows(tables / f"detections_per_image.shard{index}.csv"))
        architecture_rows.extend(load_rows(tables / f"architecture_per_image.shard{index}.csv"))
        protocol_path = output / f"protocol.shard{index}.json"
        if protocol_path.is_file():
            protocols.append(json.loads(protocol_path.read_text()))

    seen: set[str] = set()
    unique_detections: list[dict] = []
    for row in detection_rows:
        stem = str(row.get("stem", ""))
        if stem in seen:
            continue
        seen.add(stem)
        unique_detections.append(row)

    seen_arch: set[str] = set()
    unique_architecture: list[dict] = []
    for row in architecture_rows:
        stem = str(row.get("stem", ""))
        if stem in seen_arch:
            continue
        seen_arch.add(stem)
        unique_architecture.append(row)

    protocol = dict(protocols[0]) if protocols else {}
    source = str(output)
    if protocols and " from " in str(protocols[0].get("cohort", "")):
        source = str(protocols[0]["cohort"]).split(" from ", 1)[1]
    protocol["n_images"] = len(unique_detections)
    protocol["cohort"] = f"{len(unique_detections)} images from {source}"
    protocol["device"] = " + ".join(
        str(item.get("device", "")) for item in protocols if item.get("device")
    ) or protocol.get("device", "")
    protocol["shard_count"] = shard_count
    protocol["merged_utc"] = utc_now()
    protocol.pop("shard_index", None)

    excel = output / "Plant_Architecture_and_Detections.xlsx"
    write_workbook(excel, unique_detections, unique_architecture, protocol)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    tables.mkdir(exist_ok=True)
    pd.DataFrame(unique_detections).to_csv(tables / "detections_per_image.csv", index=False)
    pd.DataFrame(unique_architecture).to_csv(tables / "architecture_per_image.csv", index=False)
    (output / "run_summary.json").write_text(
        json.dumps(
            {
                "finished_utc": utc_now(),
                "images": len(unique_detections),
                "excel": str(excel),
                "output": str(output),
                "shard_count": shard_count,
            },
            indent=2,
        )
    )
    print(f"merged {len(unique_detections)} images -> {excel}", flush=True)
    return excel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge the per-GPU shard tables of one output folder into one Excel workbook."
    )
    parser.add_argument("--output", required=True, help="Output folder shared by all shards")
    parser.add_argument("--shard-count", type=int, required=True, help="Number of shards that were run")
    args = parser.parse_args()
    if args.shard_count < 1:
        raise ValueError("shard-count must be >= 1")
    merge_output(Path(args.output), args.shard_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
