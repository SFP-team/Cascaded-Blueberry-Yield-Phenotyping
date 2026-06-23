"""Download the blueberry ripe/unripe detection dataset from Roboflow.

The API key is read from the ROBOFLOW_API_KEY environment variable — it is NEVER hardcoded
or committed. Get the key from Roboflow (Settings -> API) and export it for the session:

    export ROBOFLOW_API_KEY=your_key_here
    .venv-yolo/bin/python download_data.py

Optional env overrides: ROBOFLOW_FORMAT (default "yolov11"), ROBOFLOW_VERSION (default 6).
The export format only affects the data.yaml/layout — the label files are standard YOLO txt
that any Ultralytics model (YOLO11, YOLO26, ...) trains on.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WORKSPACE = "smart-fruit-phenomics-team"
PROJECT = "blueberry_raghavapp"
DEST = Path(__file__).resolve().parent / "datasets"


def main() -> int:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("ERROR: set ROBOFLOW_API_KEY in your environment first (see this file's docstring).")

    fmt = os.environ.get("ROBOFLOW_FORMAT", "yolov11")
    version_num = int(os.environ.get("ROBOFLOW_VERSION", "6"))

    from roboflow import Roboflow

    DEST.mkdir(parents=True, exist_ok=True)
    location = DEST / f"{PROJECT}-v{version_num}"

    rf = Roboflow(api_key=key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(version_num)
    try:
        dataset = version.download(fmt, location=str(location))
    except Exception as e:
        # roboflow may not know a given format string on older installs — fall back to yolov11
        print(f"download('{fmt}') failed ({e}); retrying with 'yolov11'")
        dataset = version.download("yolov11", location=str(location))

    data_yaml = Path(dataset.location) / "data.yaml"
    print("\nDownloaded to:", dataset.location)
    print("data.yaml:    ", data_yaml)
    if data_yaml.exists():
        print("\n--- data.yaml ---")
        print(data_yaml.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
