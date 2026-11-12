from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and resolve its paths.

    Relative paths are resolved against the folder above the config folder
    (``segmentation/benchmark`` for ``configs/fixed_split.yaml``).
    Environment variables such as ``${BLUEBERRY_SEG_DATA}`` are expanded.
    """

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    project_dir = config_path.parent.parent
    cfg["_project_dir"] = str(project_dir)
    for key in ("dataset", "manifest", "runs", "artifacts"):
        if key not in cfg["paths"]:
            continue
        value = Path(os.path.expandvars(str(cfg["paths"][key]))).expanduser()
        if not value.is_absolute():
            value = project_dir / value
        cfg["paths"][key] = str(value.resolve())
    return cfg


def run_dir(cfg: dict[str, Any], model: str, protocol: str, fold: int) -> Path:
    if cfg.get("data", {}).get("split_mode") == "fixed":
        return Path(cfg["paths"]["runs"]) / model / protocol / "fixed_split"
    return Path(cfg["paths"]["runs"]) / model / protocol / f"fold_{fold}"
