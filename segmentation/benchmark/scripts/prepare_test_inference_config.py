#!/usr/bin/env python3
"""Build a validation-calibration + test manifest and its inference config.

The derived manifest contains

  * every validation record of the training manifest (stem prefixed
    ``valcal__``), used only to select the probability threshold,
  * one training record (prefixed ``traincal__``), required by the YOLO26
    exporter, and
  * every test record (split ``test``; ``reference_path`` becomes ``mask_path``).

The derived config is the training config with ``paths.manifest`` replaced and,
optionally, ``paths.runs`` / ``paths.artifacts`` redirected. Running
``plant_benchmark.cli predict`` (and ``train_center_birefnet.py predict``) with
it selects the threshold on validation records only and then writes binary
masks for the test images. Test reference masks are never used for threshold
or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train-val-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runs-root",
        type=Path,
        help="Checkpoint/prediction root for this inference run "
        "(default: the base config's runs root, i.e. the training checkpoints).",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        help="Preprocessing-artifact root for this inference run. Use a separate "
        "folder so the YOLO26 training export is not replaced.",
    )
    return parser.parse_args()


def prepare_inference_config(
    train_val_manifest: Path,
    test_manifest: Path,
    base_config: Path,
    output: Path,
    runs_root: Path | None = None,
    artifacts_root: Path | None = None,
) -> dict:
    train_val_manifest = train_val_manifest.expanduser().resolve()
    test_manifest = test_manifest.expanduser().resolve()
    base_config = base_config.expanduser().resolve()
    output = output.expanduser().resolve()
    training = pd.read_csv(train_val_manifest)
    train = training[
        training["split"].astype(str).str.strip().str.lower().eq("train")
    ].head(1).copy()
    validation = training[
        training["split"].astype(str).str.strip().str.lower().eq("val")
    ].copy()
    if train.empty:
        raise ValueError("Training manifest contains no training images")
    if validation.empty:
        raise ValueError("Training manifest contains no validation images")
    # Prefix calibration-only records so a same-name test photograph cannot
    # collide in threshold-selection lookups or prediction filenames.
    validation["source_stem"] = validation["stem"].astype(str)
    validation["stem"] = "valcal__" + validation["stem"].astype(str)
    validation["split"] = "val"
    # YOLO's fixed-split exporter requires a training partition even though
    # inference uses only validation calibration and test records. Prefixing
    # these rows keeps all three partitions collision-free in the derived view.
    train["source_stem"] = train["stem"].astype(str)
    train["stem"] = "traincal__" + train["stem"].astype(str)
    train["split"] = "train"

    test = pd.read_csv(test_manifest).copy()
    test["mask_path"] = test["reference_path"]
    if "reference_sha256" not in test:
        raise ValueError("The test manifest requires reference_sha256")
    test["mask_sha256"] = test["reference_sha256"]
    test["manually_edited"] = True
    test["split"] = "test"
    test["source_stem"] = test["stem"].astype(str)
    if "seed_x" not in test or "seed_y" not in test:
        raise ValueError("The test manifest requires seed_x and seed_y")
    combined = pd.concat([train, validation, test], ignore_index=True, sort=False)
    if combined["stem"].duplicated().any():
        raise RuntimeError("Combined inference manifest stems are not unique")

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "val_calibration_plus_test.csv"
    combined.to_csv(manifest_path, index=False)
    config = yaml.safe_load(base_config.read_text())
    base_project = base_config.resolve().parent.parent
    for key, value in config.get("paths", {}).items():
        candidate = Path(os.path.expandvars(str(value))).expanduser()
        if not candidate.is_absolute():
            candidate = base_project / candidate
        config["paths"][key] = str(candidate.resolve())
    config.setdefault("data", {})["split_mode"] = "fixed"
    config["data"]["folds"] = 1
    config.setdefault("paths", {})["manifest"] = str(manifest_path.resolve())
    if runs_root is not None:
        config["paths"]["runs"] = str(runs_root.expanduser().resolve())
    if artifacts_root is not None:
        config["paths"]["artifacts"] = str(artifacts_root.expanduser().resolve())
    config_path = output / "test_inference_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    payload = {
        "status": "pass",
        "training_export_images": len(train),
        "validation_calibration_images": len(validation),
        "test_images": len(test),
        "combined_rows": len(combined),
        "manifest": str(manifest_path.resolve()),
        "config": str(config_path.resolve()),
        "runs_root": config["paths"].get("runs"),
        "artifacts_root": config["paths"].get("artifacts"),
        "threshold_policy": (
            "Select the hard-mask threshold once on the frozen validation records; "
            "never use a test reference mask for threshold selection."
        ),
    }
    (output / "TEST_INFERENCE_PROVENANCE.json").write_text(
        json.dumps(payload, indent=2)
    )
    return payload


def main() -> None:
    args = parse_args()
    payload = prepare_inference_config(
        args.train_val_manifest,
        args.test_manifest,
        args.base_config,
        args.output,
        args.runs_root,
        args.artifacts_root,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
