#!/usr/bin/env python3
"""Evaluate staged seven-model masks against the Manual Mask references.

Reads ``<predictions-root>/predictions/<condition>/<category>/<stem>.mask.png``
(written by stage_test_predictions.py) and the test manifest, and writes:

  tables/per_image_metrics.csv      one row per image and model
  tables/summary_long.csv           mean/median/std and bootstrap 95% CI,
                                    overall and per manifest ``category`` (age group)
  tables/overall_metrics.csv        overall means, one row per model
  tables/metrics_by_age_long.csv    the per-category rows of summary_long.csv
  statistics/friedman_tests.csv
  statistics/pairwise_wilcoxon_holm.csv
  VALIDATION.json, EVALUATION_NOTE.md

To evaluate a subset of the test set, pass a manifest that contains only those
rows and a matching --subset-name and --expected-images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_LABELS = OrderedDict(
    [
        ("sam3_original", "SAM3"),
        ("sam21_finetuned", "SAM2.1"),
        ("birefnet_hr_finetuned", "BiRefNet-HR"),
        ("birefnet_hr_center_balanced", "Center BiRefNet"),
        ("segformer_b2_finetuned", "SegFormer-B2"),
        ("deeplabv3plus_r50_finetuned", "DeepLabV3+"),
        ("yolo26_finetuned", "YOLO26-Seg"),
    ]
)
EXTRA_METRICS = ("signed_area_error_percent",)
AGE_LABELS = {
    "Age_1_Year_W24": "1 year (W24)",
    "Age_2_Years_W23": "2 years (W23)",
    "Age_3_Years_W22": "3 years (W22)",
    "Age_Unspecified_W20_W21": "Unspecified (W20/W21)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--predictions-root",
        type=Path,
        required=True,
        help="Directory containing predictions/<condition>/... or the predictions directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--analysis-max-dimension", type=int, default=1024)
    parser.add_argument("--bootstrap-iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def find_prediction(
    root: Path, key: str, category: str, stem: str
) -> Path:
    if (root / "predictions").is_dir():
        root = root / "predictions"
    candidates = (
        root / key / category / f"{stem}.mask.png",
        root / key / f"{stem}.mask.png",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing {key} prediction for {stem}; checked "
        + ", ".join(str(path) for path in candidates)
    )


def bootstrap_summary(
    frame: pd.DataFrame,
    metric_names: list[str],
    iterations: int,
    seed: int,
    analysis: object,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    groupings = [("Overall", frame)]
    groupings.extend((str(group), part) for group, part in frame.groupby("category"))
    for group, group_frame in groupings:
        for (key, label), model_frame in group_frame.groupby(
            ["model_key", "model"], sort=False
        ):
            for metric in metric_names:
                values = model_frame[metric].to_numpy(dtype=float)
                low, high = analysis.bootstrap_ci(values, rng, iterations)
                records.append(
                    {
                        "group": group,
                        "group_label": AGE_LABELS.get(group, group),
                        "model_key": key,
                        "model": label,
                        "metric": metric,
                        "n": len(values),
                        "mean": float(values.mean()),
                        "median": float(np.median(values)),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "src"))
    from plant_benchmark import mask_metrics as analysis

    manifest = pd.read_csv(args.manifest)
    if len(manifest) != args.expected_images:
        raise ValueError(
            f"{args.subset_name}: expected {args.expected_images} images, "
            f"found {len(manifest)}"
        )
    if manifest["stem"].nunique() != len(manifest):
        raise ValueError("Test stems are not unique")

    records: list[dict[str, object]] = []
    prediction_counts = {key: 0 for key in MODEL_LABELS}
    for row in manifest.itertuples(index=False):
        reference_path = Path(row.reference_path)
        reference_full = analysis.load_binary(reference_path)
        small_size = analysis.analysis_size(
            int(row.width), int(row.height), args.analysis_max_dimension
        )
        reference_small = analysis.load_binary(reference_path, small_size)
        if reference_full.shape != (int(row.height), int(row.width)):
            raise ValueError(
                f"{reference_path}: shape={reference_full.shape}, expected="
                f"{(int(row.height), int(row.width))}"
            )
        for key, label in MODEL_LABELS.items():
            prediction_path = find_prediction(
                args.predictions_root, key, str(row.category), str(row.stem)
            )
            prediction_full = analysis.load_binary(
                prediction_path, (int(row.width), int(row.height))
            )
            prediction_small = analysis.load_binary(prediction_path, small_size)
            metrics = analysis.calculate_metrics(
                prediction_full,
                reference_full,
                prediction_small,
                reference_small,
            )
            target_area = int(metrics["tp"]) + int(metrics["fn"])
            predicted_area = int(metrics["tp"]) + int(metrics["fp"])
            metrics["signed_area_error_percent"] = (
                100.0 * (predicted_area - target_area) / max(target_area, 1)
            )
            records.append(
                {
                    "subset": args.subset_name,
                    "stem": str(row.stem),
                    "category": str(row.category),
                    "age": AGE_LABELS.get(str(row.category), str(row.category)),
                    "year_prefix": getattr(row, "year_prefix", ""),
                    "model_key": key,
                    "model": label,
                    "prediction_path": str(prediction_path.resolve()),
                    "reference_path": str(reference_path.resolve()),
                    **metrics,
                }
            )
            prediction_counts[key] += 1

    metrics = pd.DataFrame(records)
    metric_names = list(analysis.METRICS) + list(EXTRA_METRICS)
    numeric = metrics[metric_names].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        bad = np.argwhere(~np.isfinite(numeric))[:10].tolist()
        raise RuntimeError(f"Non-finite metrics at indexes {bad}")
    summary = bootstrap_summary(
        metrics,
        metric_names,
        args.bootstrap_iterations,
        args.seed,
        analysis,
    )
    friedman, pairs = analysis.statistical_tests(metrics, dict(MODEL_LABELS))
    friedman["group_label"] = friedman["group"].map(AGE_LABELS).fillna(
        friedman["group"]
    )
    pairs["group_label"] = pairs["group"].map(AGE_LABELS).fillna(pairs["group"])

    tables = args.output / "tables"
    statistics = args.output / "statistics"
    tables.mkdir(parents=True, exist_ok=True)
    statistics.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "per_image_metrics.csv", index=False)
    summary.to_csv(tables / "summary_long.csv", index=False)
    friedman.to_csv(statistics / "friedman_tests.csv", index=False)
    pairs.to_csv(statistics / "pairwise_wilcoxon_holm.csv", index=False)

    overall = summary[summary["group"] == "Overall"].pivot(
        index="model", columns="metric", values="mean"
    )
    overall = overall.reindex(list(MODEL_LABELS.values()))
    overall.to_csv(tables / "overall_metrics.csv")
    by_age = summary[summary["group"] != "Overall"].copy()
    by_age.to_csv(tables / "metrics_by_age_long.csv", index=False)

    expected_rows = len(manifest) * len(MODEL_LABELS)
    validation = {
        "status": "pass",
        "subset": args.subset_name,
        "images": len(manifest),
        "models": len(MODEL_LABELS),
        "metric_rows": len(metrics),
        "expected_metric_rows": expected_rows,
        "all_metrics_finite": True,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "predictions_root": str(args.predictions_root.resolve()),
        "prediction_counts": prediction_counts,
        "model_labels": dict(MODEL_LABELS),
        "analysis_max_dimension": args.analysis_max_dimension,
        "bootstrap_iterations": args.bootstrap_iterations,
    }
    if len(metrics) != expected_rows or any(
        count != len(manifest) for count in prediction_counts.values()
    ):
        validation["status"] = "fail"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "VALIDATION.json").write_text(json.dumps(validation, indent=2))
    note = [
        f"# {args.subset_name} evaluation note",
        "",
        f"This evaluation contains {len(manifest)} test images and "
        f"{len(MODEL_LABELS)} models.",
    ]
    (args.output / "EVALUATION_NOTE.md").write_text("\n".join(note) + "\n")
    if validation["status"] != "pass":
        raise RuntimeError(validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
