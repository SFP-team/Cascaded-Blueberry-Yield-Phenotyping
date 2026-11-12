#!/usr/bin/env python3
"""Compare detections and phenotypes of every mask condition against the Manual Mask.

Reads the outputs of detect_within_mask.py and extract_traits.py and writes per-image tables,
aggregates with 95% confidence intervals, paired Wilcoxon signed-rank tests with Holm
correction, per-trait Spearman and Pearson correlations with the Manual Mask, and figures.

Detection counts use confidence >= 0.35; 0.25 and 0.50 are kept as sensitivity thresholds.
"Manual agreement" compares the detections of a condition with the detections the same detector
produces on the Manual Mask input. It is an agreement proxy, not object-detection precision.

Outputs (under --output):
  tables/detection_per_image.csv, detection_overall.csv, detection_by_<group>.csv,
  tables/detection_paired_wilcoxon_holm.csv
  tables/architecture_error_per_image.csv, architecture_trait_errors_long.csv,
  tables/architecture_overall.csv, architecture_by_<group>.csv,
  tables/architecture_per_feature_summary.csv, architecture_pairwise_wilcoxon_holm.csv
  tables/segmentation_downstream_per_image.csv, segmentation_downstream_spearman.csv
      (only with --segmentation-metrics)
  figures/*.png
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from common import (
    CLASS_NAMES,
    CONDITIONS,
    GROUP_COLUMNS,
    MASK_CONDITIONS,
    MODEL_MASK_CONDITIONS,
    REFERENCE_CONDITION,
    add_input_arguments,
    load_manifest,
)


THRESHOLDS = (0.25, 0.35, 0.50)
HEADLINE_THRESHOLD = 0.35
CONDITION_ORDER = list(CONDITIONS)

KEY_ARCHITECTURE_TRAITS = (
    "foreground_fraction",
    "area_px",
    "canopy_height_px",
    "canopy_width_px",
    "perimeter_px",
    "solidity",
    "extent",
    "aspect_ratio_hw",
    "eccentricity",
    "circularity",
    "convexity",
    "top_base_width_ratio",
    "widest_height_frac",
    "lateral_lean",
    "top_heaviness",
    "asymmetry",
    "holes_area_frac",
    "gap_fraction",
    "frac_green",
    "edge_density",
)

ARCHITECTURE_DISPLAY_LABELS = {
    "foreground_fraction": "LDI Whole (%)",
    "area_px": "Canopy Area (px²)",
    "canopy_height_px": "Canopy Height (px)",
    "canopy_width_px": "Canopy Width (px)",
    "perimeter_px": "Canopy Perimeter (px)",
    "solidity": "LDI Hull/Solidity (%)",
    "extent": "LDI Bounded (%)",
    "convex_hull_area_px": "Hull Area (px²)",
    "gli_mean": "Mean GLI",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_arguments(parser, masks=False)
    parser.add_argument(
        "--segmentation-metrics",
        type=Path,
        help=(
            "Optional per-image segmentation metrics CSV with columns stem, condition and dice "
            "(condition keys as in common.CONDITIONS). Enables the Dice-versus-downstream "
            "Spearman correlations."
        ),
    )
    return parser.parse_args()


def finite(value, default=np.nan) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except Exception:
        return default


def safe_div(num: float, den: float, zero_value: float = 0.0) -> float:
    return float(num / den) if den else float(zero_value)


def box_iou_pairs(a: list[dict], b: list[dict], threshold: float = 0.5) -> int:
    """Greedy class-aware IoU matching from a single vectorised pair matrix per class."""
    matches = 0
    for class_id in CLASS_NAMES:
        aa = np.asarray([x["xyxy"] for x in a if int(x["category_id"]) == class_id], dtype=float)
        bb = np.asarray([x["xyxy"] for x in b if int(x["category_id"]) == class_id], dtype=float)
        if len(aa) == 0 or len(bb) == 0:
            continue
        ix1 = np.maximum(aa[:, None, 0], bb[None, :, 0])
        iy1 = np.maximum(aa[:, None, 1], bb[None, :, 1])
        ix2 = np.minimum(aa[:, None, 2], bb[None, :, 2])
        iy2 = np.minimum(aa[:, None, 3], bb[None, :, 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_a = np.maximum(0.0, aa[:, 2] - aa[:, 0]) * np.maximum(0.0, aa[:, 3] - aa[:, 1])
        area_b = np.maximum(0.0, bb[:, 2] - bb[:, 0]) * np.maximum(0.0, bb[:, 3] - bb[:, 1])
        iou = inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)
        candidates = np.argwhere(iou >= threshold)
        if not len(candidates):
            continue
        scores = iou[candidates[:, 0], candidates[:, 1]]
        used_a: set[int] = set()
        used_b: set[int] = set()
        for idx in np.argsort(scores)[::-1]:
            ai, bi = (int(v) for v in candidates[idx])
            if ai not in used_a and bi not in used_b:
                matches += 1
                used_a.add(ai)
                used_b.add(bi)
    return matches


def agreement(matches: int, predicted: int, reference: int) -> tuple[float, float, float]:
    if predicted == 0 and reference == 0:
        return 1.0, 1.0, 1.0
    precision = safe_div(matches, predicted)
    recall = safe_div(matches, reference)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def load_detection_tables(manifest: list[dict[str, str]], output_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for image_index, meta in enumerate(manifest, 1):
        payloads: dict[str, dict] = {}
        for condition in CONDITION_ORDER:
            path = output_root / "detection/predictions" / condition / f"{meta['stem']}.json"
            obj = json.loads(path.read_text())
            if obj.get("status") != "complete":
                raise RuntimeError(f"Incomplete detector record: {path}")
            payloads[condition] = obj

        for threshold in THRESHOLDS:
            selected = {
                condition: [
                    d for d in obj["detections"] if finite(d.get("confidence"), -1) >= threshold
                ]
                for condition, obj in payloads.items()
            }
            raw = selected["raw_unmasked"]
            manual = selected[REFERENCE_CONDITION]
            for condition in CONDITION_ORDER:
                detections = selected[condition]
                manual_matches = (
                    len(manual)
                    if condition == REFERENCE_CONDITION
                    else box_iou_pairs(detections, manual)
                )
                raw_matches = (
                    len(raw) if condition == "raw_unmasked" else box_iou_pairs(detections, raw)
                )
                ap, ar, af1 = agreement(manual_matches, len(detections), len(manual))
                rp, rr, rf1 = agreement(raw_matches, len(detections), len(raw))
                confidences = [finite(d["confidence"]) for d in detections]
                on_target = sum(bool(d.get("center_in_manual_reference")) for d in detections)
                record = {
                    "stem": meta["stem"],
                    **{column: meta[column] for column in GROUP_COLUMNS},
                    "condition": condition,
                    "condition_display": CONDITIONS[condition],
                    "threshold": threshold,
                    "detection_count": len(detections),
                    "immature_berry_count": sum(int(d["category_id"]) == 0 for d in detections),
                    "mature_berry_count": sum(int(d["category_id"]) == 1 for d in detections),
                    "flowers_count": sum(int(d["category_id"]) == 2 for d in detections),
                    "on_target_count": on_target,
                    "outside_target_count": len(detections) - on_target,
                    "on_target_rate": safe_div(on_target, len(detections), zero_value=1.0),
                    "mean_confidence": float(np.mean(confidences)) if confidences else np.nan,
                    "median_confidence": float(np.median(confidences)) if confidences else np.nan,
                    "manual_detection_count": len(manual),
                    "manual_count_difference": len(detections) - len(manual),
                    "manual_count_absolute_error": abs(len(detections) - len(manual)),
                    "manual_matches_iou50": manual_matches,
                    "manual_agreement_precision": ap,
                    "manual_agreement_recall": ar,
                    "manual_agreement_f1": af1,
                    "raw_detection_count": len(raw),
                    "raw_count_difference": len(detections) - len(raw),
                    "raw_matches_iou50": raw_matches,
                    "raw_retention": rr,
                    "raw_agreement_precision": rp,
                    "raw_agreement_f1": rf1,
                    "input_mask_fraction": finite(payloads[condition].get("input_mask_fraction")),
                    "manual_reference_fraction": finite(
                        payloads[condition].get("manual_reference_fraction")
                    ),
                    "runtime_seconds": finite(payloads[condition].get("runtime_seconds")),
                }
                rows.append(record)
        if image_index % 25 == 0:
            print(f"Detection aggregation: {image_index}/{len(manifest)}", flush=True)
    return pd.DataFrame(rows)


def mean_ci95(values: pd.Series) -> tuple[float, float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    mean = float(x.mean())
    if len(x) < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(x) - 1) * stats.sem(x))
    return mean, mean - half, mean + half


def aggregate_detection(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    out: list[dict] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["image_count"] = int(group["stem"].nunique())
        for field in (
            "detection_count",
            "immature_berry_count",
            "mature_berry_count",
            "flowers_count",
            "outside_target_count",
            "on_target_count",
        ):
            row[f"total_{field}"] = int(group[field].sum())
            row[f"mean_{field}"] = float(group[field].mean())
            row[f"median_{field}"] = float(group[field].median())
        row["weighted_on_target_rate"] = safe_div(
            group["on_target_count"].sum(), group["detection_count"].sum(), zero_value=1.0
        )
        for field in (
            "on_target_rate",
            "mean_confidence",
            "manual_count_absolute_error",
            "manual_agreement_precision",
            "manual_agreement_recall",
            "manual_agreement_f1",
            "raw_retention",
            "raw_agreement_f1",
            "runtime_seconds",
        ):
            mean, low, high = mean_ci95(group[field])
            row[f"mean_{field}"] = mean
            row[f"ci95_low_{field}"] = low
            row[f"ci95_high_{field}"] = high
        out.append(row)
    return pd.DataFrame(out)


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    n = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (n - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def detection_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon tests of each condition against the raw image and the Manual Mask."""
    data = frame[np.isclose(frame["threshold"], HEADLINE_THRESHOLD)]
    tests: list[dict] = []
    for baseline in ("raw_unmasked", REFERENCE_CONDITION):
        base = data[data["condition"] == baseline].set_index("stem")
        for condition in CONDITION_ORDER:
            if condition == baseline:
                continue
            comp = data[data["condition"] == condition].set_index("stem")
            for metric in (
                "detection_count",
                "on_target_count",
                "outside_target_count",
                "mean_confidence",
            ):
                pair = base[[metric]].join(comp[[metric]], lsuffix="_base", rsuffix="_test").dropna()
                delta = pair[f"{metric}_test"] - pair[f"{metric}_base"]
                try:
                    result = stats.wilcoxon(delta, zero_method="pratt", alternative="two-sided")
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                except ValueError:
                    statistic, p_value = 0.0, 1.0
                tests.append(
                    {
                        "baseline": baseline,
                        "condition": condition,
                        "metric": metric,
                        "n_pairs": len(pair),
                        "mean_delta": float(delta.mean()),
                        "median_delta": float(delta.median()),
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                    }
                )
    table = pd.DataFrame(tests)
    table["p_holm"] = holm_adjust(table["p_value"].tolist())
    table["significant_holm_0_05"] = table["p_holm"] < 0.05
    return table


def architecture_tables(
    output_root: Path, image_count: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    traits = pd.read_csv(output_root / "tables/architecture_traits_per_image.csv")
    expected = image_count * len(MASK_CONDITIONS)
    if len(traits) != expected:
        raise RuntimeError(f"Architecture rows {len(traits)} != expected {expected}")

    metadata = {
        "stem",
        *GROUP_COLUMNS,
        "condition",
        "condition_display",
        "mask_path",
        "no_plant",
        "manually_edited",
        "mask_confidence",
        "img_height",
        "img_width",
    }
    numeric = [c for c in traits.columns if c not in metadata and pd.api.types.is_numeric_dtype(traits[c])]
    references = traits[traits["condition"] == REFERENCE_CONDITION].set_index("stem")
    long_rows: list[dict] = []
    per_image_rows: list[dict] = []
    for _, row in traits.iterrows():
        ref = references.loc[row["stem"]]
        groups = {column: row[column] for column in GROUP_COLUMNS}
        feature_errors: list[tuple[str, float]] = []
        key_errors: list[float] = []
        for trait in numeric:
            predicted = finite(row[trait])
            reference = finite(ref[trait])
            if not (math.isfinite(predicted) and math.isfinite(reference)):
                continue
            absolute = abs(predicted - reference)
            smape = 200.0 * absolute / max(abs(predicted) + abs(reference), 1e-9)
            relative = 100.0 * absolute / max(abs(reference), 1e-9)
            feature_errors.append((trait, smape))
            if trait in KEY_ARCHITECTURE_TRAITS:
                key_errors.append(smape)
            long_rows.append(
                {
                    "stem": row["stem"],
                    **groups,
                    "condition": row["condition"],
                    "condition_display": row["condition_display"],
                    "trait": trait,
                    "predicted": predicted,
                    "manual_reference": reference,
                    "signed_error": predicted - reference,
                    "absolute_error": absolute,
                    "relative_absolute_error_pct": relative,
                    "smape_pct": smape,
                }
            )

        def standard_pct(trait: str) -> float:
            p, r = finite(row.get(trait)), finite(ref.get(trait))
            return 100.0 * abs(p - r) / max(abs(r), 1e-9)

        per_image_rows.append(
            {
                "stem": row["stem"],
                **groups,
                "condition": row["condition"],
                "condition_display": row["condition_display"],
                "key_trait_mean_smape_pct": float(np.mean(key_errors)) if key_errors else np.nan,
                "key_trait_median_smape_pct": float(np.median(key_errors)) if key_errors else np.nan,
                "all_trait_median_smape_pct": (
                    float(np.median([v for _, v in feature_errors])) if feature_errors else np.nan
                ),
                "area_absolute_error_pct": standard_pct("area_px"),
                "height_absolute_error_pct": standard_pct("canopy_height_px"),
                "width_absolute_error_pct": standard_pct("canopy_width_px"),
                "solidity_absolute_error": abs(finite(row.get("solidity")) - finite(ref.get("solidity"))),
                "aspect_ratio_absolute_error": abs(
                    finite(row.get("aspect_ratio_hw")) - finite(ref.get("aspect_ratio_hw"))
                ),
            }
        )
    return traits, pd.DataFrame(per_image_rows), pd.DataFrame(long_rows)


def aggregate_architecture(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    out: list[dict] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["image_count"] = int(group["stem"].nunique())
        for metric in (
            "key_trait_mean_smape_pct",
            "key_trait_median_smape_pct",
            "all_trait_median_smape_pct",
            "area_absolute_error_pct",
            "height_absolute_error_pct",
            "width_absolute_error_pct",
            "solidity_absolute_error",
            "aspect_ratio_absolute_error",
        ):
            mean, low, high = mean_ci95(group[metric])
            row[f"mean_{metric}"] = mean
            row[f"median_{metric}"] = float(group[metric].median())
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        out.append(row)
    return pd.DataFrame(out)


def architecture_feature_summary(long: pd.DataFrame) -> pd.DataFrame:
    """Per condition and trait: error versus the Manual Mask and Spearman/Pearson correlation."""
    rows: list[dict] = []
    for (condition, trait), group in long.groupby(["condition", "trait"], sort=False):
        x = group["predicted"].to_numpy(float)
        y = group["manual_reference"].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
            rho, rho_p = stats.spearmanr(x, y)
            pearson, pearson_p = stats.pearsonr(x, y)
        else:
            rho = rho_p = pearson = pearson_p = np.nan
        rows.append(
            {
                "condition": condition,
                "condition_display": CONDITIONS[condition],
                "trait": trait,
                "image_count": len(x),
                "mae": float(group["absolute_error"].mean()),
                "rmse": float(np.sqrt(np.mean(np.square(group["signed_error"])))),
                "mean_bias": float(group["signed_error"].mean()),
                "median_smape_pct": float(group["smape_pct"].median()),
                "mean_smape_pct": float(group["smape_pct"].mean()),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
                "pearson_r": float(pearson),
                "pearson_p": float(pearson_p),
            }
        )
    return pd.DataFrame(rows)


def architecture_statistics(per_image: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wilcoxon tests between automatic masks on the per-image key-trait error."""
    model_data = per_image[per_image["condition"].isin(MODEL_MASK_CONDITIONS)]
    tests: list[dict] = []
    conditions = list(MODEL_MASK_CONDITIONS)
    for i, first in enumerate(conditions):
        a = model_data[model_data["condition"] == first].set_index("stem")
        for second in conditions[i + 1 :]:
            b = model_data[model_data["condition"] == second].set_index("stem")
            pair = a[["key_trait_median_smape_pct"]].join(
                b[["key_trait_median_smape_pct"]], lsuffix="_a", rsuffix="_b"
            ).dropna()
            delta = pair.iloc[:, 0] - pair.iloc[:, 1]
            try:
                result = stats.wilcoxon(delta, zero_method="pratt")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            tests.append(
                {
                    "condition_a": first,
                    "condition_b": second,
                    "n_pairs": len(pair),
                    "mean_error_a": float(pair.iloc[:, 0].mean()),
                    "mean_error_b": float(pair.iloc[:, 1].mean()),
                    "mean_delta_a_minus_b": float(delta.mean()),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
    table = pd.DataFrame(tests)
    table["p_holm"] = holm_adjust(table["p_value"].tolist())
    table["significant_holm_0_05"] = table["p_holm"] < 0.05
    return table


def downstream_correlations(
    detection: pd.DataFrame, architecture: pd.DataFrame, segmentation_metrics: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman correlation of per-image Dice with detection and phenotype agreement."""
    det = detection[
        np.isclose(detection["threshold"], HEADLINE_THRESHOLD)
        & detection["condition"].isin(MODEL_MASK_CONDITIONS)
    ][
        [
            "stem",
            "condition",
            "manual_agreement_f1",
            "manual_count_absolute_error",
            "on_target_rate",
            "outside_target_count",
        ]
    ]
    arch = architecture[architecture["condition"].isin(MODEL_MASK_CONDITIONS)][
        ["stem", "condition", "key_trait_median_smape_pct"]
    ]
    seg = pd.read_csv(segmentation_metrics)
    merged = seg.merge(det, on=["stem", "condition"], how="inner").merge(
        arch, on=["stem", "condition"], how="inner"
    )
    rows: list[dict] = []
    for condition, group in list(merged.groupby("condition")) + [("ALL_MODELS_POOLED", merged)]:
        for outcome in (
            "manual_agreement_f1",
            "manual_count_absolute_error",
            "on_target_rate",
            "outside_target_count",
            "key_trait_median_smape_pct",
        ):
            clean = group[["dice", outcome]].dropna()
            if len(clean) >= 3 and clean["dice"].std() > 0 and clean[outcome].std() > 0:
                rho, p_value = stats.spearmanr(clean["dice"], clean[outcome])
            else:
                rho = p_value = np.nan
            rows.append(
                {
                    "condition": condition,
                    "outcome": outcome,
                    "n": len(clean),
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                }
            )
    return merged, pd.DataFrame(rows)


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "condition" in out:
        out["condition"] = pd.Categorical(out["condition"], CONDITION_ORDER, ordered=True)
        out = out.sort_values("condition", kind="stable")
        out["condition"] = out["condition"].astype(str)
    return out


def make_figures(
    output_root: Path,
    detection_overall: pd.DataFrame,
    detection_by_group: dict[str, pd.DataFrame],
    architecture: pd.DataFrame,
    feature_summary: pd.DataFrame,
    downstream: pd.DataFrame | None,
) -> list[Path]:
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    outputs: list[Path] = []
    head = ordered(detection_overall[np.isclose(detection_overall["threshold"], HEADLINE_THRESHOLD)])
    labels = [CONDITIONS[x] for x in head["condition"]]

    path = figures / "detection_class_counts_conf035.png"
    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(head))
    for field, label, color in (
        ("total_immature_berry_count", "Immature berries", "#e6ad25"),
        ("total_mature_berry_count", "Mature berries", "#7a3db8"),
        ("total_flowers_count", "Flowers", "#50a65d"),
    ):
        vals = head[field].to_numpy(float)
        ax.bar(labels, vals, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_ylabel("Total detections (confidence ≥ 0.35)")
    ax.set_title("Detection counts under raw and mask-restricted inputs")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)

    path = figures / "detection_target_attribution_conf035.png"
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    axes[0].bar(labels, 100 * head["weighted_on_target_rate"], color="#2b8cbe")
    axes[0].set_ylabel("On-target detections (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Bounding-box centers inside the Manual Mask")
    axes[1].bar(labels, head["mean_outside_target_count"], color="#d95f5f")
    axes[1].set_ylabel("Mean outside-target detections per image")
    axes[1].set_title("Neighbour/background attribution proxy")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)

    path = figures / "detection_manual_agreement_conf035.png"
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(labels, head["mean_manual_agreement_f1"], color="#4c78a8")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Box-set agreement F1")
    ax.set_title("Agreement with detections on the Manual Mask input (proxy, not precision)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)

    for column, table in detection_by_group.items():
        part = table[np.isclose(table["threshold"], HEADLINE_THRESHOLD)]
        if part[column].nunique() < 2:
            continue
        path = figures / f"detection_by_{column}_conf035.png"
        pivot = part.pivot(index="condition", columns=column, values="mean_detection_count").reindex(CONDITION_ORDER)
        pivot.index = [CONDITIONS[x] for x in pivot.index]
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
        ax.set_title(f"Mean detections per image by {column} (confidence ≥ 0.35)")
        ax.set_xlabel(column)
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        outputs.append(path)

    path = figures / "architecture_error_overall.png"
    arch = architecture[architecture["condition"].isin(MODEL_MASK_CONDITIONS)].copy()
    order = sorted(
        MODEL_MASK_CONDITIONS,
        key=lambda x: arch.loc[arch["condition"] == x, "key_trait_median_smape_pct"].mean(),
    )
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(
        data=arch,
        x="condition",
        y="key_trait_median_smape_pct",
        order=order,
        errorbar=("ci", 95),
        color="#59a14f",
        ax=ax,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CONDITIONS[x] for x in order], rotation=30, ha="right")
    ax.set_ylabel("Median key-trait sMAPE (%), lower is better")
    ax.set_xlabel("")
    ax.set_title("Phenotype error relative to the Manual Mask")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)

    path = figures / "architecture_key_trait_heatmap.png"
    fs = feature_summary[
        feature_summary["condition"].isin(MODEL_MASK_CONDITIONS)
        & feature_summary["trait"].isin(KEY_ARCHITECTURE_TRAITS)
    ]
    pivot = fs.pivot(index="condition", columns="trait", values="median_smape_pct").reindex(MODEL_MASK_CONDITIONS)
    pivot.index = [CONDITIONS[x] for x in pivot.index]
    pivot = pivot.rename(columns=ARCHITECTURE_DISPLAY_LABELS)
    fig, ax = plt.subplots(figsize=(20, 7))
    sns.heatmap(pivot, cmap="mako_r", annot=False, ax=ax)
    ax.set_title("Median phenotype error versus the Manual Mask (%)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)

    if downstream is not None:
        path = figures / "segmentation_dice_vs_downstream.png"
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        sns.scatterplot(
            data=downstream,
            x="dice",
            y="manual_agreement_f1",
            hue="condition",
            alpha=0.55,
            s=32,
            legend=False,
            ax=axes[0],
        )
        axes[0].set_title("Segmentation Dice vs Manual Mask detection agreement")
        sns.scatterplot(
            data=downstream,
            x="dice",
            y="key_trait_median_smape_pct",
            hue="condition",
            alpha=0.55,
            s=32,
            ax=axes[1],
        )
        axes[1].set_title("Segmentation Dice vs phenotype error")
        axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs


def write_csv(frame: pd.DataFrame, output_root: Path, name: str) -> Path:
    path = output_root / "tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    output_root: Path = args.output
    manifest = load_manifest(args.manifest)

    detection = ordered(load_detection_tables(manifest, output_root))
    write_csv(detection, output_root, "detection_per_image.csv")
    detection_overall = aggregate_detection(
        detection, ["condition", "condition_display", "threshold"]
    )
    write_csv(ordered(detection_overall), output_root, "detection_overall.csv")
    detection_by_group: dict[str, pd.DataFrame] = {}
    for column in GROUP_COLUMNS:
        table = aggregate_detection(
            detection, [column, "condition", "condition_display", "threshold"]
        )
        detection_by_group[column] = table
        write_csv(table, output_root, f"detection_by_{column}.csv")
    write_csv(detection_statistics(detection), output_root, "detection_paired_wilcoxon_holm.csv")

    _, architecture_per_image, architecture_long = architecture_tables(output_root, len(manifest))
    write_csv(architecture_per_image, output_root, "architecture_error_per_image.csv")
    write_csv(architecture_long, output_root, "architecture_trait_errors_long.csv")
    architecture_overall = aggregate_architecture(
        architecture_per_image, ["condition", "condition_display"]
    )
    write_csv(ordered(architecture_overall), output_root, "architecture_overall.csv")
    for column in GROUP_COLUMNS:
        write_csv(
            aggregate_architecture(architecture_per_image, [column, "condition", "condition_display"]),
            output_root,
            f"architecture_by_{column}.csv",
        )
    feature_summary = architecture_feature_summary(architecture_long)
    write_csv(feature_summary, output_root, "architecture_per_feature_summary.csv")
    write_csv(
        architecture_statistics(architecture_per_image),
        output_root,
        "architecture_pairwise_wilcoxon_holm.csv",
    )

    downstream = None
    if args.segmentation_metrics is not None:
        downstream, correlations = downstream_correlations(
            detection, architecture_per_image, args.segmentation_metrics
        )
        write_csv(downstream, output_root, "segmentation_downstream_per_image.csv")
        write_csv(correlations, output_root, "segmentation_downstream_spearman.csv")

    figures = make_figures(
        output_root,
        detection_overall,
        detection_by_group,
        architecture_per_image,
        feature_summary,
        downstream,
    )
    print(json.dumps({"status": "complete", "figures": [str(p) for p in figures]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
