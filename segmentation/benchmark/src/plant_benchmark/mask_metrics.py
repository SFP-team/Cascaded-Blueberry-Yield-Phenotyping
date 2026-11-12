"""Per-image mask metrics, bootstrap intervals and paired tests.

Region metrics use full-resolution masks. Boundary, distance, shape and
S-measure metrics use masks resized (nearest neighbour) so that the longer
side is at most ``analysis_max_dimension`` pixels; HD95 and similar distances
are normalised by that analysis image diagonal.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, stats


METRICS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "f_beta_beta2_0_3",
    "f2",
    "mcc",
    "boundary_f1",
    "boundary_iou",
    "hausdorff_normalized",
    "hd95_normalized",
    "hd95_analysis_px",
    "hd95_percent_diagonal",
    "assd_normalized",
    "structure_measure",
    "relative_area_error",
    "relative_area_error_percent",
    "centroid_error_normalized",
    "dice_loss",
    "jaccard_loss",
    "tversky_loss",
    "boundary_error",
]


def load_binary(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("L")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        return np.asarray(image) > 127


def analysis_size(width: int, height: int, maximum: int) -> tuple[int, int]:
    scale = min(1.0, maximum / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def boundary(mask: np.ndarray) -> np.ndarray:
    return mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool))


def boundary_band(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Return the inward boundary region used by the official Boundary-IoU API."""
    dilation = max(1, int(round(dilation_ratio * math.hypot(*mask.shape))))
    eroded = ndimage.binary_erosion(
        mask,
        structure=np.ones((3, 3), bool),
        iterations=dilation,
        border_value=0,
    )
    return mask & ~eroded


def boundary_metrics(
    pred: np.ndarray, target: np.ndarray, tolerance: int
) -> tuple[float, float, float, float, float, float]:
    pb = boundary(pred)
    tb = boundary(target)
    diagonal = math.hypot(*pred.shape)
    if not pb.any() and not tb.any():
        return 1.0, 1.0, 0.0, 0.0, 0.0, 0.0
    if not pb.any() or not tb.any():
        return 0.0, 0.0, 1.0, 1.0, diagonal, 1.0
    pdist = ndimage.distance_transform_edt(~pb)
    tdist = ndimage.distance_transform_edt(~tb)
    pred_to_true = tdist[pb]
    true_to_pred = pdist[tb]
    precision = float(np.mean(pred_to_true <= tolerance))
    recall = float(np.mean(true_to_pred <= tolerance))
    bf1 = 2 * precision * recall / max(precision + recall, 1e-12)
    pred_band = boundary_band(pred)
    target_band = boundary_band(target)
    band_union = np.count_nonzero(pred_band | target_band)
    boundary_iou = (
        float(np.count_nonzero(pred_band & target_band) / band_union)
        if band_union
        else 1.0
    )
    hausdorff_px = max(float(pred_to_true.max()), float(true_to_pred.max()))
    hd95_px = max(
        float(np.quantile(pred_to_true, 0.95)),
        float(np.quantile(true_to_pred, 0.95)),
    )
    assd_px = 0.5 * (float(pred_to_true.mean()) + float(true_to_pred.mean()))
    return (
        bf1,
        boundary_iou,
        hausdorff_px / diagonal,
        hd95_px / diagonal,
        hd95_px,
        assd_px / diagonal,
    )


def _s_object(values: np.ndarray) -> float:
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return 2.0 * mean / (mean * mean + 1.0 + std + 1e-12)


def _region_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    x = float(pred.mean())
    y = float(target.mean())
    count = pred.size
    denom = max(count - 1, 1)
    sigma_x = float(np.sum((pred - x) ** 2) / denom)
    sigma_y = float(np.sum((target - y) ** 2) / denom)
    sigma_xy = float(np.sum((pred - x) * (target - y)) / denom)
    alpha = 4.0 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x + sigma_y)
    if abs(alpha) > 1e-12:
        return alpha / (beta + 1e-12)
    return 1.0 if abs(beta) <= 1e-12 else 0.0


def structure_measure(
    pred: np.ndarray, target: np.ndarray, alpha: float = 0.5
) -> float:
    """Fan et al. S-measure, evaluated uniformly on final binary masks."""
    prediction = pred.astype(np.float64)
    truth = target.astype(bool)
    foreground_fraction = float(truth.mean())
    if foreground_fraction == 0.0:
        return float(1.0 - prediction.mean())
    if foreground_fraction == 1.0:
        return float(prediction.mean())

    object_score = (
        foreground_fraction * _s_object(prediction[truth])
        + (1.0 - foreground_fraction) * _s_object((1.0 - prediction)[~truth])
    )

    height, width = truth.shape
    center_y, center_x = np.argwhere(truth).mean(axis=0).round().astype(int) + 1
    center_y = min(max(int(center_y), 1), height)
    center_x = min(max(int(center_x), 1), width)
    regions = (
        (slice(0, center_y), slice(0, center_x)),
        (slice(0, center_y), slice(center_x, width)),
        (slice(center_y, height), slice(0, center_x)),
        (slice(center_y, height), slice(center_x, width)),
    )
    region_score = 0.0
    total_area = float(height * width)
    for rows, columns in regions:
        current = prediction[rows, columns]
        weight = current.size / total_area
        if weight:
            region_score += weight * _region_ssim(current, truth[rows, columns])
    return float(max(0.0, min(1.0, alpha * object_score + (1.0 - alpha) * region_score)))


def centroid(mask: np.ndarray) -> tuple[float, float]:
    indexes = np.argwhere(mask)
    if not len(indexes):
        return float("nan"), float("nan")
    return float(indexes[:, 0].mean()), float(indexes[:, 1].mean())


def calculate_metrics(
    pred_full: np.ndarray,
    target_full: np.ndarray,
    pred_small: np.ndarray,
    target_small: np.ndarray,
) -> dict[str, float | int]:
    tp = int(np.count_nonzero(pred_full & target_full))
    fp = int(np.count_nonzero(pred_full & ~target_full))
    fn = int(np.count_nonzero(~pred_full & target_full))
    tn = int(np.count_nonzero(~pred_full & ~target_full))
    eps = 1e-12
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    beta_squared = 0.3
    f_beta_beta2_0_3 = (
        (1 + beta_squared)
        * precision
        * recall
        / (beta_squared * precision + recall + eps)
    )
    f2 = 5 * precision * recall / (4 * precision + recall + eps)
    mcc_denom = math.sqrt(
        max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0)
    )
    mcc = (tp * tn - fp * fn) / (mcc_denom + eps)
    bf1, biou, hd, hd95, hd95_px, assd = boundary_metrics(
        pred_small, target_small, tolerance=3
    )
    target_area = tp + fn
    pred_area = tp + fp
    relative_area_error = abs(pred_area - target_area) / (target_area + eps)
    pc = centroid(pred_small)
    tc = centroid(target_small)
    if any(math.isnan(v) for v in (*pc, *tc)):
        centroid_error = 1.0
    else:
        centroid_error = math.hypot(pc[0] - tc[0], pc[1] - tc[1]) / math.hypot(
            *pred_small.shape
        )
    components = int(ndimage.label(pred_small)[1])
    alpha, beta = 0.3, 0.7
    tversky = tp / (tp + alpha * fp + beta * fn + eps)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "f_beta_beta2_0_3": f_beta_beta2_0_3,
        "f2": f2,
        "mcc": mcc,
        "boundary_f1": bf1,
        "boundary_iou": biou,
        "hausdorff_normalized": hd,
        "hd95_normalized": hd95,
        "hd95_analysis_px": hd95_px,
        "hd95_percent_diagonal": 100.0 * hd95,
        "assd_normalized": assd,
        "structure_measure": structure_measure(pred_small, target_small),
        "relative_area_error": relative_area_error,
        "relative_area_error_percent": 100.0 * relative_area_error,
        "centroid_error_normalized": centroid_error,
        "predicted_components": components,
        "dice_loss": 1 - dice,
        "jaccard_loss": 1 - iou,
        "tversky_loss": 1 - tversky,
        "boundary_error": 1 - bf1,
    }


def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, iterations: int
) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indexes = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[indexes].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def statistical_tests(
    metrics: pd.DataFrame, models: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Friedman test per group/metric and Holm-adjusted paired Wilcoxon tests.

    ``models`` maps condition keys to display labels, in table order.
    """
    friedman_rows = []
    pair_rows = []
    for group in ["Overall", *sorted(metrics.category.unique())]:
        subset = metrics if group == "Overall" else metrics[metrics.category == group]
        for metric in (
            "dice",
            "iou",
            "f2",
            "f_beta_beta2_0_3",
            "boundary_f1",
            "boundary_iou",
            "hd95_normalized",
            "structure_measure",
        ):
            pivot = subset.pivot(index="stem", columns="model_key", values=metric)
            pivot = pivot[list(models)]
            statistic, pvalue = stats.friedmanchisquare(
                *[pivot[column].to_numpy() for column in pivot]
            )
            friedman_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "n": len(pivot),
                    "statistic": statistic,
                    "p_value": pvalue,
                }
            )
            current = []
            keys = list(models)
            for i, left in enumerate(keys):
                for right in keys[i + 1 :]:
                    delta = pivot[left] - pivot[right]
                    try:
                        test = stats.wilcoxon(delta, zero_method="zsplit")
                        p = float(test.pvalue)
                        statistic_w = float(test.statistic)
                    except ValueError:
                        p, statistic_w = 1.0, 0.0
                    current.append(
                        {
                            "group": group,
                            "metric": metric,
                            "model_a": models[left],
                            "model_b": models[right],
                            "mean_delta_a_minus_b": float(delta.mean()),
                            "median_delta_a_minus_b": float(np.median(delta)),
                            "wilcoxon_statistic": statistic_w,
                            "p_value": p,
                        }
                    )
            adjusted = holm_adjust([row["p_value"] for row in current])
            for row, value in zip(current, adjusted):
                row["p_holm"] = value
                row["significant_0_05"] = value < 0.05
            pair_rows.extend(current)
    return pd.DataFrame(friedman_rows), pd.DataFrame(pair_rows)
