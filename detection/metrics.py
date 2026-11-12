"""Detection metrics on whole photos: COCO-style AP, P/R/F1 and count errors at a threshold, bootstrap intervals.

AP follows pycocotools (COCOeval, area "all"): per class and photo, detections in score order take the unmatched
ground-truth box with the highest IoU at or above the threshold; precision is made monotone and read at 101 recall
points. evaluate.py score --check compares these numbers with pycocotools itself.
"""

from __future__ import annotations

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.951, 0.05), 2)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)
SIZE_BINS = ((0, 16), (16, 32), (32, 64), (64, 1e9))  # sqrt(area) in pixels


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-12)).astype(np.float32)


class Matches:
    """Greedy COCO matching of one photo's detections of one class, at every IoU threshold."""

    def __init__(self, det_boxes: np.ndarray, det_scores: np.ndarray, gt_boxes: np.ndarray, max_dets: int = 5000):
        order = np.argsort(-det_scores, kind="mergesort")[:max_dets]
        self.scores = det_scores[order]
        self.n_gt = len(gt_boxes)
        self.gt_size = np.sqrt(np.clip((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]), 0, None))
        self.matched = np.zeros((len(IOU_THRESHOLDS), len(order)), bool)
        self.gt_of_det = np.full(len(order), -1)  # ground-truth index matched at IoU 0.5, -1 when none
        ious = box_iou(det_boxes[order], gt_boxes)
        candidates = []
        for row in ious:
            idx = np.nonzero(row >= IOU_THRESHOLDS[0])[0]
            idx = idx[np.argsort(-row[idx], kind="mergesort")]
            candidates.append((idx, row[idx]))
        for ti, threshold in enumerate(IOU_THRESHOLDS):
            taken = np.zeros(self.n_gt, bool)
            for d, (idx, values) in enumerate(candidates):
                for j, value in zip(idx, values):
                    if value < threshold:
                        break
                    if not taken[j]:
                        taken[j] = True
                        self.matched[ti, d] = True
                        if ti == 0:
                            self.gt_of_det[d] = j
                        break


def average_precision(scores: np.ndarray, matched: np.ndarray, n_gt: int) -> float | None:
    if n_gt == 0:
        return None
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    hits = matched[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(~hits)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    index = np.searchsorted(recall, RECALL_POINTS, side="left")
    sampled = np.zeros(len(RECALL_POINTS))
    inside = index < len(precision)
    sampled[inside] = precision[index[inside]]
    return float(sampled.mean())


def class_ap(per_image: list[Matches], threshold_index: int = 0) -> float | None:
    n_gt = sum(m.n_gt for m in per_image)
    scores = np.concatenate([m.scores for m in per_image]) if per_image else np.zeros(0)
    matched = np.concatenate([m.matched[threshold_index] for m in per_image]) if per_image else np.zeros(0, bool)
    return average_precision(scores, matched, n_gt)


def class_ap_50_95(per_image: list[Matches]) -> float | None:
    values = [class_ap(per_image, ti) for ti in range(len(IOU_THRESHOLDS))]
    return None if values[0] is None else float(np.mean(values))


def prf(per_image: list[Matches], conf: float) -> dict:
    tp = sum(int(m.matched[0][m.scores >= conf].sum()) for m in per_image)
    n_pred = sum(int((m.scores >= conf).sum()) for m in per_image)
    n_gt = sum(m.n_gt for m in per_image)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": n_pred - tp, "fn": n_gt - tp}


def recall_by_size(per_image: list[Matches], conf: float) -> dict:
    found = {b: 0 for b in SIZE_BINS}
    total = {b: 0 for b in SIZE_BINS}
    for m in per_image:
        hit = np.zeros(m.n_gt, bool)
        hit[m.gt_of_det[(m.scores >= conf) & (m.gt_of_det >= 0)]] = True
        for low, high in SIZE_BINS:
            inside = (m.gt_size >= low) & (m.gt_size < high)
            found[(low, high)] += int(hit[inside].sum())
            total[(low, high)] += int(inside.sum())
    return {f"{int(lo)}-{'' if hi > 1e8 else int(hi)}px": {"recall": found[(lo, hi)] / total[(lo, hi)] if total[(lo, hi)] else None,
                                                           "n": total[(lo, hi)]} for lo, hi in SIZE_BINS}


def count_errors(pred_counts: np.ndarray, gt_counts: np.ndarray) -> dict:
    error = pred_counts - gt_counts
    r = float(np.corrcoef(pred_counts, gt_counts)[0, 1]) if len(error) > 2 and gt_counts.std() > 0 and pred_counts.std() > 0 else None
    return {"mae": float(np.abs(error).mean()), "mean_signed": float(error.mean()),
            "relative_mae": float(np.abs(error).sum() / max(gt_counts.sum(), 1)), "r": r,
            "pred_total": int(pred_counts.sum()), "gt_total": int(gt_counts.sum())}


def bootstrap(names: list[str], statistic, rounds: int = 1000, seed: int = 0) -> list[float]:
    """95% percentile interval of statistic(list of photo names) over photo resamples."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        sample = [names[i] for i in rng.integers(0, len(names), len(names))]
        value = statistic(sample)
        if value is not None:
            values.append(value)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else [None, None]
