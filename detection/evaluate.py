#!/usr/bin/env python3
"""Score the detector on whole photos the way the pipeline runs it.

    python evaluate.py predict --weights WEIGHTS --name NAME --split test [--root DATA_ROOT]   # writes <root>/evals/NAME/test
    python evaluate.py score EVAL_DIR [--subset all|field] [--count-conf 0.35] [--check]
    python evaluate.py compare EVAL_DIR_A EVAL_DIR_B [--subset all|field]   # paired: same photos, same labels

Inference: 640 px slices with 20% overlap at native resolution, no extra whole-image pass, and the slice detections
merged across classes with GREEDYNMM (IOS 0.5). Slicing and merging follow sahi 0.12 (get_slice_bboxes and
GreedyNMMPostprocess) and are implemented below. For AP, boxes are kept from conf 0.01; counts, precision, recall
and F1 use conf 0.35. The pipeline also keeps only boxes whose centre lies inside the target-plant mask; that needs
masks, so it is not applied here.

Subsets: all (the whole split), field (only camera photos: without photos whose split.json family is "other", the web
and stock images).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import config
import metrics

NAMES = ("immature", "mature", "flowers")


# ---------------------------------------------------------------- slicing and merging (as sahi 0.12)
def slice_windows(w: int, h: int, size: int = config.SLICE, overlap: float = config.SLICE_OVERLAP) -> list[tuple]:
    """sahi.slicing.get_slice_bboxes: step size - int(overlap * size); the last window is moved back inside the
    image; an image smaller than a slice is a single window of its own size."""
    windows, y_min, y_max, step_back = [], 0, 0, int(overlap * size)
    while y_max < h:
        x_min = x_max = 0
        y_max = y_min + size
        while x_max < w:
            x_max = x_min + size
            if y_max > h or x_max > w:
                xmax, ymax = min(w, x_max), min(h, y_max)
                windows.append((max(0, xmax - size), max(0, ymax - size), xmax, ymax))
            else:
                windows.append((x_min, y_min, x_max, y_max))
            x_min = x_max - step_back
        y_min = y_max - step_back
    return windows


def _ios(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    iw = np.clip(np.minimum(box[2], others[:, 2]) - np.maximum(box[0], others[:, 0]), 0, None)
    ih = np.clip(np.minimum(box[3], others[:, 3]) - np.maximum(box[1], others[:, 1]), 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return iw * ih / np.maximum(np.minimum(area, areas), 1e-12)


def greedy_nmm(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, threshold: float = 0.5):
    """sahi GreedyNMMPostprocess(match_metric="IOS", class_agnostic=True): the best box absorbs the boxes whose IOS
    with it is >= threshold; each absorbed box that still overlaps the grown box by IOS > threshold widens it to the
    union; the score is the maximum and the class that of the higher-scoring box."""
    if not len(boxes):
        return boxes, scores, classes
    order = np.argsort(scores, kind="stable")
    clusters = []
    while len(order):
        keep, order = order[-1], order[:-1]
        if not len(order):
            clusters.append((keep, []))
            break
        hit = _ios(boxes[keep], boxes[order]) >= threshold
        clusters.append((keep, list(order[hit][::-1])))
        rest = order[~hit]
        order = rest[np.argsort(scores[rest], kind="stable")]
    out_boxes, out_scores, out_classes = [], [], []
    for keep, members in clusters:
        box, score, cls = boxes[keep].copy(), scores[keep], classes[keep]
        for m in members:
            if _ios(box, boxes[m][None])[0] > threshold:
                box = np.array([min(box[0], boxes[m][0]), min(box[1], boxes[m][1]),
                                max(box[2], boxes[m][2]), max(box[3], boxes[m][3])])
                if scores[m] >= score:
                    cls = classes[m]
                score = max(score, scores[m])
        out_boxes.append(box)
        out_scores.append(score)
        out_classes.append(cls)
    return np.array(out_boxes), np.array(out_scores), np.array(out_classes)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


class YoloSlices:
    def __init__(self, weights: str, device: str):
        from ultralytics import YOLO

        self.model, self.device = YOLO(weights), device

    def __call__(self, crops: list[np.ndarray], conf: float) -> list[tuple]:
        # Ultralytics treats numpy input as BGR, as SAHI's wrapper does (it reverses the RGB slice).
        results = self.model.predict([np.ascontiguousarray(c[:, :, ::-1]) for c in crops], conf=conf, imgsz=config.SLICE,
                                     max_det=config.MAX_DET, device=self.device, verbose=False)
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)) for r in results]


def predict_sliced(image: np.ndarray, model, conf: float, batch: int = 16):
    h, w = image.shape[:2]
    windows = slice_windows(w, h)
    boxes, scores, classes = [], [], []
    for start in range(0, len(windows), batch):
        chunk = windows[start:start + batch]
        for (x0, y0, x1, y1), (b, s, c) in zip(chunk, model([image[y0:y1, x0:x1] for x0, y0, x1, y1 in chunk], conf)):
            if not len(b):
                continue
            b = np.clip(b, 0, None) + np.array([x0, y0, x0, y0])
            valid = (b[:, 0] < b[:, 2]) & (b[:, 1] < b[:, 3])
            boxes.append(b[valid])
            scores.append(s[valid])
            classes.append(c[valid])
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return greedy_nmm(np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes), config.MERGE["match_threshold"])


def split_records(dataset: Path) -> dict[str, dict]:
    return json.loads((dataset / "split.json").read_text())["images"]


def run_predict(args) -> int:
    root = config.data_root(args.root)
    dataset = config.dataset_dir(root)
    out = config.evals_dir(root) / args.name / args.split
    out.mkdir(parents=True, exist_ok=True)
    records = {n: r for n, r in sorted(split_records(dataset).items()) if r["split"] == args.split}
    if args.limit:
        records = dict(list(records.items())[: args.limit])
    model = YoloSlices(args.weights, args.device)
    images, started = {}, time.time()
    for index, name in enumerate(records, 1):
        path = dataset / args.split / "images" / name
        boxes, scores, classes = predict_sliced(load_rgb(path), model, args.conf)
        images[name] = {"boxes": np.round(boxes, 2).tolist(), "scores": np.round(scores, 5).tolist(),
                        "classes": [int(c) for c in classes]}
        if index % 10 == 0 or index == len(records):
            print(f"{index}/{len(records)} {name} {len(scores)} boxes {time.time() - started:.0f}s", flush=True)
    meta = {"weights": str(Path(args.weights).resolve()), "split": args.split, "conf": args.conf, "slice": config.SLICE,
            "overlap": config.SLICE_OVERLAP, "merge": config.MERGE, "max_det": config.MAX_DET,
            "seconds": round(time.time() - started, 1), "images": len(images)}
    (out / "predictions.json").write_text(json.dumps({"meta": meta, "images": images}))
    print(json.dumps(meta), flush=True)
    return 0


# ---------------------------------------------------------------- scoring
SUBSETS = {
    "all": lambda r: True,
    "field": lambda r: r["family"] != "other",
}
SELECTORS = {"immature": lambda c: c == 0, "mature": lambda c: c == 1, "flowers": lambda c: c == 2, "berry": lambda c: c < 2}


def load_gt(dataset: Path, name: str, record: dict) -> tuple[np.ndarray, np.ndarray]:
    path = dataset / record["split"] / "labels" / f"{Path(name).stem}.txt"
    a = np.array([line.split()[:5] for line in path.read_text().splitlines() if line.strip()], float).reshape(-1, 5)
    w, h = record["w"], record["h"]
    boxes = np.stack([(a[:, 1] - a[:, 3] / 2) * w, (a[:, 2] - a[:, 4] / 2) * h,
                      (a[:, 1] + a[:, 3] / 2) * w, (a[:, 2] + a[:, 4] / 2) * h], 1)
    return a[:, 0].astype(int), boxes


def load_predictions(payload: dict, names: list[str]) -> dict[str, tuple]:
    out = {}
    for n in names:
        p = payload["images"][n]
        out[n] = (np.array(p["boxes"], float).reshape(-1, 4), np.array(p["scores"], float), np.array(p["classes"], int))
    return out


def match_all(preds: dict, gt: dict, names: list[str]) -> dict[str, dict]:
    """metrics.Matches per class group in SELECTORS and per photo."""
    matches = {}
    for key, select in SELECTORS.items():
        matches[key] = {}
        for n in names:
            (gc, gb), (b, s, c) = gt[n], preds[n]
            matches[key][n] = metrics.Matches(b[select(c)], s[select(c)], gb[select(gc)])
    return matches


def pycocotools_check(names, records, gt, preds) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    annotations, detections = [], []
    for i, name in enumerate(names):
        for c, (x1, y1, x2, y2) in zip(*gt[name]):
            annotations.append({"id": len(annotations) + 1, "image_id": i, "category_id": int(c) + 1, "iscrowd": 0,
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1)})
        boxes, scores, classes = preds[name]
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
            detections.append({"image_id": i, "category_id": int(c) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    coco_gt = COCO()
    coco_gt.dataset = {"images": [{"id": i, "width": records[n]["w"], "height": records[n]["h"]} for i, n in enumerate(names)],
                       "annotations": annotations, "categories": [{"id": k + 1, "name": NAMES[k]} for k in range(3)]}
    coco_gt.createIndex()
    evaluator = COCOeval(coco_gt, coco_gt.loadRes(detections), "bbox")
    evaluator.params.maxDets = [1, 100, 5000]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]  # [iou, recall, class, area, maxdet]
    out = {}
    for k in range(3):
        p = precision[:, :, k, 0, 2]
        out[NAMES[k]] = {"ap50": float(p[0][p[0] > -1].mean()), "ap50_95": float(p[p > -1].mean())}
    return out


def run_score(args) -> int:
    folder = Path(args.dir)
    payload = json.loads((folder / "predictions.json").read_text())
    dataset = config.dataset_dir(config.data_root(args.root))
    records = split_records(dataset)
    names = [n for n in sorted(payload["images"]) if SUBSETS[args.subset](records[n])]
    gt = {n: load_gt(dataset, n, records[n]) for n in names}
    preds = load_predictions(payload, names)
    conf = args.count_conf
    selectors = SELECTORS
    matches = match_all(preds, gt, names)
    result = {"subset": args.subset, "images": len(names), "count_conf": conf, "meta": payload["meta"], "classes": {}}
    for key in selectors:
        group = [matches[key][n] for n in names]
        result["classes"][key] = {"ap50": metrics.class_ap(group), "ap50_95": metrics.class_ap_50_95(group),
                                  "n_gt": sum(m.n_gt for m in group), "at_count_conf": metrics.prf(group, conf),
                                  "recall_by_size": metrics.recall_by_size(group, conf)}
    present = [result["classes"][k] for k in NAMES if result["classes"][k]["ap50"] is not None]
    result["mean3"] = {"ap50": float(np.mean([c["ap50"] for c in present])), "ap50_95": float(np.mean([c["ap50_95"] for c in present]))}

    def counts(threshold: float) -> dict[str, np.ndarray]:
        out = {}
        for key, select in selectors.items():
            out[key] = (np.array([int(((preds[n][1] >= threshold) & select(preds[n][2])).sum()) for n in names]),
                        np.array([int(select(gt[n][0]).sum()) for n in names]))
        return out

    at_conf = counts(conf)
    result["counts"] = {key: metrics.count_errors(*at_conf[key]) for key in selectors}
    result["count_sweep"] = []
    for threshold in np.round(np.arange(0.15, 0.701, 0.05), 2):
        berries = counts(float(threshold))["berry"]
        row = metrics.count_errors(*berries)
        row.update(conf=float(threshold), berry_f1=metrics.prf([matches["berry"][n] for n in names], float(threshold))["f1"])
        result["count_sweep"].append(row)
    result["families"] = {}
    for family in sorted({records[n]["family"] for n in names}):
        members = [i for i, n in enumerate(names) if records[n]["family"] == family]
        group = [matches["berry"][names[i]] for i in members]
        result["families"][family] = {"images": len(members), "berry_gt": sum(m.n_gt for m in group),
                                      "berry_ap50": metrics.class_ap(group),
                                      "berry_count": metrics.count_errors(at_conf["berry"][0][members], at_conf["berry"][1][members])}
    index = {n: i for i, n in enumerate(names)}

    def mean3(sample):
        values = [metrics.class_ap([matches[k][n] for n in sample]) for k in NAMES]
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    result["bootstrap95"] = {
        "mean3_ap50": metrics.bootstrap(names, mean3, rounds=args.rounds),
        "berry_ap50": metrics.bootstrap(names, lambda s: metrics.class_ap([matches["berry"][n] for n in s]), rounds=args.rounds),
        "berry_count_mae": metrics.bootstrap(names, lambda s: float(np.mean([abs(at_conf["berry"][0][index[n]] - at_conf["berry"][1][index[n]]) for n in s])), rounds=args.rounds),
    }
    if args.check:
        result["pycocotools"] = pycocotools_check(names, records, gt, preds)
    (folder / f"metrics_{args.subset}.json").write_text(json.dumps(result, indent=1))
    (folder / f"summary_{args.subset}.md").write_text(summary_markdown(result))
    print(summary_markdown(result))
    return 0


def summary_markdown(r: dict) -> str:
    f = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    lines = [f"# {Path(r['meta']['weights']).name} on {r['meta']['split']} ({r['subset']}, {r['images']} photos)", "",
             f"Count threshold {r['count_conf']}. AP from conf {r['meta']['conf']}, maxDets 5000.", "",
             "| Class | AP50 | AP50-95 | GT | P | R | F1 |", "|---|---|---|---|---|---|---|"]
    for key, c in r["classes"].items():
        p = c["at_count_conf"]
        lines.append(f"| {key} | {f(c['ap50'])} | {f(c['ap50_95'])} | {c['n_gt']} | {p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} |")
    b = r["bootstrap95"]
    lines += [f"| mean of 3 | {f(r['mean3']['ap50'])} | {f(r['mean3']['ap50_95'])} | | | | |", "",
              f"95% bootstrap: mean-of-3 AP50 {f(b['mean3_ap50'][0])}-{f(b['mean3_ap50'][1])}, berry AP50 "
              f"{f(b['berry_ap50'][0])}-{f(b['berry_ap50'][1])}, berry count MAE {f(b['berry_count_mae'][0])}-{f(b['berry_count_mae'][1])}", "",
              "| Count | MAE | mean signed | r | predicted | GT |", "|---|---|---|---|---|---|"]
    for key, c in r["counts"].items():
        lines.append(f"| {key} | {c['mae']:.1f} | {c['mean_signed']:+.1f} | {f(c['r'])} | {c['pred_total']} | {c['gt_total']} |")
    lines += ["", "| Family | photos | berry GT | berry AP50 | count MAE | mean signed |", "|---|---|---|---|---|---|"]
    for family, c in r["families"].items():
        lines.append(f"| {family} | {c['images']} | {c['berry_gt']} | {f(c['berry_ap50'])} | {c['berry_count']['mae']:.1f} | {c['berry_count']['mean_signed']:+.1f} |")
    sizes = r["classes"]["berry"]["recall_by_size"]
    lines += ["", "Berry recall by size: " + ", ".join(f"{k} {f(v['recall'])} (n={v['n']})" for k, v in sizes.items())]
    if "pycocotools" in r:
        lines += ["", "pycocotools check: " + ", ".join(f"{k} AP50 {v['ap50']:.4f} / AP {v['ap50_95']:.4f}" for k, v in r["pycocotools"].items())]
    return "\n".join(lines) + "\n"


def run_compare(args) -> int:
    """Two prediction folders on the same photos and labels. Each bootstrap resample scores both models on the same
    photos, so the interval is for the difference itself, much narrower than two separate intervals."""
    dataset = config.dataset_dir(config.data_root(args.root))
    records = split_records(dataset)
    folders = [Path(args.base), Path(args.new)]
    payloads = [json.loads((f / "predictions.json").read_text()) for f in folders]
    names = [n for n in sorted(set(payloads[0]["images"]) & set(payloads[1]["images"])) if SUBSETS[args.subset](records[n])]
    gt = {n: load_gt(dataset, n, records[n]) for n in names}
    base, new = (match_all(load_predictions(p, names), gt, names) for p in payloads)
    conf = args.count_conf

    def ap(matches, key, sample):
        return metrics.class_ap([matches[key][n] for n in sample])

    def mean3(matches, sample):
        values = [v for v in (ap(matches, k, sample) for k in NAMES) if v is not None]
        return float(np.mean(values)) if values else None

    def difference(score):
        def statistic(sample):
            a, b = score(base, sample), score(new, sample)
            return None if a is None or b is None else b - a
        return statistic

    rows = {}
    for key in [*SELECTORS, "mean of 3"]:
        score = mean3 if key == "mean of 3" else (lambda m, s, k=key: ap(m, k, s))
        a, b = score(base, names), score(new, names)
        row = {"base_ap50": a, "new_ap50": b, "delta_ap50": None if a is None or b is None else b - a,
               "delta_ap50_ci95": metrics.bootstrap(names, difference(score), rounds=args.rounds)}
        if key in SELECTORS:
            gb, gn = [base[key][n] for n in names], [new[key][n] for n in names]
            row.update(n_gt=sum(m.n_gt for m in gb), base_ap50_95=metrics.class_ap_50_95(gb), new_ap50_95=metrics.class_ap_50_95(gn),
                       base_at_count_conf=metrics.prf(gb, conf), new_at_count_conf=metrics.prf(gn, conf),
                       base_recall_by_size=metrics.recall_by_size(gb, conf), new_recall_by_size=metrics.recall_by_size(gn, conf))
        rows[key] = row
    result = {"base": str(folders[0]), "new": str(folders[1]), "subset": args.subset, "images": len(names),
              "count_conf": conf, "rounds": args.rounds, "classes": rows}
    stem = f"compare_{folders[0].parent.name}_{args.subset}"
    (folders[1] / f"{stem}.json").write_text(json.dumps(result, indent=1))
    (folders[1] / f"{stem}.md").write_text(compare_markdown(result))
    print(compare_markdown(result))
    return 0


def compare_markdown(r: dict) -> str:
    f = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    sf = lambda v: "-" if v is None else f"{v:+.3f}"  # noqa: E731
    pr = lambda d: "-" if not d else f"{d['precision']:.3f} / {d['recall']:.3f}"  # noqa: E731
    base, new = Path(r["base"]).parent.name, Path(r["new"]).parent.name
    lines = [f"# {new} against {base} on {Path(r['new']).name} ({r['subset']}, {r['images']} photos)", "",
             f"Paired: both models on the same photos and labels; 95% interval of the AP50 difference from {r['rounds']} "
             f"photo resamples. P and R at count threshold {r['count_conf']}.", "",
             f"| Class | GT | {base} AP50 | {new} AP50 | difference | 95% interval | {base} P / R | {new} P / R |",
             "|---|---|---|---|---|---|---|---|"]
    for key, c in r["classes"].items():
        lo, hi = c["delta_ap50_ci95"]
        lines.append(f"| {key} | {c.get('n_gt', '')} | {f(c['base_ap50'])} | {f(c['new_ap50'])} | {sf(c['delta_ap50'])} | "
                     f"{sf(lo)} to {sf(hi)} | {pr(c.get('base_at_count_conf'))} | {pr(c.get('new_at_count_conf'))} |")
    flowers = r["classes"]["flowers"]
    lines += ["", f"Flower recall by size at {r['count_conf']}:", "", f"| size | GT | {base} | {new} |", "|---|---|---|---|"]
    for size, v in flowers["base_recall_by_size"].items():
        lines.append(f"| {size} | {v['n']} | {f(v['recall'])} | {f(flowers['new_recall_by_size'][size]['recall'])} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    root_help = f"data root (default: ${config.ROOT_ENV} or {config.DEFAULT_ROOT.relative_to(config.REPO_ROOT)})"
    p = sub.add_parser("predict", help="predict on whole photos of one split, write predictions.json")
    p.add_argument("--weights", required=True, help="trained detector, e.g. ../weights/blueberry_yolo26l.pt")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--name", default="yolo26l", help="evaluation name; output goes to <root>/evals/NAME/SPLIT")
    p.add_argument("--conf", type=float, default=config.EVAL_CONF)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--root", help=root_help)
    p.add_argument("--limit", type=int, default=0, help="only the first N photos (checks)")
    s = sub.add_parser("score", help="score predictions.json against the labels")
    s.add_argument("dir")
    s.add_argument("--root", help=root_help)
    s.add_argument("--subset", choices=sorted(SUBSETS), default="all")
    s.add_argument("--count-conf", type=float, default=config.COUNT_CONF)
    s.add_argument("--rounds", type=int, default=1000, help="bootstrap resamples")
    s.add_argument("--check", action="store_true", help="also compute AP with pycocotools")
    c = sub.add_parser("compare", help="paired comparison of two prediction folders")
    c.add_argument("base", help="reference prediction folder")
    c.add_argument("new", help="prediction folder to compare with it; the report is written here")
    c.add_argument("--root", help=root_help)
    c.add_argument("--subset", choices=sorted(SUBSETS), default="all")
    c.add_argument("--count-conf", type=float, default=config.COUNT_CONF)
    c.add_argument("--rounds", type=int, default=1000, help="bootstrap resamples")
    args = parser.parse_args()
    return {"predict": run_predict, "score": run_score, "compare": run_compare}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
