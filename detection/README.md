# Berry and flower detection

This folder trains and evaluates the berry and flower detector used in the cascade. The detector is YOLO26-L with three
classes:

| id | class |
|---|---|
| 0 | immature berry |
| 1 | mature berry |
| 2 | flowers |

At inference the pipeline slices each whole photo into 640 px windows and merges the detections. `evaluate.py` runs
the detector the same way, so its scores describe the detector as the pipeline uses it. The pipeline then keeps only
detections inside the target-plant mask. `evaluate.py` does not apply that step because it works without masks.

## Training data

The training data are mostly field RGB photos of southern highbush blueberry plants taken with rover-mounted and
handheld cameras, plus a small number of web and stock photos (family `other` in `split.json`). Every immature
berry, mature berry and flower carries a bounding box, with one box per flower. Photos are split into train, val and
test by group, so near-duplicates (views of the same plant, frames shot within a short time of each other) always
land in the same split. The photos and labels are not distributed with this repository.

## Data layout

All scripts read one data root. Pass it with `--root`, or set `BLUEBERRY_DETECTION_ROOT`. Without either, the root is
`data/detection` under the repository root.

```
<root>/
  dataset/
    split.json
    train/images/*.jpg   train/labels/*.txt
    val/images/*.jpg     val/labels/*.txt
    test/images/*.jpg    test/labels/*.txt
  tiles_640/             written by prepare_tiles.py
  runs/                  written by train.py
  evals/                 written by evaluate.py predict, score and compare
```

Labels use the YOLO format, one `class x_center y_center width height` line per box with coordinates normalised to
the photo size. A label file has the same stem as its photo. Store the photos upright with no EXIF rotation:
`prepare_tiles.py` reads the raw pixels, while `evaluate.py` applies the EXIF orientation as the pipeline does.

`split.json` lists every photo under `images`, keyed by file name:

```json
{
  "images": {
    "plant_0001.jpg": {"split": "train", "w": 4000, "h": 3000, "family": "handheld"},
    "plant_0002.jpg": {"split": "test", "w": 5184, "h": 3456, "family": "rover"}
  }
}
```

The scripts read these fields:

- `split`: `train`, `val` or `test`. It must match the folder that holds the photo.
- `w`, `h`: photo width and height in pixels. `evaluate.py` uses them to convert the labels to pixels.
- `family`: the camera or source the photo came from, as free text, with `other` for web and stock photos.
  `evaluate.py score` reports berry AP and count error per family. The `field` subset scores only the camera
  photos, dropping those whose family is `other`.

Other fields are ignored.

## Commands

Install the packages in `requirements.txt` (Python 3.10). Run every command from this folder.

1. Cut the train and val photos into training tiles:

   ```
   python prepare_tiles.py
   ```

   Photos with a long side above 640 px and small boxes become 640 px tiles with 20% overlap. Other photos stay whole.
   A box cut by a tile edge is kept when at least 30% of its area is inside the tile. Each training photo keeps up
   to 3 tiles without boxes. Validation keeps every tile. The script writes `tiles_640/{train,val}/{images,labels}`,
   `tiles_640/data.yaml` and `tiles_640/tile_stats.json`, and refuses to overwrite an existing `tiles_640`.

2. Train the detector on one GPU:

   ```
   python train.py --name yolo26l --device 0
   ```

   The script starts from the COCO-pretrained `yolo26l.pt`, which Ultralytics downloads if it is missing. It trains
   for 60 epochs with MuSGD (lr0 0.01, momentum 0.9), a cosine schedule to the last epoch, batch 6, imgsz 640,
   scale jitter 0.3, mosaic off for the last 15 epochs and seed 42. The docstring of `train.py` lists every
   hyperparameter. The run goes to `runs/yolo26l/`, with `weights/best.pt` and `train_meta.json`, which records the
   hyperparameters, package versions and checksums of the pretrained weights and `split.json`.

3. Copy the trained weights to the file the pipeline reads:

   ```
   ROOT="${BLUEBERRY_DETECTION_ROOT:-../data/detection}"
   mkdir -p ../weights
   cp "$ROOT/runs/yolo26l/weights/best.pt" ../weights/blueberry_yolo26l.pt
   ```

4. Predict on the whole test photos, then score the predictions:

   ```
   python evaluate.py predict --weights ../weights/blueberry_yolo26l.pt --name yolo26l --split test
   python evaluate.py score "$ROOT/evals/yolo26l/test"
   python evaluate.py score "$ROOT/evals/yolo26l/test" --subset field
   ```

   `predict` slices each photo into 640 px windows with 20% overlap, keeps boxes from confidence 0.01, and merges
   them across classes with greedy non-maximum merging (IOS 0.5). It writes
   `<root>/evals/<name>/<split>/predictions.json`. `score` writes `metrics_<subset>.json` and `summary_<subset>.md`
   into the same folder. They contain AP50 and AP50-95 per class
   and for both berry classes together, with precision, recall, F1 and count errors at confidence 0.35
   (`--count-conf`). The files also hold berry recall by box size, a sweep of the berry count threshold, per-family
   results and 95% bootstrap intervals over photos (`--rounds`). `--check` recomputes AP with pycocotools.

5. Optional: compare two prediction folders made on the same split, for example two training seeds:

   ```
   python evaluate.py compare "$ROOT/evals/run_a/test" "$ROOT/evals/run_b/test"
   ```

   Both prediction sets are scored on the photos they share. The bootstrap resamples photos once for both, so the
   95% interval is for the AP50 difference itself. The report `compare_<name of base run>_<subset>.{json,md}` is
   written into the second folder.

## Files

| file | role |
|---|---|
| `config.py` | data root, tiling and inference constants |
| `prepare_tiles.py` | training tiles from the train and val photos |
| `train.py` | YOLO26-L training |
| `evaluate.py` | whole-photo prediction, scoring and paired comparison |
| `metrics.py` | COCO-style AP, precision/recall/F1, count errors, bootstrap intervals |

## Weights

`train.py` produces `runs/<name>/weights/best.pt`. Step 3 copies it to `weights/blueberry_yolo26l.pt` at the
repository root, the file the pipeline loads. The weights are not distributed.
