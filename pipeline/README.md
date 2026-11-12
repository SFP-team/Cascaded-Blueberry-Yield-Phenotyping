# Inference pipeline

`run.py` takes a folder of field photographs and writes, for every photo, the target-plant mask, the berry and
flower detections inside that mask, and the image-derived plant traits. Everything is collected in one Excel
workbook.

Steps for each photo:

1. The photo is loaded upright (EXIF orientation applied).
2. `center_seed.py` finds a stem-base seed point of the centre plant from Excess Green in the central 30 % of the
   image width.
3. Center BiRefNet (BiRefNet-HR with a fourth input channel holding a heat map around the seed) predicts the
   target-plant mask on the photo resized to 1024 px wide by 1536 px high. The probability map is thresholded with the value stored in
   `weights/threshold.json` and resized back to full resolution.
4. The traits in `../traits` are computed from the mask and the photo.
5. YOLO26-L runs on 640 px slices with 20 % overlap (SAHI, class-agnostic greedy merging). Detections with
   confidence of at least 0.25 are stored.
6. A detection is counted when its confidence is at least 0.35 and the centre of its box lies inside the
   target-plant mask. `total_berries = unripe + ripe`; flowers are counted separately and are not berries.

Classes: 0 `immature_berry` (unripe), 1 `mature_berry` (ripe), 2 `flowers`.

## Install

Python 3.10 and an NVIDIA GPU are expected (CPU works with `--device cpu` but is slow).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r pipeline/requirements.txt
```

The BiRefNet-HR architecture is loaded from Hugging Face (`ZhengPeng7/BiRefNet_HR`, `trust_remote_code=True`) the
first time the pipeline runs, so the first run needs network access. Set `HF_HOME` to choose where the Hugging Face
cache lives.

## Weights

Weights are not distributed. The pipeline reads these files, relative to the repository root (paths are set in
`configs/pipeline.yaml`):

| File | Model | Produced by |
|---|---|---|
| `weights/birefnet_hr.pt` | fine-tuned BiRefNet-HR, the base of Center BiRefNet | BiRefNet-HR training in `segmentation/` (see `segmentation/README.md`) |
| `weights/center_birefnet.pt` | Center BiRefNet fine-tuning delta, applied on top of `birefnet_hr.pt` | Center BiRefNet training in `segmentation/` |
| `weights/threshold.json` | mask probability thresholds chosen on validation (`balanced`, `center_strict`) | Center BiRefNet prediction step in `segmentation/` |
| `weights/blueberry_yolo26l.pt` | berry and flower detector | `detection/train.py` (see `detection/README.md`) |

Both BiRefNet checkpoints store a `fold` value; it must equal `segmentation.fold` in the config. The default
operating point is `balanced`; `--operating-point center_strict` uses the other threshold in `threshold.json`.

## Input

`--input` is one image or a folder. Folders are searched recursively for `.jpg`, `.jpeg`, `.png`, `.tif` and
`.tiff` files. Hidden files, `*.mask.png` files and folders named like pipeline outputs (`masks`,
`mask_overlays`, `overlays`, `predictions`, `tables`, `output`, ...) are skipped.

Each photo gets an identity in the workbook:

- `stem`: the file name without extension. If two photos share a stem, the later one gets its relative path joined
  with `__`.
- `week`: the first folder named `Age_*` in the path relative to `--input`; otherwise the first folder after
  `phenome-raw/`, or the first folder, that is not a year (`20xx`), a genotype folder (`NN.N...` or `NN-N...`) or
  one of `phenome-raw`, `phenome`, `raw`, `inputs`.
- `year_class`: `W<yy>` from an `Age_<n>_Year_W<yy>` folder; otherwise a `20xx` folder in the path; otherwise a
  `W<yy>` prefix of the file name; otherwise a `20xx` folder anywhere in the absolute path.

These columns are only labels for sorting and joining; they may be blank. For example:

```
photos/
  2025/
    week_1/
      W21_plant_001.jpg
      W21_plant_002.jpg
```

gives `week = week_1` and `year_class = 2025`.

## Run

One GPU:

```bash
python pipeline/run.py --input /path/to/photos --output results/my_run --device cuda:0
```

Several GPUs: start one shard per GPU on the same `--output`, then merge. Photo `i` (in sorted order) goes to shard
`i mod shard-count`.

```bash
for gpu in 0 1; do
  python pipeline/run.py --input /path/to/photos --output results/my_run \
    --device cuda:$gpu --shard-index $gpu --shard-count 2 &
done
wait
python pipeline/merge_shards.py --output results/my_run --shard-count 2
```

Other options (`python pipeline/run.py --help`):

- `--limit N`: only the first N photos.
- `--overwrite`: recompute masks and detections. Without it, a photo whose `masks/<stem>.mask.png` and
  `.meta.json` exist reuses the mask (traits are recomputed), and a photo with a complete
  `predictions/<stem>.json` reuses its detections, so an interrupted run can be restarted with the same command.
- `--no-overlays`: do not write detection overlays.
- `--detect-only`: skip segmentation; use the masks already in `<output>/masks` and copy the
  Plant_Architecture sheet from the existing workbook of the same shard.
- `--excel-every N`: rewrite the workbook every N photos during detection.
- `--config`: another YAML with the same keys as `configs/pipeline.yaml`.

## Outputs

```
<output>/
  Plant_Architecture_and_Detections.xlsx
  protocol.json                 settings of the run
  run_summary.json
  tables/detections_per_image.csv
  tables/architecture_per_image.csv
  masks/<stem>.mask.png         binary target-plant mask (0/255), full resolution
  masks/<stem>.meta.json        image size, seed point, threshold, operating point
  mask_overlays/<stem>.jpg      photo with the mask tinted green
  predictions/<stem>.json       every detection with confidence >= 0.25: class, confidence, box, box centre,
                                whether the centre is inside the mask; plus the counts
  overlays/<stem>.jpg           counted boxes on the photo (orange unripe, purple ripe, white flowers)
```

A sharded run writes `Plant_Architecture_and_Detections.shard<i>.xlsx`, `protocol.shard<i>.json` and
`tables/*.shard<i>.csv` per shard; `merge_shards.py` combines them into the files above, keeping the first row of
each `stem`.

Workbook sheets:

| Sheet | Contents |
|---|---|
| `Per_image` | `image_name`, `stem`, `week`, `year_class`, `unripe`, `ripe`, `flowers`, `total_berries`, `total_detections`, image size, `mask_foreground_pixels`, `runtime_seconds`, `status`, `image_path`, `mask_path` |
| `Summary` | totals over all photos, count threshold, filtering rule, detector |
| `Protocol` | the key/value pairs of `protocol.json` |
| `Plant_Architecture` | 8 identity columns and the 92 trait columns described in `../traits/README.md` |

Rows that failed have `status = error: ...` and zero counts or blank traits.

## Files

| File | Role |
|---|---|
| `run.py` | command-line entry point |
| `merge_shards.py` | merges shard tables into one workbook |
| `discover.py` | finds photos and derives `stem`, `week`, `year_class` |
| `center_seed.py` | stem-base seed point of the centre plant |
| `segment.py` | Center BiRefNet loading, input tensor with the seed heat map, mask prediction, mask files |
| `detect.py` | SAHI + YOLO26-L inference, centre-in-mask rule, counts, overlays |
| `workbook.py` | writes the four-sheet workbook |
| `configs/pipeline.yaml` | image size, weight paths, thresholds, slicing settings |

## Tests

```bash
python -m unittest discover -s pipeline/tests -t .
```

The tests check the counting rule and the seed point on synthetic arrays; they need numpy, opencv-python and Pillow
only.
