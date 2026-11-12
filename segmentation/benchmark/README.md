# Target-plant segmentation benchmark

Code for training and evaluating the target-plant segmentation models on a
fixed image-level train/validation split and a held-out test set:

| Condition key (outputs)        | Model                                                   |
|--------------------------------|---------------------------------------------------------|
| `sam21_finetuned`              | SAM2.1 (Hiera-L), mask decoder fine-tuned with one positive point |
| `segformer_b2_finetuned`       | SegFormer-B2                                            |
| `deeplabv3plus_r50_finetuned`  | DeepLabV3+ with a ResNet-50 encoder                     |
| `yolo26_finetuned`             | YOLO26-Seg (Ultralytics semantic-segmentation checkpoint `yolo26m-sem.pt`) |
| `birefnet_hr_finetuned`        | BiRefNet-HR, fine-tuned end to end                      |
| `birefnet_hr_center_balanced`  | Center BiRefNet (BiRefNet-HR plus a fourth center-guidance channel) |
| `sam3_original`                | SAM3 zero-shot masks from `../sam3_annotation` (not trained) |

Training uses the training split; early stopping, checkpoint selection
(`best.pt` = epoch with the lowest finite validation loss) and the probability
threshold (pooled-pixel Dice over a fixed candidate list) use the validation
split only. The threshold is fixed before the test images are predicted.
YOLO26-Seg outputs a hard class map and has no threshold.

## Install

```bash
cd segmentation/benchmark
pip install -r requirements.txt
export PYTHONPATH=src
```

Pretrained weights are downloaded on first use: `facebook/sam2.1-hiera-large`,
`ZhengPeng7/BiRefNet_HR` and `nvidia/segformer-b2-finetuned-ade-512-512` from
Hugging Face, the ResNet-50 ImageNet encoder by segmentation-models-pytorch, and
`yolo26m-sem.pt` by Ultralytics (or place the file in `segmentation/benchmark`).
Training was run on CUDA GPUs with 24 GiB of memory.

All commands below are run from `segmentation/benchmark` with `PYTHONPATH=src`.

## Configuration

`configs/fixed_split.yaml` holds the paper protocol (image size 1024 x 1536,
loss weights, learning rates, batch sizes, epochs, early stopping, threshold
candidates). Paths are relative to `segmentation/benchmark` and may use
environment variables, for example `dataset: ${BLUEBERRY_SEG_DATA}/train_val_source`:

| Key              | Default                                 | Content |
|------------------|-----------------------------------------|---------|
| `paths.dataset`  | `data/train_val_source`                 | annotated training/validation photographs |
| `paths.manifest` | `data/manifests/train_val_manifest.csv` | written by `build_fixed_split_manifest.py` |
| `paths.artifacts`| `outputs/artifacts`                     | YOLO26 export (upright JPEGs, class-map PNGs) |
| `paths.runs`     | `outputs/runs`                          | `runs/<model>/finetuned/fixed_split/` |

## Data layout

Training and validation photographs (`paths.dataset`) use the output tree of
the SAM3 annotation tool after manual correction (see
`../sam3_annotation/README.md`):

```text
<dataset>/<session>/<cohort>/raw/<stem>.jpg
<dataset>/<session>/<cohort>/results/masks/<stem>.mask.npy
<dataset>/<session>/<cohort>/results/masks/<stem>.meta.json   (manually_edited, edited_mask_area, seed_xy)
<dataset>/<session>/<cohort>/results/mask_pngs/<stem>.mask.png (Manual Mask, 0/255)
```

Cohort folder names start with `W20`. Masks have the size of the
EXIF-corrected (upright) photograph.

Test images are listed in a CSV with the columns
`stem,image_path,mask_path,metadata_path` (the image file stem should equal
`stem`; `metadata_path` is the SAM3 tool's `<stem>.meta.json`, whose `seed_xy`
is the image-only stem seed). The first three characters of the stem give the
age group: `W24` one year, `W23` two years, `W22` three years, other prefixes
unspecified.

Manifest columns used by training (`train_val_manifest.csv`): `stem` (unique;
`<session>__<cohort>__<file stem>`), `image_path`, `mask_path`, `width`,
`height` (upright), `seed_x`, `seed_y` (SAM2.1 positive point), `seed_source`,
`seed_repaired`, `seed_x_original`, `seed_y_original`, `seed_source_original`
(Center BiRefNet location prior), `manually_edited`, `split` (`train`/`val`).

## Workflow

### 1. Test manifest

```bash
python scripts/build_test_manifest.py \
    --source-csv data/test_source.csv --output data/test
```

Writes `data/test/benchmark_manifest.csv`, `data/test/reference_lock.json`,
and hard links (or copies) under `data/test/inputs/raw/<category>/` and
`data/test/references/manually_corrected/<category>/`. Use `--expected-images N`
to fail on an unexpected count.

### 2. Train/validation manifest

```bash
python scripts/build_fixed_split_manifest.py \
    --config configs/fixed_split.yaml \
    --test-manifest data/test/benchmark_manifest.csv
```

Validates every record (binary, non-empty, upright-size mask that matches
`edited_mask_area`), takes the prompt point from `seed_xy` (or from the image-only
stem anchor when absent), moves points that fall outside the Manual Mask to the
nearest mask pixel for SAM2.1 while keeping the original point for Center
BiRefNet, assigns `val` to `--validation-fraction` (default 0.15) of each
session/cohort group with `--seed` (default 2026), and flags training images
that are byte-identical to, or share a name with, a test image
(`overlaps_test_exact_image`, `overlaps_test_stem`). Exclude a record with
`--quarantine <file stem>` (repeatable). Writes the manifest,
`quarantine.csv` and `manifest_summary.json`.

### 3. Train the six models

```bash
./run_fixed_split.sh smoke   # one forward+backward check per model
./run_fixed_split.sh train
```

`train` runs SAM2.1 and BiRefNet-HR, then SegFormer-B2 and DeepLabV3+, then
YOLO26-Seg and Center BiRefNet, two at a time on `GPU_A`/`GPU_B` (default 0 and
1; set both to the same id to run one at a time). Center BiRefNet starts from
the trained BiRefNet-HR `best.pt`. Runs resume from `last.pt` after an
interruption and are bound to the config and manifest by
`experiment_fingerprint.json`; a changed config or manifest needs a new
`paths.runs`. The individual commands are:

```bash
python -m plant_benchmark.cli doctor --config configs/fixed_split.yaml
python -m plant_benchmark.cli train --config configs/fixed_split.yaml \
    --model {sam21,birefnet_hr,segformer_b2,deeplabv3plus_r50,yolo26} \
    --protocol finetuned --fold 0
python scripts/train_center_birefnet.py --config configs/fixed_split.yaml train --fold 0 \
    --epochs 15 --patience 4 --learning-rate 1e-5 --gradient-accumulation 4 \
    --max-gradient-norm 1.0 --side-negative-weight 2.0
```

Each run folder `outputs/runs/<model>/finetuned/fixed_split/` contains
`best.pt`, `last.pt`, `history.csv`, `training_complete.json` and
`experiment_fingerprint.json` (YOLO26-Seg also keeps the Ultralytics run in
`ultralytics/`). The Center BiRefNet `best.pt` stores only the retrained
decoder and input-projection weights plus the SHA-256 of the BiRefNet-HR
`best.pt` it was trained from; both files are needed to load it.

### 4. SAM3 zero-shot masks for the test images

From the repository root, in the SAM3 environment:

```bash
python -m segmentation.sam3_annotation.run_recursive \
    segmentation/benchmark/data/test/inputs/raw \
    segmentation/benchmark/outputs/sam3_test /path/to/sam3.pt cuda
```

These masks are not corrected.

### 5. Test-set prediction and evaluation

```bash
TEST_MANIFEST=data/test/benchmark_manifest.csv \
SAM3_OUTPUT=outputs/sam3_test \
SAM3_CHECKPOINT=/path/to/sam3.pt \
./run_test_evaluation.sh
```

The script runs these steps (outputs under `outputs/test_evaluation/`):

1. `scripts/prepare_test_inference_config.py` builds a manifest of the
   validation records (prefixed `valcal__`) plus the test records and a derived
   config (`inference/test_inference_config.yaml`).
2. `python -m plant_benchmark.cli prepare-yolo` and
   `python -m plant_benchmark.cli predict --model <model> --protocol finetuned --fold 0`
   for the five models, and `scripts/train_center_birefnet.py predict --fold 0`.
   Each selects its threshold on the validation records, writes
   `threshold.json` and writes binary masks to
   `outputs/runs/<model>/finetuned/fixed_split/predictions/{val,test}/`
   (Center BiRefNet: `predictions/{balanced,center_strict}/{val,test}/`).
3. `scripts/stage_sam3_predictions.py` checks the SAM3 masks, metadata and
   prompt defaults and stages them.
4. `scripts/stage_test_predictions.py` collects the seven mask sets in
   `staged/predictions/<condition>/<category>/<stem>.mask.png`. Center BiRefNet
   is evaluated at the `balanced` (validation Dice) threshold.
5. `scripts/evaluate_predictions.py` writes `results/full_test/`:
   `tables/per_image_metrics.csv`, `tables/summary_long.csv` (mean, median,
   standard deviation and bootstrap 95% interval, overall and per age group),
   `tables/overall_metrics.csv`, `tables/metrics_by_age_long.csv`,
   `statistics/friedman_tests.csv`, `statistics/pairwise_wilcoxon_holm.csv`
   and `VALIDATION.json`.

Metrics include Dice, IoU, precision, recall, HD95 as a percentage of the image
diagonal (`hd95_percent_diagonal`), absolute and signed area error in percent
(`relative_area_error_percent`, `signed_area_error_percent`), boundary F1,
boundary IoU, S-measure and others. Region metrics use full-resolution masks;
boundary and distance metrics use masks resized to a longer side of 1024
pixels (`--analysis-max-dimension`).

To evaluate a subset of the test images (for example the plants with
hand-harvest data), pass a manifest that contains only those rows:

```bash
python scripts/evaluate_predictions.py \
    --manifest data/test/subset_manifest.csv \
    --predictions-root outputs/test_evaluation/staged \
    --output outputs/test_evaluation/results/subset \
    --subset-name subset --expected-images <rows in subset_manifest.csv>
```

### 6. Masks and overlays

```bash
python scripts/export_masks_overlays.py \
    --manifest data/test/benchmark_manifest.csv \
    --predictions-root outputs/test_evaluation/staged \
    --output outputs/test_evaluation/per_image_outputs --workers 4
```

Writes `<condition>/masks/<stem>.mask.png` and
`<condition>/overlays/<stem>.overlay.jpg` for the seven conditions and the
Manual Mask (`reference`), plus `export_manifest.csv`.

### 7. Age-group figure

```bash
python scripts/make_age_representative_grid.py \
    --manifest data/test/benchmark_manifest.csv \
    --predictions-root outputs/test_evaluation/staged \
    --metrics outputs/test_evaluation/results/full_test/tables/per_image_metrics.csv \
    --output outputs/figures/age_grid
```

One row per age group with the Manual Mask and the seven model masks, labelled
with per-image Dice and IoU. The default rows are the test images shown in the
paper; choose others with `--row "1 year:<stem>"` (repeatable).

## Weights used by the deployment pipeline

`../../pipeline` loads three files from `weights/` in the repository root.
They are not distributed; create them from the runs above:

| File                        | Written by | Source |
|-----------------------------|------------|--------|
| `weights/birefnet_hr.pt`    | `python -m plant_benchmark.cli train --model birefnet_hr` (step 3) | `outputs/runs/birefnet_hr/finetuned/fixed_split/best.pt` |
| `weights/center_birefnet.pt`| `scripts/train_center_birefnet.py train` (step 3) | `outputs/runs/birefnet_hr_centered/finetuned/fixed_split/best.pt` |
| `weights/threshold.json`    | `scripts/train_center_birefnet.py predict` (step 5) | `outputs/runs/birefnet_hr_centered/finetuned/fixed_split/threshold.json` |

```bash
mkdir -p ../../weights
cp outputs/runs/birefnet_hr/finetuned/fixed_split/best.pt ../../weights/birefnet_hr.pt
cp outputs/runs/birefnet_hr_centered/finetuned/fixed_split/best.pt ../../weights/center_birefnet.pt
cp outputs/runs/birefnet_hr_centered/finetuned/fixed_split/threshold.json ../../weights/threshold.json
```

Copy the files byte for byte: the Center BiRefNet checkpoint is checked against
the SHA-256 of the BiRefNet-HR checkpoint it was trained from.
`threshold.json` holds the `balanced` (validation Dice) and `center_strict`
(validation F-beta, beta 0.8) thresholds.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
