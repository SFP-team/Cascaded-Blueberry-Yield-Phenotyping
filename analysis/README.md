# Downstream analysis

This folder holds the code for the downstream analyses of the paper. It covers:

1. Berry and flower detection inside each target-plant mask, one run per mask condition.
2. Image-derived plant phenotypes per mask condition.
3. Statistics that compare every mask condition with the Manual Mask.
4. Agreement of berry counts and phenotypes between each model mask and the Manual Mask.
5. Prediction of the hand-harvested berry count with Ridge regression and PLSR (R).
6. The timing experiment for automatic and manually corrected masks.

No weights, images or masks ship with this folder. The paper's hand-harvest data (the harvest
table, the ground truth and a link to the photos) is in [`../validation_data`](../validation_data).

## Layout

```
analysis/
  R/harvest_prediction.R           Ridge and PLSR with nested leave-one-genotype-out CV
  python/common.py                 condition names, manifest and mask helpers
  python/detect_within_mask.py     detector on raw and mask-restricted images
  python/extract_traits.py         phenotypes per mask condition (uses ../traits)
  python/compare_conditions.py     per-image tables, Wilcoxon + Holm, Spearman, figures
  python/build_harvest_table.py    input table of the R script
  python/phenotype_agreement.py    Pearson r, MAE, RMSE of each model mask vs the Manual Mask
  python/time_sam3_masks.py        timing of automatic SAM3 masks
  python/time_center_birefnet_masks.py  timing of automatic Center BiRefNet masks
  python/plot_annotation_times.py  timing figures
  requirements.txt
```

## Mask conditions

The Python scripts use these condition keys. The display name is the `Model` value in the
harvest table and in the R output.

| Key | Display name |
|---|---|
| `raw_unmasked` | Raw image (detection only, no phenotypes) |
| `manual_mask` | Manual Mask |
| `sam3` | SAM3 |
| `sam21` | SAM2.1 |
| `birefnet_hr` | BiRefNet-HR |
| `center_birefnet` | Center BiRefNet |
| `segformer_b2` | SegFormer-B2 |
| `deeplabv3plus` | DeepLabV3+ |
| `yolo26_seg` | YOLO26-Seg |

## Inputs

**Test-image manifest.** A CSV with one row per test image:

```
stem,image_path,age_group,canopy_architecture
PLANT0001,images/PLANT0001.jpg,Age_2_Years,Upright
```

`stem` and `image_path` are required. Relative paths resolve against the manifest folder.
The optional `age_group` and `canopy_architecture` columns produce stratified tables. Missing
values become `Unspecified`. Optional `seed_x`, `seed_y` and `metadata_path` (a JSON file
merged into the trait metadata) are passed to the phenotype extractor. Without a seed, the
extractor uses the bottom center of the image.

**Masks.** One folder per condition key under `--masks-root`, with one binary PNG per image
named `<stem>.mask.png` (nonzero = plant) at the full EXIF-upright image size:

```
masks/manual_mask/PLANT0001.mask.png
masks/sam3/PLANT0001.mask.png
...
```

`--mask-dir KEY=DIR` points one condition at another folder. The Manual Mask comes from the
SAM3-assisted annotation in `../segmentation`. The model masks come from the segmentation
models in `../segmentation`.

**Detector weights.** `weights/blueberry_yolo26l.pt` at the repository root, the default of
`--weights`. `../detection` documents how to train it.

**Hand-harvest ground truth.** A CSV with the columns
`Image Name, Genotype_gt, Immature_gt, Mature_gt, Observed_Total`. `Image Name` equals the
manifest `stem`. The paper's file is `../validation_data/ground_truth.csv`.

## Steps

Run the Python scripts from `analysis/python`, or call them by path. All of them write under
`--output` (default `./analysis_outputs`).

**1. Detection per mask condition.** For each image and condition, pixels outside the mask are
set to black and the YOLO26-L detector runs through SAHI sliced inference (640 px slices,
0.2 overlap, GREEDYNMM with IOS 0.5, class-agnostic merging, confidence 0.25, at most 3000
boxes). The raw condition runs on the full image. Every box is also marked on-target when its
center lies inside the Manual Mask. Completed outputs are skipped on rerun.

```
python detect_within_mask.py --manifest manifest.csv --masks-root masks --output out
```

On a shared GPU, pin the device with `CUDA_VISIBLE_DEVICES`. If you also set
`BLUEBERRY_RESPECT_CUDA_VISIBLE_DEVICES=1`, the script keeps the device ordinal SAHI would
otherwise remap.

**2. Phenotypes per mask condition.** Runs `traits.architecture.analyze_plant` on every mask
and the RGB image, and adds `foreground_fraction` (mask pixels divided by image pixels).

```
python extract_traits.py --manifest manifest.csv --masks-root masks --output out
```

**3. Comparison with the Manual Mask.** Counts use confidence 0.35. The tables keep 0.25 and
0.50 as sensitivity thresholds.

```
python compare_conditions.py --manifest manifest.csv --output out \
    [--segmentation-metrics per_image_dice.csv]
```

- Detection: per-image counts by class, on-target and outside-target counts, and box agreement
  with the Manual Mask input and the raw input (class-aware greedy IoU matching at 0.5). It also
  writes paired Wilcoxon signed-rank tests of each condition against the raw image and against
  the Manual Mask (Pratt zero handling, Holm correction over the whole table).
- Phenotypes: per-trait error against the Manual Mask, a per-image key-trait sMAPE, Spearman and
  Pearson correlations with the Manual Mask per condition and trait, and pairwise Wilcoxon tests
  between the automatic masks with Holm correction.
- `--segmentation-metrics` takes a CSV with columns `stem, condition, dice`. It adds the
  Spearman correlation between per-image Dice and the downstream agreement measures.

**4. Harvest table.** Joins the within-mask berry count with the 19 phenotypes and the
ground truth for every mask condition except the raw image. The berry count is the number of
immature plus mature berry boxes at confidence 0.35 on the mask-restricted image. Flowers are
not counted.

```
python build_harvest_table.py --output out --ground-truth ground_truth.csv --table harvest_table.csv
```

The 19 phenotypes are LDI Whole (%), Std Dev Hue, Mean Hue, Canopy Perimeter (px),
Yellow (%), Texture ASM, Mean VARI, Canopy Height (px), Mean Value, Mean Saturation,
Orientation, Mean GLI, Canopy Width (px), Texture Contrast, Texture Energy,
Std Dev Saturation, Fractal Dimension, Brown (%) and Hull Area (px²). LDI Whole (%) is
stored as a fraction. Yellow (%) and Brown (%) are percentages.

**5. Agreement with the Manual Mask (Fig. 2a and Fig. S1).** For every model mask and every
variable of the harvest table, plants are paired with their Manual Mask row. The script then
computes the Pearson r, MAE and RMSE. The summary gives r, MAE and RMSE of the berry count and
the mean r over the 19 phenotypes. The heatmap shows every r. Its layout is not the paper
figure's.

```
python phenotype_agreement.py --table harvest_table.csv --out-dir out/agreement
```

**6. Harvest prediction (R).**

```
Rscript R/harvest_prediction.R --data harvest_table.csv --out-dir out/harvest_prediction
```

The script fits one model per mask condition with the berry count and the 19 phenotypes as
predictors (`predictor_names` in the script). Each outer fold holds out one genotype. Before
fitting, predictors are converted to numeric, missing values get the median of the training
fold, and zero-variance predictors are dropped.

- Ridge (`glmnet`, alpha 0, standardized) picks lambda at the minimum of a genotype-grouped
  inner CV.
- PLSR (`pls`, SIMPLS, scaled) picks the number of components, at most 15, at the minimum
  inner-CV RMSEP. Each inner fold leaves out one genotype.

Outputs:

| File | Content |
|---|---|
| `logo_predictions.csv` | out-of-fold prediction per plant, method and mask |
| `logo_performance.csv` | N, Pearson R, R² of the correlation, predictive R², RMSE, MAE, RMSE rank |
| `method_comparison.csv` | Ridge and PLSR side by side per mask |
| `final_tuning.csv` | lambda (min and 1se) and PLS components fitted on all plants |
| `full_predictions.csv`, `full_fit_performance.csv` | in-sample fit on all plants, not a validation result |
| `performance_table.csv` | rounded LOGO performance joined with the tuning values |
| `logo_ridge.png/.pdf`, `logo_plsr.png/.pdf` | observed vs LOGO-predicted count, one panel per mask |

R packages: dplyr, tidyr, purrr, glmnet, pls, ggplot2, scales. The script ran with R 4.6.1,
dplyr 1.2.1, tidyr 1.3.2, purrr 1.2.2, glmnet 5.0, pls 2.9.0, ggplot2 4.0.3 and scales 1.4.0.
Expect `cv.glmnet` warnings about `grouped=FALSE`. They come from genotypes with fewer than
three plants.

**7. Timing experiment.** The two timing scripts need one CUDA GPU. They time automatic mask
generation per image on a folder of `.jpg` images and exclude warm-up passes.

```
python time_sam3_masks.py --images timing_images --output out/timing_sam3 --checkpoint sam3.pt
python time_center_birefnet_masks.py --images timing_images --output out/timing_center_birefnet
python plot_annotation_times.py --manual-times manual_times.csv \
    --sam3-timing out/timing_sam3/per_image_timing.csv \
    --center-timing out/timing_center_birefnet/per_image_timing.csv \
    --output out/timing_plots
```

- `time_sam3_masks.py` runs the zero-shot SAM3 center-plant segmenter from
  `../segmentation/sam3_annotation` with Meta's SAM3 checkpoint. It stops if the segmenter's
  prompt settings differ from the protocol listed in the script.
- `time_center_birefnet_masks.py` loads Center BiRefNet with the deployment loader in
  `../pipeline/segment.py`. The weights come from `weights/birefnet_hr.pt`,
  `weights/center_birefnet.pt` and `weights/threshold.json`, and the settings from
  `../pipeline/configs/pipeline.yaml`. The center prompt is the exact image center.
- `manual_times.csv` holds the stopwatch times: one row per image with `image_name`,
  `manual_seconds` (drawing the mask by hand), `sam3_corrected_seconds` and
  `center_birefnet_corrected_seconds` (correcting the automatic mask by hand).

## Paper data

[`../validation_data`](../validation_data) holds the harvest table and the ground truth of the
paper: 59 plants of 32 genotypes, 8 mask conditions. Its README describes the columns and how to
download the 59 photos.

```
Rscript R/harvest_prediction.R --data ../validation_data/harvest_table.csv --out-dir out/harvest_prediction
python python/phenotype_agreement.py --table ../validation_data/harvest_table.csv --out-dir out/agreement
```

The R run takes under a minute on a laptop.

## Weights

This folder produces no weights. It reads `weights/blueberry_yolo26l.pt` (detection),
`weights/birefnet_hr.pt`, `weights/center_birefnet.pt` and `weights/threshold.json`
(Center BiRefNet timing).
