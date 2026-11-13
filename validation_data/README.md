# Hand-harvest validation data

The data behind the prediction of hand-harvested berry count in the paper: 59 plants of 32 genotypes from the
University of Florida Blueberry Breeding Program field at Waldo, Florida, photographed by the rover in March and April
2026 and then harvested by hand.

| File | Content |
|---|---|
| `harvest_table.csv` | input of [`../analysis/R/harvest_prediction.R`](../analysis/R/harvest_prediction.R): one row per plant and mask condition, 59 × 8 = 472 rows |
| `ground_truth.csv` | hand-harvest counts, one row per plant |
| `images_sha256.txt` | SHA-256 of every photo |
| `images/` (download, see below) | the 59 photos, `<Image Name>.jpg` |

## Photos

The photos are attached to the release
[validation-data-v1](https://github.com/SFP-team/Cascaded-Blueberry-Yield-Phenotyping/releases/tag/validation-data-v1)
as one zip of 472 MB. From this folder:

```bash
curl -L -O https://github.com/SFP-team/Cascaded-Blueberry-Yield-Phenotyping/releases/download/validation-data-v1/validation_data_images.zip
unzip validation_data_images.zip                   # creates images/
(cd images && shasum -a 256 -c ../images_sha256.txt)
```

They are the unedited camera files: Canon EOS Rebel T6, JPEG, 5184 × 3456 pixels as stored, with EXIF orientation 6.
Rotate them 90° clockwise to see the plant upright, as `PIL.ImageOps.exif_transpose` does; the code in this
repository reads them that way. The file name is the plant code, `Image Name` in both tables.

## `harvest_table.csv`

| Column | Content |
|---|---|
| `Model` | mask condition: Manual Mask, SAM3, SAM2.1, BiRefNet-HR, Center BiRefNet, SegFormer-B2, DeepLabV3+ or YOLO26-Seg |
| `Image Name` | plant code, the photo's file name |
| `Genotype` | genotype of the plant; the cross-validation leaves out one genotype at a time |
| `total_detection` | berry detections (immature and mature) inside the mask of that condition |
| 19 phenotypes | measured from the photo and the mask of that condition; names below |
| `Immature_gt`, `Mature_gt`, `Observed_Total` | hand-harvest counts, the same in every row of a plant (see `ground_truth.csv`) |

The phenotype columns are LDI Whole (%), Std Dev Hue, Mean Hue, Canopy Perimeter (px), Yellow (%), Texture ASM,
Mean VARI, Canopy Height (px), Mean Value, Mean Saturation, Orientation, Mean GLI, Canopy Width (px),
Texture Contrast, Texture Energy, Std Dev Saturation, Fractal Dimension, Brown (%) and Hull Area (px²).
LDI Whole (%) is stored as a fraction; Yellow (%) and Brown (%) are percentages. Lengths and areas are in pixels of
the upright full-resolution photo.

The values are those of the spreadsheet used for the paper, copied cell by cell.

## `ground_truth.csv`

| Column | Content |
|---|---|
| `Image Name` | plant code |
| `Genotype_gt` | genotype |
| `Immature_gt` | immature berries counted at harvest |
| `Mature_gt` | mature berries counted at harvest |
| `Observed_Total` | total berries counted at harvest, the response of the harvest-count models |

`Observed_Total` is the recorded total. For four plants it differs from `Immature_gt + Mature_gt` by 2 to 6 berries
(W20101006, W22111307, W22130314, W23021108).

## Running the analysis

From the repository root:

```bash
Rscript analysis/R/harvest_prediction.R --data validation_data/harvest_table.csv --out-dir analysis/out/harvest_prediction
python analysis/python/phenotype_agreement.py --table validation_data/harvest_table.csv --out-dir analysis/out/agreement
```

[`../analysis/README.md`](../analysis/README.md) describes both scripts and their outputs.
