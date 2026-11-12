# Plant-architecture traits

Image-derived plant phenotypes computed from one RGB photograph and the binary mask of the target plant. Only the
pixels inside the mask are used, so neighbouring plants and the background do not contribute. No model weights are
involved.

| File | Role |
|---|---|
| `architecture.py` | `analyze_plant(mask, rgb, meta)`, the trait groups, and a command-line report for a folder of masks |
| `columns.py` | `TRAIT_COLUMNS` (table header, trait key, scale) and `architecture_excel_row`, which builds one Plant_Architecture row |
| `tests/` | unit tests on synthetic masks and images |

The pipeline (`../pipeline`) calls `analyze_plant` for every photo and writes the Plant_Architecture sheet with
`architecture_excel_row`.

## Install

```bash
pip install -r traits/requirements.txt
```

## Input

```python
from traits import analyze_plant, architecture_excel_row

traits = analyze_plant(mask, rgb, meta)
traits["foreground_fraction"] = float(mask.mean())
```

- `mask`: boolean array, H x W, True on the target plant.
- `rgb`: uint8 array, H x W x 3, the upright photograph aligned with the mask. If it is `None` or has another size,
  only geometry traits are computed.
- `meta`: dict with `image_name` (required) and optionally `seed_xy` (stem-base point `[x, y]` in pixels; default
  bottom centre), `height`, `width`, `no_plant`, `manually_edited`, `mask_confidence`.
- `scale_cm_per_px` (optional): adds `canopy_height_cm`, `canopy_width_cm` and `area_cm2`.

An empty mask returns only the identity fields with `no_plant = True`.

`foreground_fraction` (the `LDI Whole (%)` column) is set by the caller; `architecture_excel_row` falls back to
`area_px / (img_height * img_width)` when it is missing.

## Output columns

A Plant_Architecture row has 100 columns: 8 identity columns (`image_name`, `stem`, `week`, `year_class`, `status`,
`architecture_runtime_seconds`, `image_path`, `mask_path`) and the 92 trait columns below, in this order. Lengths and
areas are in pixels. Column names ending in `(%)` keep the names of the original trait table but hold fractions
between 0 and 1. Region traits use the largest 8-connected component of the mask; row profiles, holes and the
fraction below the stem base use the whole mask.

**Size**

| Column | Meaning |
|---|---|
| LDI Whole (%) | mask pixels / image pixels |
| Image Height (px), Image Width (px) | photo size |
| Canopy Area (px²) | pixel area of the plant |
| Canopy Height (px), Canopy Width (px) | bounding-box height and width |
| Canopy Perimeter (px) | four-direction Crofton perimeter |
| Equivalent Diameter (px) | diameter of a circle with the same area |
| Maximum Feret Diameter (px) | longest distance between two boundary points |
| Canopy Below Stem Base | fraction of plant area at or below the stem-base row |

**Shape**

| Column | Meaning |
|---|---|
| LDI Hull/Solidity (%) | area / convex hull area |
| Hull Area (px²) | convex hull area |
| LDI Bounded (%) | area / bounding-box area |
| Height-to-Width Ratio | bounding-box height / width |
| Eccentricity | eccentricity of the moment-equivalent ellipse |
| Circularity | 4 pi area / contour perimeter² |
| Convexity | hull perimeter / contour perimeter (at most 1) |
| Rectangularity | area / minimum-area rotated rectangle |
| Elongation | 1 - minor axis / major axis |
| Major Axis Length (px), Minor Axis Length (px), Orientation | moment-equivalent ellipse |
| Fractal Dimension, Fractal Dimension R² | box-counting dimension of the one-pixel boundary over dyadic box sizes, and the R² of the log-log fit |
| Hu Moment 1 to 7 | sign-preserving -log10 of the seven Hu moments |

**Vertical and horizontal profile (bush habit)**

| Column | Meaning |
|---|---|
| Mean Top Width (px), Mean Middle Width (px), Mean Base Width (px) | mean row width in the top, middle and bottom fifth of the plant height |
| Top-to-Base Width Ratio | top width / base width |
| Widest Point Height | height of the widest row, 0 at the base and 1 at the top |
| Lateral Lean | (centroid x - stem-base x) / bounding-box width, signed |
| Centroid Height | centroid height above the lowest plant row / plant height |
| Top Heaviness | area above the middle row / area below it |
| Left-Right Balance | (area right - area left of the stem base) / total, signed |
| Asymmetry | absolute value of Left-Right Balance |
| Revolution Volume (px³) | each row's width revolved about a vertical axis, summed (volume surrogate) |

**Porosity**

| Column | Meaning |
|---|---|
| Hole Area (px²) | enclosed background pixels inside the plant outline |
| Hole Area Fraction | hole area / filled outline area |
| Number of Holes | enclosed holes of at least 25 px |
| Gap Fraction | 1 - solidity |

**Colour indices.** Computed on the plant pixels after a small erosion (about 1 % of the plant's bounding box)
that keeps soil and sky edge pixels out. With R, G, B the channel values and r, g, b the chromatic coordinates
(R / (R + G + B), ...):

| Column | Meaning |
|---|---|
| Excess Green Mean, 10th Percentile, Median, 90th Percentile | ExG = 2g - r - b |
| Excess Red Mean | ExR = 1.4r - g |
| Excess Green Minus Red Mean | ExG - ExR |
| Mean GLI | (2G - R - B) / (2G + R + B) |
| Mean VARI | (g - r) / (g + r - b), clipped to [-1, 1] |
| Normalized Green-Red Difference Mean | (G - R) / (G + R) |
| Modified Green-Red Vegetation Index Mean | (G² - R²) / (G² + R²) |
| RGB Vegetation Index Mean | (G² - BR) / (G² + BR) |
| Triangular Greenness Index Mean | -0.5 (190 (R - G) - 120 (R - B)) / 255 |
| Green Chromatic Coordinate Mean, 10th Percentile, Median, 90th Percentile | g |

**Colour coverage.** Fractions of eroded plant pixels in fixed OpenCV HSV ranges (H 0-179, S and V 0-255). They are
coverage proxies, not counts, and may overlap slightly.

| Column | HSV range |
|---|---|
| Green Pixel Fraction | H 36-89, S >= 40, V >= 40 |
| Yellow Pixel Fraction | H 20-35, S >= 50, V >= 70 |
| Brown Pixel Fraction | H 5-24, S >= 30, 25 <= V < 120 |
| Red Pixel Fraction | H <= 9 or H >= 165, S >= 60, V >= 40 |
| Blue Fruit Colour Fraction | H 100-140, S >= 35, V >= 18 |
| Flower Colour Fraction | S < 35 and V > 165, or H 140-164, S >= 30, V > 120 |

**Hue, saturation, value and Lab**

| Column | Meaning |
|---|---|
| Circular Hue Mean, Circular Hue Spread | circular mean (OpenCV 0-179 scale) and circular standard deviation of hue, eroded mask |
| Circular Mean Hue (Preferred), Circular Hue Spread (Preferred) | the same, weighted by saturation so near-grey pixels count less |
| Mean Hue, Std Dev Hue, Mean Saturation, Std Dev Saturation, Mean Value, Std Dev Value | plain mean and standard deviation of H, S, V on the full mask |
| Yellow (%), Brown (%) | fraction of full-mask pixels with H 20-40, S >= 50, V >= 50 (yellow) or H 0-20, S 40-150, V 40-180 (brown) |
| Lab a Mean, Lab b Mean | mean a* and b* (OpenCV Lab, centred at 0), eroded mask |

**Texture.** On the grey-scale plant pixels (eroded mask).

| Column | Meaning |
|---|---|
| Grayscale Mean, Grayscale Standard Deviation | grey level statistics |
| Edge Density | fraction of plant pixels on Canny edges (thresholds 50 and 150) |
| Texture Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM | grey-level co-occurrence matrix properties, averaged over distances 1 and 3 and angles 0, 45, 90 and 135 degrees |
| GLCM Entropy | entropy of the normalised co-occurrence matrix, same average |

The co-occurrence matrix uses 32 fixed grey levels on the crop downsampled so its long side is at most 256 px;
background pixels are excluded.

`analyze_plant` also returns keys that are not table columns: `stem`, `no_plant`, `manually_edited`,
`mask_confidence`, and HSV saturation and value means and standard deviations on the eroded mask
(`hsv_saturation_mean`, `hsv_saturation_std`, `hsv_value_mean`, `hsv_value_std`).

## Command-line report for a folder of masks

For masks written by the SAM3 annotation tool in `segmentation/` (`masks/<stem>.mask.npy` and
`masks/<stem>.meta.json`, where the meta file holds `image_name`, `source_path` of the photo and `seed_xy`):

```bash
python traits/architecture.py path/to/annotation_output --out-dir results/traits
python traits/architecture.py path/to/annotation_output --scale 0.05 --no-figures
```

It writes `architecture_report.csv` and `architecture_report.json` (one row per plant, trait keys as column names)
and, unless `--no-figures`, one figure per plant in `figures/` (outline, convex hull, bounding box, centroid, stem
base, width profile and colour fractions). Masks from `pipeline/run.py` already have their traits in the workbook.

## Tests

From the repository root:

```bash
python -m unittest discover -s traits/tests -t .
```

The tests build small masks and images in memory (rectangles, a triangle, a disc, a synthetic bush with a gap and
blue berries) and check sizes, hole counts, profile signs, colour fractions and the 100-column row layout.
