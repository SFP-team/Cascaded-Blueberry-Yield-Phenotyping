#!/usr/bin/env python3
"""Build the harvest-prediction input table used by ../R/harvest_prediction.R.

One row per (mask condition, harvested plant) with the within-mask berry count, the 19
image-derived phenotypes and the hand-harvest counts. Plants are matched on
``Image Name`` = manifest ``stem``.

Inputs:
  --output          analysis output root written by detect_within_mask.py and extract_traits.py
  --ground-truth    CSV with columns Image Name, Genotype_gt, Immature_gt, Mature_gt, Observed_Total
Output:
  --table           CSV with columns Model, Image Name, Genotype, total_detection,
                    <19 phenotypes>, Immature_gt, Mature_gt, Observed_Total

total_detection is the number of immature plus mature berry detections at confidence >= 0.35
on the mask-restricted image (flowers are not counted).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import CONDITIONS, MASK_CONDITIONS


HEADLINE_THRESHOLD = 0.35
BERRY_CLASSES = (0, 1)

# Column header in the harvest table -> trait key in tables/architecture_traits_per_image.csv.
PHENOTYPE_COLUMNS = {
    "LDI Whole (%)": "foreground_fraction",
    "Std Dev Hue": "legacy_hue_std",
    "Mean Hue": "legacy_hue_mean",
    "Canopy Perimeter (px)": "perimeter_px",
    "Yellow (%)": "legacy_yellow_pct",
    "Texture ASM": "glcm_asm",
    "Mean VARI": "vari_mean",
    "Canopy Height (px)": "canopy_height_px",
    "Mean Value": "legacy_value_mean",
    "Mean Saturation": "legacy_saturation_mean",
    "Orientation": "orientation_deg",
    "Mean GLI": "gli_mean",
    "Canopy Width (px)": "canopy_width_px",
    "Texture Contrast": "glcm_contrast",
    "Texture Energy": "glcm_energy",
    "Std Dev Saturation": "legacy_saturation_std",
    "Fractal Dimension": "fractal_dimension_boxcount",
    "Brown (%)": "legacy_brown_pct",
    "Hull Area (px²)": "convex_hull_area_px",
}
GROUND_TRUTH_COLUMNS = ("Image Name", "Genotype_gt", "Immature_gt", "Mature_gt", "Observed_Total")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("analysis_outputs"), help="Analysis output root.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True, help="Destination CSV.")
    return parser.parse_args()


def berry_count(prediction_json: Path) -> int:
    payload = json.loads(prediction_json.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"Incomplete detector record: {prediction_json}")
    return sum(
        float(d["confidence"]) >= HEADLINE_THRESHOLD and int(d["category_id"]) in BERRY_CLASSES
        for d in payload["detections"]
    )


def main() -> int:
    args = parse_args()
    ground_truth = pd.read_csv(args.ground_truth, dtype={"Image Name": str, "Genotype_gt": str})
    missing = set(GROUND_TRUTH_COLUMNS) - set(ground_truth.columns)
    if missing:
        raise SystemExit(f"{args.ground_truth} lacks columns: {sorted(missing)}")
    traits = pd.read_csv(
        args.output / "tables/architecture_traits_per_image.csv", dtype={"stem": str}
    ).set_index(["condition", "stem"])

    rows: list[dict] = []
    skipped: list[str] = []
    for _, plant in ground_truth.iterrows():
        stem = plant["Image Name"]
        if any((condition, stem) not in traits.index for condition in MASK_CONDITIONS):
            skipped.append(stem)
            continue
        for condition in MASK_CONDITIONS:
            trait_row = traits.loc[(condition, stem)]
            row = {
                "Model": CONDITIONS[condition],
                "Image Name": stem,
                "Genotype": plant["Genotype_gt"],
                "total_detection": berry_count(
                    args.output / "detection/predictions" / condition / f"{stem}.json"
                ),
            }
            row.update({header: trait_row[key] for header, key in PHENOTYPE_COLUMNS.items()})
            row.update(
                {
                    "Immature_gt": plant["Immature_gt"],
                    "Mature_gt": plant["Mature_gt"],
                    "Observed_Total": plant["Observed_Total"],
                }
            )
            rows.append(row)

    args.table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.table, index=False)
    print(f"Wrote {len(rows)} rows to {args.table}")
    if skipped:
        print(f"Plants in the ground truth without detections/traits (skipped): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
