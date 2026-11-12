#!/usr/bin/env python3
"""Write the SYNTHETIC example inputs of the analysis folder.

All values are random draws with plausible ranges; they are not field data. The files have the
same column schema as the real inputs:
  synthetic_ground_truth.csv   Image Name, Genotype_gt, Immature_gt, Mature_gt, Observed_Total
  synthetic_harvest_table.csv  input of ../R/harvest_prediction.R and ../python/phenotype_agreement.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = (
    "Manual Mask",
    "SAM3",
    "SAM2.1",
    "BiRefNet-HR",
    "Center BiRefNet",
    "SegFormer-B2",
    "DeepLabV3+",
    "YOLO26-Seg",
)
# Phenotype: (low, high) of the uniform draw for the Manual Mask value.
PHENOTYPES = {
    "LDI Whole (%)": (0.10, 0.50),
    "Std Dev Hue": (10.0, 30.0),
    "Mean Hue": (30.0, 60.0),
    "Canopy Perimeter (px)": (5000.0, 20000.0),
    "Yellow (%)": (0.0, 15.0),
    "Texture ASM": (0.001, 0.05),
    "Mean VARI": (-0.10, 0.20),
    "Canopy Height (px)": (1500.0, 3500.0),
    "Mean Value": (80.0, 160.0),
    "Mean Saturation": (60.0, 140.0),
    "Orientation": (-90.0, 90.0),
    "Mean GLI": (0.0, 0.20),
    "Canopy Width (px)": (1000.0, 3000.0),
    "Texture Contrast": (20.0, 200.0),
    "Texture Energy": (0.03, 0.22),
    "Std Dev Saturation": (30.0, 60.0),
    "Fractal Dimension": (1.10, 1.50),
    "Brown (%)": (0.0, 10.0),
    "Hull Area (px²)": (1.0e6, 8.0e6),
}
PLANTS = 60
GENOTYPES = 32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # 28 genotypes with two plants and 4 with one plant: 60 plants, 32 genotypes.
    genotypes = [f"SYN_G{g:02d}" for g in range(1, GENOTYPES + 1)]
    plant_genotypes = [g for g in genotypes[:28] for _ in range(2)] + genotypes[28:]
    genotype_effect = dict(zip(genotypes, rng.normal(0.0, 0.35, GENOTYPES)))

    plants = []
    for i, genotype in enumerate(plant_genotypes, start=1):
        size = float(np.clip(rng.normal(1.0 + genotype_effect[genotype], 0.25), 0.2, 2.5))
        total = int(round(np.clip(1500.0 * size + rng.normal(0, 250), 100, 5000)))
        mature_share = float(rng.uniform(0.05, 0.45))
        mature = int(round(total * mature_share))
        plants.append(
            {
                "Image Name": f"SYN{i:04d}",
                "Genotype_gt": genotype,
                "Immature_gt": total - mature,
                "Mature_gt": mature,
                "Observed_Total": total,
                "size": size,
            }
        )
    ground_truth = pd.DataFrame(plants)

    manual = {}
    for plant in plants:
        values = {}
        for name, (low, high) in PHENOTYPES.items():
            values[name] = float(rng.uniform(low, high))
        for name in ("Canopy Height (px)", "Canopy Width (px)", "Hull Area (px²)", "Canopy Perimeter (px)"):
            low, high = PHENOTYPES[name]
            values[name] = float(np.clip(values[name] * (0.6 + 0.4 * plant["size"]), low, high * 1.3))
        visibility = float(rng.uniform(0.12, 0.30))
        values["total_detection"] = float(round(plant["Observed_Total"] * visibility + rng.normal(0, 20)))
        manual[plant["Image Name"]] = values

    rows = []
    for model in MODELS:
        noise = 0.0 if model == "Manual Mask" else float(rng.uniform(0.02, 0.10))
        for plant in plants:
            base = manual[plant["Image Name"]]
            count = base["total_detection"] * (1.0 + rng.normal(0.0, noise))
            row = {
                "Model": model,
                "Image Name": plant["Image Name"],
                "Genotype": plant["Genotype_gt"],
                "total_detection": max(0, int(round(count))),
            }
            for name, (low, high) in PHENOTYPES.items():
                row[name] = base[name] + rng.normal(0.0, noise * (high - low) / 2.0)
            row["LDI Whole (%)"] = float(np.clip(row["LDI Whole (%)"], 0.01, 0.95))
            row["Immature_gt"] = plant["Immature_gt"]
            row["Mature_gt"] = plant["Mature_gt"]
            row["Observed_Total"] = plant["Observed_Total"]
            rows.append(row)
    table = pd.DataFrame(rows)
    numeric = table.select_dtypes("float").columns
    table[numeric] = table[numeric].round(6)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ground_truth.drop(columns="size").to_csv(args.out_dir / "synthetic_ground_truth.csv", index=False)
    table.to_csv(args.out_dir / "synthetic_harvest_table.csv", index=False)
    print(f"Wrote {len(ground_truth)} plants and {len(table)} table rows to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
