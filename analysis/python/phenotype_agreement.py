#!/usr/bin/env python3
"""Correlate berry counts and phenotypes of each model mask with the Manual Mask.

Input is the harvest table (see build_harvest_table.py and ../example_data). For every model
mask and every variable (total_detection and the 19 phenotypes), plants are paired on
``Image Name`` with the Manual Mask row and the Pearson correlation r, MAE and RMSE are computed.

Outputs (under --out-dir):
  phenotype_agreement_long.csv     Model, Variable, n, pearson_r, mae, rmse
  phenotype_agreement_summary.csv  per Model: r of total_detection, MAE and RMSE of
                                   total_detection, mean r over the 19 phenotypes
  phenotype_agreement_heatmap.png  r for every Model x Variable
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REFERENCE_MODEL = "Manual Mask"
COUNT_VARIABLE = "total_detection"
ID_COLUMNS = ("Model", "Image Name", "Genotype", "Immature_gt", "Mature_gt", "Observed_Total")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, required=True, help="Harvest table CSV.")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table = pd.read_csv(args.table, dtype={"Image Name": str, "Genotype": str})
    variables = [c for c in table.columns if c not in ID_COLUMNS]
    if COUNT_VARIABLE not in variables:
        raise SystemExit(f"{args.table} has no {COUNT_VARIABLE} column")
    models = list(dict.fromkeys(table["Model"]))
    if REFERENCE_MODEL not in models:
        raise SystemExit(f"{args.table} has no '{REFERENCE_MODEL}' rows")
    reference = table[table["Model"] == REFERENCE_MODEL].set_index("Image Name")

    rows: list[dict] = []
    for model in models:
        if model == REFERENCE_MODEL:
            continue
        current = table[table["Model"] == model].set_index("Image Name")
        for variable in variables:
            pair = pd.concat(
                [current[variable].rename("model"), reference[variable].rename("manual")],
                axis=1,
                join="inner",
            ).astype(float).dropna()
            x = pair["model"].to_numpy()
            y = pair["manual"].to_numpy()
            r = float(stats.pearsonr(x, y)[0]) if len(x) >= 3 and x.std() > 0 and y.std() > 0 else np.nan
            rows.append(
                {
                    "Model": model,
                    "Variable": variable,
                    "n": len(x),
                    "pearson_r": r,
                    "mae": float(np.mean(np.abs(x - y))),
                    "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
                }
            )
    long = pd.DataFrame(rows)

    summary_rows = []
    for model, group in long.groupby("Model", sort=False):
        count = group[group["Variable"] == COUNT_VARIABLE].iloc[0]
        phenotypes = group[group["Variable"] != COUNT_VARIABLE]
        summary_rows.append(
            {
                "Model": model,
                "count_pearson_r": count["pearson_r"],
                "count_mae": count["mae"],
                "count_rmse": count["rmse"],
                "phenotype_count": len(phenotypes),
                "phenotype_mean_pearson_r": float(phenotypes["pearson_r"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.out_dir / "phenotype_agreement_long.csv", index=False)
    summary.to_csv(args.out_dir / "phenotype_agreement_summary.csv", index=False)

    pivot = long.pivot(index="Model", columns="Variable", values="pearson_r").reindex(
        index=[m for m in models if m != REFERENCE_MODEL], columns=variables
    )
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(variables), 1.6 + 0.5 * len(pivot)))
    image = ax.imshow(pivot.to_numpy(float), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(variables)))
    ax.set_xticklabels(variables, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color="black")
    ax.set_title("Pearson r with the Manual Mask", fontsize=11)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "phenotype_agreement_heatmap.png", dpi=220)
    plt.close(fig)
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
