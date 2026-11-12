#!/usr/bin/env python3
"""Plot manual versus automatic (and automatic + corrected) mask-generation time per image.

Inputs:
  --manual-times    CSV, one row per image: image_name, manual_seconds,
                    sam3_corrected_seconds, center_birefnet_corrected_seconds
                    (stopwatch times: drawing the mask by hand, and correcting the automatic
                    SAM3 / Center BiRefNet mask by hand)
  --sam3-timing     per_image_timing.csv from time_sam3_masks.py
  --center-timing   per_image_timing.csv from time_center_birefnet_masks.py
Automatic time per image is complete_generation_ms / 1000 of the timing CSVs.

Outputs (under --output, PNG and PDF):
  plot1_model_stages_boxplot     time per image of the four model stages, manual mean as a line
  plot3_paired_workflow_per_image  the same values paired per image
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patheffects
import numpy as np


NAVY = "#173f5f"
INK = "#17202a"
MUTED = "#5c6773"
GRID = "#e5eaee"
SAM = "#8a5ab5"
CENTER = "#158466"
WHITE = "#ffffff"

STAGES = (
    ("SAM3 Auto", "samAuto", SAM, ("SAM3", "Auto")),
    ("SAM3 corrected", "sam", SAM, ("SAM3", "corrected")),
    ("Center BiRefNet Auto", "auto", CENTER, ("Center BiRefNet", "Auto")),
    ("Center BiRefNet corrected", "center", CENTER, ("Center BiRefNet", "corrected")),
)

LOG_TICKS = (
    (0.5, "0.5s"),
    (1, "1s"),
    (2, "2s"),
    (5, "5s"),
    (10, "10s"),
    (30, "30s"),
    (60, "1m"),
    (120, "2m"),
    (300, "5m"),
)
Y_MIN = 0.5
Y_MAX = 500.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manual-times", type=Path, required=True)
    parser.add_argument("--sam3-timing", type=Path, required=True)
    parser.add_argument("--center-timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {Path(row["image_name"]).stem: row for row in csv.DictReader(handle)}


def load_rows(manual_path: Path, sam3_path: Path, center_path: Path) -> list[dict[str, float | str]]:
    manual = read_csv(manual_path)
    sam3 = read_csv(sam3_path)
    center = read_csv(center_path)
    missing = sorted((set(manual) ^ set(sam3)) | (set(manual) ^ set(center)))
    if missing:
        raise RuntimeError(f"Image sets differ between the three timing tables: {missing}")
    return [
        {
            "name": stem,
            "manual": float(row["manual_seconds"]),
            "sam": float(row["sam3_corrected_seconds"]),
            "center": float(row["center_birefnet_corrected_seconds"]),
            "samAuto": float(sam3[stem]["complete_generation_ms"]) / 1000.0,
            "auto": float(center[stem]["complete_generation_ms"]) / 1000.0,
        }
        for stem, row in manual.items()
    ]


def values(rows: list[dict[str, float | str]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=np.float64)


def box_stats(sample: np.ndarray) -> dict[str, float]:
    q1, med, q3 = np.quantile(sample, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    inside = sample[(sample >= q1 - 1.5 * iqr) & (sample <= q3 + 1.5 * iqr)]
    return {
        "q1": float(q1),
        "med": float(med),
        "q3": float(q3),
        "low": float(inside.min()),
        "high": float(inside.max()),
    }


def style_axes(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_yticks([tick for tick, _ in LOG_TICKS])
    ax.set_yticklabels([label for _, label in LOG_TICKS])
    ax.set_ylabel("Observed time per image (log scale)", color=INK, fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", colors=INK, labelsize=11)
    ax.tick_params(axis="x", colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#aab5be")
        spine.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(WHITE)


def add_manual_mean(ax: plt.Axes, manual_mean: float) -> None:
    ax.axhline(manual_mean, color=NAVY, linewidth=1.6, linestyle=(0, (7, 5)), zorder=2)
    ax.annotate(
        f"Manual mean {manual_mean:.2f} s",
        xy=(0.995, manual_mean),
        xycoords=("axes fraction", "data"),
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=INK,
        fontweight="bold",
        clip_on=False,
    )


def set_stage_ticks(ax: plt.Axes, xs: np.ndarray) -> None:
    ax.set_xticks(xs)
    ax.set_xticklabels([""] * len(STAGES))
    ymin = ax.get_ylim()[0]
    for x, (_, _, _, lines) in zip(xs, STAGES):
        is_sam = lines[0] == "SAM3"
        ax.annotate(
            lines[0],
            xy=(x, ymin),
            xytext=(0, -5),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=13 if is_sam else 11.5,
            color=INK,
            fontweight="bold",
            clip_on=False,
        )
        ax.annotate(
            lines[1],
            xy=(x, ymin),
            xytext=(0, -22),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=12 if is_sam else 11,
            color=INK,
            clip_on=False,
        )


def time_label(value: float) -> str:
    return f"{value:.1f}s"


def annotate_time(ax: plt.Axes, x: float, y: float, value: float, offset: tuple[float, float]) -> None:
    text = ax.annotate(
        time_label(value),
        (x, y),
        textcoords="offset points",
        xytext=offset,
        ha="center",
        va="bottom",
        fontsize=10,
        color=INK,
        fontweight="bold",
        zorder=8,
    )
    text.set_path_effects(
        [patheffects.Stroke(linewidth=3.4, foreground=WHITE), patheffects.Normal()]
    )


def draw_boxplot(rows: list[dict[str, float | str]], dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=220)
    fig.patch.set_facecolor(WHITE)
    xs = np.arange(1, len(STAGES) + 1)
    means = []
    rng = np.random.default_rng(20)
    for x, (_, key, color, _) in zip(xs, STAGES):
        sample = values(rows, key)
        means.append(float(sample.mean()))
        stats = box_stats(sample)
        ax.vlines(x, stats["low"], stats["high"], color=color, linewidth=1.6, zorder=3)
        ax.hlines([stats["low"], stats["high"]], x - 0.11, x + 0.11, color=color, linewidth=1.6, zorder=3)
        ax.add_patch(
            Rectangle(
                (x - 0.22, stats["q1"]),
                0.44,
                max(stats["q3"] - stats["q1"], 1e-6),
                linewidth=1.6,
                edgecolor=color,
                facecolor=(*mcolors.to_rgb(color), 0.18),
                zorder=3,
            )
        )
        ax.hlines(stats["med"], x - 0.22, x + 0.22, color=color, linewidth=2.4, zorder=4)
        jitter = rng.uniform(-0.12, 0.12, size=sample.size)
        ax.scatter(
            np.full(sample.shape, x) + jitter,
            sample,
            s=22,
            color=color,
            alpha=0.72,
            edgecolors=WHITE,
            linewidths=0.45,
            zorder=5,
        )

    means = np.array(means)
    offsets = ((0, 11), (-38, 8), (0, 12), (-34, 8))
    for x, mean, (_, _, color, _), offset in zip(xs, means, STAGES, offsets):
        ax.scatter(
            [x],
            [mean],
            s=78,
            marker="D",
            color=color,
            edgecolors=WHITE,
            linewidths=1.05,
            zorder=6,
        )
        annotate_time(ax, x, mean, mean, offset)

    style_axes(ax)
    add_manual_mean(ax, float(values(rows, "manual").mean()))
    ax.set_xlim(0.45, 4.55)
    set_stage_ticks(ax, xs)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.19)
    for path in (dest.with_suffix(".png"), dest.with_suffix(".pdf")):
        fig.savefig(path, dpi=260, facecolor=WHITE)
    plt.close(fig)


def draw_paired(rows: list[dict[str, float | str]], dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=220)
    fig.patch.set_facecolor(WHITE)
    xs = np.arange(1, len(STAGES) + 1)
    samples = [values(rows, key) for _, key, _, _ in STAGES]
    means = np.array([sample.mean() for sample in samples])

    for image_idx in range(len(rows)):
        for left, right in ((0, 1), (2, 3)):
            ax.plot(
                [xs[left], xs[right]],
                [samples[left][image_idx], samples[right][image_idx]],
                color="#9da8b0",
                alpha=0.48,
                linewidth=1.0,
                zorder=2,
            )
        for stage_idx, (_, _, color, _) in enumerate(STAGES):
            ax.scatter(
                xs[stage_idx],
                samples[stage_idx][image_idx],
                s=18,
                color=color,
                alpha=0.72,
                edgecolors="none",
                zorder=4,
            )

    for left, right in ((0, 1), (2, 3)):
        ax.plot(
            [xs[left], xs[right]],
            [means[left], means[right]],
            color=NAVY,
            linewidth=2.2,
            zorder=5,
        )
    offsets = ((-18, 9), (-4, 12), (8, 11), (-20, 9))
    for x, mean, (_, _, color, _), offset in zip(xs, means, STAGES, offsets):
        ax.scatter(
            [x],
            [mean],
            s=86,
            marker="D",
            color=color,
            edgecolors=WHITE,
            linewidths=1.1,
            zorder=6,
        )
        annotate_time(ax, x, mean, mean, offset)

    style_axes(ax)
    add_manual_mean(ax, float(values(rows, "manual").mean()))
    ax.set_xlim(0.45, 4.55)
    set_stage_ticks(ax, xs)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.19)
    for path in (dest.with_suffix(".png"), dest.with_suffix(".pdf")):
        fig.savefig(path, dpi=260, facecolor=WHITE)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.manual_times, args.sam3_timing, args.center_timing)
    args.output.mkdir(parents=True, exist_ok=True)
    draw_boxplot(rows, args.output / "plot1_model_stages_boxplot.png")
    draw_paired(rows, args.output / "plot3_paired_workflow_per_image.png")
    print(args.output / "plot1_model_stages_boxplot.png")
    print(args.output / "plot3_paired_workflow_per_image.png")


if __name__ == "__main__":
    main()
