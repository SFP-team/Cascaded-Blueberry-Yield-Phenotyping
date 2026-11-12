"""Map ``analyze_plant`` keys onto the Plant_Architecture table columns.

A Plant_Architecture row has 100 columns: the 8 identity columns written by
``architecture_excel_row`` followed by the 92 trait columns in ``TRAIT_COLUMNS``.
"""

from __future__ import annotations

from typing import Any

# Trait columns of the Plant_Architecture table, in order.
TRAIT_COLUMNS: list[tuple[str, str, float]] = [
    # (excel header, internal key, scale)  the value is multiplied by scale; 0.01 turns a percentage into a fraction
    ("LDI Whole (%)", "foreground_fraction", 1.0),
    ("Image Height (px)", "img_height", 1.0),
    ("Image Width (px)", "img_width", 1.0),
    ("Canopy Area (px²)", "area_px", 1.0),
    ("Canopy Height (px)", "canopy_height_px", 1.0),
    ("Canopy Width (px)", "canopy_width_px", 1.0),
    ("Canopy Perimeter (px)", "perimeter_px", 1.0),
    ("Equivalent Diameter (px)", "equiv_diameter_px", 1.0),
    ("Maximum Feret Diameter (px)", "feret_max_px", 1.0),
    ("Canopy Below Stem Base", "skirt_below_base_frac", 1.0),
    ("LDI Hull/Solidity (%)", "solidity", 1.0),
    ("Hull Area (px²)", "convex_hull_area_px", 1.0),
    ("LDI Bounded (%)", "extent", 1.0),
    ("Height-to-Width Ratio", "aspect_ratio_hw", 1.0),
    ("Eccentricity", "eccentricity", 1.0),
    ("Circularity", "circularity", 1.0),
    ("Convexity", "convexity", 1.0),
    ("Rectangularity", "rectangularity", 1.0),
    ("Elongation", "elongation", 1.0),
    ("Major Axis Length (px)", "major_axis_px", 1.0),
    ("Minor Axis Length (px)", "minor_axis_px", 1.0),
    ("Orientation", "orientation_deg", 1.0),
    ("Fractal Dimension", "fractal_dimension_boxcount", 1.0),
    ("Fractal Dimension R²", "fractal_dimension_r2", 1.0),
    ("Hu Moment 1", "hu1", 1.0),
    ("Hu Moment 2", "hu2", 1.0),
    ("Hu Moment 3", "hu3", 1.0),
    ("Hu Moment 4", "hu4", 1.0),
    ("Hu Moment 5", "hu5", 1.0),
    ("Hu Moment 6", "hu6", 1.0),
    ("Hu Moment 7", "hu7", 1.0),
    ("Mean Top Width (px)", "width_top_px", 1.0),
    ("Mean Middle Width (px)", "width_mid_px", 1.0),
    ("Mean Base Width (px)", "width_base_px", 1.0),
    ("Top-to-Base Width Ratio", "top_base_width_ratio", 1.0),
    ("Widest Point Height", "widest_height_frac", 1.0),
    ("Lateral Lean", "lateral_lean", 1.0),
    ("Centroid Height", "centroid_height_frac", 1.0),
    ("Top Heaviness", "top_heaviness", 1.0),
    ("Left-Right Balance", "lr_balance", 1.0),
    ("Asymmetry", "asymmetry", 1.0),
    ("Revolution Volume (px³)", "revolution_volume_px3", 1.0),
    ("Hole Area (px²)", "holes_area_px", 1.0),
    ("Hole Area Fraction", "holes_area_frac", 1.0),
    ("Number of Holes", "n_holes", 1.0),
    ("Gap Fraction", "gap_fraction", 1.0),
    ("Excess Green Mean", "exg_mean", 1.0),
    ("Excess Green 10th Percentile", "exg_p10", 1.0),
    ("Excess Green Median", "exg_p50", 1.0),
    ("Excess Green 90th Percentile", "exg_p90", 1.0),
    ("Excess Red Mean", "exr_mean", 1.0),
    ("Excess Green Minus Red Mean", "exgr_mean", 1.0),
    ("Mean GLI", "gli_mean", 1.0),
    ("Mean VARI", "vari_mean", 1.0),
    ("Normalized Green-Red Difference Mean", "ngrdi_mean", 1.0),
    ("Modified Green-Red Vegetation Index Mean", "mgrvi_mean", 1.0),
    ("RGB Vegetation Index Mean", "rgbvi_mean", 1.0),
    ("Triangular Greenness Index Mean", "tgi_mean", 1.0),
    ("Green Chromatic Coordinate Mean", "gcc_mean", 1.0),
    ("Green Chromatic Coordinate 10th Percentile", "gcc_p10", 1.0),
    ("Green Chromatic Coordinate Median", "gcc_p50", 1.0),
    ("Green Chromatic Coordinate 90th Percentile", "gcc_p90", 1.0),
    ("Green Pixel Fraction", "frac_green", 1.0),
    ("Yellow Pixel Fraction", "frac_yellow", 1.0),
    ("Brown Pixel Fraction", "frac_brown", 1.0),
    ("Red Pixel Fraction", "frac_red", 1.0),
    ("Blue Fruit Colour Fraction", "frac_blue_fruit", 1.0),
    ("Flower Colour Fraction", "frac_flower", 1.0),
    ("Circular Hue Mean", "hue_circ_mean_deg", 1.0),
    ("Circular Hue Spread", "hue_circ_std", 1.0),
    ("Circular Mean Hue (Preferred)", "hue_sat_weighted_mean_deg", 1.0),
    ("Circular Hue Spread (Preferred)", "hue_sat_weighted_std", 1.0),
    ("Mean Hue", "legacy_hue_mean", 1.0),
    ("Std Dev Hue", "legacy_hue_std", 1.0),
    ("Mean Saturation", "legacy_saturation_mean", 1.0),
    ("Std Dev Saturation", "legacy_saturation_std", 1.0),
    ("Mean Value", "legacy_value_mean", 1.0),
    ("Std Dev Value", "legacy_value_std", 1.0),
    ("Yellow (%)", "legacy_yellow_pct", 0.01),
    ("Brown (%)", "legacy_brown_pct", 0.01),
    ("Lab a Mean", "lab_a_mean", 1.0),
    ("Lab b Mean", "lab_b_mean", 1.0),
    ("Grayscale Mean", "gray_mean", 1.0),
    ("Grayscale Standard Deviation", "gray_std", 1.0),
    ("Edge Density", "edge_density", 1.0),
    ("Texture Contrast", "glcm_contrast", 1.0),
    ("Texture Dissimilarity", "glcm_dissimilarity", 1.0),
    ("Texture Homogeneity", "glcm_homogeneity", 1.0),
    ("Texture Energy", "glcm_energy", 1.0),
    ("Texture Correlation", "glcm_correlation", 1.0),
    ("Texture ASM", "glcm_asm", 1.0),
    ("GLCM Entropy", "glcm_entropy", 1.0),
]


def _scale(value: Any, factor: float) -> Any:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if factor == 1.0:
        return value
    return number * factor


def architecture_excel_row(
    record: dict[str, Any],
    traits: dict[str, Any],
    mask_path: str,
    runtime_seconds: float,
    status: str,
) -> dict[str, Any]:
    foreground = traits.get("foreground_fraction")
    if foreground is None and "area_px" in traits and traits.get("img_height") and traits.get("img_width"):
        height = float(traits["img_height"])
        width = float(traits["img_width"])
        foreground = float(traits["area_px"]) / max(height * width, 1.0)
    traits = dict(traits)
    if foreground is not None:
        traits["foreground_fraction"] = foreground
    row = {
        "image_name": record["image_name"],
        "stem": record["stem"],
        "week": record.get("week", ""),
        "year_class": record.get("year_class", ""),
        "status": status,
        "architecture_runtime_seconds": runtime_seconds,
        "image_path": record["image_path"],
        "mask_path": mask_path,
    }
    for header, key, factor in TRAIT_COLUMNS:
        row[header] = _scale(traits.get(key, ""), factor)
    return row
