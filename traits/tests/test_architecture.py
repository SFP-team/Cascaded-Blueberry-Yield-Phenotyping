"""Trait extraction on small synthetic masks and images built in the test (no photographs)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from traits.architecture import analyze_plant
from traits.columns import TRAIT_COLUMNS, architecture_excel_row

SOIL = (120, 90, 60)
LEAF = (40, 160, 40)
BERRY = (40, 60, 160)


def synthetic_plant(height: int = 240, width: int = 180) -> tuple[np.ndarray, np.ndarray, dict]:
    """Elliptical green canopy on a thin stem, with one interior gap and a few blue berries."""
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width // 2, int(height * 0.42)
    canopy = ((xx - cx) / (width * 0.35)) ** 2 + ((yy - cy) / (height * 0.32)) ** 2 <= 1.0
    stem = (np.abs(xx - cx) <= 3) & (yy >= cy) & (yy < height * 0.85)
    mask = canopy | stem
    gap = (np.abs(xx - cx) <= 6) & (np.abs(yy - cy) <= 6)
    mask &= ~gap

    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:] = SOIL
    rgb[mask] = LEAF
    for bx, by in ((cx - 25, cy - 20), (cx + 20, cy + 15), (cx - 10, cy + 30)):
        berry = ((xx - bx) ** 2 + (yy - by) ** 2 <= 16) & mask
        rgb[berry] = BERRY

    meta = {
        "image_name": "synthetic_plant.jpg",
        "height": height,
        "width": width,
        "seed_xy": [int(cx), int(height * 0.8)],
        "no_plant": False,
        "mask_confidence": 1.0,
    }
    return mask, rgb, meta


def rectangle(height: int, width: int, top: int, left: int, rows: int, cols: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[top : top + rows, left : left + cols] = True
    return mask


def meta_for(mask: np.ndarray, seed_xy: tuple[int, int] | None = None) -> dict:
    meta = {"image_name": "synthetic.png", "height": mask.shape[0], "width": mask.shape[1]}
    if seed_xy is not None:
        meta["seed_xy"] = list(seed_xy)
    return meta


class FullTraitRowTest(unittest.TestCase):
    def test_every_table_trait_is_computed_and_finite(self) -> None:
        mask, rgb, meta = synthetic_plant()
        traits = analyze_plant(mask, rgb, meta)
        # foreground_fraction is added by the caller (float(mask.mean())), not by analyze_plant.
        for _, key, _ in TRAIT_COLUMNS:
            if key == "foreground_fraction":
                continue
            self.assertIn(key, traits)
            self.assertIsNotNone(traits[key], key)
            self.assertTrue(math.isfinite(float(traits[key])), key)

    def test_identity_fields(self) -> None:
        mask, rgb, meta = synthetic_plant()
        traits = analyze_plant(mask, rgb, meta)
        self.assertEqual(traits["stem"], "synthetic_plant")
        self.assertFalse(traits["no_plant"])
        self.assertEqual(traits["img_height"], 240)
        self.assertEqual(traits["img_width"], 180)

    def test_colour_traits_follow_pixel_colours(self) -> None:
        mask, rgb, meta = synthetic_plant()
        traits = analyze_plant(mask, rgb, meta)
        self.assertGreater(traits["frac_green"], 0.9)
        self.assertGreater(traits["frac_blue_fruit"], 0.0)
        self.assertGreater(traits["exg_mean"], 0.0)
        self.assertGreater(traits["gcc_mean"], 1.0 / 3.0)
        self.assertLess(traits["lab_a_mean"], 0.0)  # green is negative on Lab a*

    def test_interior_gap_is_one_hole(self) -> None:
        mask, rgb, meta = synthetic_plant()
        traits = analyze_plant(mask, rgb, meta)
        self.assertEqual(traits["n_holes"], 1)
        self.assertEqual(traits["holes_area_px"], 13 * 13)


class GeometryTest(unittest.TestCase):
    def test_rectangle_size_and_shape(self) -> None:
        mask = rectangle(100, 80, top=10, left=20, rows=60, cols=40)
        traits = analyze_plant(mask, None, meta_for(mask))
        self.assertEqual(traits["area_px"], 60 * 40)
        self.assertEqual(traits["canopy_height_px"], 60)
        self.assertEqual(traits["canopy_width_px"], 40)
        self.assertAlmostEqual(traits["extent"], 1.0)
        self.assertAlmostEqual(traits["solidity"], 1.0)
        self.assertAlmostEqual(traits["aspect_ratio_hw"], 1.5)
        self.assertAlmostEqual(traits["equiv_diameter_px"], round(math.sqrt(4 * 2400 / math.pi), 2))
        self.assertEqual(traits["n_holes"], 0)
        self.assertEqual(traits["holes_area_px"], 0)
        self.assertAlmostEqual(traits["top_base_width_ratio"], 1.0)

    def test_square_hole(self) -> None:
        mask = rectangle(100, 100, top=20, left=20, rows=60, cols=60)
        mask[45:55, 45:55] = False
        traits = analyze_plant(mask, None, meta_for(mask))
        self.assertEqual(traits["n_holes"], 1)
        self.assertEqual(traits["holes_area_px"], 100)
        self.assertAlmostEqual(traits["holes_area_frac"], round(100 / 3600, 4))

    def test_tiny_holes_are_not_counted(self) -> None:
        mask = rectangle(100, 100, top=20, left=20, rows=60, cols=60)
        mask[50:52, 50:52] = False  # 4 px, below the 25 px hole size
        traits = analyze_plant(mask, None, meta_for(mask))
        self.assertEqual(traits["n_holes"], 0)
        self.assertEqual(traits["holes_area_px"], 4)

    def test_triangle_is_wider_at_the_base(self) -> None:
        height, width = 120, 120
        mask = np.zeros((height, width), dtype=bool)
        for row in range(10, 110):
            half = (row - 10) // 2
            mask[row, 60 - half : 60 + half + 1] = True
        traits = analyze_plant(mask, None, meta_for(mask, (60, 109)))
        self.assertLess(traits["top_base_width_ratio"], 0.5)
        self.assertLess(traits["widest_height_frac"], 0.1)
        self.assertLess(traits["centroid_height_frac"], 0.5)
        self.assertLess(traits["top_heaviness"], 1.0)
        self.assertAlmostEqual(traits["lr_balance"], 0.0, delta=0.05)

    def test_balance_relative_to_stem_base(self) -> None:
        mask = rectangle(100, 100, top=20, left=50, rows=60, cols=30)
        traits = analyze_plant(mask, None, meta_for(mask, (50, 79)))
        self.assertAlmostEqual(traits["lr_balance"], 1.0)
        self.assertAlmostEqual(traits["asymmetry"], 1.0)
        self.assertGreater(traits["lateral_lean"], 0.0)

    def test_skirt_below_stem_base(self) -> None:
        mask = rectangle(100, 100, top=20, left=20, rows=60, cols=60)
        traits = analyze_plant(mask, None, meta_for(mask, (50, 50)))
        self.assertAlmostEqual(traits["skirt_below_base_frac"], 0.5)

    def test_largest_component_drives_region_traits(self) -> None:
        mask = rectangle(120, 120, top=10, left=10, rows=50, cols=50)
        mask[100:105, 100:105] = True
        traits = analyze_plant(mask, None, meta_for(mask))
        self.assertEqual(traits["area_px"], 2500)

    def test_fractal_dimension_of_a_smooth_outline_is_near_one(self) -> None:
        yy, xx = np.mgrid[0:256, 0:256]
        mask = (xx - 128) ** 2 + (yy - 128) ** 2 <= 100**2
        traits = analyze_plant(mask, None, meta_for(mask))
        self.assertAlmostEqual(traits["fractal_dimension_boxcount"], 1.0, delta=0.15)
        self.assertGreater(traits["fractal_dimension_r2"], 0.95)


class EdgeCaseTest(unittest.TestCase):
    def test_empty_mask_is_no_plant(self) -> None:
        mask = np.zeros((50, 40), dtype=bool)
        traits = analyze_plant(mask, None, meta_for(mask))
        self.assertTrue(traits["no_plant"])
        self.assertNotIn("area_px", traits)

    def test_without_rgb_only_geometry(self) -> None:
        mask, _, meta = synthetic_plant()
        traits = analyze_plant(mask, None, meta)
        self.assertIn("area_px", traits)
        self.assertNotIn("exg_mean", traits)
        self.assertNotIn("glcm_contrast", traits)

    def test_rgb_of_other_size_is_ignored(self) -> None:
        mask, rgb, meta = synthetic_plant()
        traits = analyze_plant(mask, rgb[:-1], meta)
        self.assertNotIn("exg_mean", traits)

    def test_scale_adds_cm_columns(self) -> None:
        mask = rectangle(100, 80, top=10, left=20, rows=60, cols=40)
        traits = analyze_plant(mask, None, meta_for(mask), scale_cm_per_px=0.5)
        self.assertAlmostEqual(traits["canopy_height_cm"], 30.0)
        self.assertAlmostEqual(traits["canopy_width_cm"], 20.0)
        self.assertAlmostEqual(traits["area_cm2"], 600.0)


class ExcelRowTest(unittest.TestCase):
    RECORD = {
        "image_name": "synthetic_plant.jpg",
        "stem": "synthetic_plant",
        "week": "week_1",
        "year_class": "2025",
        "image_path": "photos/synthetic_plant.jpg",
    }

    def test_row_has_identity_and_trait_columns(self) -> None:
        mask, rgb, meta = synthetic_plant()
        traits = analyze_plant(mask, rgb, meta)
        row = architecture_excel_row(self.RECORD, traits, "masks/synthetic_plant.mask.png", 0.1, "complete")
        self.assertEqual(len(TRAIT_COLUMNS), 92)
        self.assertEqual(len(row), 100)
        self.assertEqual(list(row)[8:], [header for header, _, _ in TRAIT_COLUMNS])
        self.assertEqual(row["Canopy Area (px²)"], traits["area_px"])
        self.assertAlmostEqual(row["Yellow (%)"], traits["legacy_yellow_pct"] * 0.01)

    def test_foreground_fraction_falls_back_to_area(self) -> None:
        mask = rectangle(100, 80, top=10, left=20, rows=60, cols=40)
        traits = analyze_plant(mask, None, meta_for(mask))
        row = architecture_excel_row(self.RECORD, traits, "", 0.0, "complete")
        self.assertAlmostEqual(row["LDI Whole (%)"], 2400 / 8000)

    def test_error_row_has_blank_traits(self) -> None:
        row = architecture_excel_row(self.RECORD, {}, "", 0.0, "error: missing photo")
        self.assertEqual(row["status"], "error: missing photo")
        self.assertEqual(row["Canopy Area (px²)"], "")


if __name__ == "__main__":
    unittest.main()
