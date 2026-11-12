"""Counting rule and stem seed on synthetic inputs (no models, no photographs)."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.center_seed import find_stem_seed
from pipeline.detect import counts_from_detections, point_in_mask


def detection(category_id: int, confidence: float, center_in_mask: bool) -> dict:
    return {"category_id": category_id, "confidence": confidence, "center_in_mask": center_in_mask}


class CountingRuleTest(unittest.TestCase):
    def test_confidence_and_mask_filter(self) -> None:
        detections = [
            detection(0, 0.90, True),   # immature berry, counted
            detection(0, 0.30, True),   # below the count threshold
            detection(1, 0.35, True),   # mature berry, counted (threshold is inclusive)
            detection(1, 0.95, False),  # centre outside the target-plant mask
            detection(2, 0.80, True),   # flower, counted but not a berry
        ]
        counts = counts_from_detections(detections, 0.35)
        self.assertEqual(
            counts,
            {"unripe": 1, "ripe": 1, "flowers": 1, "total_berries": 2, "total_detections": 3},
        )

    def test_point_in_mask_rounds_and_clips(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True
        self.assertTrue(point_in_mask(mask, 3.4, 3.6))
        self.assertFalse(point_in_mask(mask, 5.6, 3.0))
        self.assertFalse(point_in_mask(mask, -20.0, 50.0))


class StemSeedTest(unittest.TestCase):
    def test_seed_on_central_vegetation(self) -> None:
        rgb = np.full((200, 150, 3), (120, 90, 60), dtype=np.uint8)
        rgb[40:160, 65:85] = (40, 160, 40)
        x, y, fallback = find_stem_seed(rgb)
        self.assertFalse(fallback)
        self.assertTrue(65 <= x < 85)
        self.assertTrue(40 <= y < 160)
        self.assertGreater(y, 100)  # 70 % of the vegetation rows lie above the seed

    def test_fallback_without_vegetation(self) -> None:
        rgb = np.full((200, 150, 3), (120, 90, 60), dtype=np.uint8)
        self.assertEqual(find_stem_seed(rgb), (75, 110, True))


if __name__ == "__main__":
    unittest.main()
