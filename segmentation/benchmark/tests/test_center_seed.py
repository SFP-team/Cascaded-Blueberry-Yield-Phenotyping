"""Center BiRefNet must never use a prompt point derived from the Manual Mask."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plant_benchmark.center_seed import resolve_center_seed, sam21_seed

PROJECT = Path(__file__).resolve().parents[1]


class CenterSeedPolicyTests(unittest.TestCase):
    def test_repaired_sam_point_uses_preserved_original_for_center(self) -> None:
        row = {
            "stem": "sample",
            "width": 100,
            "height": 80,
            "seed_x": 55.0,
            "seed_y": 60.0,
            "seed_source": "metadata_seed_xy_reference_checked_nearest_foreground",
            "seed_repaired": True,
            "seed_x_original": 51.5,
            "seed_y_original": 75.0,
            "seed_source_original": "metadata_seed_xy",
        }
        center = resolve_center_seed(row)
        self.assertEqual((center.x, center.y), (51.5, 75.0))
        self.assertTrue(center.used_preserved_original)
        self.assertFalse(center.reference_mask_read)
        # SAM2.1 keeps its existing reference-checked positive prompt.
        self.assertEqual(sam21_seed(row), (55.0, 60.0))

    def test_unrepaired_image_only_point_is_used_directly(self) -> None:
        center = resolve_center_seed(
            {
                "stem": "sample",
                "width": 100,
                "height": 80,
                "seed_x": 50.0,
                "seed_y": 72.0,
                "seed_source": "sam3_image_only_stem_anchor",
                "seed_repaired": False,
            }
        )
        self.assertEqual((center.x, center.y), (50.0, 72.0))
        self.assertEqual(center.policy, "unmodified_manifest_original_coordinate")

    def test_reference_repair_without_original_provenance_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks a complete, approved"):
            resolve_center_seed(
                {
                    "stem": "sample",
                    "width": 100,
                    "height": 80,
                    "seed_x": 50.0,
                    "seed_y": 72.0,
                    "seed_source": "metadata_seed_xy_reference_checked_nearest_foreground",
                    "seed_repaired": True,
                }
            )

    def test_legacy_manifest_uses_metadata_without_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "sample.meta.json"
            metadata.write_text(json.dumps({"seed_xy": [10, 15]}), encoding="utf-8")
            center = resolve_center_seed(
                {
                    "stem": "sample",
                    "width": 30,
                    "height": 40,
                    "seed_x": 999,
                    "seed_y": 999,
                    "metadata_path": str(metadata),
                }
            )
            self.assertEqual((center.x, center.y), (10.0, 15.0))
            self.assertEqual(center.policy, "metadata_only_legacy_manifest_fallback")
            self.assertFalse(center.reference_mask_read)

    def test_sam21_training_and_prediction_still_use_manifest_seed_columns(self) -> None:
        engine = (PROJECT / "src/plant_benchmark/engine.py").read_text(encoding="utf-8")
        cli = (PROJECT / "src/plant_benchmark/cli.py").read_text(encoding="utf-8")
        self.assertIn(
            "sam_inputs(processor, image, row.seed_x, row.seed_y, mask)",
            engine,
        )
        self.assertIn(
            "sam_inputs(processor, image, row.seed_x, row.seed_y)",
            engine,
        )
        self.assertIn(
            "sam_inputs(processor, image, row.seed_x, row.seed_y, mask)",
            cli,
        )
        self.assertIn(
            "sam_inputs(processor, image, row.seed_x, row.seed_y)",
            cli,
        )


if __name__ == "__main__":
    unittest.main()
