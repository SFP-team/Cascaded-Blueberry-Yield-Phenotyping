from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from plant_benchmark.config import run_dir
from plant_benchmark.data import (
    load_mask,
    prepare_yolo26_dataset,
    sha256,
    split_frame,
    validate_fixed_manifest,
    yolo_partition_root,
)
from plant_benchmark.engine import (
    assert_finite_state_dict,
    ensure_run_fingerprint,
    experiment_fingerprint,
    guarded_amp_step,
    mark_training_complete,
    training_is_complete,
)
from plant_benchmark.models import enable_birefnet_swin_checkpointing
from plant_benchmark.yolo26 import (
    _atomic_verified_checkpoint_copy,
    _yolo_training_outcome,
)


class FixedSplitTests(unittest.TestCase):
    def test_yolo_training_outcome_handles_one_and_zero_based_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            results = output / "ultralytics/results.csv"
            results.parent.mkdir(parents=True)
            results.write_text(
                "epoch,metrics/mIoU\n1,0.70\n2,0.80\n32,0.85\n",
                encoding="utf-8",
            )
            self.assertEqual(_yolo_training_outcome(output, 80)[0], 32)
            results.write_text(
                "epoch,metrics/mIoU\n0,0.70\n1,0.80\n31,0.85\n",
                encoding="utf-8",
            )
            self.assertEqual(_yolo_training_outcome(output, 80)[0], 32)

    def test_yolo_checkpoint_copy_is_atomic_and_content_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "native.pt"
            destination = root / "best.pt"
            source.write_bytes(b"trained-yolo-checkpoint")
            expected = _atomic_verified_checkpoint_copy(source, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(
                _atomic_verified_checkpoint_copy(source, destination), expected
            )
            self.assertFalse(list(root.glob(".best.pt.*.tmp")))

            destination.write_bytes(b"different-checkpoint")
            with self.assertRaisesRegex(RuntimeError, "Refusing to replace"):
                _atomic_verified_checkpoint_copy(source, destination)

    def test_guarded_amp_step_skips_overflow_and_downscales(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = torch.amp.GradScaler(
            "cpu", init_scale=1024.0, growth_interval=2000
        )
        before = parameter.detach().clone()
        finite_loss = parameter * torch.tensor(3.0e38)
        self.assertTrue(bool(torch.isfinite(finite_loss)))
        scaler.scale(finite_loss).backward()
        result = guarded_amp_step(scaler, optimizer, [parameter], 1.0)
        self.assertFalse(result["finite"])
        self.assertEqual(result["scale_before"], 1024.0)
        self.assertLess(result["scale_after"], result["scale_before"])
        self.assertGreaterEqual(result["scale_after"], 1.0)
        torch.testing.assert_close(parameter.detach(), before)
        self.assertTrue(bool(torch.isfinite(parameter)))
        self.assertIsNone(parameter.grad)

    def test_nonfinite_checkpoint_tensor_is_rejected(self) -> None:
        assert_finite_state_dict({"weight": torch.ones(2)}, "finite.pt")
        with self.assertRaisesRegex(RuntimeError, "Non-finite tensor 'weight'"):
            assert_finite_state_dict(
                {"weight": torch.tensor([1.0, float("nan")])}, "bad.pt"
            )

    def test_birefnet_swin_checkpointing_fails_closed(self) -> None:
        class Stage(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.use_checkpoint = False

            def forward(self, value, height, width):
                return value, height, width, value, height, width

        class Backbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList([Stage(), Stage(), Stage(), Stage()])

            def forward(self, value):
                return (value,)

        class Remote(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bb = Backbone()

        class Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Remote()

        model = Wrapper()
        self.assertEqual(enable_birefnet_swin_checkpointing(model), 4)
        self.assertTrue(
            model.model.bb._plant_backbone_checkpointing
            and all(not stage.use_checkpoint for stage in model.model.bb.layers)
        )
        self.assertEqual(enable_birefnet_swin_checkpointing(model), 4)
        with self.assertRaisesRegex(RuntimeError, "model.model.bb.layers"):
            enable_birefnet_swin_checkpointing(torch.nn.Linear(2, 1))

    def test_swin_checkpoint_wrapper_captures_each_scale_dimensions(self) -> None:
        class SpatialBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
                self.H = None
                self.W = None

            def forward(self, value, unused_mask):
                self.assert_shape(value)
                return (value * self.weight).square()

            def assert_shape(self, value) -> None:
                self_test.assertEqual(value.shape[1], self.H * self.W)

        class Stage(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.use_checkpoint = False
                self.block = SpatialBlock()

            def forward(self, value, height, width):
                self.block.H, self.block.W = height, width
                result = self.block(value, None)
                return result, height, width, result, height, width

        class Backbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList([Stage()])

            def forward(self, value):
                height, width = value.shape[-2:]
                tokens = value.flatten(2).transpose(1, 2)
                return (self.layers[0](tokens, height, width)[0],)

        class Remote(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bb = Backbone()

        class Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Remote()

        self_test = self
        model = Wrapper().train()
        enable_birefnet_swin_checkpointing(model)
        full = torch.randn(1, 2, 2, 3)
        half = torch.randn(1, 2, 1, 2)
        full_output = model.model.bb(full)[0]
        half_output = model.model.bb(half)[0]
        (full_output.sum() + half_output.sum()).backward()
        weight_grad = model.model.bb.layers[0].block.weight.grad
        self.assertIsNotNone(weight_grad)
        self.assertTrue(bool(torch.isfinite(weight_grad)))
        self.assertGreater(float(weight_grad.abs()), 0.0)

    def test_png_and_npy_masks_are_equivalent(self) -> None:
        expected = np.array([[False, True], [True, False]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npy = root / "sample.mask.npy"
            png = root / "sample.mask.png"
            np.save(npy, expected.astype(np.uint8))
            Image.fromarray(expected.astype(np.uint8) * 255, mode="L").save(png)
            np.testing.assert_array_equal(load_mask(npy), expected)
            np.testing.assert_array_equal(load_mask(png), expected)

    def test_fixed_split_uses_explicit_split_column(self) -> None:
        frame = pd.DataFrame(
            {
                "stem": ["week10__a", "week11__a", "week12__b"],
                "split": ["train", "val", "train"],
            }
        )
        self.assertEqual(
            split_frame(frame, fold=0, split="train", n_folds=1)["stem"].tolist(),
            ["week10__a", "week12__b"],
        )
        self.assertEqual(
            split_frame(frame, fold=0, split="val", n_folds=1)["stem"].tolist(),
            ["week11__a"],
        )

    def test_fixed_manifest_rejects_colliding_output_stems(self) -> None:
        frame = pd.DataFrame(
            {
                "stem": ["same", "same"],
                "image_path": ["a.jpg", "b.jpg"],
                "mask_path": ["a.png", "b.png"],
                "width": [10, 10],
                "height": [20, 20],
                "seed_x": [5, 5],
                "seed_y": [10, 10],
                "manually_edited": [True, True],
                "split": ["train", "val"],
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate unique stems"):
            validate_fixed_manifest(frame)

    def test_fixed_paths_are_isolated(self) -> None:
        cfg = {
            "paths": {"runs": "/data/experiment/runs", "artifacts": "/data/experiment/artifacts"},
            "data": {"split_mode": "fixed"},
        }
        self.assertEqual(
            run_dir(cfg, "sam21", "finetuned", 0),
            Path("/data/experiment/runs/sam21/finetuned/fixed_split"),
        )
        self.assertEqual(
            yolo_partition_root(cfg, 0),
            Path("/data/experiment/artifacts/yolo26/fixed_split"),
        )

    def test_fixed_manifest_rejects_nonfinite_or_out_of_bounds_seed(self) -> None:
        frame = pd.DataFrame(
            {
                "stem": ["train", "val"],
                "image_path": ["train.jpg", "val.jpg"],
                "mask_path": ["train.png", "val.png"],
                "width": [10, 10],
                "height": [20, 20],
                "seed_x": [float("nan"), 10],
                "seed_y": [10, 10],
                "manually_edited": [True, True],
                "split": ["train", "val"],
            }
        )
        with self.assertRaisesRegex(ValueError, "non-finite seed_x"):
            validate_fixed_manifest(frame)
        frame.loc[0, "seed_x"] = 5
        with self.assertRaisesRegex(ValueError, "out-of-bounds prompt seeds"):
            validate_fixed_manifest(frame)

    def test_resume_binding_changes_with_manifest_or_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text("stem,split\na,train\n", encoding="utf-8")
            cfg = {
                "seed": 1,
                "paths": {"manifest": str(manifest), "runs": str(root / "runs")},
                "data": {"split_mode": "fixed"},
                "training": {"epochs": 2},
                "_project_dir": str(root),
            }
            first = experiment_fingerprint(cfg)
            cfg["training"]["epochs"] = 3
            self.assertNotEqual(first, experiment_fingerprint(cfg))
            cfg["training"]["epochs"] = 2
            manifest.write_text("stem,split\na,train\nb,val\n", encoding="utf-8")
            self.assertNotEqual(first, experiment_fingerprint(cfg))

    def test_completion_marker_is_parsed_and_fingerprint_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "best.pt").write_bytes(b"checkpoint")
            mark_training_complete(output, "model", 3, 0.2, "expected")
            self.assertTrue(training_is_complete(output, "expected"))
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                training_is_complete(output, "different")
            (output / "training_complete.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Unreadable completion marker"):
                training_is_complete(output, "expected")

    def test_run_binding_refuses_mixed_or_unbound_checkpoint_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "clean"
            output.mkdir()
            ensure_run_fingerprint(output, "first")
            ensure_run_fingerprint(output, "first")
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                ensure_run_fingerprint(output, "second")

            unbound = Path(directory) / "unbound"
            unbound.mkdir()
            (unbound / "last.pt").write_bytes(b"old")
            with self.assertRaisesRegex(RuntimeError, "Unbound checkpoint state"):
                ensure_run_fingerprint(unbound, "first")

    def test_yolo_export_invalidates_same_size_stale_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index, split in enumerate(("train", "train", "val")):
                stem = f"sample_{index}"
                image_path = root / f"{stem}.jpg"
                mask_path = root / f"{stem}.png"
                Image.new("RGB", (8, 6), (255, 0, 0)).save(image_path)
                mask = np.zeros((6, 8), dtype=np.uint8)
                mask[1:5, 2:6] = 255
                Image.fromarray(mask, mode="L").save(mask_path)
                records.append(
                    {
                        "stem": stem,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "image_sha256": sha256(image_path),
                        "mask_sha256": sha256(mask_path),
                        "width": 8,
                        "height": 6,
                        "split": split,
                    }
                )
            frame = pd.DataFrame(records)
            cfg = {
                "_project_dir": str(root),
                "paths": {"artifacts": str(root / "artifacts")},
                "data": {"split_mode": "fixed", "folds": 1},
            }
            exported = prepare_yolo26_dataset(cfg, frame)
            output = exported / "images_all" / "sample_0.jpg"
            staged_image = exported / "fixed_split" / "images" / "train" / "sample_0.jpg"
            staged_mask = exported / "fixed_split" / "masks" / "train" / "sample_0.png"
            smoke_image = exported / "smoke" / "images" / "train" / "sample_0.jpg"
            self.assertTrue(staged_image.is_file())
            self.assertTrue(staged_mask.is_file())
            self.assertTrue(smoke_image.is_file())
            self.assertFalse(staged_image.is_symlink())
            self.assertFalse(staged_mask.is_symlink())
            self.assertFalse(smoke_image.is_symlink())
            self.assertEqual(sha256(staged_image), sha256(output))
            with Image.open(output) as exported_image:
                before = np.asarray(exported_image.convert("RGB"))[0, 0].copy()

            source = Path(frame.loc[0, "image_path"])
            Image.new("RGB", (8, 6), (0, 0, 255)).save(source)
            frame.loc[0, "image_sha256"] = sha256(source)
            prepare_yolo26_dataset(cfg, frame)
            with Image.open(output) as exported_image:
                after = np.asarray(exported_image.convert("RGB"))[0, 0].copy()
            self.assertFalse(np.array_equal(before, after))
            self.assertEqual(sha256(staged_image), sha256(output))
            index = json.loads((exported / "export_index.json").read_text())
            self.assertEqual(index["sample_0"]["image_sha256"], sha256(source))


if __name__ == "__main__":
    unittest.main()
