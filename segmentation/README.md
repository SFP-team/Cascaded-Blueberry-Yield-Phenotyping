# Target-plant segmentation

This folder covers the first stage of the workflow: separating the target
(center) blueberry plant from neighboring plants and background in field RGB
photographs.

| Folder              | Purpose |
|---------------------|---------|
| `sam3_annotation/`  | SAM3-assisted annotation: automatic target-plant masks with frozen SAM3, browser and matplotlib tools for manual correction. The corrected masks form the Manual Mask reference set; the uncorrected masks are the SAM3 zero-shot baseline. |
| `benchmark/`        | Fixed train/validation manifest, training of SAM2.1, SegFormer-B2, DeepLabV3+, YOLO26-Seg, BiRefNet-HR and Center BiRefNet, validation threshold selection, test-set prediction and evaluation (per image, overall, per age group), mask and overlay export, and the age-group figure. |

The two parts use separate Python environments: SAM 3 has its own install
requirements (see `sam3_annotation/README.md`), the benchmark uses
`benchmark/requirements.txt` (the same file is referenced by
`requirements.txt` here).

## Order of work

1. Run `sam3_annotation` on the photographs (`segment` or `run_recursive`).
2. Correct every mask in the review app and save; the saved
   `mask_pngs/<stem>.mask.png` files are the Manual Masks.
3. In `benchmark/`: build the test manifest and the train/validation manifest,
   train the six models, run SAM3 (uncorrected) on the test images, predict the
   test images, evaluate, export masks and overlays, draw the figure.
4. Copy the BiRefNet-HR and Center BiRefNet checkpoints and the Center BiRefNet
   thresholds to `weights/` for the deployment pipeline.

## Weights

No weights are distributed. The files in `weights/` (repository root) that the
pipeline needs come from the benchmark runs:

| File                         | Produced by | Copied from |
|------------------------------|-------------|-------------|
| `weights/birefnet_hr.pt`     | `python -m plant_benchmark.cli train --model birefnet_hr --protocol finetuned --fold 0` | `benchmark/outputs/runs/birefnet_hr/finetuned/fixed_split/best.pt` |
| `weights/center_birefnet.pt` | `python scripts/train_center_birefnet.py train --fold 0` | `benchmark/outputs/runs/birefnet_hr_centered/finetuned/fixed_split/best.pt` |
| `weights/threshold.json`     | `python scripts/train_center_birefnet.py --config <test inference config> predict --fold 0` (run by `run_test_evaluation.sh`) | `benchmark/outputs/runs/birefnet_hr_centered/finetuned/fixed_split/threshold.json` |

The copy commands are in `benchmark/README.md`. SAM3 weights come from Meta
(`facebook/sam3` on Hugging Face).
