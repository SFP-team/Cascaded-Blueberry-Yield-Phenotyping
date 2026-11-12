# SAM3-assisted target-plant annotation

This tool produces the target-plant masks used as the Manual Mask reference set,
and the uncorrected SAM3 zero-shot masks used as a baseline in the segmentation
benchmark.

Workflow:

1. **Auto mask.** SAM3 (frozen weights) segments the center (target) plant in
   every photograph. A deterministic image step estimates the stem base and the
   upper central canopy. The prompts are one positive stem seed, three positive
   spine points, a center box, negative side points and side boxes, and the text
   prompt `blueberry bush` with fallbacks `blueberry plant` and `plant`.
   Candidate masks are ranked, refined with a second prompt pass, closed, and
   only the connected component that contains (or is nearest to) the stem seed
   is kept. Defaults are in `config.py`.
2. **Human correction.** Annotators open the masks in the browser review app
   (or the matplotlib editor), add missing plant regions, remove neighboring
   vegetation and background, and save. Saving overwrites the mask in place and
   records `manually_edited: true` and `edited_mask_area` in the metadata.
3. **Manual Mask set.** The saved `mask_pngs/<stem>.mask.png` files are the
   reference masks used to train and evaluate the segmentation models
   (`../benchmark`).

The SAM3 zero-shot baseline is step 1 alone, run on the test images without
correction.

## Install

The tool needs Meta's SAM 3 package, which is not included here. Follow the
installation instructions of the SAM 3 repository (Python and PyTorch versions
as listed there), then install this tool's requirements in the same
environment:

```bash
git clone https://github.com/facebookresearch/sam3.git
pip install -e ./sam3
pip install -r segmentation/sam3_annotation/requirements.txt
```

SAM 3 weights are gated on Hugging Face (`facebook/sam3`). Either log in with
`huggingface-cli login` after accepting the license, or download `sam3.pt` and
pass it with `--sam3-checkpoint`. Without a checkpoint path the weights are
loaded from Hugging Face.

All commands below are run from the repository root, so that the package is
importable as `segmentation.sam3_annotation`.

## 1. Auto mask

One folder of images:

```bash
python -m segmentation.sam3_annotation segment <input_dir> <output_dir> \
    --sam3-checkpoint /path/to/sam3.pt --device cuda
```

Options (defaults match the paper): `--text-prompt "blueberry bush"`,
`--fallback-prompts "blueberry plant" plant`, `--sam3-confidence-thresh 0.35`,
`--plant-box-width-frac 0.75`, `--negative-box-frac 0.22`,
`--rotate {0,90,180,270}` (instead of EXIF orientation),
`--report-format {csv,json,both}`, `--no-save-neighbor-masks`.
`--device` accepts `auto`, `cuda`, `mps` or `cpu`.

A nested tree (for example `<session>/<cohort>/raw/*.jpg`), loading the model
once and mirroring the folder tree in the output:

```bash
python -m segmentation.sam3_annotation.run_recursive \
    <input_root> <output_root> /path/to/sam3.pt cuda
```

Output of one folder:

```text
<output_dir>/
  masks/<stem>.mask.npy        boolean mask (the file edited by the review tools)
  masks/<stem>.meta.json       source_path, seed_xy, prompt_used, confidence, ...
  mask_pngs/<stem>.mask.png    binary PNG (0/255), same size as the upright image
  mask_previews/<stem>.mask_preview.png
  overlays/<stem>.overlay.png  mask, prompts and seed drawn on the photo
  neighbor_masks/              non-selected candidate masks (optional)
  report.csv, report.json
```

## 2. Human correction

Browser app (local only, binds to 127.0.0.1):

```bash
python -m segmentation.sam3_annotation.review_app <output_dir> [--device cpu] [--port 7860]
```

The page lists all images. Tools: click-delete a stray blob, erase or paint
with a brush, smart-add (click or draw a box and SAM 3 segments the missed
region), keep only the stem-seed component, separate thin bridges, fill holes,
undo, reset, save. Saving writes `masks/<stem>.mask.npy`, sets
`manually_edited` and `edited_mask_area` in `<stem>.meta.json`, and regenerates
`mask_pngs/`, `mask_previews/` and `overlays/` for that image.

On macOS, `launch_mask_review.command` starts the same app with a double-click.
Set `SAM3_PYTHON` to the interpreter of the SAM 3 environment and
`SAM3_REVIEW_DIR` to the output folder (default `sam3_results` in the
repository root).

A lighter matplotlib editor with the same core functions is also available:

```bash
python -m segmentation.sam3_annotation.mask_review <output_dir> [--stem <stem>]
```

Keys: click removes a blob, `b` toggles the erase brush, `[` and `]` change the
brush size, `k` keeps only the stem-seed blob, `u` undo, `r` reset, `s` save,
`n` or right arrow next image, `q` quit.

## 3. Manual Mask set for training

`../benchmark/scripts/build_fixed_split_manifest.py` expects the corrected
outputs next to the photographs, one output folder per cohort:

```text
<dataset>/<session>/<cohort>/raw/<stem>.jpg
<dataset>/<session>/<cohort>/results/masks/<stem>.mask.npy, <stem>.meta.json
<dataset>/<session>/<cohort>/results/mask_pngs/<stem>.mask.png
```

which is what `segment <session>/<cohort>/raw <session>/<cohort>/results`
produces. Cohort folder names start with `W20` (for example `W2023`). Every
image must have been saved in a review tool (`manually_edited: true`).

## Tests

```bash
pip install pytest
python -m pytest segmentation/sam3_annotation/tests
```

The tests use stub SAM 3 objects and do not need the SAM 3 package or weights.
