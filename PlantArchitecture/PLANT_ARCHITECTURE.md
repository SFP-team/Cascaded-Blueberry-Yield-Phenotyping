# Plant Architecture — Phase 2 (start-here context for a fresh chat)

This repo has two phases. **Phase 1 (done): segment the center blueberry plant** with SAM 3 →
one clean mask per image. **Phase 2 (this is what's next): measure *plant architecture* traits
*inside* those masks** — i.e. describe each plant's structure using only the masked region
("the mask area only"). Read this whole file first; it is the handoff.

> "Mask area only" = compute everything from the segmented center-plant mask (+ the photo pixels
> under it). Neighbour/background plants are already excluded by Phase 1, so architecture metrics
> are attributed to the correct plant.

---

## 1. What Phase 1 already does (short — UI excluded)

A rover drives a blueberry row and photographs one target plant per frame, but neighbour plants
share the frame. The pipeline isolates **only the center plant**.

Pipeline (`python -m SAM3_new.cli segment <in> <out> --device cpu`), in [`SAM3_new/`](SAM3_new):

1. **Ingest** (`io_utils`) — load EXIF-upright RGB from a folder.
2. **Stem anchor** (`stem_anchor`) — Excess-Green vegetation map → ground-line estimate → a
   **stem seed** placed low in the central canopy (70th-percentile row of central vegetation);
   from it a center-biased plant box + positive/negative geometric prompt boxes.
3. **Segment** (`segmenter.Sam3HybridSegmenter`) — SAM 3 `set_image`, then a concept-prompt chain
   `"blueberry bush"→"blueberry plant"→"plant"` with geometric boxes; candidates **scored** by
   spatial priors (center, seed-inside, box overlap, ground contact, size, side penalty); best one
   refined with a two-pass `predict_inst`. Stem-only fallback if grounding finds nothing.
4. **Clean** (`mask_ops`) — **keep only the stem-seed-connected component**. This is the decisive
   step: SAM often returns the target plant *plus* a disconnected background row along the horizon;
   keeping the seed's component drops the background and keeps the rooted plant whole.
5. **Report** — overlays, mask PNGs, `masks/*.mask.npy` + meta, reports.

Result on the 41-image eval set: a leak-free single-component mask for every image, mean confidence
**0.83**. Only genuinely hard case = a background bush physically *touching* the center canopy
(one connected component → can't be split geometrically; cleaned by hand). Validated **CPU-only on
Apple Silicon**.

*(There is also a browser/matplotlib mask editor — intentionally out of scope here; Phase 2 doesn't
touch the UI.)*

---

## 2. What's on disk — the inputs to Phase 2

```
Testimages/                         # 41 source JPEGs (rover photos), EXIF-upright
output_Testimages/
  masks/<stem>.mask.npy             # ★ the plant mask: bool ndarray, FULL image res e.g. (5184, 3456)
  masks/<stem>.meta.json            # ★ per-plant metadata (see fields below)
  overlays/<stem>.overlay.png       # green mask + seed/box, confidence caption (for eyeballing)
  mask_pngs/<stem>.mask.png         # binary mask as PNG
  mask_previews/<stem>.mask_preview.png
  neighbor_masks/<stem>.npy         # non-selected candidates (ignore for architecture)
  segmentation_report.{csv,json}    # per-image confidence/area summary
  sam3_diagnostics.{csv,json}       # prompt_used, candidate scores, fallback_reason
```

**`<stem>.meta.json` fields you'll use:**
`image_name`, `source_path` (abs path to the JPEG), `height`, `width` (full image px),
`mask_confidence`, **`seed_xy` = `[x, y]` of the stem base** (low-central, near the ground — the
natural *base reference* for height/architecture), `pred_iou`, `prompt_used`,
`manually_edited` (bool — true if a human corrected it), `no_plant`.

**Load a plant in Python:**
```python
import json, numpy as np
from PIL import Image
stem = "W24120501"
mask = np.load(f"output_Testimages/masks/{stem}.mask.npy")          # bool (H, W), True = plant
meta = json.load(open(f"output_Testimages/masks/{stem}.meta.json"))
rgb  = np.asarray(Image.open(meta["source_path"]).convert("RGB"))   # same (H, W) as mask
base_x, base_y = meta["seed_xy"]                                     # stem base in px
```
The mask and the RGB photo are the **same resolution and aligned**, so every architecture metric is
just geometry on `mask` (+ colour from `rgb[mask]` when needed).

---

## 3. Environment / how to run

- **`.venv-sam3/`** — Python 3.12 venv, `numpy<2` (REQUIRED). Use `.venv-sam3/bin/python`.
- **`../sam3/`** — patched SAM 3 repo (REQUIRED only for *re-segmenting*; Phase 2 architecture math
  needs only numpy / opencv / scikit-image / Pillow, **no torch, no SAM weights**).
- Re-run segmentation (rarely needed):
  `HF_TOKEN=… HF_HUB_DISABLE_XET=1 PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-sam3/bin/python -m SAM3_new.cli segment <in> <out> --device cpu`
- For pure mask analysis you can install `scikit-image scipy` into `.venv-sam3` if missing.

---

## 4. Phase 2 — "plant architecture" inside the mask

**Goal:** for each masked plant, produce a row of **structural / morphological traits** (and the
intermediate visualisations to trust them). Everything is computed *only* from the mask region.

Candidate traits (2D, derivable from one mask + photo — **confirm the wanted set with the user
before building**, this list is a starting point, not a decision):

| group | trait | how (sketch) |
|---|---|---|
| size | canopy height, width, area | mask bbox extents; `mask.sum()` |
| size | crown/base width vs top width, height:width aspect | width profile near `seed_xy` vs top |
| shape | convex-hull area, **solidity** (area/hull), compactness | `cv2.convexHull` / contour |
| shape | width-as-function-of-height profile, centroid offset | row-wise mask widths |
| shape | erectness / bush form (upright vs spreading) | profile + aspect + hull |
| structure | **skeleton / medial axis**, main-cane count, branch points | `skimage.morphology.skeletonize` |
| density | canopy porosity / fill (gaps inside hull), green-pixel density | hull-minus-mask; `rgb[mask]` ExG |
| reference | plant base = `seed_xy`, ground line (re-derivable from `stem_anchor`) | for absolute height |

**Open questions to settle with the user in the new chat (do not assume):**
1. **Which traits matter** for their study (phenotyping? yield model? a fixed agronomic list?).
2. **Pixel → real-world scale** — are heights/widths needed in **cm**? If so we need a scale
   (camera height + GSD, a reference object/marker, or a known row/stake width). Without it, report
   **pixel-based + ratio** traits (ratios like solidity/aspect are scale-free and still useful).
3. **Output format** — a `architecture_report.csv` (one row per plant) + per-plant annotated
   figures? Where should it live (e.g. `output_*/architecture/`)?
4. **Skeleton/branch metrics** are the hard part (canopy occlusion, 2D ambiguity) — wanted, or keep
   to robust size/shape/density traits first?

**Suggested approach when you start:** add a new module `SAM3_new/architecture.py` (pure
mask→traits, no torch), a CLI `python -m SAM3_new.architecture <output_dir>` that reads the existing
`masks/*.mask.npy` + meta and writes `architecture_report.csv` + per-plant overlays, and a test in
`tests/`. Keep it decoupled from segmentation so it re-runs instantly on existing masks.

**Note on YOLO berry detection (separate, later):** the user will train a YOLO berry detector on
their Roboflow annotations and run it *inside* these masks to attribute berries per plant. That is a
distinct downstream step — not part of segmentation and not the same as architecture; mentioned only
so it isn't conflated. SAM/architecture comes first.

---

## 5. Constraints (carry over — still in effect)

- The web app / any server binds to **127.0.0.1 only** — nothing leaves the machine.
- **No Hugging Face token in any file or commit.** SAM 3 weights are **gated Meta weights** — do not
  redistribute (don't copy into the repo, don't push them).
- Commits are authored as **raghav-rathi `<rathi.raghav2409@gmail.com>`** with **no AI/Claude
  attribution**.
- Save/edit operations overwrite masks in place — no automatic backup beyond session undo.
