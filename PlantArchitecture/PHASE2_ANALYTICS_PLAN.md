# Phase 2 — Blueberry Plant Analytics Plan (what we can extract from one mask + RGB)

**Input per plant:** one full-res binary mask (`True` = center plant), the aligned RGB photo, and
the stem-base pixel `seed_xy = (x0, y0)` near the ground (from `PLANT_ARCHITECTURE.md`).
**Compute everything from `mask` (geometry) + `rgb[mask]` (color).** No depth, no torch — pure
`numpy / opencv / scikit-image / scipy`, optionally `PlantCV`.

This plan is the synthesis of three literature reviews (blueberry agronomy, 2D-mask computer-vision
phenotyping, RGB color/health + scale). Sources are listed at the bottom.

---

## 0. The one hard constraint: this is a single 2D side-view

A side silhouette is a *projection*. It captures the outline and the visible surface color — not the
solid, not depth, not the occluded interior. This cleanly splits every candidate trait into four
tiers. **The most-cited anchor for the 2D-vs-3D line is Wu et al. 2019 (Plant Methods):** 1-D side
measurements (height, in-row width) hit R²≈0.92–0.95 vs LiDAR, while volume, crown diameter, and true
shape descriptors *required* multi-angle 3D to beat occlusion. Build confidently on the first; label
the proxies as proxies; don't claim the rest.

---

## 1. Master trait catalog (tiered by how much we can trust it)

### TIER 1 — Robust core (build first; survives occlusion; high agronomic value)

| Trait | Agronomic meaning (blueberry) | Compute | Scale |
|---|---|---|---|
| **Canopy height** | growth/vigor, yield potential, pruning & harvester header height | top mask row → `seed_xy.y` ground line | px → cm w/ scale |
| **In-row canopy width** | growth, spacing/hedging, biomass | mask col extent (bbox) | px → cm |
| **Silhouette area** | size / biomass / vigor proxy | `mask.sum()` | px² → cm² |
| **Perimeter, equiv. diameter, Feret max** | size scalars | `regionprops` / `cv2` | px → cm |
| **Aspect ratio (H/W)** | **bush habit: upright vs spreading** — top cultivar descriptor; drives machine-harvest fit | bbox H/W, or ellipse axes | **scale-free** |
| **Solidity = area/convex-hull** | canopy openness/density; compact vs ragged/spreading | `regionprops.solidity` | **scale-free** |
| **Extent, eccentricity, circularity, convexity** | habit / compactness fingerprint | `regionprops` + `cv2` | **scale-free** |
| **Hu moments (7)** | rotation/scale-invariant shape signature for clustering cultivars | `cv2.HuMoments` | **scale-free** |
| **Width-vs-height profile `w(y)`** | the silhouette shape curve: where the bush is widest | per-row `mask[y,:].sum()` | **scale-free (normalized)** |
| **Crown/base vs top width ratio** | **vase vs columnar vs mounding** form (harvestability) | mean `w(y)` top 20% ÷ bottom 20% | **scale-free** |
| **Max-width height fraction** | top-heavy (spreading) vs mid (ovate) | `argmax(w(y))` / height | **scale-free** |
| **Centroid-vs-base offset & lean** | lateral lean, vertical balance | `regionprops.centroid` − `seed_xy` | **scale-free** |
| **Top-heaviness, L/R asymmetry** | mass distribution, lopsidedness | mass split at mid-height / at `x0` | **scale-free** |
| **Basal/"skirt" clearance height** | catch-plate seating for mechanical harvest | gap from `seed_xy` to lowest dense canopy | px → cm |
| **Color health indices** (ExG, GLI, VARI, NGRDI, TGI, GCC) | **vigor**, greenness/biomass, N/chlorophyll proxy | per-pixel on `rgb[mask]`, then mean/percentiles | **scale-free** |
| **Color fractions** green / yellow / brown / red (% canopy) | chlorosis, senescence, anthocyanin/stress load | threshold hue + Lab `a*/b*` bins | **scale-free** |
| **Hue circular mean/std, Lab a*/b* means** | health fingerprint, illumination-robust | `PlantCV analyze_color` (circular stats) | **scale-free** |

> These directly answer the questions a blueberry lab asks: *how big, what habit (upright/spreading),
> how vigorous, how healthy, how harvestable.* Most are scale-free, so they work even with **no cm
> calibration**.

### TIER 2 — Useful proxies (build second; report explicitly as proxies)

| Trait | Meaning | Compute | Caveat |
|---|---|---|---|
| **Canopy porosity / gap fraction** = 1 − solidity; internal holes | openness → light, airflow, disease/SWD pressure, pruning need | hull−mask; `binary_fill_holes`; `euler_number` | only *outline-level* openness, not true interior light penetration |
| **Canopy fill ratio (extent)** | density within envelope | `regionprops.extent` | — |
| **2D-area → volume surrogate** | volume correlates w/ yield (r≈0.74) | area × width assumption (ellipsoid) | needs a shape assumption; state it |
| **Texture (GLCM contrast/homogeneity/entropy, LBP, edge density)** | canopy roughness/density, leaf-edge fineness | `skimage.feature.graycomatrix` / `local_binary_pattern` on `rgb[mask]` | illumination-sensitive; only compare matched lighting |
| **Color phenology coverage** — % flower-colored (white/pink), % ripe-blue pixels | bloom load, ripe-fruit load proxy (yield) | HSV/Lab thresholds inside an eroded mask | **coverage proxy, not counts** — YOLO does counts |

### TIER 3 — Exploratory (behind a flag; noisy; never report as absolute)

| Trait | Why tempting | Why risky |
|---|---|---|
| **Skeleton / medial axis, branch points, endpoints, branch angles, tortuosity, hierarchy** | "architecture," cane/branch structure | a 2D projection merges overlapping branches; medial-axis spawns **spurious spurs** on every boundary bump → over-/mis-counted, unstable run-to-run. Mandatory aggressive prune; treat as *relative* only. (`PlantCV morphology.*`) |
| **Fractal dimension (box-counting), lacunarity** | canopy complexity / gappiness | scale-window- and grid-placement-sensitive; regression breaks over limited plant scales. Average over grid origins, restrict scale window, report cautiously. |

### TIER 4 — NOT obtainable from one 2D mask (needs stereo/3D/LiDAR or manual)

True canopy **volume**, **across-row width (depth)**, **crown diameter** (occluded, the key
machine-harvest ground-loss metric), **individual cane number/age**, validated **3D shape descriptors**,
interior **fruiting-surface distribution / light penetration**, roots. *Don't attempt these from the
silhouette — flag them as "requires multi-view/3D" if the lab wants them later.*

---

## 2. Proposed deliverables

1. **`SAM3_new/architecture.py`** — pure `mask → traits` (no torch). CLI:
   `python -m SAM3_new.architecture <output_dir> [--scale cm_per_px] [--tier 1|2|3]`.
   Reads existing `masks/*.mask.npy` + `*.meta.json` + the source JPEG; re-runs instantly on existing
   masks (decoupled from segmentation).
2. **`output_*/architecture/architecture_report.csv`** — one row per plant, columns grouped by tier;
   every length column emitted in **both px and cm** (cm only if a scale is given), ratios unitless.
3. **`output_*/architecture/figures/<stem>.png`** — per-plant annotated panel: mask + bbox + convex
   hull + stem base + width-profile curve + color-fraction bar — so each number is eyeball-verifiable.
4. **`tests/test_architecture.py`** — synthetic masks (rectangle, ellipse, asymmetric blob) with
   known solidity/aspect/area to lock the math.

**Reuse, don't reinvent:** `PlantCV pcv.analyze.size()` and `skimage.regionprops_table` give most of
Tier 1 size+shape in one call; `pcv.analyze.color()` gives the color histograms + circular hue stats;
`pcv.morphology.*` is the whole Tier-3 skeleton suite. We only hand-roll the profiles (Group 3),
fractal/lacunarity, and color-fraction binning.

---

## 3. Phasing

- **Phase 2a (core, ~1 module):** Tier 1 → CSV + figures + tests. Delivers habit, vigor, health, size
  on day one; mostly scale-free so it's useful before any calibration.
- **Phase 2b (proxies):** add Tier 2 (porosity, texture, 2D-volume surrogate, color phenology).
- **Phase 2c (exploratory, flagged):** Tier 3 skeleton + fractal, clearly labeled "relative."
- **Scale module (parallel, only if a reference exists):** ArUco/known-object/known-distance →
  `cm_per_px`; otherwise Tier 1/2 ratios + color carry the analysis.

---

## 4. Decisions (LOCKED 2026-06-16)

1. **Primary goal → cultivar/phenotyping + yield modeling + health/stress mapping** (not pruning-led).
   So the leads are: shape ratios / profiles / Hu moments / color fingerprint (phenotyping); size,
   area, 2D-volume surrogate, color flower-&-ripe-fruit coverage (yield); color indices + canopy
   color fractions (health). Pruning-specific traits (skirt clearance, asymmetry) stay in — they're
   cheap — but are not prioritized.
2. **Scale → ratios/pixels only.** No reference object available. Emit **pixel + scale-free ratio +
   color** traits. Lengths are reported in **px**; an optional `--scale cm_per_px` flag adds cm
   columns later without code changes. All ratio/shape/color traits need no calibration.
3. **Scope → Tier 1 + Tier 2.** Build both. Skeleton/fractal (Tier 3) deferred. Tier 4 (3D) out.
4. **Color phenology → include as a coverage proxy** (training-free %flower-colored / %ripe-blue),
   clearly labeled as coverage, not counts — YOLO does counts downstream.

---

## 5. Sources (fetched)

**Blueberry agronomy:** Wu et al. 2019 *3D point cloud to characterize shrub size/shape* (Plant
Methods, PMC6441659) · Bian et al. 2024 *Blueberry Crop Ontology* (HortScience 59(10):1433) ·
Nuñez *Shining a Light on the Blueberry Canopy* (FL Blueberry Growers) · UVM & UConn pruning guides ·
Cmeddad et al. *flower-bud → yield regression* (Scientia Horticulturae) · Strik *pruning severity /
balance* (HortScience, OSU; Growing Produce).
**2D CV phenotyping:** scikit-image `regionprops` · OpenCV contour properties · PlantCV
`analyze.size` / `morphology` / `analyze.color` · Frontiers 2025 & Bucksch on skeleton pitfalls ·
ImageJ FracLac (box-counting + lacunarity) · ScienceDirect on box-counting limits.
**Color / phenology / scale:** Sukhova et al. (PMC12073610, RGB indices vs NDVI/PRI) · Hunt et al.
(TGI) · SegVeg & maize senescence (color senescence) · TREx anthocyanin index · PlantCV `analyze_color` ·
wild-blueberry blue-pixel yield (R²≈0.98, ScienceDirect S1537511010000905) · UGA blueberry phenology ·
Pix4D/InertialLabs GSD · ArUco/fiducial size calibration (Pysource, PyImageSearch) · geometric
morphometrics on scale-invariant shape (PMC6041757).
