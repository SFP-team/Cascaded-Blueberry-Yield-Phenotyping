# Plant Architecture (Stage 2)

Extracts **81 plant-architecture / phenotyping traits** per plant from the SAM 3 center-plant
masks produced by [../Segmentation](../Segmentation). Pure-Python (scikit-image, OpenCV, scipy),
ratios/pixels only — no scale needed unless you pass `--scale`. `architecture.py` is standalone
(no `SAM3_new` import) and uses the **shared** Python 3.12 venv `../.venv-sam3`.

## Run

```bash
# from the repo root:
.venv-sam3/bin/python PlantArchitecture/architecture.py Segmentation/results --out-dir PlantArchitecture/results

# or from inside this folder:
../.venv-sam3/bin/python architecture.py ../Segmentation/results --out-dir results
```

Reads `<output_dir>/masks/*.mask.npy` + `*.meta.json`; writes `architecture_report.{csv,json}`
(one row per plant × 81 trait columns) + per-plant `figures/` into `--out-dir`.
Flags: `--scale CM_PER_PX` (adds cm/cm² columns), `--no-figures`, `--limit N`.

## Contents
- `architecture.py` — analytics module (size, shape, profile, porosity, colour/vegetation indices, GLCM texture).
- `PHASE2_ANALYTICS_PLAN.md` — trait catalog + literature + design decisions.
- `PLANT_ARCHITECTURE.md` — project handoff notes.
- `tests/test_architecture.py` — synthetic-mask unit tests: `../.venv-sam3/bin/python tests/test_architecture.py`
- `results/` — the generated report + figures (41 plants × 81 traits).
