# BlueberryDetector — ripe / unripe blueberry detection (YOLO)

A self-contained YOLO training + inference pipeline for detecting **ripe and unripe blueberries on
the plant**, trained on the Roboflow dataset `smart-fruit-phenomics-team/blueberry_raghavapp` (v6).
Kept in its own folder with its own venv so it can't disturb the SAM3 segmentation env
(`../.venv-sam3`, which is pinned to numpy<2).

This is the **downstream berry step** referenced in `../PLANT_ARCHITECTURE.md`: run the detector
*inside* the SAM3 center-plant masks to attribute berries per plant (see `predict.py --mask`).

---

## Which model? (research-backed default)

Default = **`yolo26s.pt`** — Ultralytics **YOLO26**, the latest generation (NMS-free / end-to-end,
explicitly optimized for **small-object** detection and edge/CPU inference). The installed
`ultralytics` (8.4.x) ships the full YOLO26 family, including a **`yolo26-p2.yaml`** high-resolution
P2 head that is purpose-built for tiny objects like individual berries.

Pick a variant with `--model`:

| model | when to use |
|---|---|
| `yolo26n.pt` | fastest / smallest, weakest accuracy |
| `yolo26s.pt` | **default** — best speed/accuracy balance, pretrained for transfer learning |
| `yolo26m.pt` | more accuracy if you have GPU/Colab |
| `yolo26-p2.yaml` | maximum small-berry recall (extra high-res head); trains from scratch, needs more data + compute |
| `yolo11s.pt` | proven YOLO11 fallback |

Why this and not "YOLOv6": YOLOv6 (2022) is several generations old. The Roboflow `yolo26` export
is just standard YOLO txt labels, so the dataset trains on any Ultralytics model — we use the newest.
Small, clustered berries are the key challenge, so the pipeline defaults to **`imgsz=1280`** and
keeps mosaic augmentation; raise resolution / use the P2 head for the hardest cases.

### What the literature says (researched 2026-06)

- **YOLO26 is the current best fit for fruit ripeness.** It is NMS-free / end-to-end, adds the MuSGD
  optimizer + ProgLoss/STAL, and is explicitly tuned for **small-object** accuracy and fast CPU/edge
  inference (~43% faster CPU than YOLO11n). In a published tomato-ripeness benchmark, **YOLO26n
  reached 95.2% precision vs 91.8% (YOLO12s) and 90.4% (YOLO11s)**. See
  [Roboflow YOLO26](https://blog.roboflow.com/yolo26/),
  [YOLO26 architecture (arXiv 2509.25164)](https://arxiv.org/pdf/2509.25164),
  [YOLO evolution overview (arXiv 2510.09653)](https://arxiv.org/pdf/2510.09653),
  [YOLO26-RipeLoc tomato ripeness (arXiv 2605.27129)](https://arxiv.org/html/2605.27129).
- **Blueberry-on-bush detection is hard** — leaf occlusion + tiny, clustered fruit. Dedicated models
  (STF-YOLO, YOLO-Blueberry) report ~**80% mAP50** and ~72% counting accuracy across blue/purple/green
  maturity in real orchards.
  [STF-YOLO (PMC12695768)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12695768/),
  [YOLO-Blueberry (preprints.org)](https://www.preprints.org/frontend/manuscript/36e0fbd71ae86268a81c1da2e55c1e17/download_pub).
- **Small objects in high-res frames → train at high `imgsz`, or use SAHI slicing.** At `imgsz=640`
  a berry in a 5184×3456 rover photo is a few pixels — letterboxing shrinks it further. Train at
  1280 (default here), use the `yolo26-p2.yaml` high-res head, and for the full rover frames consider
  **SAHI** sliced training/inference (tile into 640px patches so each berry keeps its pixels). See
  [Ultralytics small-object / SAHI discussion](https://github.com/ultralytics/ultralytics/issues/24732).

**Bottom line:** default to **YOLO26** with transfer learning from COCO, high resolution + strong
augmentation; escalate to the P2 head and/or SAHI for the hardest occluded/tiny-berry frames; and run
the detector **inside the SAM3 plant masks** (`predict.py --mask`) so berry counts attribute to the
correct plant.

---

## Setup

The venv and deps are already installed by the assistant. To recreate from scratch:

```bash
python3 -m venv .venv-yolo
.venv-yolo/bin/python -m pip install -U pip wheel
.venv-yolo/bin/python -m pip install -r requirements.txt
```

## 1. Download the dataset

The Roboflow API key is read from an environment variable and **never stored in any file**:

```bash
export ROBOFLOW_API_KEY=your_key_here          # get from Roboflow → Settings → API
.venv-yolo/bin/python download_data.py
```

Lands in `datasets/blueberry_raghavapp-v6/` with `train/ valid/ test/` and a `data.yaml`.
(Override format/version with `ROBOFLOW_FORMAT`, `ROBOFLOW_VERSION`.)

> ⚠️ **Security:** never paste your API key into a file or commit it. If a key was ever shared in
> plaintext (chat, screenshot), rotate it in Roboflow. `.gitignore` already excludes `datasets/`,
> `runs/`, `*.pt`, `.env`, and `.venv-yolo/`.

## 2. Train

```bash
.venv-yolo/bin/python train.py --epochs 100 --imgsz 1280
# faster smoke test:
.venv-yolo/bin/python train.py --model yolo26n.pt --epochs 3 --imgsz 640 --batch 4 --name smoke
```

Auto-detects device (CUDA → MPS → CPU). On Apple Silicon it uses the **MPS GPU** but is slow for
full runs — prefer a CUDA GPU / Colab for the real 100-epoch train. Outputs (weights, curves,
confusion matrix, val predictions) go to `runs/<name>/`.

## 3. Predict / evaluate

```bash
# annotated detections on a folder
.venv-yolo/bin/python predict.py --weights runs/blueberry_yolo26s/weights/best.pt --source ../Testimages

# attribute berries to one plant via its SAM3 mask (per-plant ripe/unripe counts)
.venv-yolo/bin/python predict.py --weights runs/blueberry_yolo26s/weights/best.pt \
    --source ../Testimages/W24120501.jpg --mask ../output_Testimages/masks/W24120501.mask.npy
```

`val` metrics (mAP50, mAP50-95, per-class P/R) are computed automatically at the end of training and
saved under `runs/<name>/`.

---

## Files

| file | purpose |
|---|---|
| `download_data.py` | pull the Roboflow dataset (env-var API key) |
| `train.py` | train a YOLO detector (model/imgsz/epochs/batch configurable) |
| `predict.py` | run inference; optional per-plant attribution via a SAM3 `.npy` mask |
| `requirements.txt` | `ultralytics`, `roboflow` |
| `datasets/`, `runs/`, `.venv-yolo/` | gitignored (data, outputs, venv) |
