# Cascaded Image Segmentation and Object Detection for Yield Phenotyping in Blueberry Breeding

Code for the paper of the same name (under review). The workflow assigns berry and flower counts and plant traits to
the right genotype in field photos of southern highbush blueberry, where neighbouring bushes overlap the plant being
measured:

1. **Target-plant segmentation.** A mask of the plant in the centre of the photo, separated from neighbouring bushes,
   sky and ground. SAM3 drafts the training masks, people correct them, and supervised models learn from the
   corrected set. The deployed pipeline uses Center BiRefNet.
2. **Berry and flower detection.** A YOLO26-L detector finds immature berries, mature berries and flowers on
   640 px slices of the full-resolution photo. A detection is counted only when the centre of its box lies inside
   the target-plant mask.
3. **Image-derived phenotypes.** Canopy size, shape, colour and texture traits measured from the photo and its mask.
4. **Hand-harvest prediction.** Within-mask berry counts and selected phenotypes predict the hand-harvested berry
   count of each plant with Ridge regression and PLSR under leave-one-genotype-out cross-validation.

![Pipeline output on a test photo](examples/pipeline_example_01.jpg)

## Repository layout

| Folder | Contents |
|---|---|
| [`segmentation/sam3_annotation`](segmentation/sam3_annotation) | SAM3-assisted mask drafting and the review tool used to correct masks by hand |
| [`segmentation/benchmark`](segmentation/benchmark) | training and test-set evaluation of SAM2.1, SegFormer-B2, DeepLabV3+, YOLO26-Seg, BiRefNet-HR and Center BiRefNet, and the SAM3 zero-shot baseline |
| [`detection`](detection) | tiling, training and evaluation of the berry and flower detector |
| [`traits`](traits) | image-derived plant traits from a photo and its target-plant mask |
| [`pipeline`](pipeline) | end-to-end inference: a folder of photos in, masks, counts, traits and one Excel workbook out |
| [`analysis`](analysis) | within-mask detection for every mask condition, trait comparison, statistics, timing, and the R harvest-count models |
| [`examples`](examples) | pipeline outputs on three test photos and one test photo segmented by every model |
| [`validation_data`](validation_data) | the hand-harvest data of the paper: harvest table, ground truth and a link to the 59 photos |

Each folder has its own README with the data layout it expects, the commands in order, and its outputs.

## Workflow

1. Draft and correct target-plant masks with [`segmentation/sam3_annotation`](segmentation/sam3_annotation). The
   corrected masks are the Manual Mask set.
2. Train and evaluate the segmentation models with [`segmentation/benchmark`](segmentation/benchmark). This writes
   `weights/birefnet_hr.pt`, `weights/center_birefnet.pt` and `weights/threshold.json`.
3. Train and evaluate the detector with [`detection`](detection). This writes `weights/blueberry_yolo26l.pt`.
4. Run [`pipeline`](pipeline) on new photos.
5. Compare mask conditions and fit the harvest-count models with [`analysis`](analysis).

## Installation

Python 3.10 and an NVIDIA GPU are expected; the code was run with PyTorch 2.6.0 and CUDA 12.4.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r pipeline/requirements.txt        # inference: pipeline + traits
pip install -r segmentation/requirements.txt    # segmentation training and evaluation
pip install -r detection/requirements.txt       # detector training and evaluation
pip install -r analysis/requirements.txt        # downstream analysis (Python part)
```

The SAM3 annotation tool needs Meta's `sam3` package and checkpoint; see
[`segmentation/sam3_annotation`](segmentation/sam3_annotation). The harvest-count models need R; see
[`analysis`](analysis).

## Running the pipeline

```bash
python pipeline/run.py --input path/to/photos --output results/run_1 --device cuda:0
```

The run writes one workbook (`Plant_Architecture_and_Detections.xlsx`) with a row per photo, plus the masks,
mask overlays, detection overlays and per-photo detection files. Sharding across several GPUs is described in
[`pipeline/README.md`](pipeline/README.md).

## Weights and data

Trained weights are not distributed. The pipeline reads them from `weights/` at the repository root:

| File | Written by |
|---|---|
| `weights/birefnet_hr.pt` | BiRefNet-HR training in [`segmentation/benchmark`](segmentation/benchmark) |
| `weights/center_birefnet.pt` | Center BiRefNet training in [`segmentation/benchmark`](segmentation/benchmark) |
| `weights/threshold.json` | Center BiRefNet prediction on the validation set in [`segmentation/benchmark`](segmentation/benchmark) |
| `weights/blueberry_yolo26l.pt` | [`detection/train.py`](detection/train.py) |

[`validation_data`](validation_data) holds the hand-harvest data of the paper: the counts of 59 hand-harvested plants
(32 genotypes), the table the harvest-count models were fitted on, and a download of the 59 photos. The other field
photos and the annotations are not public. [`examples`](examples) shows pipeline outputs on three test photos.

## Examples

![One test photo segmented by every model](examples/segmentation_models_example_01.jpg)

More in [`examples`](examples).

## License

The code in this repository is released under the GNU Affero General Public License v3.0 ([`LICENSE`](LICENSE)).
Third-party packages and pretrained weights keep their own licenses: Ultralytics (AGPL-3.0), BiRefNet (MIT),
SAM 2 (Apache-2.0), and SAM 3 and the pretrained SegFormer weights under their owners' terms.

## Citation

The paper is under review. Citation details will be added here once it is published.
