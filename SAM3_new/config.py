"""Configuration for the standalone SAM3_new pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Sam3NewConfig:
    device: str = "auto"
    sam3_checkpoint: Path | None = None
    sam3_bpe_path: str | None = None
    rotate: int | None = None

    text_prompt: str = "blueberry bush"
    fallback_prompts: tuple[str, ...] = ("blueberry plant", "plant")
    sam3_confidence_thresh: float = 0.35
    plant_box_width_frac: float = 0.75
    negative_box_frac: float = 0.22

    use_negative_points: bool = True
    use_negative_boxes: bool = True
    save_neighbor_masks: bool = True

    min_mask_area_frac: float = 0.04
    max_mask_area_frac: float = 0.90
    min_component_area: int = 1500
    keep_seed_component: bool = True        # keep ONLY the stem-seed-connected blob (drops background plants)
    seed_component_close_frac: float = 0.0  # optional MORPH_CLOSE before seed-component keep (frac of min(H,W))
    dilation_margin: int = 0
    core_erosion_frac: float = 0.02

    w_pred_iou: float = 0.40
    w_agreement: float = 0.20
    w_text_iou: float = 0.20
    w_center_score: float = 0.20

    output_dir: Path = Path("output")
    report_format: str = "both"

    prompt_score_floor: float = 0.25
    fallback_to_stem_only: bool = True

    extra_prompt_labels: list[str] = field(default_factory=lambda: ["blueberry plant", "plant"])

