from __future__ import annotations

import numpy as np

from segmentation.sam3_annotation.config import Sam3NewConfig
from segmentation.sam3_annotation.report import result_row, diagnostic_row, write_reports
from segmentation.sam3_annotation.segmenter import Candidate, Sam3HybridSegmenter, SegmentationResult, _score_bundle
from segmentation.sam3_annotation.stem_anchor import StemSeed


class FakeProcessor:
    def __init__(self, prompt_map):
        self.prompt_map = prompt_map
        self.conf = 0.35
        self.last_prompt = None

    def set_image(self, image, state=None):
        return {
            "original_height": image.height if hasattr(image, "height") else image.shape[0],
            "original_width": image.width if hasattr(image, "width") else image.shape[1],
            "backbone_out": {"sam2_backbone_out": {}},
        }

    def reset_all_prompts(self, state):
        return state

    def set_confidence_threshold(self, threshold, state=None):
        self.conf = threshold
        return state

    def set_text_prompt(self, prompt, state):
        self.last_prompt = prompt
        payload = self.prompt_map[prompt]
        state.update(payload)
        return state

    def add_geometric_prompt(self, box, label, state):
        state.setdefault("geo", []).append((box, label))
        return state


class FakeModel:
    def __init__(self, refine_mask, refine_score):
        self.refine_mask = refine_mask
        self.refine_score = refine_score

    def predict_inst(self, state, **kwargs):
        if kwargs.get("multimask_output", True):
            masks = state["masks"]
            scores = state["scores"]
            logits = np.asarray(masks, dtype=np.float32)
            return masks, scores, logits
        return np.asarray([self.refine_mask], dtype=bool), np.asarray([self.refine_score], dtype=np.float32), np.asarray([self.refine_mask], dtype=np.float32)


def test_center_scoring_prefers_central_candidate():
    cfg = Sam3NewConfig()
    rgb_shape = (100, 100, 3)
    seed = StemSeed(
        points=np.array([[50, 80]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
        center_box_xyxy=np.array([30, 30, 70, 95], dtype=np.float32),
        prompt_boxes=[],
        ground_line_y=95,
        confidence=1.0,
        is_fallback=False,
    )
    center = Candidate(mask=np.pad(np.ones((20, 20), bool), ((40, 40), (40, 40))), score=0.9, box=np.array([35, 35, 65, 92], dtype=np.float32), index=0, prompt="plant")
    side = Candidate(mask=np.pad(np.ones((20, 20), bool), ((40, 40), (0, 80))), score=0.95, box=np.array([0, 35, 20, 92], dtype=np.float32), index=1, prompt="plant")
    scores, idx = _score_bundle([center, side], seed, rgb_shape)
    assert idx == 0
    assert scores[0] > scores[1]


def test_report_schema_is_compatible(tmp_path):
    result = SegmentationResult(
        image_name="x.jpg",
        height=10,
        width=10,
        mask=np.zeros((10, 10), dtype=bool),
        core=np.zeros((10, 10), dtype=bool),
        seed_xy=(5, 5),
        seed_is_fallback=False,
        no_plant=True,
        pred_iou=0.1,
        agreement=0.2,
        mask_confidence=0.3,
    )
    row = result_row(result)
    assert list(row) == ["image", "width", "height", "no_plant", "seed_x", "seed_y", "seed_is_fallback", "mask_area", "mask_area_frac", "core_area", "pred_iou", "agreement", "mask_confidence"]
    diag = diagnostic_row(result)
    assert "fallback_reason" in diag
    written = write_reports([result], tmp_path, "both")
    assert any(p.name == "segmentation_report.csv" for p in written)
    assert any(p.name == "sam3_diagnostics.csv" for p in written)


def test_segmenter_falls_back_when_refine_loses_seed():
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    prompt_map = {
        "blueberry bush": {
            "masks": np.asarray([mask]),
            "boxes": np.asarray([[5, 5, 15, 15]], dtype=np.float32),
            "scores": np.asarray([0.9], dtype=np.float32),
        }
    }
    cfg = Sam3NewConfig(fallback_to_stem_only=False)
    seed = StemSeed(
        points=np.array([[10, 10]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
        center_box_xyxy=np.array([5, 5, 15, 15], dtype=np.float32),
        prompt_boxes=[],
        ground_line_y=15,
        confidence=1.0,
        is_fallback=False,
    )
    bad_refine = np.zeros((20, 20), dtype=bool)
    bad_refine[1:8, 1:8] = True
    seg = Sam3HybridSegmenter(cfg, model=FakeModel(refine_mask=bad_refine, refine_score=0.4), processor=FakeProcessor(prompt_map))
    seg.set_image(rgb)
    state, cands, chosen, bundle_scores, idx = seg._choose_candidate(rgb, seed, "blueberry bush")
    refined_mask, refined_score, reason = seg._refine_candidate(rgb, seed, state, chosen)
    assert reason == "refine_lost_seed"
    assert refined_mask.sum() > 0
