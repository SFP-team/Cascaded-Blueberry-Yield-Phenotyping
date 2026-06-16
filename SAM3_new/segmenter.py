"""SAM3 hybrid segmentation backend for the standalone pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Sam3NewConfig
from .mask_ops import extract_core, mask_iou, multimask_agreement, postprocess_mask
from .stem_anchor import StemSeed, find_stem_seed

log = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Candidate:
    mask: np.ndarray
    score: float
    box: np.ndarray | None
    index: int
    prompt: str


@dataclass
class SegmentationResult:
    image_name: str
    height: int
    width: int
    mask: np.ndarray
    core: np.ndarray
    seed_xy: tuple[int, int]
    seed_is_fallback: bool
    no_plant: bool
    pred_iou: float
    agreement: float
    mask_confidence: float
    box: np.ndarray | None = None
    seed_points: np.ndarray | None = None
    seed_labels: np.ndarray | None = None
    confidence_map: np.ndarray | None = None
    neighbor_masks: list[np.ndarray] = field(default_factory=list)
    prompt_used: str | None = None
    sam3_candidate_count: int = 0
    sam3_selected_score: float = 0.0
    sam3_text_iou: float | None = None
    sam3_refined_iou: float | None = None
    neighbor_candidate_count: int = 0
    fallback_reason: str | None = None
    center_selection_score: float = 0.0

    @property
    def mask_area(self) -> int:
        return int(self.mask.sum())

    @property
    def core_area(self) -> int:
        return int(self.core.sum())


def _mask_area_frac(mask: np.ndarray) -> float:
    area = mask.shape[0] * mask.shape[1] or 1
    return float(mask.sum() / area)


def _to_numpy(value, dtype=None) -> np.ndarray:
    """Convert SAM3 torch/np/list outputs to CPU numpy arrays."""
    if value is None:
        arr = np.asarray([])
    elif hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _box_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - inter
    return float(inter / union) if union else 0.0


def _candidate_center_score(
    cand: Candidate,
    seed: StemSeed,
    rgb_shape: tuple[int, int, int],
) -> float:
    h, w = rgb_shape[:2]
    ys, xs = np.nonzero(cand.mask)
    if xs.size == 0:
        return 0.0
    cx, cy = float(xs.mean()), float(ys.mean())
    center_dist = np.hypot(cx - w / 2, cy - h / 2) / (np.hypot(w / 2, h / 2) + 1e-6)
    center_score = 1.0 - min(1.0, center_dist)
    seed_x, seed_y = seed.foreground_xy
    seed_score = 1.0 if cand.mask[min(h - 1, seed_y), min(w - 1, seed_x)] else 0.0
    box_overlap = _box_iou(cand.box, seed.center_box_xyxy)
    bottom = float(ys.max())
    ground_y = seed.ground_line_y if seed.ground_line_y is not None else h * 0.9
    ground_score = 1.0 - min(1.0, abs(bottom - ground_y) / max(1.0, h * 0.20))
    side_touch = float(cand.mask[:, :3].any() or cand.mask[:, -3:].any())
    side_penalty = 1.0 if side_touch else 0.0
    area_frac = _mask_area_frac(cand.mask)
    size_score = 1.0 - min(1.0, abs(area_frac - 0.25) / 0.25)
    base = (
        0.28 * float(cand.score)
        + 0.22 * center_score
        + 0.22 * seed_score
        + 0.14 * box_overlap
        + 0.08 * ground_score
        + 0.06 * size_score
        - 0.10 * side_penalty
    )
    return float(max(0.0, min(1.0, base)))


def _score_bundle(cands: list[Candidate], seed: StemSeed, rgb_shape) -> tuple[list[float], int]:
    scores = [_candidate_center_score(c, seed, rgb_shape) for c in cands]
    idx = int(np.argmax(scores)) if scores else -1
    return scores, idx


class Sam3HybridSegmenter:
    def __init__(
        self,
        cfg: Sam3NewConfig,
        model=None,
        processor=None,
    ):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.model = model
        self.processor = processor
        self._state = None

        if self.model is None or self.processor is None:
            self._load_backend()

    def _load_backend(self) -> None:
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            raise RuntimeError(
                "SAM 3 is not installed. Clone facebookresearch/sam3 and install it in the same env."
            ) from e

        bpe = self.cfg.sam3_bpe_path
        ckpt = str(self.cfg.sam3_checkpoint) if self.cfg.sam3_checkpoint else None
        try:
            self.model = build_sam3_image_model(
                bpe_path=bpe,
                device=self.device,
                enable_inst_interactivity=True,
                checkpoint_path=ckpt,
                load_from_HF=ckpt is None,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build SAM 3 backend: {e}") from e

        if self.device != "cuda":
            self.model = self.model.float()
        self.processor = Sam3Processor(self.model, device=self.device, confidence_threshold=self.cfg.sam3_confidence_thresh)

    def set_image(self, rgb: np.ndarray) -> dict:
        from PIL import Image

        self._state = self.processor.set_image(Image.fromarray(rgb))
        return self._state

    def _apply_prompt_boxes(self, state: dict, seed: StemSeed) -> dict:
        for box, label in seed.prompt_boxes:
            state = self.processor.add_geometric_prompt(box.tolist(), bool(label), state)
        return state

    def _prompt_candidates(self, rgb: np.ndarray, seed: StemSeed, prompt: str) -> tuple[dict, list[Candidate]]:
        state = self._state
        self.processor.reset_all_prompts(state)
        state = self.processor.set_confidence_threshold(self.cfg.sam3_confidence_thresh, state)
        try:
            state = self.processor.set_text_prompt(prompt, state)
        except RuntimeError as e:
            if "storage of a tensor" in str(e) and "mps" in str(e):
                log.warning("SAM3 text grounding failed due CPU/MPS geometry encoder mismatch; falling back to interactive prompts")
                return state, []
            raise
        try:
            state = self._apply_prompt_boxes(state, seed)
        except RuntimeError as e:
            if "storage of a tensor" in str(e) and "mps" in str(e):
                log.warning("SAM3 geometric box prompts failed due CPU/MPS mismatch; using text candidates without box prompts")
            else:
                raise
        masks = _to_numpy(state.get("masks", []), dtype=bool)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        boxes = _to_numpy(state.get("boxes", []), dtype=np.float32)
        scores = _to_numpy(state.get("scores", []), dtype=np.float32).reshape(-1)

        cands: list[Candidate] = []
        h, w = rgb.shape[:2]
        for i in range(len(masks)):
            mask = masks[i]
            area_frac = _mask_area_frac(mask)
            if area_frac < self.cfg.min_mask_area_frac or area_frac > self.cfg.max_mask_area_frac:
                continue
            box = boxes[i] if i < len(boxes) else None
            if box is not None and len(box) == 4:
                box = np.asarray(box, dtype=np.float32)
                if box[2] < box[0] or box[3] < box[1]:
                    box = None
            score = float(scores[i]) if i < len(scores) else 0.0
            cands.append(Candidate(mask=mask, score=score, box=box, index=i, prompt=prompt))
        return state, cands

    def _choose_candidate(self, rgb: np.ndarray, seed: StemSeed, prompt: str) -> tuple[dict, list[Candidate], Candidate | None, list[float], int]:
        state, cands = self._prompt_candidates(rgb, seed, prompt)
        scores, idx = _score_bundle(cands, seed, rgb.shape) if cands else ([], -1)
        chosen = cands[idx] if idx >= 0 else None
        return state, cands, chosen, scores, idx

    def _fallback_to_stem_only(self, rgb: np.ndarray, seed: StemSeed, prompt: str) -> tuple[dict, Candidate | None, list[Candidate], list[float], int, str]:
        state = self.set_image(rgb)
        self.processor.reset_all_prompts(state)
        state = self.processor.set_confidence_threshold(self.cfg.sam3_confidence_thresh, state)
        # Use the stem seed as a purely interactive prompt.
        refined, scores, low_res = self.model.predict_inst(
            state,
            point_coords=seed.points,
            point_labels=seed.labels,
            box=seed.center_box_xyxy,
            mask_input=None,
        multimask_output=True,
        )
        cands = []
        refined_np = _to_numpy(refined, dtype=bool)
        if refined_np.ndim == 4 and refined_np.shape[1] == 1:
            refined_np = refined_np[:, 0]
        scores_np = _to_numpy(scores, dtype=np.float32).reshape(-1)
        for i, mask in enumerate(refined_np):
            area_frac = _mask_area_frac(mask)
            if area_frac < self.cfg.min_mask_area_frac or area_frac > self.cfg.max_mask_area_frac:
                continue
            cands.append(Candidate(mask=mask, score=float(scores_np[i]), box=seed.center_box_xyxy, index=i, prompt=prompt))
        bundle_scores, idx = _score_bundle(cands, seed, rgb.shape) if cands else ([], -1)
        chosen = cands[idx] if idx >= 0 else None
        return state, chosen, cands, bundle_scores, idx, "concept_grounding_failed"

    def _refine_candidate(
        self,
        rgb: np.ndarray,
        seed: StemSeed,
        state: dict,
        chosen: Candidate,
    ) -> tuple[np.ndarray, float, str | None]:
        if chosen is None:
            return np.zeros(rgb.shape[:2], dtype=bool), 0.0, "no_selected_candidate"

        ys, xs = np.nonzero(chosen.mask)
        if xs.size == 0:
            return chosen.mask, 0.0, "empty_selected_candidate"

        centroid = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)
        refine_coords = np.vstack([seed.points, centroid[None, :]])
        refine_labels = np.append(seed.labels, 1)

        pass1_masks, pass1_scores, pass1_low_res = self.model.predict_inst(
            state,
            point_coords=refine_coords,
            point_labels=refine_labels,
            box=chosen.box if chosen.box is not None else seed.center_box_xyxy,
            mask_input=None,
            multimask_output=True,
        )
        pass1_masks_np = _to_numpy(pass1_masks, dtype=bool)
        if pass1_masks_np.ndim == 4 and pass1_masks_np.shape[1] == 1:
            pass1_masks_np = pass1_masks_np[:, 0]
        pass1_scores_np = _to_numpy(pass1_scores, dtype=np.float32).reshape(-1)
        low_res_np = _to_numpy(pass1_low_res)
        if pass1_masks_np.size == 0:
            return chosen.mask, 0.0, "refine_no_masks"

        overlap_scores = np.array([mask_iou(m, chosen.mask) for m in pass1_masks_np], dtype=np.float32)
        combined = 0.55 * overlap_scores + 0.45 * pass1_scores_np[: len(overlap_scores)]
        pass1_idx = int(np.argmax(combined))

        refined, refined_scores, _ = self.model.predict_inst(
            state,
            point_coords=refine_coords,
            point_labels=refine_labels,
            box=chosen.box if chosen.box is not None else seed.center_box_xyxy,
            mask_input=low_res_np[pass1_idx : pass1_idx + 1],
            multimask_output=False,
        )
        refined_np = _to_numpy(refined, dtype=bool)
        if refined_np.ndim == 4 and refined_np.shape[1] == 1:
            refined_np = refined_np[:, 0]
        refined_scores_np = _to_numpy(refined_scores, dtype=np.float32).reshape(-1)
        refined_mask = refined_np[0]
        refined_score = float(refined_scores_np[0])

        area_frac = _mask_area_frac(refined_mask)
        if area_frac < self.cfg.min_mask_area_frac or area_frac > self.cfg.max_mask_area_frac:
            return chosen.mask, 0.0, "refine_area_out_of_range"

        seed_x, seed_y = seed.foreground_xy
        if not refined_mask[min(rgb.shape[0] - 1, seed_y), min(rgb.shape[1] - 1, seed_x)]:
            return chosen.mask, refined_score, "refine_lost_seed"

        if mask_iou(refined_mask, chosen.mask) < 0.20:
            return chosen.mask, refined_score, "refine_disagreed"

        return refined_mask, refined_score, None

    def segment(self, rgb: np.ndarray, image_name: str = "image") -> SegmentationResult:
        h, w = rgb.shape[:2]
        seed = find_stem_seed(rgb, self.cfg)
        self.set_image(rgb)

        prompt_chain = [self.cfg.text_prompt, *self.cfg.fallback_prompts]
        state = None
        chosen = None
        candidates: list[Candidate] = []
        scores: list[float] = []
        idx = -1
        used_prompt = None
        for prompt in prompt_chain:
            state, cands, maybe_chosen, bundle_scores, maybe_idx = self._choose_candidate(rgb, seed, prompt)
            if cands:
                state = self.processor.set_confidence_threshold(self.cfg.sam3_confidence_thresh, state)
                chosen = maybe_chosen
                candidates = cands
                scores = bundle_scores
                idx = maybe_idx
                used_prompt = prompt
                if chosen is not None:
                    break

        fallback_reason = None
        if chosen is None:
            if self.cfg.fallback_to_stem_only:
                state, chosen, candidates, scores, idx, fallback_reason = self._fallback_to_stem_only(rgb, seed, prompt_chain[0])
                used_prompt = f"fallback:{prompt_chain[0]}"
            else:
                empty = np.zeros((h, w), dtype=bool)
                return SegmentationResult(
                    image_name=image_name,
                    height=h,
                    width=w,
                    mask=empty,
                    core=empty,
                    seed_xy=seed.foreground_xy,
                    seed_is_fallback=seed.is_fallback,
                    no_plant=True,
                    pred_iou=0.0,
                    agreement=0.0,
                    mask_confidence=0.0,
                    seed_points=seed.points,
                    seed_labels=seed.labels,
                    prompt_used=None,
                    fallback_reason="no_candidates",
                )

        if chosen is None:
            empty = np.zeros((h, w), dtype=bool)
            return SegmentationResult(
                image_name=image_name,
                height=h,
                width=w,
                mask=empty,
                core=empty,
                seed_xy=seed.foreground_xy,
                seed_is_fallback=seed.is_fallback,
                no_plant=True,
                pred_iou=0.0,
                agreement=0.0,
                mask_confidence=0.0,
                seed_points=seed.points,
                seed_labels=seed.labels,
                prompt_used=used_prompt,
                fallback_reason=fallback_reason or "no_candidates",
            )

        neighbor_masks = [c.mask for i, c in enumerate(candidates) if i != idx]
        agreement = multimask_agreement(np.asarray([c.mask for c in candidates], dtype=bool)) if candidates else 1.0
        refined_mask, refined_score, refine_reason = self._refine_candidate(rgb, seed, state, chosen)
        final_mask = refined_mask if refined_mask.any() else chosen.mask
        pred_iou = refined_score if refined_mask.any() else 0.0
        text_iou = mask_iou(chosen.mask, final_mask)
        center_selection_score = scores[idx] if idx >= 0 and idx < len(scores) else 0.0

        if refine_reason is not None:
            fallback_reason = refine_reason
        if not final_mask.any():
            fallback_reason = fallback_reason or "empty_final_mask"

        full = postprocess_mask(
            final_mask, seed.foreground_xy, self.cfg.min_component_area, self.cfg.dilation_margin,
            keep_seed_component=self.cfg.keep_seed_component,
            seed_component_close_frac=self.cfg.seed_component_close_frac,
        )
        core = extract_core(final_mask, seed.foreground_xy, self.cfg.core_erosion_frac)
        no_plant = not full.any()

        mask_confidence = self._combine_confidence(
            pred_iou=pred_iou,
            agreement=agreement,
            text_iou=text_iou,
            center_selection_score=center_selection_score,
        )

        return SegmentationResult(
            image_name=image_name,
            height=h,
            width=w,
            mask=full,
            core=core,
            seed_xy=seed.foreground_xy,
            seed_is_fallback=seed.is_fallback,
            no_plant=no_plant,
            pred_iou=pred_iou,
            agreement=agreement,
            mask_confidence=mask_confidence,
            box=seed.center_box_xyxy,
            seed_points=seed.points,
            seed_labels=seed.labels,
            neighbor_masks=neighbor_masks,
            prompt_used=used_prompt,
            sam3_candidate_count=len(candidates),
            sam3_selected_score=float(chosen.score),
            sam3_text_iou=float(text_iou),
            sam3_refined_iou=float(refined_score),
            neighbor_candidate_count=len(neighbor_masks),
            fallback_reason=fallback_reason,
            center_selection_score=float(center_selection_score),
        )

    def _combine_confidence(
        self,
        pred_iou: float,
        agreement: float,
        text_iou: float,
        center_selection_score: float,
    ) -> float:
        terms = [
            (self.cfg.w_pred_iou, pred_iou),
            (self.cfg.w_agreement, agreement),
            (self.cfg.w_text_iou, text_iou),
            (self.cfg.w_center_score, center_selection_score),
        ]
        wsum = sum(w for w, _ in terms)
        return float(sum(w * v for w, v in terms) / wsum) if wsum else 0.0
