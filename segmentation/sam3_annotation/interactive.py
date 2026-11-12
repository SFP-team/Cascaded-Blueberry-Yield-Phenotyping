"""Interactive SAM 3 "smart add" for the mask-review app.

A thin, lazily-built wrapper around the existing `Sam3HybridSegmenter` that turns
a single click (point) or a dragged box into a fresh SAM 3 mask, so a reviewer can
*complete* a center-plant mask where the automatic pipeline missed a region — the
caller then unions the returned mask into the working mask.

The heavy backend (torch + SAM 3 weights) is only built on first use, so reviewers
who merely erase/paint never pay the load cost. Per-image `set_image` state is
cached by the caller (the review server) so repeat clicks on the same photo are fast.
"""

from __future__ import annotations

import numpy as np

# Avoid the full-frame grab: a single click should never return (almost) the whole
# image. Generous so legitimately large plants still come through.
MAX_ADD_AREA_FRAC = 0.85


class InteractiveSAM:
    """Lazy SAM 3 interactive segmenter for point/box "add" prompts."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._seg = None  # built on first use

    # ------------------------------------------------------------------ backend
    def _ensure(self):
        if self._seg is None:
            from .config import Sam3NewConfig
            from .segmenter import Sam3HybridSegmenter
            self._seg = Sam3HybridSegmenter(Sam3NewConfig(device=self.device))
        return self._seg

    @property
    def loaded(self) -> bool:
        return self._seg is not None

    def set_image(self, rgb: np.ndarray) -> dict:
        """Encode `rgb` and return the per-image SAM state (cache it per stem)."""
        return self._ensure().set_image(rgb)

    # -------------------------------------------------------------------- prompts
    def _predict(self, state, *, point_coords=None, point_labels=None, box=None):
        from .segmenter import _to_numpy
        masks, scores, _ = self._seg.model.predict_inst(
            state,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=None,
            multimask_output=True,
        )
        masks_np = _to_numpy(masks, dtype=bool)
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        scores_np = _to_numpy(scores, dtype=np.float32).reshape(-1)
        return masks_np, scores_np

    @staticmethod
    def _pick(masks_np, scores_np, *, contains=None, max_area_frac=MAX_ADD_AREA_FRAC):
        """Choose the best candidate: prefer masks that contain `contains=(x,y)` and
        sit under the area cap, ranked by SAM score; fall back to the smallest mask."""
        if masks_np is None or len(masks_np) == 0:
            return None
        h, w = masks_np.shape[1:]
        order = list(range(len(masks_np)))
        areas = [float(masks_np[i].mean()) for i in order]

        def ok(i):
            if areas[i] <= 0 or areas[i] > max_area_frac:
                return False
            if contains is not None:
                cx, cy = int(np.clip(contains[0], 0, w - 1)), int(np.clip(contains[1], 0, h - 1))
                if not masks_np[i, cy, cx]:
                    return False
            return True

        eligible = [i for i in order if ok(i)]
        if eligible:
            best = max(eligible, key=lambda i: scores_np[i] if i < len(scores_np) else 0.0)
            return masks_np[best]
        # nothing matched the constraints — take the smallest non-empty mask
        nonempty = [i for i in order if areas[i] > 0]
        if not nonempty:
            return None
        return masks_np[min(nonempty, key=lambda i: areas[i])]

    def add_from_point(self, state, x: float, y: float,
                       max_area_frac: float = MAX_ADD_AREA_FRAC) -> np.ndarray | None:
        """Full-res boolean mask for a single positive click at (x, y) — or None."""
        pts = np.array([[float(x), float(y)]], dtype=np.float32)
        lab = np.array([1], dtype=np.int32)
        masks_np, scores_np = self._predict(state, point_coords=pts, point_labels=lab)
        return self._pick(masks_np, scores_np, contains=(x, y), max_area_frac=max_area_frac)

    def add_from_box(self, state, box, max_area_frac: float = MAX_ADD_AREA_FRAC) -> np.ndarray | None:
        """Full-res boolean mask for a dragged box [x0, y0, x1, y1] — or None."""
        b = np.asarray(box, dtype=np.float32).reshape(-1)
        try:
            masks_np, scores_np = self._predict(state, box=b)
        except Exception:
            # box-only prompting unsupported on this build — use the box centre point
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            return self.add_from_point(state, cx, cy, max_area_frac=max_area_frac)
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        return self._pick(masks_np, scores_np, contains=(cx, cy), max_area_frac=max_area_frac)
