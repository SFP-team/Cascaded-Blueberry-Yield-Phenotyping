"""Manual mask-review / cleanup UI for SAM3_new.

A lightweight click-to-remove editor for the persisted center-plant masks
(`masks/<stem>.mask.npy` + `<stem>.meta.json`). When the automatic seed-component
fix still leaves a stray background blob (or drops something it shouldn't), open the
image, click unwanted blobs to delete those connected components, and save the
cleaned mask back in place so all downstream use consumes the corrected mask.

Core functions are GUI-free and unit-testable; `InteractiveMaskEditor` is a thin
matplotlib front-end over them.

Run:  python -m SAM3_new.mask_review <output_dir> [--stem W24140706]
Keys: click = remove the clicked blob (COMPONENT mode) | b = toggle BRUSH mode
      (drag to erase — for background plants merged into the center plant as one
      blob) | [ / ] = brush size | k = keep only the stem-seed blob | u = undo |
      r = reset | s = save | n/right = next image | q = quit
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .io_utils import load_upright

MASKS_SUBDIR = "masks"
PREVIEW_SUBDIR = "review_edits"


# --------------------------------------------------------------------------- core

def list_stems(output_dir: Path) -> list[str]:
    masks_dir = Path(output_dir) / MASKS_SUBDIR
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"No '{MASKS_SUBDIR}/' in {output_dir}.")
    return sorted(p.name[: -len(".mask.npy")] for p in masks_dir.glob("*.mask.npy"))


def load(output_dir: Path, stem: str):
    """Return (rgb, mask, meta, mask_path, meta_path) for one stem."""
    masks_dir = Path(output_dir) / MASKS_SUBDIR
    mask_path = masks_dir / f"{stem}.mask.npy"
    meta_path = masks_dir / f"{stem}.meta.json"
    mask = np.load(mask_path).astype(bool)
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    rgb = load_upright(Path(meta["source_path"]), meta.get("rotate")) if meta.get("source_path") else None
    if rgb is not None and rgb.shape[:2] != mask.shape:
        rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]))
    return rgb, mask, meta, mask_path, meta_path


def component_at(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    """Boolean mask of the connected component containing pixel (x, y); empty if off-mask."""
    h, w = mask.shape
    x, y = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
    if not mask[y, x]:
        return np.zeros_like(mask)
    n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    return labels == labels[y, x]


def remove_component_at(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    """Mask with the connected component under (x, y) removed (no-op if off-mask)."""
    return mask & ~component_at(mask, x, y)


def keep_only_seed(mask: np.ndarray, seed_xy) -> np.ndarray:
    """Keep only the seed-connected component (the one-click auto cleanup)."""
    if not mask.any() or seed_xy is None:
        return mask
    h, w = mask.shape
    n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    x, y = int(np.clip(seed_xy[0], 0, w - 1)), int(np.clip(seed_xy[1], 0, h - 1))
    lab = labels[y, x]
    if lab == 0:
        ys, xs = np.nonzero(mask)
        nn = np.argmin((xs - x) ** 2 + (ys - y) ** 2)
        lab = labels[ys[nn], xs[nn]]
    return labels == lab


def erase_disk(mask: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    """Erase a filled disk of `radius` px centered at (x, y) — for connected leaks
    where the background plants merge into the center plant (one component, so
    component-delete can't separate them). Returns a new mask."""
    out = mask.astype(np.uint8).copy()
    cv2.circle(out, (int(x), int(y)), max(1, int(radius)), 0, -1)
    return out.astype(bool)


def open_separate(mask: np.ndarray, seed_xy, radius: int) -> np.ndarray:
    """Morphological opening that severs thin bridges then keeps the seed-connected
    component, restored to its original extent. Snaps apart background plants that
    are only thinly connected to the center plant (then brush any remnants)."""
    if not mask.any() or seed_xy is None:
        return mask
    k = max(3, int(radius) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(mask.astype(np.uint8), ker).astype(bool)
    if not eroded.any():
        return mask
    kept = keep_only_seed(eroded, seed_xy)
    grown = cv2.dilate(kept.astype(np.uint8), ker).astype(bool)
    return grown & mask


def save(output_dir: Path, stem: str, mask: np.ndarray, rgb=None) -> dict:
    """Overwrite the persisted mask, mark the meta edited, and write a preview PNG."""
    rgb_, _, meta, mask_path, meta_path = load(output_dir, stem)
    rgb = rgb if rgb is not None else rgb_
    np.save(mask_path, mask.astype(bool))
    meta["manually_edited"] = True
    meta["edited_mask_area"] = int(mask.sum())
    if meta_path.is_file() or meta:
        meta_path.write_text(json.dumps(meta, indent=2))
    if rgb is not None:
        prev_dir = Path(output_dir) / PREVIEW_SUBDIR
        prev_dir.mkdir(parents=True, exist_ok=True)
        vis = rgb.astype(np.float32)
        vis[mask] = vis[mask] * 0.45 + np.array([0, 200, 0], np.float32) * 0.55
        cv2.imwrite(str(prev_dir / f"{stem}.edited.png"), cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR))
    return meta


# ----------------------------------------------------------------- interactive UI

class InteractiveMaskEditor:
    """matplotlib click-to-remove editor over one output directory."""

    def __init__(self, output_dir: Path, stems: list[str]):
        self.output_dir = Path(output_dir)
        self.stems = stems
        self.idx = 0
        self._load_current()

    def _load_current(self):
        self.stem = self.stems[self.idx]
        self.rgb, self.mask, self.meta, _, _ = load(self.output_dir, self.stem)
        self.original = self.mask.copy()
        self.history: list[np.ndarray] = []
        self.dirty = False
        self.mode = "component"   # 'component' (click=delete blob) | 'brush' (drag=erase)
        self.brush = max(6, min(self.mask.shape) // 80)
        self._stroking = False

    def _overlay(self) -> np.ndarray:
        base = self.rgb if self.rgb is not None else np.zeros((*self.mask.shape, 3), np.uint8)
        vis = base.astype(np.float32).copy()
        vis[self.mask] = vis[self.mask] * 0.45 + np.array([0, 200, 0], np.float32) * 0.55
        return vis.astype(np.uint8)

    def run(self):  # pragma: no cover - interactive
        import matplotlib.pyplot as plt

        self.fig, self.ax = plt.subplots(figsize=(11, 9))
        self.im = self.ax.imshow(self._overlay())
        self.ax.set_axis_off()
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._title()
        plt.tight_layout()
        plt.show()

    def _title(self):  # pragma: no cover - interactive
        n = cv2.connectedComponents(self.mask.astype(np.uint8))[0] - 1
        star = " *unsaved*" if self.dirty else ""
        mode = f"BRUSH(r={self.brush})" if self.mode == "brush" else "COMPONENT"
        self.ax.set_title(
            f"[{self.idx+1}/{len(self.stems)}] {self.stem}  -  {n} blob(s)  -  mode={mode}{star}\n"
            "b=brush/component   [ ]=brush size   click/drag=erase   o=snap-separate   "
            "k=keep-seed   u=undo   r=reset   s=save   n=next   q=quit",
            fontsize=9)
        self.fig.canvas.draw_idle()

    def _push(self):  # pragma: no cover - interactive
        self.history.append(self.mask.copy())

    def _on_press(self, e):  # pragma: no cover - interactive
        if e.inaxes != self.ax or e.xdata is None:
            return
        self._push()
        if self.mode == "brush":
            self._stroking = True
            self.mask = erase_disk(self.mask, e.xdata, e.ydata, self.brush)
        else:
            new = remove_component_at(self.mask, int(e.xdata), int(e.ydata))
            if new.sum() == self.mask.sum():
                self.history.pop()  # nothing changed; don't keep a no-op undo step
                return
            self.mask = new
        self.dirty = True
        self.im.set_data(self._overlay()); self._title()

    def _on_motion(self, e):  # pragma: no cover - interactive
        if self._stroking and e.inaxes == self.ax and e.xdata is not None:
            self.mask = erase_disk(self.mask, e.xdata, e.ydata, self.brush)
            self.im.set_data(self._overlay()); self._title()

    def _on_release(self, e):  # pragma: no cover - interactive
        self._stroking = False

    def _on_key(self, e):  # pragma: no cover - interactive
        if e.key == "b":
            self.mode = "brush" if self.mode == "component" else "component"
        elif e.key == "]":
            self.brush = int(self.brush * 1.4) + 1
        elif e.key == "[":
            self.brush = max(2, int(self.brush / 1.4))
        elif e.key == "o":
            self._push(); self.mask = open_separate(self.mask, self.meta.get("seed_xy"), self.brush * 3); self.dirty = True
        elif e.key == "k":
            self._push(); self.mask = keep_only_seed(self.mask, self.meta.get("seed_xy")); self.dirty = True
        elif e.key == "u" and self.history:
            self.mask = self.history.pop(); self.dirty = True
        elif e.key == "r":
            self._push(); self.mask = self.original.copy(); self.dirty = True
        elif e.key == "s":
            save(self.output_dir, self.stem, self.mask, self.rgb); self.dirty = False
            print(f"saved {self.stem} (area {int(self.mask.sum())})")
        elif e.key in ("n", "right"):
            self.idx = (self.idx + 1) % len(self.stems); self._load_current()
            self.im.set_data(self._overlay())
        elif e.key == "q":
            import matplotlib.pyplot as plt; plt.close(self.fig); return
        self.im.set_data(self._overlay()); self._title()


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="SAM3_new.mask_review",
                                description="Manually clean center-plant masks (click to remove blobs).")
    p.add_argument("output_dir", help="A SAM3_new segment output dir containing masks/.")
    p.add_argument("--stem", default=None, help="Edit a single image (by file stem) instead of all.")
    args = p.parse_args(argv)
    stems = [args.stem] if args.stem else list_stems(Path(args.output_dir))
    if not stems:
        print("No masks found."); return 1
    InteractiveMaskEditor(Path(args.output_dir), stems).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
