"""Synthetic-mask tests for architecture.py — lock the trait math to known shapes.

Runnable with pytest, or directly from the repo root:
    ``.venv-sam3/bin/python PlantArchitecture/tests/test_architecture.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# architecture.py is the standalone module one level up (PlantArchitecture/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from architecture import analyze_plant


def _meta(stem="syn", h=200, w=200, seed=None):
    return {
        "image_name": f"{stem}.jpg",
        "height": h,
        "width": w,
        "seed_xy": seed or [w // 2, h - 1],
        "mask_confidence": 1.0,
    }


def _green_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = (30, 200, 40)  # leafy green
    return rgb


def test_rectangle_basic_geometry():
    mask = np.zeros((200, 200), dtype=bool)
    mask[40:160, 60:120] = True  # height 120, width 60, area 7200
    r = analyze_plant(mask, _green_rgb(mask), _meta())
    assert r["area_px"] == 7200
    assert r["canopy_height_px"] == 120
    assert r["canopy_width_px"] == 60
    assert abs(r["aspect_ratio_hw"] - 2.0) < 0.02
    assert r["solidity"] > 0.97          # rectangle is convex
    assert r["extent"] > 0.97            # fills its bbox
    assert r["rectangularity"] > 0.95
    assert r["circularity"] < 1.0


def test_filled_ellipse_shape():
    mask = np.zeros((240, 240), dtype=bool)
    canvas = np.zeros((240, 240), dtype=np.uint8)
    cv2.ellipse(canvas, (120, 120), (90, 40), 0, 0, 360, 255, -1)  # wide, flat ellipse
    mask = canvas.astype(bool)
    r = analyze_plant(mask, _green_rgb(mask), _meta(h=240, w=240))
    assert r["solidity"] > 0.95          # convex blob
    assert 0.70 < r["extent"] < 0.82     # ~pi/4 for an ellipse in its bbox
    assert r["eccentricity"] > 0.8       # elongated
    assert r["aspect_ratio_hw"] < 0.6    # wider than tall


def test_hole_counts_as_porosity():
    mask = np.zeros((200, 200), dtype=bool)
    mask[40:160, 40:160] = True
    mask[80:120, 80:120] = False         # one interior hole
    r = analyze_plant(mask, _green_rgb(mask), _meta())
    assert r["n_holes"] == 1
    assert r["holes_area_px"] == 1600
    assert r["holes_area_frac"] > 0.0
    assert r["solidity"] < 1.0           # hole lowers area/hull


def test_left_right_asymmetry():
    # mass shifted to the right of the stem-base x -> positive lr_balance
    mask = np.zeros((200, 200), dtype=bool)
    mask[40:160, 100:180] = True
    r = analyze_plant(mask, _green_rgb(mask), _meta(seed=[100, 199]))
    assert r["lr_balance"] > 0.5
    assert r["asymmetry"] > 0.5
    assert r["lateral_lean"] > 0.0


def test_green_canopy_colour():
    mask = np.zeros((200, 200), dtype=bool)
    mask[40:160, 40:160] = True
    r = analyze_plant(mask, _green_rgb(mask), _meta())
    assert r["exg_mean"] > 0.2           # clearly green
    assert r["frac_green"] > 0.8
    assert r["vari_mean"] <= 1.0         # clipped, not blown up
    assert abs(r["tgi_mean"]) <= 100.0   # TGI is tens-scale on 0-1 reflectance, but bounded
    assert -1.0 <= r["centroid_height_frac"] <= 1.0


def test_empty_mask_is_no_plant():
    mask = np.zeros((200, 200), dtype=bool)
    r = analyze_plant(mask, None, _meta())
    assert r["no_plant"] is True
    assert "area_px" not in r


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
