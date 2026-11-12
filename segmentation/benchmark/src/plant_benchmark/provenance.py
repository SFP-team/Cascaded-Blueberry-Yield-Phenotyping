"""Shared, immutable protocol declarations used across training and staging."""

from __future__ import annotations

from typing import Any


CENTER_SEED_PROTOCOL: dict[str, Any] = {
    "purpose": "image-location prior only",
    "coordinate_policy": (
        "pre-reference-repair metadata/image-only coordinate; no reference mask read"
    ),
    "reference_mask_reads_during_seed_selection": 0,
}
