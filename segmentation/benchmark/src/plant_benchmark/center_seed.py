"""Leakage-safe prompt selection for Center BiRefNet.

The fixed-split training manifest has two different prompt semantics:

* ``seed_x, seed_y`` are the positive points consumed by SAM2.1.  During
  manifest construction, points outside the manually corrected target were
  snapped to the nearest reference-foreground pixel.
* Center BiRefNet only needs an image-location prior.  Feeding it a point that
  was moved using the reference mask would leak target information into the
  model input.  It must therefore use the preserved pre-repair coordinates.

This module deliberately never reads a reference mask.  It fails closed if a
reference-repaired row does not preserve its original point.  A metadata-only
fallback is provided for external/test manifests that pre-date the provenance
columns; it reads only ``seed_xy`` from the metadata JSON.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


SAFE_ORIGINAL_SOURCES = frozenset(
    {
        "metadata_seed_xy",
        "sam3_image_only_stem_anchor",
    }
)
REFERENCE_DERIVED_MARKERS = (
    "reference_checked",
    "nearest_foreground",
    "reference_snapped",
)


@dataclass(frozen=True)
class CenterSeed:
    """A Center BiRefNet seed and its auditable selection provenance."""

    x: float
    y: float
    source: str
    policy: str
    used_preserved_original: bool
    reference_mask_read: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return not str(value).strip()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return False
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _finite_pair(x: Any, y: Any) -> tuple[float, float] | None:
    try:
        x_value, y_value = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x_value) and math.isfinite(y_value)):
        return None
    return x_value, y_value


def _reference_derived(source: str) -> bool:
    normalized = source.strip().lower()
    return any(marker in normalized for marker in REFERENCE_DERIVED_MARKERS)


def _validate_bounds(
    pair: tuple[float, float], width: Any, height: Any, *, label: str
) -> tuple[float, float]:
    try:
        width_value, height_value = int(width), int(height)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid image dimensions {width!r}x{height!r}") from exc
    x, y = pair
    if width_value <= 0 or height_value <= 0:
        raise ValueError(f"{label}: non-positive image dimensions")
    if not (0.0 <= x < width_value and 0.0 <= y < height_value):
        raise ValueError(
            f"{label}: seed {(x, y)} is outside {width_value}x{height_value}"
        )
    return pair


def _metadata_seed(row: Any) -> tuple[float, float] | None:
    raw_path = _get(row, "metadata_path", "")
    if _blank(raw_path):
        return None
    metadata_path = Path(str(raw_path)).expanduser()
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read seed metadata: {metadata_path}") from exc
    seed = payload.get("seed_xy")
    if not isinstance(seed, (list, tuple)) or len(seed) < 2:
        return None
    pair = _finite_pair(seed[0], seed[1])
    if pair is None:
        raise ValueError(f"{metadata_path}: seed_xy is not finite")
    return pair


def resolve_center_seed(row: Any) -> CenterSeed:
    """Resolve a Center BiRefNet prompt without consulting the target mask.

    Selection order is intentionally conservative:

    1. For a repaired/reference-derived SAM point, require and use the
       preserved original coordinates and original source.
    2. For a manifest row whose current source is already one of the two
       approved original sources, use the current coordinates.
    3. For a legacy external/test manifest without source columns, read only
       ``seed_xy`` from its metadata JSON.

    No reference mask is opened and no nearest-foreground operation is
    permitted here.
    """

    label = str(_get(row, "stem", "row"))
    width, height = _get(row, "width"), _get(row, "height")
    raw_current_source = _get(row, "seed_source", "")
    current_source = "" if _blank(raw_current_source) else str(raw_current_source).strip()
    repaired = _truthy(_get(row, "seed_repaired", False))

    if repaired or _reference_derived(current_source):
        raw_original_source = _get(row, "seed_source_original", "")
        original_source = (
            "" if _blank(raw_original_source) else str(raw_original_source).strip()
        )
        original = _finite_pair(
            _get(row, "seed_x_original"),
            _get(row, "seed_y_original"),
        )
        if original is None or original_source not in SAFE_ORIGINAL_SOURCES:
            raise ValueError(
                f"{label}: reference-repaired SAM seed lacks a complete, approved "
                "pre-repair seed provenance"
            )
        x, y = _validate_bounds(original, width, height, label=label)
        return CenterSeed(
            x=x,
            y=y,
            source=original_source,
            policy="preserved_pre_reference_repair_coordinate",
            used_preserved_original=True,
        )

    current = _finite_pair(_get(row, "seed_x"), _get(row, "seed_y"))
    if current_source in SAFE_ORIGINAL_SOURCES and current is not None:
        x, y = _validate_bounds(current, width, height, label=label)
        return CenterSeed(
            x=x,
            y=y,
            source=current_source,
            policy="unmodified_manifest_original_coordinate",
            used_preserved_original=False,
        )

    # Older test manifests do not carry seed-source provenance.  Their raw
    # metadata is an admissible image-acquisition input; the reference mask is
    # neither opened nor needed.
    metadata = _metadata_seed(row)
    if metadata is not None:
        x, y = _validate_bounds(metadata, width, height, label=label)
        return CenterSeed(
            x=x,
            y=y,
            source="metadata_seed_xy",
            policy="metadata_only_legacy_manifest_fallback",
            used_preserved_original=False,
        )

    raise ValueError(
        f"{label}: no leakage-safe Center BiRefNet seed is available. Preserve "
        "the original metadata/image-only point before reference repair."
    )


def sam21_seed(row: Any) -> tuple[float, float]:
    """Return the existing SAM2.1 positive point unchanged.

    This helper exists only to make the intentional protocol separation
    explicit.  It does not alter or reinterpret the frozen manifest values.
    """

    label = str(_get(row, "stem", "row"))
    pair = _finite_pair(_get(row, "seed_x"), _get(row, "seed_y"))
    if pair is None:
        raise ValueError(f"{label}: SAM2.1 seed is not finite")
    return _validate_bounds(
        pair,
        _get(row, "width"),
        _get(row, "height"),
        label=label,
    )
