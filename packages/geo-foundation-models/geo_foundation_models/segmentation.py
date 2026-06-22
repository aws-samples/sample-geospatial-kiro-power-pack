"""SAMGeo-style segmentation for the Foundation_Model_Server (Pillar C, Req 9).

``segment`` turns a validated imagery tile into a per-pixel label map, in the
spirit of `segment-geospatial`/SAMGeo. Two modes are supported, matching SAM's
prompt-guided and automatic mask generation:

* **Prompt-guided** - when ``prompts`` are supplied, each ``point`` / ``box``
  prompt seeds a segment. Box pixels take their box's label; remaining pixels
  are assigned to the nearest point prompt (a Voronoi partition), so the output
  is the familiar "one mask region per prompt" result.
* **Automatic** - when no prompts are supplied, the tile is partitioned by
  quantized pixel intensity into contiguous-value segments (or a single
  whole-tile segment when the tile carries no inlined pixel data).

The tile is validated **before** any segmentation work via the same
:func:`~geo_foundation_models.embedding.validate_tile` used by ``embed_tile``,
so an empty, oversized, or unsupported tile raises an ``Error_Taxonomy``
:class:`~geo_common.errors.ValidationError` and yields no mask (Requirement 9.7).
Prompt coordinates are likewise validated against the tile bounds.

Without GPU model weights the segmentation is deterministic and
content-dependent; the result shape (a label image + segment count) matches
what a real SAMGeo backend returns, so production weights can be dropped in
without changing the tool contract.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from geo_common.errors import ValidationError

from geo_foundation_models.embedding import (
    DEFAULT_SUPPORTED_FORMATS,
    coerce_raster_tile,
    validate_tile,
)
from geo_foundation_models.models import (
    MAX_TILE_DIMENSION,
    RasterTile,
    SegmentationMask,
)

__all__ = ["segment", "SAMGEO_MODEL_NAME", "DEFAULT_MAX_MASK_PIXELS"]

#: Source identifier used on errors raised by this server.
_SOURCE = "geo-foundation-models"

#: Name reported on a :class:`SegmentationMask` produced by this backend.
SAMGEO_MODEL_NAME = "SAMGeo"

#: Default cap on the number of pixels (``width * height``) a single ``segment``
#: call may return. The mask is a flat per-pixel label list, so a full
#: 1024x1024 tile would serialize ~1M integers - far too large for a calling
#: agent's context. A tile whose pixel count exceeds this cap raises a
#: ``ValidationError`` asking for a smaller tile, rather than returning an
#: unusable, oversized mask. 65,536 == a 256x256 tile. Overridable per server.
DEFAULT_MAX_MASK_PIXELS = 256 * 256

#: Maximum number of segments produced in automatic (prompt-free) mode.
_MAX_AUTO_SEGMENTS = 8


def segment(
    tile: RasterTile,
    prompts: Optional[Sequence[Dict]] = None,
    *,
    model: str = SAMGEO_MODEL_NAME,
    supported_formats: Iterable[str] = DEFAULT_SUPPORTED_FORMATS,
    max_dimension: int = MAX_TILE_DIMENSION,
    max_mask_pixels: int = DEFAULT_MAX_MASK_PIXELS,
) -> SegmentationMask:
    """Segment a tile SAMGeo-style, returning a :class:`SegmentationMask`.

    Validates the tile (and any prompt coordinates) **before** segmenting, so
    an empty / oversized / unsupported tile or an out-of-bounds prompt raises a
    ``ValidationError`` and produces no mask. With ``prompts`` the result has one
    region per prompt; without prompts it partitions by quantized intensity.

    The returned mask is a flat per-pixel label list, so a tile whose pixel
    count (``width * height``) exceeds ``max_mask_pixels`` (default 65,536, a
    256x256 tile) raises a ``ValidationError`` asking for a smaller tile rather
    than returning an oversized, hard-to-use mask.
    """
    # Normalize a raw JSON dict (uncoerced by the runtime) into a RasterTile so
    # a malformed tile is a clean ValidationError, then validate (Req 9.7).
    tile = coerce_raster_tile(tile)
    validate_tile(tile, supported_formats=supported_formats, max_dimension=max_dimension)

    pixel_count = tile.width * tile.height
    if pixel_count > max_mask_pixels:
        raise ValidationError(
            "tile %dx%d has %d pixels, exceeding the segmentation maximum of "
            "%d; use a smaller tile (<= %d pixels) so the per-pixel mask fits "
            "in the response"
            % (tile.width, tile.height, pixel_count, max_mask_pixels, max_mask_pixels),
            source=_SOURCE,
            detail={
                "parameter": "max_mask_pixels",
                "width": tile.width,
                "height": tile.height,
                "pixel_count": pixel_count,
                "max_mask_pixels": max_mask_pixels,
            },
        )

    width, height = tile.width, tile.height
    if prompts:
        labels = _segment_with_prompts(tile, prompts)
    else:
        labels = _segment_automatic(tile)

    num_segments = int(np.count_nonzero(np.unique(labels)))
    return SegmentationMask(
        model=model,
        width=width,
        height=height,
        num_segments=num_segments,
        mask=[int(v) for v in labels.reshape(-1).tolist()],
        backend="deterministic-local",
    )


def _pixel_intensity_grid(tile: RasterTile) -> Optional[np.ndarray]:
    """Return a ``height x width`` per-pixel mean-intensity grid, or ``None``.

    ``None`` when the tile carries no inlined pixel data, in which case callers
    fall back to structure-only behaviour.
    """
    if tile.data is None:
        return None
    values = np.asarray(tile.data, dtype=np.float64)
    # data is row-major, band-interleaved: reshape and average across bands.
    grid = values.reshape(tile.height, tile.width, tile.bands)
    return grid.mean(axis=2)


def _segment_automatic(tile: RasterTile) -> np.ndarray:
    """Partition the tile by quantized intensity into up to 8 segments."""
    height, width = tile.height, tile.width
    intensity = _pixel_intensity_grid(tile)
    if intensity is None:
        # No pixel data: a single segment covering the whole tile.
        return np.ones((height, width), dtype=np.int64)

    lo = float(intensity.min())
    hi = float(intensity.max())
    if hi <= lo:
        # Uniform tile: one segment.
        return np.ones((height, width), dtype=np.int64)

    # Quantize into evenly spaced intensity bins; labels start at 1.
    normalized = (intensity - lo) / (hi - lo)
    bins = np.clip(
        (normalized * _MAX_AUTO_SEGMENTS).astype(np.int64),
        0,
        _MAX_AUTO_SEGMENTS - 1,
    )
    return bins + 1


def _validate_point(prompt: Dict, index: int, width: int, height: int) -> Tuple[float, float]:
    try:
        x = float(prompt["x"])
        y = float(prompt["y"])
    except (KeyError, TypeError, ValueError):
        raise ValidationError(
            "point prompt %d must provide numeric 'x' and 'y'" % index,
            source=_SOURCE,
            detail={"prompt_index": index, "prompt": _safe_prompt(prompt)},
        )
    if not (0.0 <= x < width and 0.0 <= y < height):
        raise ValidationError(
            "point prompt %d (x=%g, y=%g) is outside the %dx%d tile"
            % (index, x, y, width, height),
            source=_SOURCE,
            detail={"prompt_index": index, "x": x, "y": y, "width": width, "height": height},
        )
    return x, y


def _validate_box(prompt: Dict, index: int, width: int, height: int) -> Tuple[int, int, int, int]:
    try:
        x_min = int(prompt["x_min"])
        y_min = int(prompt["y_min"])
        x_max = int(prompt["x_max"])
        y_max = int(prompt["y_max"])
    except (KeyError, TypeError, ValueError):
        raise ValidationError(
            "box prompt %d must provide integer 'x_min','y_min','x_max','y_max'" % index,
            source=_SOURCE,
            detail={"prompt_index": index, "prompt": _safe_prompt(prompt)},
        )
    if not (0 <= x_min <= x_max < width and 0 <= y_min <= y_max < height):
        raise ValidationError(
            "box prompt %d (%d,%d,%d,%d) is outside or inverted for the %dx%d tile"
            % (index, x_min, y_min, x_max, y_max, width, height),
            source=_SOURCE,
            detail={
                "prompt_index": index,
                "box": [x_min, y_min, x_max, y_max],
                "width": width,
                "height": height,
            },
        )
    return x_min, y_min, x_max, y_max


def _safe_prompt(prompt: Dict) -> Dict:
    """A shallow, JSON-safe copy of a prompt for error detail (no secrets)."""
    return {str(k): prompt[k] for k in prompt}


def _segment_with_prompts(tile: RasterTile, prompts: Sequence[Dict]) -> np.ndarray:
    """Assign one segment label per prompt (boxes first, then point Voronoi)."""
    width, height = tile.width, tile.height
    labels = np.zeros((height, width), dtype=np.int64)

    points: List[Tuple[float, float, int]] = []  # (x, y, label)
    boxes: List[Tuple[int, int, int, int, int]] = []  # (x_min,y_min,x_max,y_max,label)

    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            raise ValidationError(
                "prompt %d must be an object with a 'type'" % index,
                source=_SOURCE,
                detail={"prompt_index": index},
            )
        ptype = str(prompt.get("type", "point")).lower()
        label = index + 1
        if ptype == "point":
            x, y = _validate_point(prompt, index, width, height)
            points.append((x, y, label))
        elif ptype == "box":
            x_min, y_min, x_max, y_max = _validate_box(prompt, index, width, height)
            boxes.append((x_min, y_min, x_max, y_max, label))
        else:
            raise ValidationError(
                "prompt %d has unsupported type %r; expected 'point' or 'box'"
                % (index, ptype),
                source=_SOURCE,
                detail={"prompt_index": index, "type": ptype},
            )

    # Boxes claim their rectangle (later boxes win on overlap).
    for x_min, y_min, x_max, y_max, label in boxes:
        labels[y_min : y_max + 1, x_min : x_max + 1] = label

    # Remaining background pixels go to the nearest point prompt (Voronoi).
    if points:
        ys, xs = np.where(labels == 0)
        if xs.size:
            px = np.array([p[0] for p in points], dtype=np.float64)
            py = np.array([p[1] for p in points], dtype=np.float64)
            plabels = np.array([p[2] for p in points], dtype=np.int64)
            # Squared distance from each background pixel to each point.
            dx = xs[:, None] - px[None, :]
            dy = ys[:, None] - py[None, :]
            nearest = np.argmin(dx * dx + dy * dy, axis=1)
            labels[ys, xs] = plabels[nearest]

    return labels
