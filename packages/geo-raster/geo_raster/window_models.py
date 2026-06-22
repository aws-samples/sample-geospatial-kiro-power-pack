"""Windowed-read data models for the ``geo-raster`` server (Pillar B).

These models describe the inputs and output of the windowed-read and band-math
capabilities (``read_window`` / ``band_math``):

* :class:`PixelWindow` — a window expressed directly in pixel/raster
  coordinates (column/row offset plus width/height).
* :class:`GeoWindow` — a window expressed as a georeferenced bounding box in the
  asset's CRS; resolved to a pixel window using the asset's GeoTIFF
  georeferencing tags.
* :class:`RasterArray` — the returned pixel block: per-band values for exactly
  the requested window, never more (Requirement 7.2).

All models validate their own fields so a malformed window is rejected with a
pydantic error before any byte-range read is issued (Requirement 7.12); the
server layer re-raises these as the shared ``Error_Taxonomy``
:class:`~geo_common.errors.ValidationError`.

Python 3.9 compatibility: uses ``from __future__ import annotations`` with
``typing`` generics.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

__all__ = ["PixelWindow", "GeoWindow", "RasterArray"]


class PixelWindow(BaseModel):
    """A read window in pixel (raster) coordinates.

    ``col_off``/``row_off`` are the 0-based column and row of the window's
    top-left corner; ``width``/``height`` are its size in pixels. A window must
    have a strictly positive size and a non-negative origin (Requirement 7.12).
    """

    col_off: int = Field(ge=0)
    row_off: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class GeoWindow(BaseModel):
    """A read window as a georeferenced bounding box in the asset's CRS.

    ``bbox`` is ``(min_x, min_y, max_x, max_y)`` in the asset's coordinate
    reference system (degrees or projected units). ``min_x <= max_x`` and
    ``min_y <= max_y`` must hold (Requirement 7.12). The reader resolves the
    bbox to a pixel window via the asset's GeoTIFF georeferencing tags.
    """

    bbox: Tuple[float, float, float, float]
    crs: Optional[str] = None

    @model_validator(mode="after")
    def _check_ordering(self) -> "GeoWindow":
        min_x, min_y, max_x, max_y = self.bbox
        for name, value in (
            ("min_x", min_x),
            ("min_y", min_y),
            ("max_x", max_x),
            ("max_y", max_y),
        ):
            if value != value or value in (float("inf"), float("-inf")):  # NaN/inf
                raise ValueError(f"bbox component {name!r} must be finite")
        if min_x > max_x:
            raise ValueError("bbox min_x must not be greater than max_x")
        if min_y > max_y:
            raise ValueError("bbox min_y must not be greater than max_y")
        return self


class RasterArray(BaseModel):
    """A block of pixels returned for the requested window (Requirement 7.2).

    ``data`` holds one list per returned band; each band list is
    ``width * height`` values in row-major order (row 0 first, left to right).
    ``band_indices`` are the 1-based band numbers in the same order as ``data``.
    ``dtype`` names the source sample type (e.g. ``"uint16"``, ``"float32"``)
    and ``nodata`` is the asset's nodata value when declared.

    The dimensions are exactly the requested window: the reader transfers and
    returns only in-window pixels (Requirements 7.2, 12.1).
    """

    width: int = Field(ge=0)
    height: int = Field(ge=0)
    band_indices: List[int]
    dtype: str
    nodata: Optional[float] = None
    data: List[List[float]] = Field(default_factory=list)

    def band(self, band_index: int) -> List[float]:
        """Return the row-major values for the given 1-based ``band_index``."""
        try:
            pos = self.band_indices.index(band_index)
        except ValueError as exc:
            raise KeyError(f"band {band_index} not present in result") from exc
        return self.data[pos]

    def value(self, band_index: int, row: int, col: int) -> float:
        """Return the value at ``(row, col)`` for a 1-based ``band_index``."""
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IndexError("pixel coordinate out of window bounds")
        return self.band(band_index)[row * self.width + col]
