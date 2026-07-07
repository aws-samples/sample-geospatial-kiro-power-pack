"""Data models for the ``geo-foundation-models`` server (Pillar C, MVP).

This module defines the data types task 7.1 requires:

* :class:`RasterTile` - the imagery tile submitted for embedding. It carries
  the tile's pixel dimensions, band count, raster ``format``, and optional
  inlined pixel ``data``. The Foundation_Model_Server validates a tile against
  the size limit (``<= 1024 x 1024``) and the supported-format set before
  producing any embedding (Requirements 9.1, 9.7).
* :class:`EmbeddingResult` - the value returned by ``embed_tile``: the selected
  ``model``, the ``dimension`` determined by that model, and the embedding
  ``vector`` (Requirement 9.1).
* :class:`EmbeddingMetadata` - the spatial/temporal context stored alongside an
  embedding (mirrors design.md "Foundation-model embedding record + metadata",
  Req 9.1 / 9.3).
* :class:`EmbeddingRecord` - a persisted embedding plus its metadata, used by
  the expansion ``geo-embedding-search`` server (Req 9.3).
* :class:`ModelSpec` - one entry in the configurable model set, pairing a model
  name with the embedding ``dimension`` it produces (Requirement 9.2).

Python 3.9 compatibility: this module uses ``from __future__ import
annotations`` together with ``typing`` generics so the pydantic models resolve
their annotations on 3.9+.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

__all__ = [
    "MAX_TILE_DIMENSION",
    "RasterTile",
    "ModelSpec",
    "EmbeddingResult",
    "AssetChangeResult",
    "ChangeMapCell",
    "ZoneChange",
    "ChangeMapResult",
    "EmbeddingMetadata",
    "EmbeddingRecord",
    "SegmentationMask",
]

#: The maximum width and height (in pixels) of a tile accepted for embedding
#: (Requirement 9.1 / 9.7). A tile exceeding this in either dimension is
#: rejected as oversized.
MAX_TILE_DIMENSION = 1024


class RasterTile(BaseModel):
    """An imagery tile submitted to the Foundation_Model_Server for embedding.

    The tile declares its pixel ``width`` and ``height``, its ``bands`` count,
    and the raster ``format`` it is encoded in. Pixel values may be inlined as
    a flat ``data`` sequence (row-major, band-interleaved) so a deterministic,
    content-dependent embedding can be computed locally; when ``data`` is
    omitted the embedding is derived from the tile's structural attributes.

    Semantic validation (empty / oversized / unsupported-format) is performed
    by ``embed_tile`` so that failures surface as an ``Error_Taxonomy``
    ``ValidationError`` (Requirement 9.7) rather than a pydantic error. The
    field types here are intentionally permissive (e.g. ``width`` is a plain
    ``int``) so that an oversized or empty tile can be *constructed* and then
    *rejected* by the embedding logic, which is where Requirement 9.7 lives.
    """

    width: int = Field(description="Tile width in pixels.")
    height: int = Field(description="Tile height in pixels.")
    bands: int = Field(default=1, description="Number of raster bands.")
    format: str = Field(min_length=1, description="Raster format identifier, e.g. 'GTiff'.")
    data: Optional[List[float]] = Field(
        default=None,
        description=(
            "Optional flat, row-major, band-interleaved pixel values. When "
            "present its length must equal width * height * bands."
        ),
    )
    dtype: str = Field(default="uint8", description="Pixel data type label.")
    latlon: Optional[Tuple[float, float]] = Field(
        default=None,
        description=(
            "Optional (latitude, longitude) center of the tile in EPSG:4326. "
            "Passed through to the embedding backend so location-aware models "
            "(e.g. Clay) can condition on it; omit (None) to skip that context."
        ),
    )
    acquired: Optional[str] = Field(
        default=None,
        description=(
            "Optional ISO-8601 acquisition datetime of the tile. Passed to the "
            "backend for time-aware conditioning; omit (None) to skip it."
        ),
    )

    @property
    def pixel_count(self) -> int:
        """Number of pixels (``width * height``), clamped at 0 for empty tiles."""
        if self.width <= 0 or self.height <= 0:
            return 0
        return self.width * self.height


class ModelSpec(BaseModel):
    """One entry in the configurable foundation-model set (Requirement 9.2).

    Pairs a model ``name`` (e.g. ``"Clay"``) with the embedding ``dimension``
    that model produces, so the embedding vector's dimensionality is determined
    by the configured model (Requirement 9.1).
    """

    name: str = Field(min_length=1)
    dimension: int = Field(gt=0)


class EmbeddingResult(BaseModel):
    """The embedding produced for a tile (Requirement 9.1).

    ``dimension`` always equals ``len(vector)`` and is determined by the
    selected ``model``. ``backend`` records the **provenance** of the vector so
    a caller can tell a real model-weight embedding apart from the default
    deterministic local stand-in: it is ``"deterministic-local"`` for the
    built-in seeded backend, or a real backend's identifier (e.g.
    ``"clay-v1.5"``) when production weights are wired in. This makes the
    stand-in self-describing rather than silently looking like real inference.
    """

    model: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    vector: List[float]
    backend: str = Field(
        default="deterministic-local",
        description=(
            "Provenance of the embedding: 'deterministic-local' for the "
            "built-in deterministic stand-in backend, or a real model-weight "
            "backend identifier (e.g. 'clay-v1.5') when one is wired in."
        ),
    )
    structure_only: bool = Field(
        default=False,
        description=(
            "True when the tile carried no inlined pixel data, so the embedding "
            "was derived from the tile's structure (width/height/bands/format/"
            "dtype) only. Two structure-only tiles of the same shape yield the "
            "IDENTICAL embedding, which detect_change reports as 0.0 ('no "
            "change') even though no pixels were compared — treat a "
            "structure-only result as not a real measurement."
        ),
    )


class AssetChangeResult(BaseModel):
    """Change between two assets embedded server-side (read -> embed -> compare).

    ``change`` is the ``[0.0, 1.0]`` measure from :func:`detect_change`. ``backend``
    and ``structure_only`` carry the same honesty provenance as
    :class:`EmbeddingResult` (``structure_only`` is True if *either* asset window
    had no pixels). ``caveat`` is populated whenever the result is not a
    calibrated measurement — i.e. under the deterministic stand-in backend or a
    structure-only read — so the number is never mistaken for a real result.
    """

    change: float = Field(ge=0.0, le=1.0)
    model: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    backend: str = Field(default="deterministic-local")
    structure_only: bool = False
    caveat: Optional[str] = None


class ChangeMapCell(BaseModel):
    """One tile of a :class:`ChangeMapResult` — a per-tile change measurement.

    ``row``/``col`` index the tile within the change grid (row-major, origin at
    the AOI window's top-left). ``bbox`` is the tile's extent in the assets' CRS
    (``[min_x, min_y, max_x, max_y]``). ``change`` is the ``[0.0, 1.0]``
    :func:`~geo_foundation_models.change.detect_change` measure between the two
    dates for that tile. ``structure_only`` is True when the tile carried no
    pixel data on at least one date (so its change is not a real measurement).
    """

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    bbox: Tuple[float, float, float, float]
    change: float = Field(ge=0.0, le=1.0)
    structure_only: bool = False


class ZoneChange(BaseModel):
    """Per-zone summary of a change grid (optional reduction of a ChangeMapResult).

    ``mean_change`` / ``max_change`` aggregate the ``change`` of every grid tile
    whose center falls inside the zone; ``tile_count`` is how many did. A zone
    that no tile center falls in gets ``None`` statistics and ``tile_count`` 0
    (a no-data indication), mirroring ``zonal_statistics`` (Requirement 8.8).
    """

    zone_id: str
    mean_change: Optional[float] = None
    max_change: Optional[float] = None
    tile_count: int = Field(ge=0, default=0)


class ChangeMapResult(BaseModel):
    """A per-tile change grid between two co-registered assets (roadmap B#5).

    Tiles the AOI window into a ``rows x cols`` grid, embeds both dates per tile
    through the active backend, and reduces each tile to a ``[0,1]`` change
    measure (:class:`ChangeMapCell`). ``backend`` records provenance and
    ``caveat`` is populated whenever the grid is **not** a calibrated
    measurement — i.e. under the deterministic stand-in backend (where every
    tile scores ~0.5 noise) or a structure-only read — so the grid is never
    mistaken for a real result.
    """

    model: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    backend: str = Field(default="deterministic-local")
    tile_size: int = Field(gt=0)
    rows: int = Field(ge=0)
    cols: int = Field(ge=0)
    cell_count: int = Field(ge=0)
    cells: List[ChangeMapCell] = Field(default_factory=list)
    zones: Optional[List[ZoneChange]] = Field(
        default=None,
        description=(
            "Per-zone reduction of the grid, present only when 'zones' were "
            "passed: each tile's change aggregated into the vector zones it "
            "falls in (mean/max/tile_count). None when no zones were requested."
        ),
    )
    structure_only: bool = False
    calibrated: bool = Field(
        default=False,
        description=(
            "True only when a real-weight backend produced the embeddings. "
            "False under the deterministic stand-in, where the grid is ~0.5 "
            "noise everywhere; see 'caveat'."
        ),
    )
    caveat: Optional[str] = None


class EmbeddingMetadata(BaseModel):
    """Spatial/temporal context stored alongside an embedding (Req 9.1 / 9.3)."""

    bbox: Tuple[float, float, float, float]
    datetime: str
    crs: str = "EPSG:4326"
    source_asset: Optional[str] = None
    extra: Dict = Field(default_factory=dict)


class EmbeddingRecord(BaseModel):
    """A persisted embedding plus its metadata (Requirement 9.3).

    Used by the expansion ``geo-embedding-search`` server to store and retrieve
    embeddings; defined here so the embedding-producing server and the
    vector-store server share one definition.
    """

    id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    vector: List[float]
    metadata: EmbeddingMetadata


class SegmentationMask(BaseModel):
    """A SAMGeo-style segmentation result for a tile (Requirement 9, Pillar C).

    ``segment`` turns a validated :class:`RasterTile` into a per-pixel label
    map. The mask is a flat, row-major sequence of integer labels of length
    ``width * height``: ``0`` denotes background (a pixel assigned to no
    segment) and any positive integer denotes a distinct segment. ``num_segments``
    is the count of distinct positive labels actually present in ``mask``.

    Like the embedding backend, the default segmentation is deterministic and
    content-dependent so it is exercisable without GPU model weights, while the
    shape of the result matches what a real SAMGeo backend would return (a label
    image plus a segment count) so production weights can be substituted without
    changing the tool contract.
    """

    model: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    num_segments: int = Field(ge=0)
    mask: List[int]
    backend: str = Field(
        default="deterministic-local",
        description=(
            "Provenance of the mask: 'deterministic-local' for the built-in "
            "deterministic stand-in segmenter, or a real model-weight backend "
            "identifier (e.g. 'samgeo') when one is wired in."
        ),
    )
