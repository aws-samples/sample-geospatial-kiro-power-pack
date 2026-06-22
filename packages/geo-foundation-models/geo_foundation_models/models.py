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
