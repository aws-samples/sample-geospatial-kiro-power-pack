"""Read-then-embed bridges: embed a COG window server-side, and compare two.

These close the gap that :func:`geo_foundation_models.embedding.embed_tile`
leaves — it needs pixels **inlined** on the tile, and nothing reads a COG window
to build one. :func:`embed_asset` takes a COG href + window + bands, reads only
the overlapping tiles by byte range through the shared
:class:`geo_common.cog.CogReader` (no cross-server dependency), builds a
:class:`~geo_foundation_models.models.RasterTile`, and embeds it.
:func:`detect_change_from_assets` composes two ``embed_asset`` calls plus
:func:`~geo_foundation_models.change.detect_change` into a single call.

Honesty: both carry the deterministic-stand-in caveats. ``embed_asset`` reads
real pixels, so its result is never ``structure_only``; but under the default
deterministic backend the vector still has no semantic structure, so a
``detect_change_from_assets`` score is not a calibrated measurement — the
returned ``caveat`` says so. Wire a real-weight backend for meaningful scores.
"""

from __future__ import annotations

from typing import Optional, Sequence

from geo_common.cog import (
    DEFAULT_S3_REGION,
    ByteRangeReader,
    CogReader,
    HttpRangeReader,
    resolve_window,
)
from geo_common.errors import ValidationError
from geo_common.http import HttpClient

from geo_foundation_models.change import detect_change
from geo_foundation_models.embedding import (
    DEFAULT_SUPPORTED_FORMATS,
    MAX_TILE_DIMENSION,
    EmbeddingBackend,
    ModelRegistry,
    embed_tile,
)
from geo_foundation_models.models import AssetChangeResult, EmbeddingResult, RasterTile

__all__ = ["embed_asset", "detect_change_from_assets"]

_SOURCE = "geo-foundation-models"
_STANDIN_BACKEND = "deterministic-local"


def _normalize_bands(bands: object) -> "list[int]":
    """Validate ``bands`` into a non-empty list of 1-based band indices."""
    if isinstance(bands, int) and not isinstance(bands, bool):
        bands = [bands]
    if not isinstance(bands, (list, tuple)) or not bands:
        raise ValidationError(
            "bands must be a non-empty list of 1-based band indices (e.g. [1] "
            "or [1, 2, 3])",
            source=_SOURCE,
            detail={"parameter": "bands"},
        )
    out: list[int] = []
    for b in bands:
        if isinstance(b, bool) or not isinstance(b, int) or b < 1:
            raise ValidationError(
                "band indices must be positive 1-based integers",
                source=_SOURCE,
                detail={"parameter": "bands"},
            )
        out.append(b)
    return out


async def embed_asset(
    *,
    raster_href: str,
    model: str,
    window_bbox: Optional[Sequence[float]] = None,
    bands: object = (1,),
    tile_format: str = "GTiff",
    registry: Optional[ModelRegistry] = None,
    backend: Optional[EmbeddingBackend] = None,
    http: Optional[HttpClient] = None,
    region: str = DEFAULT_S3_REGION,
    reader: Optional[ByteRangeReader] = None,
    supported_formats=DEFAULT_SUPPORTED_FORMATS,
    max_dimension: int = MAX_TILE_DIMENSION,
) -> EmbeddingResult:
    """Read a COG window server-side and embed it with a selected model.

    ``raster_href`` is an ``s3://``/``http(s)://`` COG. ``window_bbox`` is an
    optional ``(min_x, min_y, max_x, max_y)`` in the asset's own CRS (omit for
    the whole asset); ``bands`` selects 1-based band indices. Only the
    overlapping tiles are read. The read pixels build a
    :class:`RasterTile` (band-interleaved) that is embedded via
    :func:`embed_tile`, so the result is a real pixel-based embedding
    (``structure_only`` is False). A window that does not overlap the asset, or
    a tile exceeding the size limit, raises an ``Error_Taxonomy``
    ``ValidationError``.
    """
    if not isinstance(raster_href, str) or not raster_href.strip():
        raise ValidationError(
            "raster_href must be a non-empty s3:// or http(s):// URL",
            source=_SOURCE,
            detail={"parameter": "raster_href"},
        )
    band_list = _normalize_bands(bands)
    bbox = tuple(float(v) for v in window_bbox) if window_bbox is not None else None

    owns_client = reader is None and http is None
    if reader is not None:
        source: ByteRangeReader = reader
        client = None
    else:
        client = http if http is not None else HttpClient()
        source = HttpRangeReader(raster_href, client, region=region, source_id=raster_href)
    try:
        cog = CogReader(source, source_id=raster_href)
        meta = await cog.open()
        for b in band_list:
            if b > meta.samples_per_pixel:
                raise ValidationError(
                    f"band {b} is out of range; asset has "
                    f"{meta.samples_per_pixel} band(s)",
                    source=_SOURCE,
                    detail={"parameter": "bands"},
                )
        col_off, row_off, width, height = resolve_window(bbox, meta, source_id=raster_href)
        if width == 0 or height == 0:
            raise ValidationError(
                "window_bbox does not overlap the asset",
                source=_SOURCE,
                detail={"parameter": "window_bbox"},
            )
        band_data = await cog.read_window_pixels(
            col_off=col_off, row_off=row_off, width=width, height=height, bands=band_list
        )
    finally:
        if owns_client and client is not None:
            await client.aclose()

    # Build a band-interleaved (row-major) flat pixel buffer for the tile.
    n = width * height
    data = [band_data[b][i] for i in range(n) for b in range(len(band_list))]
    tile = RasterTile(
        width=width,
        height=height,
        bands=len(band_list),
        format=tile_format,
        data=data,
        dtype=meta.dtype,
    )
    return embed_tile(
        tile,
        model,
        registry=registry,
        backend=backend,
        supported_formats=supported_formats,
        max_dimension=max_dimension,
    )


async def detect_change_from_assets(
    *,
    raster_href_a: str,
    raster_href_b: str,
    model: str,
    window_bbox: Optional[Sequence[float]] = None,
    bands: object = (1,),
    tile_format: str = "GTiff",
    registry: Optional[ModelRegistry] = None,
    backend: Optional[EmbeddingBackend] = None,
    http: Optional[HttpClient] = None,
    region: str = DEFAULT_S3_REGION,
    readers: Optional["dict"] = None,
    supported_formats=DEFAULT_SUPPORTED_FORMATS,
    max_dimension: int = MAX_TILE_DIMENSION,
) -> AssetChangeResult:
    """Embed the same window of two assets and return their change measure.

    Reads the same ``window_bbox`` / ``bands`` from both ``raster_href_a`` and
    ``raster_href_b``, embeds each with ``model``, and returns the
    :func:`detect_change` measure plus provenance. ``caveat`` is set whenever the
    score is not a calibrated measurement (the deterministic stand-in backend or
    a structure-only read), so a ~0.5 is never mistaken for a real result.

    ``readers`` (test-only) maps ``"a"``/``"b"`` to a pre-built
    :class:`ByteRangeReader` for in-memory assets.
    """
    readers = readers or {}
    ea = await embed_asset(
        raster_href=raster_href_a, model=model, window_bbox=window_bbox, bands=bands,
        tile_format=tile_format, registry=registry, backend=backend, http=http,
        region=region, reader=readers.get("a"), supported_formats=supported_formats,
        max_dimension=max_dimension,
    )
    eb = await embed_asset(
        raster_href=raster_href_b, model=model, window_bbox=window_bbox, bands=bands,
        tile_format=tile_format, registry=registry, backend=backend, http=http,
        region=region, reader=readers.get("b"), supported_formats=supported_formats,
        max_dimension=max_dimension,
    )
    change = detect_change(ea.vector, eb.vector)
    structure_only = ea.structure_only or eb.structure_only
    caveat: Optional[str] = None
    if ea.backend == _STANDIN_BACKEND:
        caveat = (
            "Score is NOT a calibrated measurement: embeddings came from the "
            "deterministic stand-in backend, under which any two differing "
            "windows score ~0.5. Wire a real-weight backend for a meaningful "
            "change magnitude."
        )
    elif structure_only:
        caveat = "One or both windows had no pixel data (structure-only embedding)."
    return AssetChangeResult(
        change=change,
        model=ea.model,
        dimension=ea.dimension,
        backend=ea.backend,
        structure_only=structure_only,
        caveat=caveat,
    )
