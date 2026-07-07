"""Per-tile change map between two co-registered assets (roadmap B#5).

``change_map`` tiles an area of interest, embeds both dates of a co-registered
COG pair per tile through the active embedding backend, and reduces each tile to
a ``[0,1]`` change measure — producing a change *grid* rather than the single
scalar of :func:`~geo_foundation_models.asset_embedding.detect_change_from_assets`.
It reuses the shared windowed COG reader (:class:`geo_common.cog.CogReader`) and
the embedding core (:func:`~geo_foundation_models.embedding.embed_tile`), so it
inherits the same byte-range reads and backend seam as the rest of the server.

Honesty is first-class. A change grid is only meaningful with a **real-weight**
backend; under the default deterministic stand-in every differing tile scores
~0.5 noise. So ``change_map``:

* refuses (raises) when ``require_real_backend=True`` and the active backend is
  the deterministic stand-in; and
* otherwise still runs but sets ``calibrated=False`` and a loud ``caveat`` so
  the grid is never mistaken for a real measurement.

For a large AOI the tile count is bounded (``max_tiles``); an over-large request
is rejected with guidance to coarsen the grid or route the job to distributed
compute (``aws-geo-compute``) rather than silently doing thousands of embeds.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from geo_common.cog import (
    DEFAULT_S3_REGION,
    ByteRangeReader,
    CogMetadata,
    CogReader,
    HttpRangeReader,
    resolve_window,
    window_geotransform,
)
from geo_common.errors import ValidationError
from geo_common.geometry import iter_polygons, point_in_geometry
from geo_common.http import HttpClient
from geo_common.raster_models import FeatureCollection

from geo_foundation_models.change import detect_change
from geo_foundation_models.embedding import (
    DEFAULT_SUPPORTED_FORMATS,
    MAX_TILE_DIMENSION,
    EmbeddingBackend,
    ModelRegistry,
    embed_tile,
)
from geo_foundation_models.models import (
    ChangeMapCell,
    ChangeMapResult,
    RasterTile,
    ZoneChange,
)

__all__ = ["change_map", "DEFAULT_TILE_SIZE", "DEFAULT_MAX_TILES"]

_SOURCE = "geo-foundation-models"
_STANDIN_BACKEND = "deterministic-local"

#: Default edge length (pixels) of each change-grid tile.
DEFAULT_TILE_SIZE = 256
#: Default cap on the number of tiles embedded in one call (both dates each).
DEFAULT_MAX_TILES = 256


def _normalize_bands(bands: object) -> "list[int]":
    if isinstance(bands, int) and not isinstance(bands, bool):
        bands = [bands]
    if not isinstance(bands, (list, tuple)) or not bands:
        raise ValidationError(
            "bands must be a non-empty list of 1-based band indices",
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


def _assert_aligned(a: CogMetadata, b: CogMetadata) -> None:
    """Require the two assets to share a pixel grid (Req: comparable windows)."""
    if a.width != b.width or a.height != b.height:
        raise ValidationError(
            "assets are not co-registered: dimensions differ "
            "(%dx%d vs %dx%d)" % (a.width, a.height, b.width, b.height),
            source=_SOURCE,
            detail={"parameter": "raster_href_b"},
        )
    if (a.geotransform is None) != (b.geotransform is None):
        raise ValidationError(
            "assets are not co-registered: one is georeferenced and the other "
            "is not",
            source=_SOURCE,
            detail={"parameter": "raster_href_b"},
        )
    if a.geotransform is not None and b.geotransform is not None:
        if any(abs(x - y) > 1e-6 for x, y in zip(a.geotransform, b.geotransform)):
            raise ValidationError(
                "assets are not co-registered: geotransforms differ",
                source=_SOURCE,
                detail={"parameter": "raster_href_b"},
            )


def _tile_bbox(
    meta: CogMetadata, col_off: int, row_off: int, width: int, height: int
) -> Tuple[float, float, float, float]:
    """World bbox of a pixel window, from the asset geotransform."""
    gt = window_geotransform(meta, col_off, row_off)
    ox, pw, rr, oy, cr, ph = gt

    def world(c: float, r: float) -> Tuple[float, float]:
        return (ox + pw * c + rr * r, oy + cr * c + ph * r)

    corners = [world(0, 0), world(width, 0), world(0, height), world(width, height)]
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    return (min(xs), min(ys), max(xs), max(ys))


async def _open(
    href: str,
    reader: Optional[ByteRangeReader],
    http: Optional[HttpClient],
    region: str,
) -> Tuple[CogReader, CogMetadata]:
    source = reader if reader is not None else HttpRangeReader(
        href, http, region=region, source_id=href
    )
    cog = CogReader(source, source_id=href)
    meta = await cog.open()
    return cog, meta


def _build_tile(band_data, n: int, n_bands: int, width: int, height: int,
                dtype: str, tile_format: str) -> RasterTile:
    data = [band_data[b][i] for i in range(n) for b in range(n_bands)]
    return RasterTile(
        width=width, height=height, bands=n_bands, format=tile_format,
        data=data, dtype=dtype,
    )


async def change_map(
    *,
    raster_href_a: str,
    raster_href_b: str,
    model: str,
    aoi_bbox: Sequence[float],
    tile_size: int = DEFAULT_TILE_SIZE,
    bands: object = (1,),
    zones: Optional[FeatureCollection] = None,
    tile_format: str = "GTiff",
    require_real_backend: bool = False,
    max_tiles: int = DEFAULT_MAX_TILES,
    registry: Optional[ModelRegistry] = None,
    backend: Optional[EmbeddingBackend] = None,
    http: Optional[HttpClient] = None,
    region: str = DEFAULT_S3_REGION,
    readers: Optional[dict] = None,
    supported_formats=DEFAULT_SUPPORTED_FORMATS,
    max_dimension: int = MAX_TILE_DIMENSION,
) -> ChangeMapResult:
    """Compute a per-tile change grid between two co-registered COGs.

    ``raster_href_a``/``raster_href_b`` are the two dates (``s3://``/``http(s)``)
    over the **same** pixel grid; ``aoi_bbox`` is ``(min_x, min_y, max_x, max_y)``
    in the assets' CRS. The AOI window is split into ``tile_size``-pixel tiles;
    each tile is read + embedded for both dates and reduced to a ``[0,1]`` change
    measure. Returns a :class:`ChangeMapResult`. Raises an ``Error_Taxonomy``
    ``ValidationError`` for bad input, mis-registered assets, an AOI that does
    not overlap, a tile count over ``max_tiles``, or (when
    ``require_real_backend``) a deterministic-stand-in backend.

    ``readers`` (test-only) maps ``"a"``/``"b"`` to pre-built
    :class:`ByteRangeReader` instances for in-memory assets.
    """
    if not isinstance(raster_href_a, str) or not raster_href_a.strip():
        raise ValidationError("raster_href_a must be a non-empty URL", source=_SOURCE,
                              detail={"parameter": "raster_href_a"})
    if not isinstance(raster_href_b, str) or not raster_href_b.strip():
        raise ValidationError("raster_href_b must be a non-empty URL", source=_SOURCE,
                              detail={"parameter": "raster_href_b"})
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 1:
        raise ValidationError("tile_size must be a positive integer", source=_SOURCE,
                              detail={"parameter": "tile_size"})
    if tile_size > max_dimension:
        raise ValidationError(
            "tile_size %d exceeds the %d-pixel embedding limit"
            % (tile_size, max_dimension),
            source=_SOURCE, detail={"parameter": "tile_size"},
        )
    if isinstance(max_tiles, bool) or not isinstance(max_tiles, int) or max_tiles < 1:
        raise ValidationError("max_tiles must be a positive integer", source=_SOURCE,
                              detail={"parameter": "max_tiles"})
    if aoi_bbox is None or len(tuple(aoi_bbox)) != 4:
        raise ValidationError(
            "aoi_bbox must be [min_x, min_y, max_x, max_y] in the assets' CRS",
            source=_SOURCE, detail={"parameter": "aoi_bbox"},
        )
    band_list = _normalize_bands(bands)
    bbox = tuple(float(v) for v in aoi_bbox)

    # Backend provenance and the honesty gate — decided before any I/O.
    backend_id = getattr(backend, "backend_id", _STANDIN_BACKEND) if backend else _STANDIN_BACKEND
    calibrated = backend_id != _STANDIN_BACKEND
    if not calibrated and require_real_backend:
        raise ValidationError(
            "change_map requires a real-weight embedding backend: the active "
            "backend is the deterministic stand-in, under which every tile "
            "scores ~0.5 noise. Configure a remote endpoint "
            "(GEO_FM_EMBED_ENDPOINT_URL or GEO_FM_SAGEMAKER_ENDPOINT), or call "
            "with require_real_backend=false to get an explicitly-uncalibrated grid.",
            source=_SOURCE,
            detail={"parameter": "require_real_backend", "backend": backend_id},
        )

    readers = readers or {}
    owns_client = http is None and not readers
    client = http
    if client is None and not readers:
        client = HttpClient()
    try:
        cog_a, meta_a = await _open(raster_href_a, readers.get("a"), client, region)
        cog_b, meta_b = await _open(raster_href_b, readers.get("b"), client, region)
        _assert_aligned(meta_a, meta_b)
        for b in band_list:
            if b > meta_a.samples_per_pixel:
                raise ValidationError(
                    "band %d is out of range; asset has %d band(s)"
                    % (b, meta_a.samples_per_pixel),
                    source=_SOURCE, detail={"parameter": "bands"},
                )

        col_off, row_off, win_w, win_h = resolve_window(bbox, meta_a, source_id=raster_href_a)
        if win_w == 0 or win_h == 0:
            raise ValidationError(
                "aoi_bbox does not overlap the asset",
                source=_SOURCE, detail={"parameter": "aoi_bbox"},
            )

        n_cols = (win_w + tile_size - 1) // tile_size
        n_rows = (win_h + tile_size - 1) // tile_size
        if n_cols * n_rows > max_tiles:
            raise ValidationError(
                "change grid would be %d tiles (%dx%d), over the max_tiles=%d "
                "limit; use a larger tile_size, a smaller aoi_bbox, or route the "
                "job to aws-geo-compute for distributed change mapping."
                % (n_cols * n_rows, n_cols, n_rows, max_tiles),
                source=_SOURCE, detail={"parameter": "max_tiles"},
            )

        n_bands = len(band_list)
        dtype = meta_a.dtype
        cells: List[ChangeMapCell] = []
        any_structure_only = False
        for tr in range(n_rows):
            for tc in range(n_cols):
                tcol = col_off + tc * tile_size
                trow = row_off + tr * tile_size
                tw = min(tile_size, col_off + win_w - tcol)
                th = min(tile_size, row_off + win_h - trow)
                data_a = await cog_a.read_window_pixels(
                    col_off=tcol, row_off=trow, width=tw, height=th, bands=band_list
                )
                data_b = await cog_b.read_window_pixels(
                    col_off=tcol, row_off=trow, width=tw, height=th, bands=band_list
                )
                n = tw * th
                tile_a = _build_tile(data_a, n, n_bands, tw, th, dtype, tile_format)
                tile_b = _build_tile(data_b, n, n_bands, tw, th, dtype, tile_format)
                emb_a = embed_tile(tile_a, model, registry=registry, backend=backend,
                                   supported_formats=supported_formats, max_dimension=max_dimension)
                emb_b = embed_tile(tile_b, model, registry=registry, backend=backend,
                                   supported_formats=supported_formats, max_dimension=max_dimension)
                change = detect_change(emb_a.vector, emb_b.vector)
                structure_only = emb_a.structure_only or emb_b.structure_only
                any_structure_only = any_structure_only or structure_only
                cells.append(ChangeMapCell(
                    row=tr, col=tc,
                    bbox=_tile_bbox(meta_a, tcol, trow, tw, th),
                    change=change,
                    structure_only=structure_only,
                ))
    finally:
        if owns_client and client is not None:
            await client.aclose()

    caveat: Optional[str] = None
    if not calibrated:
        caveat = (
            "Change grid is NOT calibrated: embeddings came from the "
            "deterministic stand-in backend, under which every differing tile "
            "scores ~0.5 and identical tiles score 0.0. Wire a real-weight "
            "backend (remote endpoint) for a meaningful change magnitude."
        )
    elif any_structure_only:
        caveat = "One or more tiles had no pixel data (structure-only embedding)."

    # Optional per-zone reduction: aggregate tile changes into the vector zones
    # whose polygon contains each tile's center (assets' CRS).
    zone_summaries = _reduce_to_zones(zones, cells) if zones is not None else None

    # The embedding dimension is model-determined; resolve the canonical spec.
    reg = registry if registry is not None else ModelRegistry()
    spec = reg.get(model)

    return ChangeMapResult(
        model=spec.name,
        dimension=spec.dimension,
        backend=backend_id,
        tile_size=tile_size,
        rows=n_rows,
        cols=n_cols,
        cell_count=len(cells),
        cells=cells,
        zones=zone_summaries,
        structure_only=any_structure_only,
        calibrated=calibrated,
        caveat=caveat,
    )


def _reduce_to_zones(
    zones: FeatureCollection, cells: List[ChangeMapCell]
) -> List[ZoneChange]:
    """Aggregate each tile's change into the zones its center falls in.

    A tile is assigned to a zone when the tile bbox center lies inside the zone
    polygon (even-odd test). A zone with no assigned tiles gets a no-data
    indication (``None`` stats, ``tile_count`` 0), mirroring ``zonal_statistics``.
    """
    from geo_common.zonal import _zone_id  # local import avoids a cycle at import

    centers = [
        ((c.bbox[0] + c.bbox[2]) / 2.0, (c.bbox[1] + c.bbox[3]) / 2.0, c.change)
        for c in cells
    ]
    summaries: List[ZoneChange] = []
    for index, feature in enumerate(zones.features):
        zid = _zone_id(dict(feature.properties), index)
        if feature.geometry is None:
            summaries.append(ZoneChange(zone_id=zid))
            continue
        polygons = iter_polygons(feature.geometry.to_geojson(), source=_SOURCE)
        vals = [ch for cx, cy, ch in centers if point_in_geometry(polygons, cx, cy)]
        if not vals:
            summaries.append(ZoneChange(zone_id=zid))
            continue
        summaries.append(
            ZoneChange(
                zone_id=zid,
                mean_change=sum(vals) / len(vals),
                max_change=max(vals),
                tile_count=len(vals),
            )
        )
    return summaries
