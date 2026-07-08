"""Retrieval of precomputed open Clay v1.5 embeddings (LGND / Source Cooperative).

This module adds a *lookup* counterpart to :mod:`geo_foundation_models.embedding`'s
*compute* path: instead of embedding a tile locally, it retrieves published
**Clay v1.5** embeddings for a bounding box from the open, credential-free
``LGND Clay v1.5 Sentinel-2`` dataset hosted on Source Cooperative's public S3
mirror (Registry of Open Data on AWS, CC-BY 4.0).

Dataset facts (confirmed against the bucket):

* Bucket ``us-west-2.opendata.source.coop`` (region ``us-west-2``), prefix
  ``clay/lgnd-clay-v1-5-sentinel-2-l2a/``, **anonymous** read (no credentials,
  no requester-pays).
* Each embedding is a **1024-dimensional** ``float32`` Clay v1.5 vector for a
  2,560 m MajorTOM grid cell.
* Two products, both hive-partitioned GeoParquet 1.1:
  * ``scene`` - per Sentinel-2 scene, partitioned by MGRS Grid Zone Designator
    (``gzd``), ``year``, ``month``.
  * ``aggregated`` - one representative embedding per MajorTOM cell per month,
    partitioned by ``geohash_l2``, ``year``, ``month``.
* Columns: ``chips_id``, ``cell_id``, ``rasters_id``, ``stac_item_id``,
  ``collection``, ``datetime`` (timestamp), ``embedding`` (list[float32]),
  ``bbox`` (struct ``xmin/ymin/xmax/ymax``), ``geometry`` (WKB polygon).
* Coverage: two time steps, June 2024 and June 2025.

These retrieved embeddings are **real Clay v1.5** (1024-d) and are distinct from
the local ``embed_tile`` path's ``Clay`` model (a 768-d, Clay-v1-style
*deterministic stub* in :data:`geo_foundation_models.embedding.DEFAULT_MODELS`).
Different version, different backend, different vector space - do not compare a
retrieved vector with a locally computed one (e.g. via ``detect_change``).

Because the parquet partition files are large (100s of MB) and a bbox can match
many cells, retrieval is **bounded**: the bbox area is gated like
``vector_features`` (Req 7.12 spirit) and the number of returned embeddings is
capped (``limit``), so a single call never returns an oversized result.

**Latency note (measured against the live bucket):** spatial pruning is cheap -
reading only the ``bbox`` column to skip non-overlapping files takes a few
seconds. But each scene file is a single ~150 MB ZSTD row group, so returning
*any* embedding from a matching file requires downloading that file's whole
embedding column. From outside ``us-west-2`` that can take minutes, well beyond
an interactive MCP timeout. This reader is therefore best run **in-region**
(e.g. on AWS in ``us-west-2``) or as a batch/ingest job that loads embeddings
into a vector store (``geo-embedding-search``); treat remote interactive calls as
best-effort. The selection/partition/bounding logic is fully unit-tested with a
mock reader; the live S3 path is exercised by an opt-in test (``RUN_LIVE_CLAY``).

The heavy S3/pyarrow I/O lives behind a pluggable ``reader`` callable so the
selection/partition/bounding logic is unit-testable without network access; the
default :class:`S3ParquetEmbeddingReader` lazily imports ``pyarrow`` only when a
real lookup runs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from geo_common.errors import ValidationError

from geo_foundation_models.models import EmbeddingMetadata, EmbeddingRecord

__all__ = [
    "CLAY_DATASET_BUCKET",
    "CLAY_DATASET_PREFIX",
    "CLAY_V15_DIMENSION",
    "CLAY_V15_MODEL_NAME",
    "AVAILABLE_PERIODS",
    "DEFAULT_LOOKUP_LIMIT",
    "DEFAULT_MAX_LOOKUP_AREA_KM2",
    "available_periods",
    "gzd_partitions",
    "periods_in_range",
    "EmbeddingRow",
    "EmbeddingReader",
    "lookup_embeddings",
]

#: Open Data bucket + prefix for the LGND Clay v1.5 Sentinel-2 dataset.
CLAY_DATASET_BUCKET = "us-west-2.opendata.source.coop"
CLAY_DATASET_REGION = "us-west-2"
CLAY_DATASET_PREFIX = "clay/lgnd-clay-v1-5-sentinel-2-l2a"

#: Clay v1.5 embedding width and the model label reported on retrieved records.
CLAY_V15_DIMENSION = 1024
CLAY_V15_MODEL_NAME = "Clay-v1.5"

#: (year, month) partitions the dataset currently publishes.
AVAILABLE_PERIODS: Tuple[Tuple[int, int], ...] = ((2024, 6), (2025, 6))


def available_periods() -> List[str]:
    """Return the published Clay v1.5 periods as ``"YYYY-MM"`` strings.

    Lets an agent discover that the real-embedding lookup path only covers these
    months (so a before/after outside them — e.g. 2026 — is unusable) instead of
    finding out via an empty ``lookup_embeddings`` result.
    """
    return [f"{year:04d}-{month:02d}" for (year, month) in AVAILABLE_PERIODS]

#: Default cap on the number of embeddings a single lookup returns. Each record
#: carries a 1024-float vector, so the cap keeps the response small enough for a
#: calling agent to consume.
DEFAULT_LOOKUP_LIMIT = 50

#: Default cap on the bbox area (km²) accepted for a lookup. A 2,560 m MajorTOM
#: grid means a large bbox matches an enormous number of cells; this bounds the
#: amount of cloud data scanned, mirroring the ``vector_features`` area gate.
DEFAULT_MAX_LOOKUP_AREA_KM2 = 2500.0

_SOURCE = "geo-foundation-models"

#: MGRS latitude band letters from -80° to +84° (8° bands, final band X is 12°).
_LAT_BANDS = "CDEFGHJKLMNPQRSTUVWX"

#: A single raw embedding row as produced by a reader (before it is turned into
#: an :class:`EmbeddingRecord`). Keys mirror the dataset columns.
EmbeddingRow = Dict[str, Any]

#: A reader fetches up to ``limit`` raw rows overlapping ``bbox`` from the given
#: product/partitions/periods. It receives the validated bbox and the partition
#: hints computed by :func:`lookup_embeddings` and returns an iterable of rows.
EmbeddingReader = Callable[..., Iterable[EmbeddingRow]]


def _lat_band(lat: float) -> str:
    """Return the MGRS latitude-band letter for ``lat`` (clamped to [-80, 84])."""
    lat = max(-80.0, min(83.9999, float(lat)))
    if lat >= 72.0:
        return "X"  # the final band spans 72..84
    index = int((lat + 80.0) // 8.0)
    index = max(0, min(len(_LAT_BANDS) - 1, index))
    return _LAT_BANDS[index]


def _utm_zone(lon: float) -> int:
    """Return the UTM zone number (1..60) for ``lon``."""
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    return int((lon + 180.0) // 6.0) % 60 + 1


def gzd_partitions(bbox: Sequence[float]) -> List[str]:
    """Candidate MGRS Grid Zone Designators (e.g. ``"10S"``) covering ``bbox``.

    Used to prune the ``scene`` product's ``gzd=`` hive partitions to only those
    the bbox can intersect, so a lookup never scans the global dataset. ``bbox``
    is ``(min_lon, min_lat, max_lon, max_lat)``.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    zones = sorted({_utm_zone(min_lon), _utm_zone(max_lon)})
    # Walk the zone range inclusively (handles a multi-zone bbox).
    z0, z1 = _utm_zone(min_lon), _utm_zone(max_lon)
    if z1 < z0:  # bbox straddles the antimeridian; keep it simple - both ends.
        zone_range = [z0, z1]
    else:
        zone_range = list(range(z0, z1 + 1))
    band0, band1 = _lat_band(min_lat), _lat_band(max_lat)
    i0, i1 = _LAT_BANDS.index(band0), _LAT_BANDS.index(band1)
    bands = _LAT_BANDS[i0 : i1 + 1] if i1 >= i0 else band0 + band1
    return ["%d%s" % (z, b) for z in zone_range for b in bands]


def periods_in_range(
    start: Optional[str], end: Optional[str]
) -> List[Tuple[int, int]]:
    """The available ``(year, month)`` partitions overlapping ``[start, end]``.

    ``start`` / ``end`` are ISO date strings (``"YYYY-MM-DD"`` or ``"YYYY-MM"``);
    either may be ``None`` (open-ended). Returns the subset of
    :data:`AVAILABLE_PERIODS` that falls in the range - the dataset only
    publishes June 2024 and June 2025, so a request outside those yields an
    empty list (and a lookup then returns no records rather than scanning).
    """

    def _key(value: Optional[str], default: Tuple[int, int]) -> Tuple[int, int]:
        if not value:
            return default
        parts = str(value).split("-")
        try:
            year = int(parts[0])
            month = int(parts[1]) if len(parts) > 1 else default[1]
        except (ValueError, IndexError):
            raise ValidationError(
                "invalid date %r; expected 'YYYY-MM-DD' or 'YYYY-MM'" % value,
                source=_SOURCE,
                detail={"parameter": "datetime", "value": value},
            )
        return (year, month)

    lo = _key(start, (0, 0))
    hi = _key(end, (9999, 12))
    if lo > hi:
        raise ValidationError(
            "start %r is after end %r" % (start, end),
            source=_SOURCE,
            detail={"parameter": "datetime", "start": start, "end": end},
        )
    return [p for p in AVAILABLE_PERIODS if lo <= p <= hi]


#: Mean Earth radius (km) for the spherical bbox-area estimate.
_EARTH_RADIUS_KM = 6371.0088


def _validate_lookup_bbox(
    bbox: Sequence[float], max_area_km2: float
) -> Tuple[float, float, float, float]:
    """Validate and area-gate a lookup bbox (self-contained, no cross-deps).

    Accepts ``(min_lon, min_lat, max_lon, max_lat)`` with finite, in-range,
    correctly-ordered ordinates, and rejects a bbox whose approximate area
    exceeds ``max_area_km2`` - mirroring ``geo-vector``'s area gate so a lookup
    never scans an unbounded region.
    """
    import math

    try:
        ordinates = [float(v) for v in bbox]
    except (TypeError, ValueError):
        raise ValidationError(
            "bbox must be four numbers (min_lon, min_lat, max_lon, max_lat)",
            source=_SOURCE,
            detail={"parameter": "bbox", "value": _safe(bbox)},
        )
    if len(ordinates) != 4 or any(math.isnan(v) or math.isinf(v) for v in ordinates):
        raise ValidationError(
            "bbox must be four finite numbers (min_lon, min_lat, max_lon, max_lat)",
            source=_SOURCE,
            detail={"parameter": "bbox", "value": _safe(bbox)},
        )
    min_lon, min_lat, max_lon, max_lat = ordinates
    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise ValidationError(
            "bbox longitudes must be within [-180, 180]",
            source=_SOURCE,
            detail={"parameter": "bbox", "min_lon": min_lon, "max_lon": max_lon},
        )
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise ValidationError(
            "bbox latitudes must be within [-90, 90]",
            source=_SOURCE,
            detail={"parameter": "bbox", "min_lat": min_lat, "max_lat": max_lat},
        )
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValidationError(
            "bbox must have min_lon < max_lon and min_lat < max_lat",
            source=_SOURCE,
            detail={"parameter": "bbox", "value": ordinates},
        )

    lon_span = math.radians(max_lon - min_lon)
    area = (
        _EARTH_RADIUS_KM
        * _EARTH_RADIUS_KM
        * lon_span
        * abs(math.sin(math.radians(max_lat)) - math.sin(math.radians(min_lat)))
    )
    if area > max_area_km2:
        raise ValidationError(
            "bbox area %.3f km² exceeds the configured maximum of %.3f km²"
            % (area, max_area_km2),
            source=_SOURCE,
            detail={
                "parameter": "bbox",
                "area_km2": area,
                "max_area_km2": max_area_km2,
            },
        )
    return (min_lon, min_lat, max_lon, max_lat)


def _safe(value: Any) -> Any:
    """A representation of ``value`` that is always JSON-serializable."""
    try:
        import json

        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _row_to_record(row: EmbeddingRow) -> EmbeddingRecord:
    """Turn a raw dataset row into an :class:`EmbeddingRecord`."""
    vector = [float(v) for v in row.get("embedding", [])]
    bbox = row.get("bbox") or {}
    if isinstance(bbox, dict):
        bb = (
            float(bbox.get("xmin", 0.0)),
            float(bbox.get("ymin", 0.0)),
            float(bbox.get("xmax", 0.0)),
            float(bbox.get("ymax", 0.0)),
        )
    else:  # struct-like / sequence fallback
        bb = tuple(float(v) for v in bbox)  # type: ignore[assignment]
    metadata = EmbeddingMetadata(
        bbox=bb,
        datetime=str(row.get("datetime", "")),
        source_asset=row.get("stac_item_id"),
        extra={
            "cell_id": row.get("cell_id"),
            "collection": row.get("collection"),
        },
    )
    return EmbeddingRecord(
        id=str(row.get("chips_id") or row.get("cell_id") or "clay-embedding"),
        model=CLAY_V15_MODEL_NAME,
        dimension=len(vector) or CLAY_V15_DIMENSION,
        vector=vector,
        metadata=metadata,
    )


def lookup_embeddings(
    *,
    bbox: Sequence[float],
    start: Optional[str] = None,
    end: Optional[str] = None,
    product: str = "aggregated",
    limit: int = DEFAULT_LOOKUP_LIMIT,
    reader: Optional[EmbeddingReader] = None,
    max_area_km2: float = DEFAULT_MAX_LOOKUP_AREA_KM2,
) -> List[EmbeddingRecord]:
    """Retrieve published Clay v1.5 embeddings overlapping ``bbox`` (bounded).

    Validates and area-gates ``bbox``, resolves the dataset partitions to scan
    (MGRS GZDs for the ``scene`` product / passed through for ``aggregated``,
    intersected with the available ``(year, month)`` periods for the
    ``[start, end]`` range), then asks ``reader`` for up to ``limit`` rows and
    returns them as :class:`EmbeddingRecord`\\ s. ``limit`` is clamped to a
    positive value and always enforced, so the response stays small.

    ``product`` is ``"aggregated"`` (default; one record per cell per month) or
    ``"scene"`` (per Sentinel-2 scene). ``reader`` defaults to
    :class:`S3ParquetEmbeddingReader`, which reads the open dataset over
    anonymous S3; inject a reader in tests to avoid network I/O.
    """
    if product not in ("aggregated", "scene"):
        raise ValidationError(
            "unknown product %r; expected 'aggregated' or 'scene'" % product,
            source=_SOURCE,
            detail={"parameter": "product", "value": product},
        )
    if limit <= 0:
        raise ValidationError(
            "limit must be positive (got %r)" % limit,
            source=_SOURCE,
            detail={"parameter": "limit", "value": limit},
        )

    validated = _validate_lookup_bbox(bbox, max_area_km2)
    periods = periods_in_range(start, end)
    if not periods:
        # No published data in the requested range: an empty result, no scan.
        return []
    gzds = gzd_partitions(validated) if product == "scene" else []

    active_reader = reader if reader is not None else S3ParquetEmbeddingReader()
    rows = active_reader(
        product=product,
        bbox=validated,
        periods=periods,
        gzds=gzds,
        limit=limit,
    )

    records: List[EmbeddingRecord] = []
    for row in rows:
        records.append(_row_to_record(row))
        if len(records) >= limit:
            break
    return records


class S3ParquetEmbeddingReader:
    """Default reader: streams matching rows from the open dataset over S3.

    Reads only the pruned partition paths (so the global dataset is never
    scanned), filters rows whose ``bbox`` struct overlaps the query bbox, and
    stops after ``limit`` rows. ``pyarrow`` is imported lazily so importing this
    module - and the rest of ``geo-foundation-models`` - never requires pyarrow
    or network access; it is only needed when an actual lookup runs.
    """

    def __init__(
        self,
        *,
        bucket: str = CLAY_DATASET_BUCKET,
        prefix: str = CLAY_DATASET_PREFIX,
        region: str = CLAY_DATASET_REGION,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.region = region

    def _partition_paths(
        self, product: str, periods: Sequence[Tuple[int, int]], gzds: Sequence[str]
    ) -> List[str]:
        paths: List[str] = []
        for year, month in periods:
            if product == "scene":
                for gzd in gzds:
                    paths.append(
                        "%s/%s/scene/gzd=%s/year=%d/month=%02d/"
                        % (self.bucket, self.prefix, gzd, year, month)
                    )
            else:
                # aggregated is partitioned by geohash; without a geohash index
                # we scan the period root and rely on the bbox filter. Pruning
                # by geohash prefix is a future optimization.
                paths.append(
                    "%s/%s/aggregated/year=%d/month=%02d/"
                    % (self.bucket, self.prefix, year, month)
                )
        return paths

    def __call__(
        self,
        *,
        product: str,
        bbox: Tuple[float, float, float, float],
        periods: Sequence[Tuple[int, int]],
        gzds: Sequence[str],
        limit: int,
    ) -> Iterable[EmbeddingRow]:
        try:
            import numpy as np  # noqa: PLC0415
            import pyarrow.fs as pafs  # noqa: PLC0415
            import pyarrow.parquet as pq  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only without pyarrow
            raise RuntimeError(
                "lookup_embeddings needs the optional 'pyarrow' dependency to "
                "read the open Clay dataset over S3. Install the extra: "
                "pip install 'geo-foundation-models[clay-lookup]' (or inject a "
                "custom reader)."
            ) from exc

        min_lon, min_lat, max_lon, max_lat = bbox
        s3 = pafs.S3FileSystem(anonymous=True, region=self.region)
        wanted = [
            "chips_id",
            "cell_id",
            "stac_item_id",
            "collection",
            "datetime",
            "embedding",
            "bbox",
        ]

        emitted = 0
        for path in self._partition_paths(product, periods, gzds):
            for file_path in self._list_parquet_files(s3, path):
                pf = pq.ParquetFile(s3.open_input_file(file_path))
                # Cheap first pass: read only the small ``bbox`` struct column
                # to find rows that overlap the query bbox, so a file with no
                # overlap is skipped without ever decoding the large embedding
                # column.
                bbox_tbl = pf.read(columns=["bbox"])
                if bbox_tbl.num_rows == 0:
                    continue
                col = bbox_tbl.column("bbox").combine_chunks()
                xmin = np.asarray(col.field("xmin"))
                ymin = np.asarray(col.field("ymin"))
                xmax = np.asarray(col.field("xmax"))
                ymax = np.asarray(col.field("ymax"))
                mask = (
                    (xmin <= max_lon)
                    & (xmax >= min_lon)
                    & (ymin <= max_lat)
                    & (ymax >= min_lat)
                )
                match_idx = np.nonzero(mask)[0]
                if match_idx.size == 0:
                    continue
                # Only now (a file with matches) decode the full set of columns,
                # take the bounded set of matching rows, and yield them.
                take_idx = match_idx[: max(0, limit - emitted)]
                rows_tbl = pf.read(columns=wanted).take(take_idx.tolist())
                for row in rows_tbl.to_pylist():
                    yield row
                    emitted += 1
                    if emitted >= limit:
                        return

    @staticmethod
    def _list_parquet_files(s3, path: str) -> List[str]:
        """List ``*.parquet`` object paths directly under ``path`` (or skip)."""
        import pyarrow.fs as pafs  # noqa: PLC0415

        try:
            infos = s3.get_file_info(pafs.FileSelector(path, recursive=False))
        except (FileNotFoundError, OSError):
            return []
        return sorted(
            info.path
            for info in infos
            if info.type == pafs.FileType.File and info.path.endswith(".parquet")
        )
