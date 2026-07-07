"""GeoParquet conversion for ``geo-formats`` (Requirements 8.5, 12.3).

This module holds :func:`to_geoparquet`, the engine behind ``geo-formats``'s
``to_geoparquet`` MCP tool. It accepts a vector dataset — either an in-memory
GeoJSON :class:`~geo_formats.models.FeatureCollection` (or equivalent mapping),
a local path to an existing ``.geojson``/``.parquet`` source, or a **remote
href** (``s3://``, ``gs://``, ``https://`` …) when the optional ``remote`` extra
is installed — and writes a **GeoParquet** file to ``dst_href`` via GeoPandas
(which embeds the GeoParquet ``geo`` metadata and stores geometries as WKB).

Accepting a path/href (not just inline features) lets a large vector produced by
an earlier step be referenced by location instead of pasted inline.

The conversion is exact: converting a vector dataset to GeoParquet and reading
it back preserves the feature count, the geometries coordinate-for-coordinate,
and every attribute value (design Property 3 / Requirements 12.3, 15.4).

Failure handling mirrors :mod:`geo_formats.cog` and the cloud-optimized-formats
contract (Requirement 12.7): a write failure aborts, removes any partial output
created at ``dst_href``, and raises a ``geo_common`` ``Error_Taxonomy`` error.
Malformed inputs raise a taxonomy
:class:`~geo_common.errors.ValidationError`.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

from geo_common.errors import GeoError, UpstreamError, ValidationError

from geo_formats.models import FeatureCollection, FormatResult

__all__ = ["to_geoparquet", "DEFAULT_CRS"]

#: GeoJSON's coordinate reference system is WGS 84 lon/lat by specification, so
#: in-memory feature collections without an explicit CRS are written as
#: EPSG:4326.
DEFAULT_CRS: str = "EPSG:4326"

#: The accepted in-memory source types for :func:`to_geoparquet`.
SourceType = Union[FeatureCollection, Dict[str, Any], str]


def to_geoparquet(
    src: SourceType,
    dst_href: str,
    *,
    crs: Optional[str] = None,
    source: str = "geo-formats",
) -> FormatResult:
    """Convert a vector dataset ``src`` to GeoParquet at ``dst_href`` (Req 8.5).

    ``src`` may be a :class:`FeatureCollection` (or a GeoJSON
    ``FeatureCollection`` mapping), or a path string to an existing
    ``.parquet`` or ``.geojson``/``.json`` file. The written file is a valid
    GeoParquet (geometry stored as WKB with embedded ``geo`` metadata) whose
    round-trip preserves feature count, geometries, and attributes exactly
    (Property 3).

    Returns a :class:`FormatResult`. Raises a taxonomy
    :class:`~geo_common.errors.ValidationError` for malformed input, and (after
    removing any partial output) an :class:`~geo_common.errors.UpstreamError`
    if writing the GeoParquet fails (Requirement 12.7).
    """
    if not dst_href:
        raise ValidationError(
            "to_geoparquet requires a non-empty dst_href", source=source,
            detail={"parameter": "dst_href"},
        )

    gdf = _to_geodataframe(src, crs=crs, source=source)

    existed_before = os.path.exists(dst_href)
    try:
        gdf.to_parquet(dst_href)
    except GeoError:
        _remove_partial(dst_href, existed_before=existed_before)
        raise
    except Exception as exc:
        _remove_partial(dst_href, existed_before=existed_before)
        raise UpstreamError(
            "failed to write GeoParquet to %r: %s"
            % (dst_href, str(exc) or type(exc).__name__),
            source=source,
            detail={"dst_href": dst_href},
            original=str(exc) or type(exc).__name__,
        ) from exc

    size_bytes = os.path.getsize(dst_href) if os.path.exists(dst_href) else None
    attribute_columns = [c for c in gdf.columns if c != gdf.geometry.name]
    return FormatResult(
        href=dst_href,
        fmt="GeoParquet",
        driver="Parquet",
        size_bytes=size_bytes,
        detail={
            "feature_count": int(len(gdf)),
            "columns": attribute_columns,
            "crs": str(gdf.crs) if gdf.crs is not None else None,
        },
    )


def _to_geodataframe(src: SourceType, *, crs: Optional[str], source: str):
    """Build a GeoDataFrame from the accepted ``src`` forms."""
    import geopandas as gpd

    # A path to an existing dataset.
    if isinstance(src, str):
        return _read_path(src, source=source)

    # A pydantic FeatureCollection or a GeoJSON mapping.
    if isinstance(src, FeatureCollection):
        features = [f.model_dump() for f in src.features]
    elif isinstance(src, dict):
        if src.get("type") != "FeatureCollection" or "features" not in src:
            raise ValidationError(
                "mapping source must be a GeoJSON FeatureCollection",
                source=source,
                detail={"parameter": "src"},
            )
        features = list(src.get("features") or [])
    else:
        raise ValidationError(
            "unsupported src type %r for to_geoparquet" % type(src).__name__,
            source=source,
            detail={"parameter": "src"},
        )

    geometries, records = _materialize_features(features, source=source)
    gdf = gpd.GeoDataFrame(
        records,
        geometry=geometries,
        crs=crs or DEFAULT_CRS,
    )
    return gdf


def _materialize_features(features: List[Dict[str, Any]], *, source: str):
    """Turn GeoJSON features into shapely geometries plus attribute records."""
    from shapely.geometry import shape

    geometries = []
    records: List[Dict[str, Any]] = []
    for idx, feat in enumerate(features):
        geom_mapping = feat.get("geometry") if isinstance(feat, dict) else None
        if geom_mapping is None:
            geometries.append(None)
        else:
            try:
                geometries.append(shape(geom_mapping))
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise ValidationError(
                    "feature %d has a malformed geometry: %s"
                    % (idx, str(exc) or type(exc).__name__),
                    source=source,
                    detail={"feature_index": idx},
                    original=str(exc) or type(exc).__name__,
                ) from exc
        props = feat.get("properties") if isinstance(feat, dict) else None
        records.append(dict(props) if isinstance(props, dict) else {})
    return geometries, records


#: URI schemes read from remote object stores (via the optional ``remote`` extra)
#: rather than the local filesystem.
_REMOTE_SCHEMES = ("s3://", "gs://", "gcs://", "http://", "https://", "az://", "abfs://")


def _is_remote(path: str) -> bool:
    """Whether ``path`` is a remote object-store/URL href rather than a local path."""
    lowered = path.lower()
    return any(lowered.startswith(scheme) for scheme in _REMOTE_SCHEMES)


def _extension(path: str) -> str:
    """Lowercased file extension of ``path``, ignoring any URL query string."""
    return path.split("?", 1)[0].lower()


def _read_path(path: str, *, source: str):
    """Read an existing ``.parquet``/``.geojson``/``.json`` source (local or remote).

    Local paths are read directly. Remote hrefs (``s3://``, ``gs://``,
    ``https://`` …) are read via geopandas/fsspec, which needs the optional
    ``remote`` extra (``geo-formats[remote]``); a missing backend raises a clear
    validation error telling the caller what to install.
    """
    if _is_remote(path):
        return _read_remote(path, source=source)

    import geopandas as gpd

    if not os.path.exists(path):
        raise ValidationError(
            "source path %r does not exist" % path, source=source,
            detail={"parameter": "src", "src": path},
        )
    lower = _extension(path)
    try:
        if lower.endswith(".parquet"):
            return gpd.read_parquet(path)
        if lower.endswith(".geojson") or lower.endswith(".json"):
            with open(path, "r", encoding="utf-8") as fh:
                mapping = json.load(fh)
            return _to_geodataframe(mapping, crs=None, source=source)
    except GeoError:
        raise
    except Exception as exc:
        raise ValidationError(
            "could not read vector source %r: %s"
            % (path, str(exc) or type(exc).__name__),
            source=source,
            detail={"parameter": "src", "src": path},
            original=str(exc) or type(exc).__name__,
        ) from exc
    raise ValidationError(
        "unsupported source extension for %r (expected .parquet/.geojson/.json)"
        % path,
        source=source,
        detail={"parameter": "src", "src": path},
    )


def _read_remote(path: str, *, source: str):
    """Read a remote vector href via geopandas/fsspec (optional ``remote`` extra)."""
    import geopandas as gpd

    lower = _extension(path)
    if not (lower.endswith(".parquet") or lower.endswith(".geojson") or lower.endswith(".json")):
        raise ValidationError(
            "unsupported source extension for %r (expected .parquet/.geojson/.json)"
            % path,
            source=source,
            detail={"parameter": "src", "src": path},
        )
    try:
        if lower.endswith(".parquet"):
            return gpd.read_parquet(path)
        import fsspec  # local import: only needed for remote text reads

        with fsspec.open(path, "r") as fh:
            mapping = json.load(fh)
        return _to_geodataframe(mapping, crs=None, source=source)
    except GeoError:
        raise
    except ImportError as exc:
        raise ValidationError(
            "reading a remote source %r needs the optional 'remote' extra "
            "(install geo-formats[remote]); missing backend: %s"
            % (path, getattr(exc, "name", None) or str(exc)),
            source=source,
            detail={"parameter": "src", "src": path},
            original=str(exc),
        ) from exc
    except Exception as exc:
        raise ValidationError(
            "could not read remote vector source %r: %s "
            "(remote reads need geo-formats[remote] and valid credentials)"
            % (path, str(exc) or type(exc).__name__),
            source=source,
            detail={"parameter": "src", "src": path},
            original=str(exc) or type(exc).__name__,
        ) from exc


def _remove_partial(dst_href: str, *, existed_before: bool) -> None:
    """Remove a partially written GeoParquet created by a failed conversion.

    Only removes ``dst_href`` when this call created it, mirroring
    :func:`geo_formats.cog._remove_partial` (Requirement 12.7).
    """
    if existed_before:
        return
    try:
        if os.path.exists(dst_href):
            os.remove(dst_href)
    except OSError:  # pragma: no cover - cleanup is best-effort
        pass
