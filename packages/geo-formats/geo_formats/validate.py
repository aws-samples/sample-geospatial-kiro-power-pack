"""Format validation for ``geo-formats`` (design.md "Pillar B").

This module holds :func:`validate_format`, the engine behind ``geo-formats``'s
``validate_format`` MCP tool. It checks that the output at ``href`` is actually
a well-formed instance of the claimed format:

* ``"COG"`` — a GeoTIFF that opens, is **internally tiled** (the defining COG
  property that enables partial reads), and carries georeferencing. Overviews
  are reported but not required, since a raster smaller than one tile
  legitimately has none.
* ``"GeoParquet"`` — a Parquet file that opens and carries the GeoParquet
  ``geo`` metadata key in its schema (geometry stored as WKB).

It returns a :class:`~geo_formats.models.FormatValidity` with a ``valid`` flag
and a ``reason`` that is non-empty exactly when the output is not a valid
instance of the requested format. An unknown ``fmt`` is rejected with a
taxonomy :class:`~geo_common.errors.ValidationError`.
"""

from __future__ import annotations

from typing import Any, Dict

from geo_common.errors import ValidationError

from geo_formats.models import FormatValidity

__all__ = ["validate_format", "SUPPORTED_FORMATS"]

#: Canonical (lower-cased) format names :func:`validate_format` understands.
SUPPORTED_FORMATS = ("cog", "geoparquet")


def validate_format(href: str, fmt: str, *, source: str = "geo-formats") -> FormatValidity:
    """Validate that ``href`` is a well-formed instance of ``fmt``.

    ``fmt`` is matched case-insensitively against :data:`SUPPORTED_FORMATS`
    (``"COG"`` / ``"GeoParquet"``). Returns a
    :class:`~geo_formats.models.FormatValidity`; raises a taxonomy
    :class:`~geo_common.errors.ValidationError` when ``fmt`` is unknown or
    ``href`` is empty.
    """
    if not href:
        raise ValidationError(
            "validate_format requires a non-empty href", source=source,
            detail={"parameter": "href"},
        )
    key = (fmt or "").strip().lower()
    if key not in SUPPORTED_FORMATS:
        raise ValidationError(
            "unsupported format %r (expected one of: COG, GeoParquet)" % (fmt,),
            source=source,
            detail={"parameter": "fmt", "fmt": fmt},
        )
    if key == "cog":
        return _validate_cog(href)
    return _validate_geoparquet(href)


def _validate_cog(href: str) -> FormatValidity:
    """Validate a Cloud-Optimized GeoTIFF: opens, tiled, georeferenced."""
    import rasterio
    from rasterio.errors import RasterioError, RasterioIOError

    try:
        with rasterio.open(href) as ds:
            driver = ds.driver
            is_tiled = bool(ds.profile.get("tiled", False))
            overview_levels = len(ds.overviews(1)) if ds.count >= 1 else 0
            has_crs = ds.crs is not None
            detail: Dict[str, Any] = {
                "driver": driver,
                "tiled": is_tiled,
                "overviews": overview_levels,
                "bands": ds.count,
                "has_crs": has_crs,
            }
    except (RasterioError, RasterioIOError, OSError) as exc:
        return FormatValidity(
            valid=False,
            fmt="COG",
            reason="could not open as a GeoTIFF: %s"
            % (str(exc) or type(exc).__name__),
            detail={"href": href},
        )

    if driver not in ("GTiff", "COG"):
        return FormatValidity(
            valid=False,
            fmt="COG",
            reason="output is not a GeoTIFF (driver=%r)" % driver,
            detail=detail,
        )
    if not is_tiled:
        return FormatValidity(
            valid=False,
            fmt="COG",
            reason="GeoTIFF is not internally tiled (not cloud-optimized)",
            detail=detail,
        )
    return FormatValidity(valid=True, fmt="COG", reason=None, detail=detail)


def _validate_geoparquet(href: str) -> FormatValidity:
    """Validate a GeoParquet: opens as Parquet and carries ``geo`` metadata."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - pyarrow is a declared dep
        return FormatValidity(
            valid=False,
            fmt="GeoParquet",
            reason="pyarrow is required to validate GeoParquet: %s" % exc,
            detail={"href": href},
        )

    try:
        schema = pq.read_schema(href)
    except Exception as exc:
        return FormatValidity(
            valid=False,
            fmt="GeoParquet",
            reason="could not open as Parquet: %s" % (str(exc) or type(exc).__name__),
            detail={"href": href},
        )

    metadata = schema.metadata or {}
    has_geo = b"geo" in metadata
    detail = {
        "columns": [field.name for field in schema],
        "has_geo_metadata": has_geo,
    }
    if not has_geo:
        return FormatValidity(
            valid=False,
            fmt="GeoParquet",
            reason="Parquet file is missing the GeoParquet 'geo' metadata key",
            detail=detail,
        )
    return FormatValidity(valid=True, fmt="GeoParquet", reason=None, detail=detail)
