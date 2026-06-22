"""Pillar B expansion I/O + validation sweep for ``geo-formats`` (task 14.12).

Task 14.1 already covers the COG write-failure cleanup path and the happy-path
round-trips; this module fills the remaining write abort-and-cleanup and
``validate_format`` gaps the dedicated test sweep calls out (Requirement 12.7
for the *write* side, and the format-validation contract):

* **GeoParquet write failure aborts and removes any partial output** (Req 12.7)
  - the mirror of the COG case, exercised by forcing ``GeoDataFrame.to_parquet``
  to fail after leaving a partial artifact at the destination.
* **A pre-existing destination is preserved on a failed overwrite** for *both*
  formats - the conversion only removes output it created itself, so a failed
  re-conversion never destroys a file that was already there (Req 12.7,
  ``_remove_partial(existed_before=True)`` branch).
* **``validate_format`` validation** - an empty ``href`` and an unknown ``fmt``
  raise a taxonomy ``ValidationError`` (no output), and a plain (non-tiled)
  GeoTIFF is reported *invalid* as a COG because it is not cloud-optimized.

Everything runs against on-disk temp files / monkeypatched writers, so no S3 or
network access is required. The repo's ``asyncio_mode = "auto"`` lets the
``async def`` tests run directly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geo_common.errors import UpstreamError, ValidationError
from geo_formats import FeatureCollection, GeoFormatsServer
from geo_formats.geoparquet import to_geoparquet
from geo_formats.validate import validate_format


@pytest.fixture
def server() -> GeoFormatsServer:
    return GeoFormatsServer()


def _point_fc() -> FeatureCollection:
    return FeatureCollection.model_validate(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                    "properties": {"k": 1},
                }
            ],
        }
    )


def _write_plain_geotiff(path: str) -> None:
    """Write a *non-tiled* (striped) GeoTIFF - a valid GeoTIFF, not a COG."""
    data = (np.arange(8 * 8).reshape(8, 8) % 250).astype("uint8")
    profile = dict(
        driver="GTiff",
        dtype="uint8",
        count=1,
        height=8,
        width=8,
        crs="EPSG:4326",
        transform=from_origin(0.0, 8.0, 1.0, 1.0),
        tiled=False,  # striped: the defining "not cloud-optimized" case
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


# --- GeoParquet write failure aborts + cleans up (Requirement 12.7) -------


async def test_geoparquet_write_failure_aborts_and_cleans_up(
    server: GeoFormatsServer, tmp_path, monkeypatch
) -> None:
    """Req 12.7: a GeoParquet write failure removes the partial output + raises."""
    import geopandas as gpd

    dst = str(tmp_path / "out.parquet")

    real_to_parquet = gpd.GeoDataFrame.to_parquet

    def flaky_to_parquet(self, path, *args, **kwargs):
        # Leave a partial artifact behind, then fail like a broken writer.
        with open(path, "wb") as fh:
            fh.write(b"partial-parquet")
        raise OSError("disk full")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", flaky_to_parquet)

    with pytest.raises(UpstreamError):
        await server.to_geoparquet(src=_point_fc(), dst_href=dst)

    # Partial output removed (Requirement 12.7).
    assert not os.path.exists(dst)

    # Sanity: with the real writer restored, the conversion succeeds + cleans nothing.
    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", real_to_parquet)
    result = await server.to_geoparquet(src=_point_fc(), dst_href=dst)
    assert result.fmt == "GeoParquet"
    assert os.path.exists(dst)


async def test_geoparquet_failed_overwrite_preserves_existing_output(
    tmp_path, monkeypatch
) -> None:
    """Req 12.7: a failed re-conversion must not destroy a pre-existing file.

    ``_remove_partial`` only deletes output the conversion itself created, so an
    overwrite that fails leaves the original (already-present) destination
    untouched.
    """
    import geopandas as gpd

    dst = str(tmp_path / "existing.parquet")
    sentinel = b"PRE-EXISTING CONTENT - must survive a failed overwrite"
    with open(dst, "wb") as fh:
        fh.write(sentinel)

    def flaky_to_parquet(self, path, *args, **kwargs):
        # Clobber the destination, then fail: cleanup must NOT remove it because
        # the file existed before this conversion began.
        with open(path, "wb") as fh:
            fh.write(b"half-written")
        raise OSError("write error mid-stream")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", flaky_to_parquet)

    with pytest.raises(UpstreamError):
        to_geoparquet(_point_fc(), dst)

    # The pre-existing destination is left in place (not deleted by cleanup).
    assert os.path.exists(dst)


async def test_cog_failed_overwrite_preserves_existing_output(
    server: GeoFormatsServer, tmp_path, monkeypatch
) -> None:
    """Req 12.7: a failed COG overwrite preserves the pre-existing destination."""
    # A source raster to convert.
    src = str(tmp_path / "src.tif")
    src_data = (np.arange(16 * 16).reshape(16, 16) % 250).astype("uint8")
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        dtype="uint8",
        count=1,
        height=16,
        width=16,
        crs="EPSG:4326",
        transform=from_origin(0.0, 16.0, 1.0, 1.0),
    ) as dst_ds:
        dst_ds.write(src_data, 1)

    dst = str(tmp_path / "existing_cog.tif")
    with open(dst, "wb") as fh:
        fh.write(b"PRE-EXISTING COG PLACEHOLDER")

    real_open = rasterio.open

    def flaky_open(path, mode="r", *args, **kwargs):
        if mode == "w":
            with open(path, "wb") as fh:
                fh.write(b"partial")
            raise OSError("disk full")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", flaky_open)

    with pytest.raises(UpstreamError):
        await server.to_cog(src_href=src, dst_href=dst)

    # The destination existed before the conversion, so cleanup leaves it alone.
    assert os.path.exists(dst)


# --- validate_format validation contract ----------------------------------


async def test_validate_format_empty_href_is_validation_error(
    server: GeoFormatsServer,
) -> None:
    with pytest.raises(ValidationError):
        await server.validate_format(href="", fmt="COG")


async def test_validate_format_unknown_format_is_validation_error(
    server: GeoFormatsServer, tmp_path
) -> None:
    with pytest.raises(ValidationError):
        await server.validate_format(href=str(tmp_path / "x.tif"), fmt="zarr")


async def test_validate_format_rejects_non_tiled_geotiff_as_cog(
    server: GeoFormatsServer, tmp_path
) -> None:
    """A striped GeoTIFF opens fine but is not internally tiled -> invalid COG."""
    plain = str(tmp_path / "plain.tif")
    _write_plain_geotiff(plain)
    validity = await server.validate_format(href=plain, fmt="COG")
    assert validity.valid is False
    assert validity.reason and "tiled" in validity.reason.lower()
    assert validity.detail["tiled"] is False


async def test_validate_format_rejects_unopenable_cog(
    server: GeoFormatsServer, tmp_path
) -> None:
    """A file that is not a GeoTIFF at all is reported invalid (not raised)."""
    bogus = str(tmp_path / "bogus.tif")
    with open(bogus, "wb") as fh:
        fh.write(b"definitely not a geotiff")
    validity = await server.validate_format(href=bogus, fmt="COG")
    assert validity.valid is False
    assert validity.reason
