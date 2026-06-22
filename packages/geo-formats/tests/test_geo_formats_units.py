"""Unit tests for ``geo-formats`` (task 14.1).

Cover the three tools end to end through the assembled
:class:`~geo_formats.server.GeoFormatsServer`:

* ``to_cog`` — produces a Cloud-Optimized GeoTIFF whose pixels round-trip
  exactly, including nodata (Requirements 8.4, 12.2).
* ``to_geoparquet`` — produces a GeoParquet whose features round-trip exactly
  (Requirements 8.5, 12.3).
* ``validate_format`` — recognizes valid COG/GeoParquet outputs and rejects
  non-conforming or unknown ones.
* The write-failure path aborts, removes any partial output, and raises a
  taxonomy error (Requirement 12.7).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geo_common.errors import GeoError, UpstreamError, ValidationError
from geo_common.models import CredentialClassification
from geo_formats import FeatureCollection, GeoFormatsServer


def _write_source_raster(path: str, *, bands: int = 2, nodata: float = 0.0) -> np.ndarray:
    """Write a small multi-band source GeoTIFF and return its pixel array."""
    height, width = 32, 48
    data = np.stack(
        [
            ((np.arange(height * width).reshape(height, width) + b * 7) % 65000).astype(
                "uint16"
            )
            for b in range(bands)
        ]
    )
    # Flag a couple of pixels as nodata so the round-trip exercises nodata too.
    data[:, 0, 0] = int(nodata)
    data[:, 5, 9] = int(nodata)
    profile = dict(
        driver="GTiff",
        dtype="uint16",
        count=bands,
        height=height,
        width=width,
        crs="EPSG:4326",
        transform=from_origin(10.0, 50.0, 0.01, 0.01),
        nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return data


@pytest.fixture
def server() -> GeoFormatsServer:
    return GeoFormatsServer()


def test_server_registers_tools_and_starts(server: GeoFormatsServer) -> None:
    assert set(server.tool_names()) == {"to_cog", "to_geoparquet", "validate_format"}
    # Optional AWS credentials only: starts with no configured credentials (Req 16.5).
    server.start(configured_keys=[])
    assert server.started is True
    cred_keys = {c.mcp_json_key for c in server.required_credentials()}
    assert cred_keys == {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}
    assert all(
        c.classification is CredentialClassification.OPTIONAL
        for c in server.required_credentials()
    )
    names = {e.name for e in server.catalog_entries()}
    assert names == {"to_cog", "to_geoparquet", "validate_format"}


async def test_to_cog_roundtrip_is_pixel_exact(server: GeoFormatsServer, tmp_path) -> None:
    src = str(tmp_path / "src.tif")
    dst = str(tmp_path / "out_cog.tif")
    source = _write_source_raster(src, bands=2)

    result = await server.to_cog(src_href=src, dst_href=dst, overviews=True)

    assert result.fmt == "COG"
    assert os.path.exists(dst)
    assert result.detail["bands"] == 2
    with rasterio.open(dst) as ds:
        assert ds.profile.get("tiled") is True
        back = ds.read()
    # Every band, every pixel (including nodata-flagged) is exactly preserved.
    assert np.array_equal(back, source)


async def test_to_cog_validates_as_cog(server: GeoFormatsServer, tmp_path) -> None:
    src = str(tmp_path / "src.tif")
    dst = str(tmp_path / "out_cog.tif")
    _write_source_raster(src)
    await server.to_cog(src_href=src, dst_href=dst)

    validity = await server.validate_format(href=dst, fmt="COG")
    assert validity.valid is True
    assert validity.reason is None
    assert validity.detail["tiled"] is True


async def test_to_cog_missing_source_is_validation_error(server: GeoFormatsServer, tmp_path) -> None:
    dst = str(tmp_path / "out.tif")
    with pytest.raises(ValidationError):
        await server.to_cog(src_href=str(tmp_path / "nope.tif"), dst_href=dst)
    # No partial output left behind for a source that never opened.
    assert not os.path.exists(dst)


async def test_to_cog_write_failure_aborts_and_cleans_up(server: GeoFormatsServer, tmp_path, monkeypatch) -> None:
    src = str(tmp_path / "src.tif")
    dst = str(tmp_path / "out_cog.tif")
    _write_source_raster(src)

    # Simulate a writer that fails *after* creating a partial output file.
    real_open = rasterio.open

    def flaky_open(path, mode="r", *args, **kwargs):
        if mode == "w":
            with open(path, "wb") as fh:
                fh.write(b"partial")  # leave a partial artifact on disk
            raise OSError("disk full")
        return real_open(path, mode, *args, **kwargs)

    # cog.py does ``import rasterio`` inside the function, so patching the
    # rasterio module's ``open`` is what the conversion will see.
    monkeypatch.setattr(rasterio, "open", flaky_open)

    with pytest.raises(UpstreamError):
        await server.to_cog(src_href=src, dst_href=dst)

    # Partial output removed (Requirement 12.7).
    assert not os.path.exists(dst)


async def test_to_geoparquet_roundtrip_preserves_features(server: GeoFormatsServer, tmp_path) -> None:
    import geopandas as gpd

    dst = str(tmp_path / "out.parquet")
    fc = FeatureCollection.model_validate(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1.5, 2.5]},
                    "properties": {"name": "a", "value": 10},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                    },
                    "properties": {"name": "b", "value": 20},
                },
            ],
        }
    )

    result = await server.to_geoparquet(src=fc, dst_href=dst)
    assert result.fmt == "GeoParquet"
    assert result.detail["feature_count"] == 2

    back = gpd.read_parquet(dst)
    assert len(back) == 2
    assert list(back["name"]) == ["a", "b"]
    assert list(back["value"]) == [10, 20]
    # Coordinate-for-coordinate geometry preservation.
    from shapely.geometry import shape

    expected = [shape(f.geometry.to_geojson()) for f in fc.features]
    assert all(a.equals(b) for a, b in zip(back.geometry, expected))


async def test_to_geoparquet_validates_as_geoparquet(server: GeoFormatsServer, tmp_path) -> None:
    dst = str(tmp_path / "out.parquet")
    fc = FeatureCollection.model_validate(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"k": 1},
                }
            ],
        }
    )
    await server.to_geoparquet(src=fc, dst_href=dst)
    validity = await server.validate_format(href=dst, fmt="GeoParquet")
    assert validity.valid is True
    assert validity.detail["has_geo_metadata"] is True


async def test_to_geoparquet_rejects_non_feature_collection_mapping(server: GeoFormatsServer, tmp_path) -> None:
    with pytest.raises(ValidationError):
        await server.to_geoparquet(src={"type": "Nonsense"}, dst_href=str(tmp_path / "x.parquet"))


async def test_validate_format_unknown_format_is_validation_error(server: GeoFormatsServer, tmp_path) -> None:
    with pytest.raises(ValidationError):
        await server.validate_format(href=str(tmp_path / "x"), fmt="shapefile")


async def test_validate_format_rejects_non_geoparquet(server: GeoFormatsServer, tmp_path) -> None:
    bogus = str(tmp_path / "bogus.parquet")
    with open(bogus, "wb") as fh:
        fh.write(b"not a parquet file")
    validity = await server.validate_format(href=bogus, fmt="GeoParquet")
    assert validity.valid is False
    assert validity.reason
