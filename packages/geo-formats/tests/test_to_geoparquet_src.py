"""Tests for ``to_geoparquet`` source forms: inline, local path, and remote href.

The MCP tool advertises ``src`` as an inline ``FeatureCollection`` *or* a path/
href (schema fix in geo-common). These cover the engine side: a local
``.geojson`` path round-trips, and a remote href without the optional ``remote``
backend fails with a clear, actionable validation error rather than a confusing
"does not exist".
"""

from __future__ import annotations

import json

import pytest

from geo_common.errors import ValidationError

from geo_formats.geoparquet import _is_remote, to_geoparquet


def test_is_remote_detects_object_store_and_url_schemes() -> None:
    assert _is_remote("s3://amzn-s3-demo-bucket/x.parquet")
    assert _is_remote("https://example.com/data/x.geojson")
    assert _is_remote("gs://bucket/x.parquet")
    assert not _is_remote("/tmp/local/x.geojson")
    assert not _is_remote("relative/path/x.parquet")


def test_local_geojson_path_is_read_and_converted(tmp_path) -> None:
    src = tmp_path / "in.geojson"
    src.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "a"},
                        "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dst = tmp_path / "out.parquet"
    result = to_geoparquet(str(src), str(dst))
    assert result.fmt == "GeoParquet"
    assert result.detail["feature_count"] == 1
    assert dst.exists()


def test_remote_href_without_backend_raises_actionable_error(tmp_path) -> None:
    """A remote href without the optional 'remote' extra errors clearly."""
    with pytest.raises(ValidationError) as excinfo:
        to_geoparquet(
            "s3://amzn-s3-demo-bucket/features.geojson",
            str(tmp_path / "out.parquet"),
        )
    message = str(excinfo.value).lower()
    assert "remote" in message
    assert excinfo.value.detail.get("parameter") == "src"


from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_coord = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    points=st.lists(
        st.tuples(_coord, _coord, st.text(min_size=0, max_size=8)),
        min_size=1,
        max_size=25,
    )
)
def test_local_path_src_roundtrip_preserves_features(points, tmp_path_factory) -> None:
    """Feature: geospatial-power-pack — a geojson path src round-trips to GeoParquet.

    Randomized feature collections written to a local ``.geojson`` path convert to
    GeoParquet with feature count, point coordinates, and the ``name`` attribute
    preserved (in order).
    """
    import geopandas as gpd

    tmp = tmp_path_factory.mktemp("gp")
    src = tmp / "in.geojson"
    dst = tmp / "out.parquet"
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": name},
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
            for (lon, lat, name) in points
        ],
    }
    src.write_text(json.dumps(fc), encoding="utf-8")

    result = to_geoparquet(str(src), str(dst))
    assert result.detail["feature_count"] == len(points)

    gdf = gpd.read_parquet(dst)
    assert len(gdf) == len(points)
    for (lon, lat, name), (_, row) in zip(points, gdf.iterrows()):
        assert row.geometry.x == pytest.approx(lon)
        assert row.geometry.y == pytest.approx(lat)
        assert row["name"] == name
