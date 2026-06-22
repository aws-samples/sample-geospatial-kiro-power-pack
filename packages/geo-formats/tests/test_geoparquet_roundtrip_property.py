"""Property test for GeoParquet round-trip fidelity (task 14.3).

Feature: geospatial-power-pack, Property 3: GeoParquet round-trip preserves features exactly

This module validates :func:`geo_formats.geoparquet.to_geoparquet` against
Property 3 of the design (Validates: Requirements 8.5, 12.3, 15.4):

*For any* vector dataset, converting it to GeoParquet and reading it back
preserves the **feature count**, the **geometries coordinate-for-coordinate**,
and **every attribute value** exactly.

Testability notes
------------------
The conversion is exercised directly through the public synchronous engine
``to_geoparquet`` (the same code path the ``geo-formats`` ``to_geoparquet`` MCP
tool drives). Each generated example writes a GeoParquet to a fresh temporary
file and reads it back with ``geopandas.read_parquet`` — no mocks — so the
assertions check real on-disk round-trip behavior.

The generators constrain the input space intelligently so the round-trip is
*expected* to be exact:

* **Geometries** use finite IEEE-754 doubles. GeoParquet stores geometries as
  WKB, and WKB encodes doubles bit-for-bit, so every coordinate round-trips
  exactly and ``geom.equals_exact(other, tolerance=0.0)`` must hold.
* **Attributes** use a fixed per-collection schema (each column has one stable
  type across all features) drawn from int / float / str / bool with no nulls.
  This keeps each Parquet column a clean, non-null typed column, so values
  round-trip with exact equality rather than being coerced (e.g. an int column
  gaining ``NaN`` and becoming float).

The ``@settings(deadline=None)`` decorator inherits the loaded profile's
>=100-example minimum (Requirement 15.1) configured in the repo ``conftest.py``.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from shapely.geometry import shape

from geo_formats.geoparquet import to_geoparquet

# ---------------------------------------------------------------------------
# Geometry generators — finite doubles so WKB round-trips coordinates exactly.
# ---------------------------------------------------------------------------
_coordinate = st.floats(
    min_value=-1.0e7,
    max_value=1.0e7,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_position = st.builds(lambda x, y: [x, y], _coordinate, _coordinate)

_point = st.builds(lambda p: {"type": "Point", "coordinates": p}, _position)
_linestring = st.builds(
    lambda ps: {"type": "LineString", "coordinates": ps},
    st.lists(_position, min_size=2, max_size=6),
)
_multipoint = st.builds(
    lambda ps: {"type": "MultiPoint", "coordinates": ps},
    st.lists(_position, min_size=1, max_size=6),
)


@st.composite
def _polygon(draw) -> Dict[str, Any]:
    """A single-ring polygon whose ring is explicitly closed."""
    ring = draw(st.lists(_position, min_size=3, max_size=6))
    closed = ring + [ring[0]]
    return {"type": "Polygon", "coordinates": [closed]}


_geometry = st.one_of(_point, _linestring, _multipoint, _polygon())

# ---------------------------------------------------------------------------
# Attribute generators — one stable type per column, no nulls.
# ---------------------------------------------------------------------------
_column_name = st.text(
    alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    min_size=1,
    max_size=8,
).filter(lambda s: s != "geometry")

_KIND_STRATEGY = {
    "int": st.integers(min_value=-(2**63), max_value=2**63 - 1),
    "float": st.floats(allow_nan=False, allow_infinity=False, width=64),
    "str": st.text(
        alphabet=st.characters(categories=["L", "N", "P", "Zs"]),
        max_size=12,
    ),
    "bool": st.booleans(),
}


@st.composite
def _feature_collections(draw) -> Dict[str, Any]:
    """Generate a GeoJSON FeatureCollection with a fixed attribute schema."""
    names: List[str] = draw(
        st.lists(_column_name, min_size=0, max_size=4, unique=True)
    )
    kinds: Dict[str, str] = {
        name: draw(st.sampled_from(list(_KIND_STRATEGY))) for name in names
    }
    n_features = draw(st.integers(min_value=1, max_value=6))

    features: List[Dict[str, Any]] = []
    for _ in range(n_features):
        geom = draw(_geometry)
        props = {name: draw(_KIND_STRATEGY[kinds[name]]) for name in names}
        features.append(
            {"type": "Feature", "geometry": geom, "properties": props}
        )
    return {"type": "FeatureCollection", "features": features}


def _values_equal(expected: Any, actual: Any) -> bool:
    """Exact attribute equality across the numpy/python boundary.

    Generated values carry no NaN, so plain ``==`` is a faithful exact check;
    ``bool(...)`` collapses any numpy scalar truthiness to a Python bool.
    """
    return bool(expected == actual)


@pytest.mark.property
@settings(deadline=None)  # >=100 examples inherited from the loaded profile
@given(fc=_feature_collections())
def test_geoparquet_roundtrip_preserves_features_exactly(
    fc: Dict[str, Any],
) -> None:
    """Feature: geospatial-power-pack, Property 3: GeoParquet round-trip preserves features exactly.

    Converting any vector dataset to GeoParquet and reading it back preserves
    the feature count, the geometries coordinate-for-coordinate, and every
    attribute value exactly.

    **Validates: Requirements 8.5, 12.3, 15.4**
    """
    import geopandas as gpd

    features = fc["features"]
    attribute_columns = sorted(
        {k for f in features for k in f["properties"].keys()}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dst = os.path.join(tmpdir, "out.parquet")

        result = to_geoparquet(fc, dst)
        assert result.fmt == "GeoParquet"
        assert os.path.exists(dst)

        back = gpd.read_parquet(dst)

    # 1) Feature count is preserved exactly.
    assert len(back) == len(features)
    assert result.detail["feature_count"] == len(features)

    # 2) Geometries are preserved coordinate-for-coordinate (WKB exact).
    expected_geoms = [shape(f["geometry"]) for f in features]
    actual_geoms = list(back.geometry)
    assert len(actual_geoms) == len(expected_geoms)
    for expected_geom, actual_geom in zip(expected_geoms, actual_geoms):
        assert actual_geom.equals_exact(expected_geom, 0.0), (
            f"geometry mismatch: {actual_geom.wkt} != {expected_geom.wkt}"
        )

    # 3) Every attribute value of every feature is preserved exactly.
    for col in attribute_columns:
        assert col in back.columns, f"attribute column {col!r} dropped"
        actual_col = list(back[col])
        for idx, feat in enumerate(features):
            expected_val = feat["properties"][col]
            assert _values_equal(expected_val, actual_col[idx]), (
                f"attribute {col!r} of feature {idx} mismatch: "
                f"{actual_col[idx]!r} != {expected_val!r}"
            )
