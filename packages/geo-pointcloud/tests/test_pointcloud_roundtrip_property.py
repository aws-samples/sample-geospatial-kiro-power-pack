"""Property test for point-cloud format round-trip (design Property 20).

Feature: geospatial-power-pack, Property 20: Format round-trip for point clouds

Validates: Requirements 8.11

*For any* point-cloud dataset, writing it to Cloud-Optimized Point Cloud (COPC)
format with ``write_pointcloud`` and reading it back with ``read_pointcloud``
preserves the point set.

Input space
-----------
A point-cloud dataset is a :class:`~geo_pointcloud.models.PointCloudChunk`: a
list of :class:`~geo_pointcloud.models.PointRecord` plus a CRS. Each point
carries the required ``x``/``y``/``z`` position and the optional LAS/LAZ
dimensions (intensity, classification, return numbering, RGB color, GPS time),
any of which may be ``None``. Coordinates are finite floats and the attribute
dimensions are finite ints, so every generated point is exactly serializable
(Python's JSON round-trips ``float``/``int`` losslessly) and the read-back set
is expected to equal the written set exactly.

The chunk is order-tolerant: COPC organizes points into a spatial octree, so
the preserved invariant is the *multiset* of points (count and content), not
their input sequence. The test compares multisets so duplicate points are
accounted for. Empty clouds (no points) are included as a degenerate case.
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from typing import Tuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_pointcloud import (
    PointCloudChunk,
    PointRecord,
    read_pointcloud,
    write_pointcloud,
)

# Finite, well-behaved coordinate/attribute ranges. Any finite value round-trips
# through JSON losslessly; bounded ranges simply keep generated clouds compact.
_coords = st.floats(
    min_value=-1.0e7, max_value=1.0e7, allow_nan=False, allow_infinity=False
)
_optional_int = st.one_of(st.none(), st.integers(min_value=0, max_value=65535))
_optional_float = st.one_of(
    st.none(),
    st.floats(min_value=0.0, max_value=1.0e9, allow_nan=False, allow_infinity=False),
)


@st.composite
def _point_records(draw) -> PointRecord:
    return PointRecord(
        x=draw(_coords),
        y=draw(_coords),
        z=draw(_coords),
        intensity=draw(_optional_int),
        classification=draw(_optional_int),
        return_number=draw(_optional_int),
        number_of_returns=draw(_optional_int),
        red=draw(_optional_int),
        green=draw(_optional_int),
        blue=draw(_optional_int),
        gps_time=draw(_optional_float),
    )


_crs = st.sampled_from(["EPSG:4326", "EPSG:32631", "EPSG:3857", "EPSG:25832"])


def _chunks() -> st.SearchStrategy[PointCloudChunk]:
    # Include the empty cloud (min_size=0) as a degenerate dataset.
    return st.builds(
        lambda points, crs: PointCloudChunk(points=points, crs=crs),
        points=st.lists(_point_records(), min_size=0, max_size=40),
        crs=_crs,
    )


def _multiset(chunk: PointCloudChunk) -> "Counter[Tuple]":
    """Order-independent multiset of points (count + every per-point field)."""
    return Counter(tuple(sorted(p.model_dump().items())) for p in chunk.points)


@pytest.mark.property
@settings(deadline=None)  # max_examples (>=100) inherited from the loaded profile
@given(chunk=_chunks())
def test_pointcloud_roundtrip_preserves_point_set(chunk: PointCloudChunk) -> None:
    """Feature: geospatial-power-pack, Property 20: Format round-trip for point clouds.

    Validates: Requirements 8.11
    """
    # A fresh path per example; tempfile avoids the function-scoped-fixture
    # reuse that @given would otherwise share across all generated cases.
    fd, path = tempfile.mkstemp(suffix=".copc")
    os.close(fd)
    try:
        result = write_pointcloud(points=chunk, dst_href=path)
        # write_pointcloud reports exactly the number of points it persisted.
        assert result.point_count == chunk.point_count

        back = read_pointcloud(copc_href=path)

        # The point set (multiset, order-independent) is preserved exactly.
        assert back.point_count == chunk.point_count
        assert _multiset(back) == _multiset(chunk)
    finally:
        if os.path.exists(path):
            os.remove(path)
