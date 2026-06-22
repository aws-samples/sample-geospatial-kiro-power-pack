"""Focused unit tests for ``geo-index`` (task 14.9; Requirements 8.9, 8.10).

These example-based tests pin down the ``index_cell`` contract that the
property test (task 14.10, design Property 15) verifies across the whole input
space:

* **Cell lookup (Req 8.9)** - ``index_cell`` returns a well-formed H3 cell id
  for resolutions 0-15 and a well-formed S2 cell id for levels 0-30, for both
  the pure function and the server tool, including the range endpoints.
* **Determinism (Req 8.9 / Property 15)** - the same input always yields the
  same identifier.
* **Out-of-range validation (Req 8.10)** - an unknown scheme, an out-of-range
  resolution, or a coordinate outside longitude ``[-180, 180]`` / latitude
  ``[-90, 90]`` raises a taxonomy ``ValidationError`` that names the offending
  parameter, and no output is produced.
"""

from __future__ import annotations

import h3
import pytest
import s2sphere

from geo_common.errors import ErrorCategory, ValidationError

from geo_index.indexing import (
    H3_MAX_RESOLUTION,
    S2_MAX_LEVEL,
    index_cell,
    resolution_range,
)
from geo_index.server import GeoIndexServer


# --- H3 lookup (Requirement 8.9) ------------------------------------------

@pytest.mark.parametrize("resolution", [0, 1, 7, 9, 15])
def test_h3_returns_wellformed_cell(resolution):
    """Req 8.9: H3 res 0-15 yields a valid, resolution-correct H3 cell."""
    cell = index_cell(lon=-122.0, lat=37.0, scheme="h3", resolution=resolution)
    assert isinstance(cell, str) and cell
    assert h3.is_valid_cell(cell)
    assert h3.get_resolution(cell) == resolution


def test_h3_matches_underlying_library():
    """The returned H3 id is exactly the library's lat/lng lookup."""
    cell = index_cell(lon=-122.0, lat=37.0, scheme="h3", resolution=9)
    assert cell == h3.latlng_to_cell(37.0, -122.0, 9)


# --- S2 lookup (Requirement 8.9) ------------------------------------------

@pytest.mark.parametrize("level", [0, 1, 12, 20, 30])
def test_s2_returns_wellformed_cell(level):
    """Req 8.9: S2 level 0-30 yields a valid, level-correct S2 cell token."""
    token = index_cell(lon=-122.0, lat=37.0, scheme="s2", resolution=level)
    assert isinstance(token, str) and token
    cell_id = s2sphere.CellId.from_token(token)
    assert cell_id.is_valid()
    assert cell_id.level() == level


def test_scheme_is_case_insensitive():
    """Schemes match case-insensitively (``"H3"``/``"S2"``)."""
    assert index_cell(lon=10.0, lat=20.0, scheme="H3", resolution=5) == index_cell(
        lon=10.0, lat=20.0, scheme="h3", resolution=5
    )
    assert index_cell(lon=10.0, lat=20.0, scheme=" S2 ", resolution=5) == index_cell(
        lon=10.0, lat=20.0, scheme="s2", resolution=5
    )


# --- Determinism (Requirement 8.9 / Property 15) --------------------------

@pytest.mark.parametrize("scheme", ["h3", "s2"])
def test_repeated_lookup_is_deterministic(scheme):
    """Req 8.9: the same input always produces the same identifier."""
    kwargs = dict(lon=-122.4194, lat=37.7749, scheme=scheme, resolution=10)
    first = index_cell(**kwargs)
    assert all(index_cell(**kwargs) == first for _ in range(5))


# --- Coordinate boundary acceptance (Requirement 8.10 boundaries) ---------

@pytest.mark.parametrize("scheme", ["h3", "s2"])
@pytest.mark.parametrize(
    "lon,lat",
    [(-180.0, -90.0), (180.0, 90.0), (-180.0, 90.0), (180.0, -90.0), (0.0, 0.0)],
)
def test_coordinate_bounds_are_inclusive(scheme, lon, lat):
    """Coordinates exactly on the [-180,180]/[-90,90] bounds are accepted."""
    cell = index_cell(lon=lon, lat=lat, scheme=scheme, resolution=4)
    assert isinstance(cell, str) and cell


# --- Out-of-range validation (Requirement 8.10) ---------------------------

@pytest.mark.parametrize(
    "scheme,resolution",
    [("h3", -1), ("h3", 16), ("s2", -1), ("s2", 31)],
)
def test_out_of_range_resolution_raises_validation(scheme, resolution):
    """Req 8.10: resolution outside the scheme range -> ValidationError(resolution)."""
    with pytest.raises(ValidationError) as exc_info:
        index_cell(lon=0.0, lat=0.0, scheme=scheme, resolution=resolution)
    err = exc_info.value
    assert err.category is ErrorCategory.VALIDATION
    assert err.detail and err.detail.get("parameter") == "resolution"


@pytest.mark.parametrize(
    "lon,lat,bad",
    [
        (-180.001, 0.0, "lon"),
        (180.001, 0.0, "lon"),
        (0.0, -90.001, "lat"),
        (0.0, 90.001, "lat"),
        (float("nan"), 0.0, "lon"),
        (0.0, float("inf"), "lat"),
    ],
)
@pytest.mark.parametrize("scheme", ["h3", "s2"])
def test_out_of_range_coordinate_raises_validation(scheme, lon, lat, bad):
    """Req 8.10: out-of-bounds / non-finite coordinate -> ValidationError(coord)."""
    with pytest.raises(ValidationError) as exc_info:
        index_cell(lon=lon, lat=lat, scheme=scheme, resolution=5)
    err = exc_info.value
    assert err.category is ErrorCategory.VALIDATION
    assert err.detail and err.detail.get("parameter") == bad


def test_unknown_scheme_raises_validation():
    """An unknown indexing scheme is rejected as a validation error."""
    with pytest.raises(ValidationError) as exc_info:
        index_cell(lon=0.0, lat=0.0, scheme="geohash", resolution=5)
    err = exc_info.value
    assert err.category is ErrorCategory.VALIDATION
    assert err.detail and err.detail.get("parameter") == "scheme"


def test_resolution_range_helper():
    """``resolution_range`` reports the documented inclusive ranges."""
    assert resolution_range("h3") == (0, H3_MAX_RESOLUTION)
    assert resolution_range("s2") == (0, S2_MAX_LEVEL)


# --- Server wiring (Req 8.9 via the MCP tool) -----------------------------

async def test_server_registers_and_runs_index_cell_tool():
    """The server exposes ``index_cell`` and returns the same id as the fn."""
    server = GeoIndexServer()
    assert "index_cell" in server.tool_names()
    via_tool = await server.index_cell(lon=-122.0, lat=37.0, scheme="h3", resolution=9)
    assert via_tool == index_cell(lon=-122.0, lat=37.0, scheme="h3", resolution=9)


async def test_server_tool_validates_out_of_range():
    """The server tool surfaces out-of-range input as a ValidationError."""
    server = GeoIndexServer()
    with pytest.raises(ValidationError):
        await server.index_cell(lon=200.0, lat=0.0, scheme="s2", resolution=5)


def test_server_has_no_required_credentials():
    """geo-index is local + credential-free, so it declares no credentials."""
    server = GeoIndexServer()
    assert server.required_credentials() == []
    assert server.missing_required_credentials(configured_keys=[]) == []


def test_catalog_entry_registered():
    """geo-index registers its index_cell capability in the catalog (Req 2.1)."""
    server = GeoIndexServer()
    entries = server.catalog_entries()
    assert any(e.name == "index_cell" and e.provider_server == "geo-index" for e in entries)
