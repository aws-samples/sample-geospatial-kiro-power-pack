"""Tests for ``geo-terrain`` ``elevation`` (Requirements 7.5, 7.12).

These are example-based tests driven through an ``httpx`` mock transport so the
terrain-source connectors are exercised without real network I/O. They cover:

* a single-location query returning a scalar elevation in metres (Req 7.5);
* an extent query returning a ``RasterArray`` sampled over a grid, in the
  documented north→south / west→east orientation (Req 7.5);
* source selection by name and rejection of an unknown source as an
  ``Error_Taxonomy`` validation error (Req 7.12);
* malformed-location rejection before any request is made (Req 7.12);
* the sample-grid cap (Req 7.12); and
* taxonomy-classified propagation of an upstream failure (no partial result).
"""

from __future__ import annotations

import httpx
import pytest

from geo_common.errors import ErrorCategory, GeoError, UpstreamError, ValidationError
from geo_common.http import HttpClient
from geo_common.models import CredentialClassification, OpennessTier
from geo_common.retry import RetryPolicy

from geo_terrain.elevation import (
    DEFAULT_SOURCE,
    TerrainSource,
    default_sources,
    parse_location,
)
from geo_terrain.models import Coordinate, GeoWindow, RasterArray
from geo_terrain.server import INSTALL_COMMAND, GeoTerrainServer

API_HOST = "api.opentopodata.org"

# A point in central Berlin and a small extent around it.
BERLIN = (13.405, 52.52)
SMALL_BBOX = (13.40, 52.50, 13.41, 52.51)


async def _no_sleep(_seconds: float) -> None:
    """An async sleep that returns immediately (keeps retry tests fast)."""
    return None


def _results_handler(elevations, *, calls=None, status=200):
    """Mock handler returning one OpenTopoData-style result per requested point.

    ``elevations`` is either a fixed list (returned as-is) or a callable mapping
    the parsed ``locations`` list to a list of elevations.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        raw = request.url.params.get("locations", "")
        points = [p for p in raw.split("|") if p]
        if callable(elevations):
            elevs = elevations(points)
        else:
            elevs = elevations
        results = [{"elevation": e, "location": {}} for e in elevs]
        return httpx.Response(status, json={"status": "OK", "results": results})

    return handler


def _server(handler, *, sources=None, default_source="srtm") -> GeoTerrainServer:
    """A GeoTerrainServer wired to a mock transport (no real network)."""
    client = HttpClient(
        RetryPolicy(max_attempts=1),
        transport=httpx.MockTransport(handler),
        sleep=_no_sleep,
    )
    kwargs = {"http": client, "default_source": default_source}
    if sources is not None:
        kwargs["sources"] = sources
    return GeoTerrainServer(**kwargs)


# --- Requirement 7.5: single-location elevation -> scalar -------------------


async def test_point_query_returns_scalar_elevation():
    server = _server(_results_handler([34.5]))
    try:
        value = await server.elevation(location=BERLIN)
    finally:
        await server.aclose()
    assert isinstance(value, float)
    assert value == 34.5


async def test_point_query_accepts_coordinate_and_mapping_forms():
    server = _server(_results_handler([12.0]))
    try:
        from_coord = await server.elevation(location=Coordinate(lon=13.4, lat=52.5))
        from_mapping = await server.elevation(location={"lon": 13.4, "lat": 52.5})
    finally:
        await server.aclose()
    assert from_coord == 12.0
    assert from_mapping == 12.0


async def test_point_query_sends_lat_lng_locations():
    calls: list[str] = []
    server = _server(_results_handler([5.0], calls=calls))
    try:
        await server.elevation(location=(13.405, 52.52))
    finally:
        await server.aclose()
    # OpenTopoData order is lat,lng; the dataset path is the SRTM default.
    assert any("locations=52.52%2C13.405" in c or "locations=52.52,13.405" in c for c in calls)
    assert any("/v1/srtm30m" in c for c in calls)


# --- Requirement 7.5: extent elevation -> RasterArray -----------------------


async def test_window_query_returns_raster_array_of_correct_shape():
    # Return an increasing elevation per sample so we can check ordering.
    server = _server(_results_handler(lambda pts: list(range(len(pts)))))
    try:
        result = await server.elevation(
            location={"bbox": SMALL_BBOX, "width": 3, "height": 2}
        )
    finally:
        await server.aclose()

    assert isinstance(result, RasterArray)
    assert result.width == 3
    assert result.height == 2
    assert result.bbox == SMALL_BBOX
    assert result.source == "srtm"
    assert result.units == "m"
    # 2 rows × 3 cols, row-major in the order the samples were requested.
    assert result.values == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


async def test_window_grid_points_run_north_to_south_west_to_east():
    window = GeoWindow(bbox=(0.0, 0.0, 2.0, 2.0), width=3, height=3)
    from geo_terrain.elevation import _grid_points

    points = _grid_points(window)
    # First sample is the north-west corner (max_lat, min_lon).
    assert points[0] == (0.0, 2.0)
    # Last sample is the south-east corner (min_lat, max_lon).
    assert points[-1] == (2.0, 0.0)
    assert len(points) == 9


async def test_window_nodata_sample_is_none():
    server = _server(_results_handler([10.0, None, 12.0, 13.0]))
    try:
        result = await server.elevation(
            location={"bbox": SMALL_BBOX, "width": 2, "height": 2}
        )
    finally:
        await server.aclose()
    assert result.values == [[10.0, None], [12.0, 13.0]]


async def test_large_window_is_split_into_location_batches():
    """Regression: OpenTopoData caps a request at 100 locations (a larger one is
    HTTP 400). A window with >100 samples must be split into batches of <=100
    and concatenated, so windowed elevation/slope/hillshade work at any size."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raw = request.url.params.get("locations", "")
        points = [p for p in raw.split("|") if p]
        # Each batch must respect the 100-location cap.
        assert len(points) <= 100, "a batch exceeded the 100-location cap"
        calls.append(raw)
        results = [{"elevation": 1.0, "location": {}} for _ in points]
        return httpx.Response(200, json={"status": "OK", "results": results})

    # 12 x 12 = 144 samples -> must be 2 requests (100 + 44).
    server = _server(handler)
    try:
        result = await server.elevation(
            location={"bbox": SMALL_BBOX, "width": 12, "height": 12}
        )
    finally:
        await server.aclose()

    assert isinstance(result, RasterArray)
    assert result.width == 12 and result.height == 12
    # All 144 samples resolved across exactly 2 batched requests.
    assert sum(len(c.split("|")) for c in calls) == 144
    assert len(calls) == 2


def test_terrain_source_respects_location_request_cap():
    """The per-request location cap defaults to 100 and is overridable."""
    from geo_terrain.elevation import DEFAULT_MAX_LOCATIONS_PER_REQUEST, TerrainSource

    assert DEFAULT_MAX_LOCATIONS_PER_REQUEST == 100
    assert TerrainSource("srtm", "srtm30m").max_locations_per_request == 100
    assert (
        TerrainSource("x", "y", max_locations_per_request=50).max_locations_per_request
        == 50
    )


# --- Requirement 7.5: source selection --------------------------------------


async def test_source_selection_uses_named_dataset():
    calls: list[str] = []
    server = _server(_results_handler([1.0], calls=calls))
    try:
        await server.elevation(location=BERLIN, source="srtm")
    finally:
        await server.aclose()
    assert any("/v1/srtm30m" in c for c in calls)


async def test_unknown_source_is_rejected_before_any_request():
    calls: list[str] = []
    server = _server(_results_handler([1.0], calls=calls))
    try:
        with pytest.raises(ValidationError) as exc:
            await server.elevation(location=BERLIN, source="not-a-source")
    finally:
        await server.aclose()
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail["parameter"] == "source"
    # No source was queried: the guard runs before any HTTP call.
    assert calls == []


# --- Requirement 7.12: malformed location ----------------------------------


@pytest.mark.parametrize(
    "bad_location",
    [
        (13.4,),  # too few values
        (13.4, 52.5, 1.0),  # 3 values: neither a point nor an extent
        (200.0, 52.5),  # longitude out of range
        (13.4, 95.0),  # latitude out of range
        ("13.4", 52.5),  # non-numeric ordinate
        "13.4,52.5",  # a string, not a sequence of numbers
        {"lat": 52.5},  # mapping missing lon
        {"bbox": (13.41, 52.50, 13.40, 52.51)},  # inverted longitude extent
        {"bbox": (13.40, 52.50, 13.41)},  # extent with too few ordinates
    ],
)
def test_malformed_location_raises_validation_error(bad_location):
    with pytest.raises(ValidationError) as exc:
        parse_location(bad_location)
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail["parameter"] == "location"


def test_valid_point_and_extent_parse():
    assert parse_location((13, 52)) == Coordinate(lon=13.0, lat=52.0)
    window = parse_location([13.0, 52.0, 14.0, 53.0])
    assert isinstance(window, GeoWindow)
    assert window.bbox == (13.0, 52.0, 14.0, 53.0)


async def test_oversized_sample_grid_is_rejected():
    server = _server(_results_handler([1.0]))
    try:
        with pytest.raises(ValidationError) as exc:
            await server.elevation(
                location={"bbox": SMALL_BBOX, "width": 1000, "height": 1000}
            )
    finally:
        await server.aclose()
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail["parameter"] == "location"


# --- upstream failure propagates (no partial result) ------------------------


async def test_upstream_5xx_propagates_as_taxonomy_error():
    # A 5xx is retried/raised by the shared HttpClient as an UpstreamError
    # (logical-source re-tagging is task 13.6); here we assert the category and
    # that no partial result is produced.
    server = _server(_results_handler([1.0], status=500))
    try:
        with pytest.raises(UpstreamError) as exc:
            await server.elevation(location=BERLIN)
    finally:
        await server.aclose()
    assert exc.value.category is ErrorCategory.UPSTREAM


async def test_non_retryable_4xx_names_logical_source():
    # A 400 is returned (not retried) by the HttpClient; the connector maps it
    # to an UpstreamError naming the logical terrain source.
    server = _server(_results_handler([1.0], status=400))
    try:
        with pytest.raises(UpstreamError) as exc:
            await server.elevation(location=BERLIN)
    finally:
        await server.aclose()
    assert exc.value.category is ErrorCategory.UPSTREAM
    assert exc.value.source == "srtm"


async def test_point_with_no_coverage_raises_upstream_error():
    server = _server(_results_handler([None]))
    try:
        with pytest.raises(UpstreamError) as exc:
            await server.elevation(location=BERLIN)
    finally:
        await server.aclose()
    assert exc.value.category is ErrorCategory.UPSTREAM


async def test_unreachable_source_yields_network_error():
    def timeout_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connection timed out")

    server = _server(timeout_handler)
    try:
        with pytest.raises(GeoError) as exc:
            await server.elevation(location=BERLIN)
    finally:
        await server.aclose()
    assert exc.value.category is ErrorCategory.NETWORK


# --- server scaffold + catalog + credentials --------------------------------


def test_server_registers_elevation_tool():
    server = GeoTerrainServer()
    assert server.server_name == "geo-terrain"
    assert server.pillar == "A"
    assert "elevation" in server.tool_names()
    assert "slope" in server.tool_names()
    assert "hillshade" in server.tool_names()
    assert set(server.sources) == {"srtm", "3dep"}


async def test_slope_over_extent_returns_degrees_grid():
    # Elevation rises with sample index (a ramp), so slope is well-defined.
    server = _server(_results_handler(lambda pts: [float(i) for i in range(len(pts))]))
    try:
        result = await server.slope(
            location={"bbox": SMALL_BBOX, "width": 3, "height": 3}
        )
    finally:
        await server.aclose()
    assert isinstance(result, RasterArray)
    assert result.width == 3 and result.height == 3
    assert result.units == "degrees"
    # Every cell is a finite, non-negative slope in degrees.
    assert all(0.0 <= v <= 90.0 for row in result.values for v in row if v is not None)


async def test_hillshade_over_extent_returns_0_255_grid():
    server = _server(_results_handler(lambda pts: [float(i) for i in range(len(pts))]))
    try:
        result = await server.hillshade(
            location={"bbox": SMALL_BBOX, "width": 3, "height": 3}
        )
    finally:
        await server.aclose()
    assert result.units == "hillshade"
    assert all(0 <= v <= 255 for row in result.values for v in row if v is not None)


async def test_slope_rejects_point_location():
    server = _server(_results_handler([10.0]))
    try:
        with pytest.raises(ValidationError):
            await server.slope(location={"lon": 13.4, "lat": 52.5})
    finally:
        await server.aclose()


async def test_hillshade_rejects_out_of_range_azimuth():
    server = _server(_results_handler(lambda pts: [float(i) for i in range(len(pts))]))
    try:
        with pytest.raises(ValidationError):
            await server.hillshade(
                location={"bbox": SMALL_BBOX, "width": 3, "height": 3}, azimuth=999.0
            )
    finally:
        await server.aclose()


def test_catalog_entry_names_geo_terrain_as_open_provider():
    server = GeoTerrainServer()
    entries = server.catalog_entries()
    assert len(entries) == 3
    entry = next(e for e in entries if e.name == "elevation")
    assert entry.name == "elevation"
    assert entry.pillar == "A"
    assert entry.openness_tier is OpennessTier.OPEN
    assert entry.provider_server == "geo-terrain"
    assert entry.install_command == INSTALL_COMMAND
    assert 1 <= len(entry.capability_description) <= 500


def test_optional_credential_declared_and_starts():
    server = GeoTerrainServer()
    specs = server.required_credentials()
    keys = {spec.mcp_json_key for spec in specs}
    assert keys == {"OPENTOPOGRAPHY_API_KEY"}
    assert all(
        spec.classification is CredentialClassification.OPTIONAL for spec in specs
    )
    # No Required credential -> startup is never blocked (Req 16.5).
    server.start(configured_keys=[])
    assert server.started is True


async def test_elevation_invoked_through_tool_registry():
    server = _server(_results_handler([7.0]))
    try:
        tool = server.get_tool("elevation")
        value = await tool.func(location=BERLIN)
    finally:
        await server.aclose()
    assert value == 7.0


def test_default_source_is_served_by_public_endpoint() -> None:
    """Regression: the default elevation source must work on the public endpoint.

    The default ``source`` must map to an OpenTopoData dataset the public
    ``api.opentopodata.org`` actually hosts, so a credential-free ``elevation``
    call works out of the box. ``geo-terrain`` ships ``srtm`` (default) and
    ``3dep``, whose datasets (``srtm30m`` / ``ned10m``) the public endpoint
    serves.
    """
    assert DEFAULT_SOURCE in default_sources()
    # The default's dataset is hosted by the default public OpenTopoData endpoint.
    assert default_sources()[DEFAULT_SOURCE].dataset in {"srtm30m", "ned10m"}
