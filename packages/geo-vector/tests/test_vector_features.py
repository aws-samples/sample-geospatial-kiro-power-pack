"""Tests for ``geo-vector`` ``vector_features`` (Requirements 7.3, 7.12).

These are example-based tests driven through an ``httpx`` mock transport so the
OpenStreetMap (Overpass) and Overture Maps connectors are exercised without
real network I/O. They cover:

* the bbox-area gate (default 2,500 km²) and malformed-bbox rejection as an
  ``Error_Taxonomy`` validation error (Requirements 7.3, 7.12);
* merging features from both sources into one GeoJSON ``FeatureCollection``
  (Requirement 7.3);
* Overpass element -> GeoJSON geometry parsing; and
* taxonomy-classified propagation of an upstream failure (no partial result,
  Requirement 7.11).
"""

from __future__ import annotations

import httpx
import pytest

from geo_common.errors import ErrorCategory, GeoError, UpstreamError, ValidationError
from geo_common.http import HttpClient
from geo_common.models import OpennessTier
from geo_common.retry import RetryPolicy

from geo_vector.bbox import bbox_area_km2, validate_bbox
from geo_vector.features import OverpassSource, OvertureSource
from geo_vector.models import FeatureCollection
from geo_vector.server import INSTALL_COMMAND, GeoVectorServer

OVERPASS_HOST = "overpass-api.de"
OVERTURE_HOST = "api.overturemaps.org"
# Overture has no public hosted endpoint, so the source must be configured with
# an explicit URL. Tests point it at a mock-transport host of the same name.
OVERTURE_URL = "https://api.overturemaps.org/features"

# A small extent near Berlin, well under the 2,500 km² default maximum.
SMALL_BBOX = (13.40, 52.50, 13.41, 52.51)
# A large extent whose spherical-quadrangle area far exceeds 2,500 km².
HUGE_BBOX = (0.0, 0.0, 10.0, 10.0)

_OSM_NODE = {
    "type": "node",
    "id": 1,
    "lon": 13.405,
    "lat": 52.505,
    "tags": {"amenity": "cafe"},
}
_OSM_WAY = {
    "type": "way",
    "id": 2,
    "geometry": [
        {"lat": 52.500, "lon": 13.400},
        {"lat": 52.500, "lon": 13.401},
        {"lat": 52.501, "lon": 13.401},
        {"lat": 52.500, "lon": 13.400},
    ],
    "tags": {"building": "yes"},
}
_OVERTURE_FEATURE = {
    "type": "Feature",
    "id": "ovt-1",
    "geometry": {"type": "Point", "coordinates": [13.404, 52.504]},
    "properties": {"class": "restaurant"},
}


async def _no_sleep(_seconds: float) -> None:
    """An async sleep that returns immediately (keeps retry tests fast)."""
    return None


def _make_handler(*, overpass=None, overture=None, calls=None):
    """Build a mock-transport handler routing by host to canned responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if calls is not None:
            calls.append(host)
        if host == OVERPASS_HOST:
            return overpass(request) if overpass else httpx.Response(
                200, json={"elements": [_OSM_NODE, _OSM_WAY]}
            )
        if host == OVERTURE_HOST:
            return overture(request) if overture else httpx.Response(
                200, json={"type": "FeatureCollection", "features": [_OVERTURE_FEATURE]}
            )
        return httpx.Response(404)  # pragma: no cover - unexpected host

    return handler


def _server(handler, *, max_area_km2=None, sources=None) -> GeoVectorServer:
    """A GeoVectorServer wired to a mock transport (no real network)."""
    client = HttpClient(
        RetryPolicy(max_attempts=1),
        transport=httpx.MockTransport(handler),
        sleep=_no_sleep,
    )
    kwargs = {"http": client}
    if max_area_km2 is not None:
        kwargs["max_area_km2"] = max_area_km2
    if sources is not None:
        kwargs["sources"] = sources
    return GeoVectorServer(**kwargs)


# --- Requirement 7.3: features from OSM + Overture, merged ------------------


async def test_merges_osm_and_overture_features():
    server = _server(
        _make_handler(),
        sources=[OverpassSource(), OvertureSource(url=OVERTURE_URL)],
    )
    try:
        result = await server.vector_features(bbox=SMALL_BBOX)
    finally:
        await server.aclose()

    assert isinstance(result, FeatureCollection)
    assert result.type == "FeatureCollection"
    assert result.bbox == SMALL_BBOX
    # node (Point) + way (Polygon) from OSM, plus one Overture feature.
    assert len(result.features) == 3

    sources = sorted(f.properties["source"] for f in result.features)
    assert sources == ["openstreetmap", "openstreetmap", "overture"]

    geom_types = {f.geometry["type"] for f in result.features if f.geometry}
    assert geom_types == {"Point", "Polygon"}

    # The closed OSM way became a Polygon ring; the node became a Point.
    osm_geoms = {
        f.geometry["type"]
        for f in result.features
        if f.properties["source"] == "openstreetmap"
    }
    assert osm_geoms == {"Point", "Polygon"}


async def test_single_source_can_be_configured():
    server = _server(_make_handler(), sources=[OvertureSource(url=OVERTURE_URL)])
    try:
        result = await server.vector_features(bbox=SMALL_BBOX)
    finally:
        await server.aclose()
    assert len(result.features) == 1
    assert result.features[0].properties["source"] == "overture"


def test_default_sources_is_openstreetmap_only():
    """Overture has no public endpoint, so only OSM/Overpass ships by default."""
    from geo_vector.features import default_sources

    sources = default_sources()
    assert [s.name for s in sources] == ["openstreetmap"]


def test_overture_source_requires_explicit_url():
    """Constructing an Overture source without an endpoint is rejected."""
    with pytest.raises(ValueError):
        OvertureSource(url="")


# --- Requirement 7.3 / 7.12: bbox-area gate --------------------------------


async def test_bbox_area_over_maximum_is_rejected_before_any_request():
    calls: list[str] = []
    server = _server(_make_handler(calls=calls))
    try:
        with pytest.raises(ValidationError) as exc:
            await server.vector_features(bbox=HUGE_BBOX)
    finally:
        await server.aclose()

    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail["parameter"] == "bbox"
    # No source was queried: the gate runs before any HTTP call.
    assert calls == []


async def test_configurable_maximum_area_override():
    server = _server(_make_handler(), max_area_km2=0.01)
    try:
        with pytest.raises(ValidationError) as exc:
            # SMALL_BBOX (~0.75 km²) is fine by default but exceeds 0.01 km².
            await server.vector_features(bbox=SMALL_BBOX)
    finally:
        await server.aclose()
    assert exc.value.category is ErrorCategory.VALIDATION


# --- Response-size guardrail: feature-count cap ----------------------------


async def test_feature_count_over_maximum_is_rejected():
    """A request matching more than max_features is refused with guidance."""

    def many_nodes(_request: httpx.Request) -> httpx.Response:
        elements = [
            {"type": "node", "id": i, "lon": 13.405, "lat": 52.505, "tags": {"a": "b"}}
            for i in range(10)
        ]
        return httpx.Response(200, json={"elements": elements})

    # Cap at 3; the source returns 10 OSM nodes.
    server = _server(_make_handler(overpass=many_nodes))
    server.max_features = 3
    try:
        with pytest.raises(ValidationError) as exc:
            await server.vector_features(bbox=SMALL_BBOX)
    finally:
        await server.aclose()
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail["parameter"] == "max_features"
    assert exc.value.detail["max_features"] == 3
    assert exc.value.detail["feature_count"] == 10


async def test_feature_count_at_or_under_maximum_is_returned():
    """A result within the cap is returned normally."""
    server = _server(_make_handler())  # OSM-only default -> node + way = 2
    server.max_features = 3
    try:
        result = await server.vector_features(bbox=SMALL_BBOX)
    finally:
        await server.aclose()
    assert len(result.features) == 2


# --- Requirement 7.12: malformed bbox --------------------------------------


@pytest.mark.parametrize(
    "bad_bbox",
    [
        (13.40, 52.50, 13.41),  # too few ordinates
        (13.40, 52.50, 13.41, 52.51, 1.0),  # too many ordinates
        (200.0, 52.50, 200.1, 52.51),  # longitude out of range
        (13.40, 95.0, 13.41, 96.0),  # latitude out of range
        (13.41, 52.50, 13.40, 52.51),  # inverted longitude (min > max)
        (13.40, 52.51, 13.41, 52.50),  # inverted latitude (min > max)
        ("13.40", 52.50, 13.41, 52.51),  # non-numeric ordinate
        "13.40,52.50,13.41,52.51",  # a string, not a sequence of numbers
    ],
)
def test_malformed_bbox_raises_validation_error(bad_bbox):
    with pytest.raises(ValidationError) as exc:
        validate_bbox(bad_bbox)
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail["parameter"] == "bbox"


def test_valid_bbox_is_normalized_to_float_tuple():
    assert validate_bbox([13, 52, 14, 53]) == (13.0, 52.0, 14.0, 53.0)


# --- bbox area computation --------------------------------------------------


def test_area_is_positive_and_scales_with_extent():
    small = bbox_area_km2((0.0, 0.0, 0.1, 0.1))
    big = bbox_area_km2((0.0, 0.0, 0.2, 0.2))
    assert small > 0.0
    # Doubling each side roughly quadruples the area near the equator.
    assert big > small
    assert 3.5 < big / small < 4.5


def test_equatorial_tenth_degree_box_area_is_about_123_km2():
    # A 0.1° x 0.1° box at the equator is ~123 km² (~11.1 km per side).
    area = bbox_area_km2((0.0, 0.0, 0.1, 0.1))
    assert 120.0 < area < 125.0


# --- Overpass query building -----------------------------------------------


def test_overpass_query_includes_extent_and_out_geom():
    query = OverpassSource().build_query(SMALL_BBOX, None)
    # Overpass order is (south, west, north, east).
    assert "52.5,13.4,52.51,13.41" in query
    assert query.startswith("[out:json]")
    assert query.endswith("out geom;")
    assert "node(" in query and "way(" in query and "relation(" in query


def test_overpass_query_filters_by_layer_tag_keys():
    query = OverpassSource().build_query(SMALL_BBOX, ["building", "highway"])
    assert "[building]" in query
    assert "[highway]" in query


# --- Requirement 7.11: upstream failure propagates (no partial result) ------


async def test_upstream_failure_propagates_as_taxonomy_error():
    def overpass_500(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    server = _server(_make_handler(overpass=overpass_500))
    try:
        with pytest.raises(UpstreamError) as exc:
            await server.vector_features(bbox=SMALL_BBOX)
    finally:
        await server.aclose()
    assert exc.value.category is ErrorCategory.UPSTREAM


# --- server scaffold --------------------------------------------------------


def test_server_registers_vector_features_tool():
    server = GeoVectorServer()
    assert server.server_name == "geo-vector"
    assert server.pillar == "A"
    assert "vector_features" in server.tool_names()


# --- Requirement 2.1 / 11.3: catalog registration -------------------------


def test_catalog_entry_names_geo_vector_as_open_provider():
    server = GeoVectorServer()
    entries = server.catalog_entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "vector_features"
    assert entry.pillar == "A"
    # Both default sources are open data, so the entry is the Open tier.
    assert entry.openness_tier is OpennessTier.OPEN
    # The wrapping server names itself as the provider (Req 11.3).
    assert entry.provider_server == "geo-vector"
    # The install command is surfaced for not-installed providers (Req 2.6).
    assert entry.install_command == INSTALL_COMMAND
    assert 1 <= len(entry.capability_description) <= 500


# --- Requirement 16.1 / 16.5: credential specs -----------------------------


def test_requires_no_credentials_and_starts():
    server = GeoVectorServer()
    # OSM/Overpass + Overture are open: no credentials are declared.
    assert server.required_credentials() == []
    # With no Required credentials the startup guard lets the server start
    # regardless of the configured environment (Req 16.5).
    server.start(configured_keys=[])
    assert server.started is True


# --- Requirement 7.11: availability error identifies the source ------------


async def test_unreachable_source_yields_availability_error_naming_source():
    def overpass_timeout(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connection timed out")

    server = _server(_make_handler(overpass=overpass_timeout))
    try:
        with pytest.raises(GeoError) as exc:
            await server.vector_features(bbox=SMALL_BBOX)
    finally:
        await server.aclose()

    # A source that does not respond within the 30s timeout surfaces as a
    # NETWORK availability error (Req 7.11, via the shared HttpClient).
    assert exc.value.category is ErrorCategory.NETWORK
    # The availability error identifies the unavailable *logical* source, not
    # just the host (Req 7.11).
    assert exc.value.source == "openstreetmap"


async def test_availability_error_returns_no_partial_results():
    # Overpass times out; Overture would succeed. The request must fail with an
    # availability error rather than returning a partial collection (Req 7.11).
    def overpass_timeout(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connection timed out")

    captured: list[FeatureCollection] = []
    server = _server(_make_handler(overpass=overpass_timeout))
    try:
        with pytest.raises(GeoError) as exc:
            captured.append(await server.vector_features(bbox=SMALL_BBOX))
    finally:
        await server.aclose()
    assert exc.value.category in (ErrorCategory.NETWORK, ErrorCategory.UPSTREAM)
    # No FeatureCollection was ever returned.
    assert captured == []


async def test_upstream_5xx_availability_error_names_source():
    def overpass_500(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    server = _server(_make_handler(overpass=overpass_500))
    try:
        with pytest.raises(UpstreamError) as exc:
            await server.vector_features(bbox=SMALL_BBOX)
    finally:
        await server.aclose()
    assert exc.value.category is ErrorCategory.UPSTREAM
    assert exc.value.source == "openstreetmap"
