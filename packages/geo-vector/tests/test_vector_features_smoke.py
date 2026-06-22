"""Smoke / integration tests for ``geo-vector`` ``vector_features`` (Req 7.3).

Where ``test_vector_features.py`` exercises individual behaviors (the bbox-area
gate, query building, error taxonomy) in isolation, these are higher-level
*smoke* tests: each one drives a representative end-to-end feature fetch through
a fully wired :class:`GeoVectorServer` whose shared :class:`HttpClient` is
backed by an ``httpx.MockTransport``. The transport returns realistic
OpenStreetMap (Overpass ``out:json``) and Overture Maps (GeoJSON) payloads, so
the tests confirm the whole path - tool dispatch -> bbox validation -> both
source connectors -> parse -> merge - produces one attributable GeoJSON
``FeatureCollection`` from OSM + Overture (Requirement 7.3).

No real network I/O happens; the mock transport stands in for both upstream
hosts and records the requests it sees so the tests can assert that a real
request was issued to each source.
"""

from __future__ import annotations

import json

import httpx

from geo_common.http import HttpClient
from geo_common.retry import RetryPolicy

from geo_vector.features import OverpassSource, OvertureSource
from geo_vector.models import FeatureCollection
from geo_vector.server import GeoVectorServer

OVERPASS_HOST = "overpass-api.de"
OVERTURE_HOST = "api.overturemaps.org"
# Overture ships disabled by default (no public hosted endpoint), so the smoke
# tests configure it explicitly with a mock-transport URL to exercise the
# two-source merge path end-to-end.
OVERTURE_URL = "https://api.overturemaps.org/features"

# A representative city-centre extent (a few blocks of central Berlin), well
# under the 2,500 km² default maximum so the request reaches both sources.
CITY_BBOX = (13.40, 52.50, 13.42, 52.52)

# A realistic Overpass ``out:json`` payload mixing a tagged amenity node and a
# closed building way (which the connector turns into a Polygon).
_OVERPASS_PAYLOAD = {
    "version": 0.6,
    "generator": "Overpass API",
    "elements": [
        {
            "type": "node",
            "id": 1001,
            "lon": 13.405,
            "lat": 52.515,
            "tags": {"amenity": "cafe", "name": "Cafe Kiro"},
        },
        {
            "type": "node",
            "id": 1002,
            "lon": 13.410,
            "lat": 52.510,
            "tags": {"highway": "bus_stop", "name": "Alexanderplatz"},
        },
        {
            "type": "way",
            "id": 2001,
            "geometry": [
                {"lat": 52.500, "lon": 13.400},
                {"lat": 52.500, "lon": 13.402},
                {"lat": 52.502, "lon": 13.402},
                {"lat": 52.502, "lon": 13.400},
                {"lat": 52.500, "lon": 13.400},
            ],
            "tags": {"building": "residential", "name": "Block A"},
        },
        {
            "type": "way",
            "id": 2002,
            "geometry": [
                {"lat": 52.510, "lon": 13.410},
                {"lat": 52.511, "lon": 13.412},
                {"lat": 52.512, "lon": 13.414},
            ],
            "tags": {"highway": "residential", "name": "Kiro Strasse"},
        },
    ],
}

# A realistic Overture GeoJSON FeatureCollection payload.
_OVERTURE_PAYLOAD = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "id": "overture-place-1",
            "geometry": {"type": "Point", "coordinates": [13.404, 52.504]},
            "properties": {"class": "restaurant", "name": "Overture Bistro"},
        },
        {
            "type": "Feature",
            "id": "overture-building-1",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [13.415, 52.515],
                        [13.416, 52.515],
                        [13.416, 52.516],
                        [13.415, 52.516],
                        [13.415, 52.515],
                    ]
                ],
            },
            "properties": {"class": "commercial"},
        },
    ],
}


def _record_handler(requests):
    """Mock-transport handler returning canned OSM + Overture payloads.

    Appends every received :class:`httpx.Request` to ``requests`` so a test can
    assert which hosts were actually queried end-to-end.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        host = request.url.host
        if host == OVERPASS_HOST:
            return httpx.Response(200, json=_OVERPASS_PAYLOAD)
        if host == OVERTURE_HOST:
            return httpx.Response(200, json=_OVERTURE_PAYLOAD)
        return httpx.Response(404)  # pragma: no cover - unexpected host

    return handler


def _smoke_server(requests):
    """A fully wired ``GeoVectorServer`` on a recording mock transport.

    Both sources are configured explicitly (OpenStreetMap/Overpass plus an
    Overture source pointed at the mock-transport URL) so the smoke tests cover
    the two-source merge path; the default server ships OSM-only.
    """
    client = HttpClient(
        RetryPolicy(max_attempts=1),
        transport=httpx.MockTransport(_record_handler(requests)),
    )
    return GeoVectorServer(
        http=client,
        sources=[OverpassSource(), OvertureSource(url=OVERTURE_URL)],
    )


# --- Requirement 7.3: end-to-end merged fetch from both sources -------------


async def test_end_to_end_fetch_merges_osm_and_overture():
    """A representative fetch returns one merged, attributable collection."""
    requests: list[httpx.Request] = []
    server = _smoke_server(requests)
    try:
        result = await server.vector_features(bbox=CITY_BBOX)
    finally:
        await server.aclose()

    assert isinstance(result, FeatureCollection)
    assert result.type == "FeatureCollection"
    # The collection echoes the validated request extent (Req 7.3).
    assert result.bbox == CITY_BBOX

    # 2 OSM nodes (Point) + 2 OSM ways (Polygon + LineString) + 2 Overture
    # features (Point + Polygon) = 6 merged features.
    assert len(result.features) == 6

    # Every feature stays attributable to its originating dataset.
    by_source: dict[str, int] = {}
    for feature in result.features:
        by_source[feature.properties["source"]] = (
            by_source.get(feature.properties["source"], 0) + 1
        )
    assert by_source == {"openstreetmap": 4, "overture": 2}

    # Both upstream sources were actually queried end-to-end.
    queried_hosts = sorted({r.url.host for r in requests})
    assert queried_hosts == [OVERTURE_HOST, OVERPASS_HOST] or queried_hosts == [
        OVERPASS_HOST,
        OVERTURE_HOST,
    ]


async def test_end_to_end_fetch_preserves_representative_geometry_types():
    """OSM elements and Overture geometries pass through with their types."""
    requests: list[httpx.Request] = []
    server = _smoke_server(requests)
    try:
        result = await server.vector_features(bbox=CITY_BBOX)
    finally:
        await server.aclose()

    geom_types = sorted(
        f.geometry["type"] for f in result.features if f.geometry is not None
    )
    # Point (cafe node + bus stop node + Overture place), LineString (the open
    # OSM way), Polygon (the closed OSM way + the Overture building).
    assert geom_types == [
        "LineString",
        "Point",
        "Point",
        "Point",
        "Polygon",
        "Polygon",
    ]

    # A representative OSM feature keeps its tags alongside the source marker.
    osm_named = {
        f.properties.get("name")
        for f in result.features
        if f.properties["source"] == "openstreetmap"
    }
    assert "Cafe Kiro" in osm_named


async def test_end_to_end_fetch_through_registered_tool_dispatch():
    """Driving the registered ``vector_features`` tool yields the same result."""
    requests: list[httpx.Request] = []
    server = _smoke_server(requests)
    tool = server.get_tool("vector_features")
    try:
        result = await tool.func(bbox=CITY_BBOX)
    finally:
        await server.aclose()

    assert isinstance(result, FeatureCollection)
    assert len(result.features) == 6
    assert {f.properties["source"] for f in result.features} == {
        "openstreetmap",
        "overture",
    }


async def test_end_to_end_layer_filter_reaches_both_sources():
    """A layer filter is propagated to the Overpass query and Overture params."""
    requests: list[httpx.Request] = []
    server = _smoke_server(requests)
    try:
        result = await server.vector_features(
            bbox=CITY_BBOX, layers=["building", "highway"]
        )
    finally:
        await server.aclose()

    assert isinstance(result, FeatureCollection)
    assert len(result.features) == 6

    overpass_req = next(r for r in requests if r.url.host == OVERPASS_HOST)
    overpass_query = overpass_req.content.decode("utf-8")
    # Both requested layers become Overpass tag selectors.
    assert "[building]" in overpass_query
    assert "[highway]" in overpass_query

    overture_req = next(r for r in requests if r.url.host == OVERTURE_HOST)
    # Overture receives the bbox plus the requested layers as theme types.
    assert "bbox" in overture_req.url.params
    assert overture_req.url.params.get_list("types") == ["building", "highway"]


async def test_end_to_end_result_is_json_serializable_geojson():
    """The merged collection serializes to GeoJSON-shaped JSON (Req 7.3)."""
    requests: list[httpx.Request] = []
    server = _smoke_server(requests)
    try:
        result = await server.vector_features(bbox=CITY_BBOX)
    finally:
        await server.aclose()

    payload = json.loads(result.model_dump_json())
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) == 6
    assert all(feat["type"] == "Feature" for feat in payload["features"])
    assert all("source" in feat["properties"] for feat in payload["features"])
