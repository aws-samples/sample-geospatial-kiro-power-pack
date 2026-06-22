"""Unit tests for the NWS source (geo-weather-climate).

Exercises the multi-hop ``NwsSource`` (point -> stations -> observations)
through an ``httpx.MockTransport`` that routes ``api.weather.gov`` by path. No
real network I/O happens.
"""

from __future__ import annotations

import httpx
import pytest

from geo_common.http import HttpClient
from geo_common.retry import RetryPolicy

from geo_weather_climate.observations import NwsSource, default_sources, observations

NWS_HOST = "api.weather.gov"
# A US point (Washington, DC) and a forward time range.
DC = {"lon": -77.0365, "lat": 38.8977}
RANGE = ["2024-06-01T00:00:00Z", "2024-06-01T06:00:00Z"]

_POINT = {
    "properties": {
        "observationStations": "https://api.weather.gov/gridpoints/LWX/97,71/stations"
    }
}
_STATIONS = {"features": [{"properties": {"stationIdentifier": "KDCA"}}]}
_OBSERVATIONS = {
    "features": [
        {
            "properties": {
                "timestamp": "2024-06-01T01:00:00Z",
                "temperature": {"value": 22.0, "unitCode": "wmoUnit:degC"},
                "relativeHumidity": {"value": 65.0, "unitCode": "wmoUnit:percent"},
                "windSpeed": {"value": 10.0, "unitCode": "wmoUnit:km_h-1"},
            }
        }
    ]
}


def _client(handler):
    return HttpClient(RetryPolicy(max_attempts=1), transport=httpx.MockTransport(handler))


def _nws_handler(*, point=_POINT, stations=_STATIONS, obs=_OBSERVATIONS):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == NWS_HOST
        assert request.headers.get("User-Agent")  # NWS requires a UA
        path = request.url.path
        if path.startswith("/points/"):
            return httpx.Response(200, json=point) if point is not None else httpx.Response(404)
        if path.endswith("/stations"):
            return httpx.Response(200, json=stations)
        if "/observations" in path:
            return httpx.Response(200, json=obs)
        return httpx.Response(404)  # pragma: no cover

    return handler


async def test_nws_returns_station_observations():
    client = _client(_nws_handler())
    try:
        records = await observations(
            location=DC, time_range=RANGE, http=client, sources=[NwsSource()], source="nws"
        )
    finally:
        await client.aclose()

    assert len(records) == 1
    rec = records[0]
    assert rec["source"] == "nws"
    assert rec["time"] == "2024-06-01T01:00:00Z"
    # Observation.as_dict() flattens the measured variables to the top level.
    assert rec["temperature"] == 22.0
    assert rec["temperature_unit"] == "wmoUnit:degC"
    assert rec["relativeHumidity"] == 65.0


async def test_nws_point_outside_us_returns_empty():
    # A 404 from /points means no US coverage -> no observations, no error.
    client = _client(_nws_handler(point=None))
    try:
        records = await observations(
            location={"lon": 2.29, "lat": 48.85},
            time_range=RANGE,
            http=client,
            sources=[NwsSource()],
            source="nws",
        )
    finally:
        await client.aclose()
    assert records == []


async def test_nws_no_stations_returns_empty():
    client = _client(_nws_handler(stations={"features": []}))
    try:
        records = await observations(
            location=DC, time_range=RANGE, http=client, sources=[NwsSource()], source="nws"
        )
    finally:
        await client.aclose()
    assert records == []


def test_default_sources_include_open_meteo_openaq_and_nws():
    names = [s.name for s in default_sources()]
    assert names == ["open-meteo", "openaq", "nws"]
