"""Tests for ``observations`` via :class:`GeoWeatherClimateServer` (task 13.4).

Covers the Requirement 7.6 / 7.12 contract for the weather/climate connector:

* a valid request returns the matching observations through the full server
  path (the shared :class:`~geo_common.http.HttpClient` wired to an
  ``httpx.MockTransport`` so the round-trip is deterministic with no real
  network I/O);
* a ``time_range`` whose start is later than its end is rejected with an
  ``Error_Taxonomy`` validation error naming ``time_range`` - the case task
  13.4 calls out explicitly - and no source is contacted;
* a malformed location and an unknown source are likewise validation errors;
* an unreachable source surfaces as an availability (``NETWORK``) error
  identifying the unavailable source with no partial results (Requirement 7.11);
* the catalog entry and Optional credential specs are registered (Req 2.1, 16.1).

_Requirements: 7.6, 7.12_
"""

from __future__ import annotations

from typing import Any, Dict

import httpx
import pytest

import json
from pathlib import Path

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    NetworkError,
    ValidationError,
)
from geo_common.http import HttpClient
from geo_common.models import CredentialClassification, OpennessTier
from geo_common.retry import RetryPolicy

from geo_weather_climate.server import INSTALL_COMMAND, GeoWeatherClimateServer

#: Repo root: tests/ -> geo-weather-climate/ -> packages/ -> <repo root>.
_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "bundle-manifest.json"

# A point location and a valid time range used across the scenarios.
SF = {"lon": -122.45, "lat": 37.75}
VALID_RANGE = ("2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z")


async def _noop_sleep(_seconds: float) -> None:
    """Async sleep that returns immediately so retries don't delay tests."""
    return None


def _client(handler: Any, *, max_attempts: int = 3) -> HttpClient:
    return HttpClient(
        RetryPolicy(max_attempts=max_attempts),
        transport=httpx.MockTransport(handler),
        sleep=_noop_sleep,
    )


def _open_meteo_payload() -> Dict[str, Any]:
    """A representative Open-Meteo archive ``hourly`` payload."""
    return {
        "latitude": 37.75,
        "longitude": -122.45,
        "hourly": {
            "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
            "temperature_2m": [9.1, 8.7],
            "precipitation": [0.0, 0.2],
            "relative_humidity_2m": [81, 83],
            "wind_speed_10m": [5.4, 6.0],
        },
    }


# --- Valid requests --------------------------------------------------------


async def test_observations_returns_records_for_valid_request() -> None:
    """Req 7.6: a valid location + time range yields the source's observations."""
    server = GeoWeatherClimateServer(
        http=_client(lambda req: httpx.Response(200, json=_open_meteo_payload()))
    )
    try:
        records = await server.observations(location=SF, time_range=VALID_RANGE)
    finally:
        await server.aclose()

    assert len(records) == 2
    first = records[0]
    assert first["time"] == "2023-01-01T00:00"
    assert first["source"] == "open-meteo"
    assert first["temperature_2m"] == 9.1
    assert first["lon"] == -122.45 and first["lat"] == 37.75


async def test_observations_invoked_through_tool_registry() -> None:
    """The ``observations`` capability is reachable via the tool registry and
    forwards the location/time-range params to the source."""
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json=_open_meteo_payload())

    server = GeoWeatherClimateServer(http=_client(handler))
    try:
        assert "observations" in server.tool_names()
        tool = server.get_tool("observations")
        records = await tool.func(location=SF, time_range=VALID_RANGE)
    finally:
        await server.aclose()

    assert len(records) == 2
    assert "latitude=37.75" in captured["url"]
    assert "start_date=2023-01-01" in captured["url"]
    assert "end_date=2023-01-02" in captured["url"]


async def test_observations_selects_named_source() -> None:
    """A caller can select the Open-Meteo air-quality source by name."""
    air_quality_payload = {
        "latitude": 37.75,
        "longitude": -122.45,
        "hourly": {
            "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
            "pm2_5": [12.3, 13.1],
            "pm10": [20.0, 21.2],
            "ozone": [55.0, 56.0],
        },
    }
    server = GeoWeatherClimateServer(
        http=_client(lambda req: httpx.Response(200, json=air_quality_payload))
    )
    try:
        records = await server.observations(
            location=SF, time_range=VALID_RANGE, source="air-quality"
        )
    finally:
        await server.aclose()

    assert len(records) == 2
    assert records[0]["source"] == "air-quality"
    assert records[0]["pm2_5"] == 12.3
    assert records[0]["pm10"] == 20.0


# --- Validation errors (Requirement 7.12) ----------------------------------


async def test_start_after_end_time_range_is_rejected() -> None:
    """Task 13.4 / Req 7.12: start later than end -> validation error, no fetch."""
    contacted = {"called": False}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        contacted["called"] = True
        return httpx.Response(200, json=_open_meteo_payload())

    server = GeoWeatherClimateServer(http=_client(handler))
    try:
        with pytest.raises(ValidationError) as exc_info:
            await server.observations(
                location=SF,
                time_range=("2023-02-01T00:00:00Z", "2023-01-01T00:00:00Z"),
            )
    finally:
        await server.aclose()

    err = exc_info.value
    assert err.category is ErrorCategory.VALIDATION
    assert err.detail.get("parameter") == "time_range"
    # No source was contacted: validation short-circuits before any I/O.
    assert contacted["called"] is False


async def test_unparseable_timestamp_is_rejected() -> None:
    """Req 7.12: a non-ISO timestamp -> validation error naming time_range."""
    server = GeoWeatherClimateServer(
        http=_client(lambda req: httpx.Response(200, json=_open_meteo_payload()))
    )
    try:
        with pytest.raises(ValidationError) as exc_info:
            await server.observations(
                location=SF, time_range=("not-a-date", "2023-01-02T00:00:00Z")
            )
    finally:
        await server.aclose()
    assert exc_info.value.detail.get("parameter") == "time_range"


async def test_out_of_range_location_is_rejected() -> None:
    """Req 7.12: an out-of-range latitude -> validation error naming location."""
    server = GeoWeatherClimateServer(
        http=_client(lambda req: httpx.Response(200, json=_open_meteo_payload()))
    )
    try:
        with pytest.raises(ValidationError) as exc_info:
            await server.observations(
                location={"lon": 0.0, "lat": 91.0}, time_range=VALID_RANGE
            )
    finally:
        await server.aclose()
    assert exc_info.value.detail.get("parameter") == "location"


async def test_unknown_source_is_rejected() -> None:
    """An unknown source name -> validation error naming source."""
    server = GeoWeatherClimateServer(
        http=_client(lambda req: httpx.Response(200, json=_open_meteo_payload()))
    )
    try:
        with pytest.raises(ValidationError) as exc_info:
            await server.observations(
                location=SF, time_range=VALID_RANGE, source="does-not-exist"
            )
    finally:
        await server.aclose()
    assert exc_info.value.detail.get("parameter") == "source"


# --- Availability error (Requirement 7.11) ---------------------------------


async def test_unreachable_source_yields_availability_error() -> None:
    """Req 7.11: an unreachable source -> NETWORK error naming the source."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    server = GeoWeatherClimateServer(http=_client(handler))
    try:
        with pytest.raises(NetworkError) as exc_info:
            await server.observations(location=SF, time_range=VALID_RANGE)
    finally:
        await server.aclose()

    err = exc_info.value
    assert err.category is ErrorCategory.NETWORK
    assert err.source == "open-meteo"


# --- Catalog + credential registration (Req 2.1, 16.1) ---------------------


def test_catalog_entry_names_geo_weather_climate_as_provider() -> None:
    """Req 2.1 / 11.3: the catalog entry names geo-weather-climate as provider."""
    server = GeoWeatherClimateServer()
    entries = server.catalog_entries()

    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "observations"
    assert entry.pillar == "A"
    assert entry.provider_server == "geo-weather-climate"
    assert entry.openness_tier is OpennessTier.OPEN


def test_required_credentials_are_optional_and_match_manifest() -> None:
    """Req 16.1: the wrapped-source key is declared and classified Optional."""
    server = GeoWeatherClimateServer()
    specs = server.required_credentials()

    keys = {spec.mcp_json_key for spec in specs}
    assert keys == {"NOAA_CDO_TOKEN"}
    assert all(
        spec.classification is CredentialClassification.OPTIONAL for spec in specs
    )


def test_server_starts_with_no_credentials_configured() -> None:
    """Req 16.5: only Optional credentials -> startup is never blocked."""
    server = GeoWeatherClimateServer()
    server.start(configured_keys=[])
    assert server.started is True


def test_required_credentials_match_manifest() -> None:
    """Req 16.1: in-code credential specs agree exactly with bundle-manifest.json."""
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    entry = {s["name"]: s for s in manifest["servers"]}["geo-weather-climate"]
    specs = GeoWeatherClimateServer().required_credentials()
    spec_pairs = {(s.mcp_json_key, s.classification.value) for s in specs}
    manifest_pairs = {(c["key"], c["classification"]) for c in entry["credentials"]}
    assert spec_pairs == manifest_pairs
    assert entry["uvx"] == INSTALL_COMMAND == "uvx geo-weather-climate"


async def test_authentication_failure_yields_authentication_error() -> None:
    """Req 7.8: an authentication failure -> auth error, no partial data."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "unauthorized"})

    server = GeoWeatherClimateServer(http=_client(handler))
    try:
        with pytest.raises(AuthenticationError) as exc_info:
            await server.observations(location=SF, time_range=VALID_RANGE)
    finally:
        await server.aclose()
    assert exc_info.value.category is ErrorCategory.AUTHENTICATION


# --- NOAA CDO (credentialed source) ----------------------------------------


async def test_cdo_without_token_is_authentication_error_naming_key() -> None:
    """``source="cdo"`` with no token configured -> AuthenticationError naming
    NOAA_CDO_TOKEN, and no request is ever made (credential guard)."""
    contacted = {"called": False}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        contacted["called"] = True
        return httpx.Response(200, json={})

    # No NOAA_CDO_TOKEN in env -> the CDO source is selectable but unconfigured.
    server = GeoWeatherClimateServer(http=_client(handler))
    try:
        with pytest.raises(AuthenticationError) as exc_info:
            await server.observations(
                location=SF, time_range=VALID_RANGE, source="cdo"
            )
    finally:
        await server.aclose()

    err = exc_info.value
    assert err.category is ErrorCategory.AUTHENTICATION
    assert err.detail.get("mcp_json_key") == "NOAA_CDO_TOKEN"
    assert "NOAA_CDO_TOKEN" in str(err)
    # Secret-free: the guard names the key, never a value.
    assert contacted["called"] is False


async def test_cdo_with_token_resolves_station_then_data() -> None:
    """With a token, CDO resolves a station near the point then fetches its
    daily-summary data, forwarding the ``token`` header on every request."""
    from geo_weather_climate.observations import NoaaCdoSource

    seen: Dict[str, Any] = {"paths": [], "token_headers": []}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["paths"].append(request.url.path)
        seen["token_headers"].append(request.headers.get("token"))
        if request.url.path.endswith("/stations"):
            # The station query must constrain to the requested window and
            # prefer best-covered stations so limit=1 picks one with data.
            seen["station_params"] = dict(request.url.params)
            return httpx.Response(
                200, json={"results": [{"id": "GHCND:USW00023234", "name": "SFO"}]}
            )
        # /data
        return httpx.Response(
            200,
            json={
                "results": [
                    {"date": "2023-01-01T00:00:00", "datatype": "TMAX", "value": 14.2},
                    {"date": "2023-01-01T00:00:00", "datatype": "TMIN", "value": 7.1},
                    {"date": "2023-01-02T00:00:00", "datatype": "TMAX", "value": 13.0},
                ]
            },
        )

    # Inject a CDO source with a token directly (no env dependency).
    server = GeoWeatherClimateServer(
        sources=[NoaaCdoSource(token="test-token")],
        http=_client(handler),
    )
    try:
        records = await server.observations(
            location=SF, time_range=VALID_RANGE, source="cdo"
        )
    finally:
        await server.aclose()

    # Two hops, station then data, both carrying the token header.
    assert any(p.endswith("/stations") for p in seen["paths"])
    assert any(p.endswith("/data") for p in seen["paths"])
    assert seen["token_headers"] == ["test-token", "test-token"]
    # Station selection is constrained to the requested window + best coverage.
    sp = seen["station_params"]
    assert sp.get("startdate") == "2023-01-01" and sp.get("enddate") == "2023-01-02"
    assert sp.get("sortfield") == "datacoverage"

    # Two distinct dates -> two grouped observations; TMAX/TMIN merged per date.
    assert [r["time"] for r in records] == [
        "2023-01-01T00:00:00",
        "2023-01-02T00:00:00",
    ]
    assert records[0]["source"] == "cdo"
    assert records[0]["TMAX"] == 14.2 and records[0]["TMIN"] == 7.1
    assert records[0]["station"] == "GHCND:USW00023234"
    assert records[1]["TMAX"] == 13.0


async def test_nws_normalizes_date_only_range_to_rfc3339() -> None:
    """Regression: NWS rejects a date-only start/end with HTTP 400, so the source
    must send full RFC3339 timestamps even when the caller passes plain dates."""
    from geo_weather_climate.observations import NwsSource

    seen: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/points/"):
            return httpx.Response(
                200,
                json={"properties": {"observationStations": "https://api.weather.gov/x/stations"}},
            )
        if path.endswith("/stations"):
            return httpx.Response(
                200, json={"features": [{"properties": {"stationIdentifier": "KDCA"}}]}
            )
        # observations: capture the start/end query params.
        seen["start"] = request.url.params.get("start")
        seen["end"] = request.url.params.get("end")
        return httpx.Response(200, json={"features": []})

    server = GeoWeatherClimateServer(sources=[NwsSource()], http=_client(handler))
    try:
        await server.observations(
            location={"lon": -77.04, "lat": 38.90},
            time_range=("2024-06-01", "2024-06-02"),
            source="nws",
        )
    finally:
        await server.aclose()

    # Date-only inputs are expanded to RFC3339 instants with an explicit zone.
    assert seen["start"] == "2024-06-01T00:00:00Z"
    assert seen["end"] == "2024-06-02T23:59:59Z"


def test_to_rfc3339_normalization() -> None:
    """The RFC3339 normalizer handles date-only, naive, and zoned inputs."""
    from geo_weather_climate.observations import _to_rfc3339

    assert _to_rfc3339("2024-06-01") == "2024-06-01T00:00:00Z"
    assert _to_rfc3339("2024-06-02", end=True) == "2024-06-02T23:59:59Z"
    assert _to_rfc3339("2024-06-01T12:30:00") == "2024-06-01T12:30:00Z"
    assert _to_rfc3339("2024-06-01T12:30:00Z") == "2024-06-01T12:30:00Z"


def test_cdo_source_is_selectable_in_default_set() -> None:
    """The credentialed CDO source is part of the server's source set (so
    ``source="cdo"`` resolves) even though it is not the default selection."""
    server = GeoWeatherClimateServer()
    names = {s.name for s in server.sources}
    assert "cdo" in names
    assert server.default_source == "open-meteo"
