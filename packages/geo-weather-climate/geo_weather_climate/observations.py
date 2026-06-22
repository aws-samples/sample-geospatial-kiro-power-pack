"""Weather/climate observation retrieval (Requirements 7.6, 7.12).

This module implements ``observations``: given a point ``location`` and a
``(start, end)`` time range, it validates the request, resolves the named
source, queries it through the shared
:class:`~geo_common.http.HttpClient`, and returns the matching observations as
a list of plain mappings (the design's ``list[dict]`` contract).

Three open, no-credential sources ship by default:

* :class:`OpenMeteoSource` - hourly weather observations from the Open-Meteo
  archive API (the default source).
* :class:`OpenAQSource` - air-quality measurements from the OpenAQ API.
* :class:`NwsSource` - US National Weather Service station observations
  (``api.weather.gov``; US coverage only), selectable via ``source="nws"``.

Every outbound call goes through the shared :class:`HttpClient`, so all sources
inherit the same retry/backoff, rate-limit handling, and 30-second per-request
timeout (Requirement 5) - which is how the 30-second response bound of
Requirement 7.6 is enforced. A source that is unreachable or does not respond
within the timeout surfaces as an ``Error_Taxonomy`` availability error
(``NETWORK``/``UPSTREAM``) re-tagged with the logical source name so the error
identifies the unavailable source (Requirement 7.11), and no partial list is
returned.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    GeoError,
    NetworkError,
    UpstreamError,
    ValidationError,
)
from geo_common.http import HttpClient

from geo_weather_climate.models import Coordinate, Observation
from geo_weather_climate.timerange import validate_location, validate_time_range

__all__ = [
    "WeatherSource",
    "OpenMeteoSource",
    "OpenAQSource",
    "NwsSource",
    "NoaaCdoSource",
    "default_sources",
    "DEFAULT_SOURCE",
    "observations",
]

_SERVER_NAME = "geo-weather-climate"

#: A descriptive User-Agent. The NWS API (api.weather.gov) requires every
#: request to identify the client, so the NWS source always sends one.
_USER_AGENT = "geo-weather-climate/0.1 (Geospatial Power Pack)"

#: The source used when a caller does not name one explicitly. Open-Meteo is
#: open and requires no credential.
DEFAULT_SOURCE = "open-meteo"

#: Taxonomy categories that represent a configured source being unavailable -
#: unreachable or not responding within the 30-second per-request timeout
#: (Requirement 7.11). ``observations`` re-tags such an error with the logical
#: source name so the availability error names the unavailable source.
_AVAILABILITY_CATEGORIES = (ErrorCategory.NETWORK, ErrorCategory.UPSTREAM)

#: Default public endpoints. Each is overridable per source instance so a
#: deployment (or a test) can point at a mirror or a mock transport.
DEFAULT_OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_OPENAQ_URL = "https://api.openaq.org/v2/measurements"
#: NWS API root (api.weather.gov). Open, no credential; US coverage only.
DEFAULT_NWS_URL = "https://api.weather.gov"
#: NOAA Climate Data Online (CDO) v2 API root. Credentialed: every request
#: carries a ``token`` header read from ``NOAA_CDO_TOKEN``.
DEFAULT_CDO_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"


class WeatherSource:
    """Base class for a weather/climate source queried by ``observations``.

    A source exposes a human-readable :attr:`name` and an async :meth:`fetch`
    that returns its observations for a validated location and time range.
    Subclasses make all I/O through the shared :class:`HttpClient` and raise a
    taxonomy-classified :class:`~geo_common.errors.GeoError` on failure.
    """

    name: str = "weather-source"

    async def fetch(
        self,
        http: HttpClient,
        location: Coordinate,
        start: str,
        end: str,
    ) -> List[Observation]:  # pragma: no cover - abstract
        raise NotImplementedError


class OpenMeteoSource(WeatherSource):
    """Hourly weather observations via the Open-Meteo archive API."""

    name = "open-meteo"

    #: Hourly variables requested by default; overridable per instance.
    DEFAULT_HOURLY = (
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
    )

    def __init__(
        self,
        url: str = DEFAULT_OPEN_METEO_URL,
        *,
        hourly: Optional[Sequence[str]] = None,
    ) -> None:
        self.url = url
        self.hourly = tuple(hourly) if hourly is not None else self.DEFAULT_HOURLY

    async def fetch(
        self,
        http: HttpClient,
        location: Coordinate,
        start: str,
        end: str,
    ) -> List[Observation]:
        params: Dict[str, Any] = {
            "latitude": location.lat,
            "longitude": location.lon,
            "start_date": _date_part(start),
            "end_date": _date_part(end),
            "hourly": ",".join(self.hourly),
            "timezone": "UTC",
        }
        try:
            response = await http.get(self.url, params=params)
        except GeoError:
            raise
        if response.status_code >= 400:
            raise UpstreamError(
                "Open-Meteo returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "Open-Meteo returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )
        return _parse_open_meteo(payload, location)


class OpenAQSource(WeatherSource):
    """Air-quality measurements via the OpenAQ API."""

    name = "openaq"

    def __init__(self, url: str = DEFAULT_OPENAQ_URL, *, radius_m: int = 25000) -> None:
        self.url = url
        self.radius_m = radius_m

    async def fetch(
        self,
        http: HttpClient,
        location: Coordinate,
        start: str,
        end: str,
    ) -> List[Observation]:
        params: Dict[str, Any] = {
            "coordinates": "%g,%g" % (location.lat, location.lon),
            "radius": self.radius_m,
            "date_from": start,
            "date_to": end,
            "limit": 1000,
        }
        try:
            response = await http.get(self.url, params=params)
        except GeoError:
            raise
        if response.status_code >= 400:
            raise UpstreamError(
                "OpenAQ returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "OpenAQ returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )
        return _parse_openaq(payload, location)


class NwsSource(WeatherSource):
    """Surface weather observations via the US National Weather Service API.

    ``api.weather.gov`` is open (no credential) but **US coverage only** and
    station-based, so a reading is resolved in up to three hops: the point's
    metadata (``/points/{lat},{lon}``) → its observation stations → the nearest
    station's observations over the time range. A point outside US coverage
    yields no observations. Every request carries a descriptive User-Agent, as
    the NWS API requires.
    """

    name = "nws"

    def __init__(self, url: str = DEFAULT_NWS_URL) -> None:
        self.url = url.rstrip("/")

    async def _get_json(self, http: HttpClient, url: str, params=None) -> Any:
        try:
            response = await http.get(
                url, params=params, headers={"User-Agent": _USER_AGENT, "Accept": "application/geo+json"}
            )
        except GeoError:
            raise
        if response.status_code == 404:
            return None  # outside US coverage / no data at this hop
        if response.status_code >= 400:
            raise UpstreamError(
                "NWS returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            return response.json()
        except ValueError as exc:
            raise UpstreamError(
                "NWS returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )

    async def fetch(
        self,
        http: HttpClient,
        location: Coordinate,
        start: str,
        end: str,
    ) -> List[Observation]:
        point = await self._get_json(
            http, "%s/points/%g,%g" % (self.url, location.lat, location.lon)
        )
        stations_url = None
        if isinstance(point, dict):
            props = point.get("properties")
            if isinstance(props, dict):
                stations_url = props.get("observationStations")
        if not isinstance(stations_url, str) or not stations_url:
            return []  # no US coverage for this point

        stations = await self._get_json(http, stations_url)
        station_id = _first_station_id(stations)
        if not station_id:
            return []

        obs = await self._get_json(
            http,
            "%s/stations/%s/observations" % (self.url, station_id),
            params={
                "start": _to_rfc3339(start),
                "end": _to_rfc3339(end, end=True),
            },
        )
        return _parse_nws(obs, location)


class NoaaCdoSource(WeatherSource):
    """Climate observations via the NOAA Climate Data Online (CDO) v2 API.

    CDO is **credentialed** (it reads a ``token`` from ``NOAA_CDO_TOKEN``) and
    **station-based with US/global station coverage**, so a reading is resolved
    in two hops: stations near the point (``/stations?extent=<bbox around the
    point>``) → the nearest station's daily-summary data
    (``/data?datasetid=GHCND&stationid=...``) over the time range. When no token
    is configured the source raises an ``Error_Taxonomy``
    :class:`~geo_common.errors.AuthenticationError` naming the ``mcp.json`` key
    (never the secret value) before any request is made.
    """

    name = "cdo"

    #: Half-width (degrees) of the bounding box built around the query point to
    #: find nearby CDO stations via the ``extent`` filter.
    _EXTENT_HALF_DEG = 0.5

    def __init__(
        self,
        url: str = DEFAULT_CDO_URL,
        *,
        token: Optional[str] = None,
        datasetid: str = "GHCND",
    ) -> None:
        self.url = url.rstrip("/")
        self.token = token or None
        self.datasetid = datasetid

    async def _get_json(self, http: HttpClient, path: str, params=None) -> Any:
        try:
            response = await http.get(
                "%s/%s" % (self.url, path.lstrip("/")),
                params=params,
                headers={"token": self.token, "User-Agent": _USER_AGENT},
            )
        except GeoError:
            raise
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise UpstreamError(
                "NOAA CDO returned HTTP %d" % response.status_code,
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            return response.json()
        except ValueError as exc:
            raise UpstreamError(
                "NOAA CDO returned a non-JSON response",
                source=self.name,
                original=str(exc),
            )

    async def fetch(
        self,
        http: HttpClient,
        location: Coordinate,
        start: str,
        end: str,
    ) -> List[Observation]:
        if not self.token:
            # Credential guard: deny before any request and name the key only.
            raise AuthenticationError(
                "NOAA CDO is not configured: set NOAA_CDO_TOKEN in mcp.json",
                source=self.name,
                detail={"mcp_json_key": "NOAA_CDO_TOKEN"},
            )

        half = self._EXTENT_HALF_DEG
        # CDO extent is "minlat,minlng,maxlat,maxlng".
        extent = "%g,%g,%g,%g" % (
            location.lat - half,
            location.lon - half,
            location.lat + half,
            location.lon + half,
        )
        # Constrain station selection to the requested window and prefer the
        # best-covered station, so ``limit=1`` lands on a station that actually
        # has data for that range (an unconstrained pick can return a station
        # with no data in the window, yielding an empty - and misleading -
        # result).
        stations = await self._get_json(
            http,
            "stations",
            params={
                "datasetid": self.datasetid,
                "extent": extent,
                "startdate": _date_part(start),
                "enddate": _date_part(end),
                "sortfield": "datacoverage",
                "sortorder": "desc",
                "limit": 1,
            },
        )
        station_id = _first_cdo_station_id(stations)
        if not station_id:
            return []

        data = await self._get_json(
            http,
            "data",
            params={
                "datasetid": self.datasetid,
                "stationid": station_id,
                "startdate": _date_part(start),
                "enddate": _date_part(end),
                "units": "metric",
                "limit": 1000,
            },
        )
        return _parse_cdo(data, location, station_id)


def default_sources() -> List[WeatherSource]:
    """The default source set: Open-Meteo (weather) + OpenAQ (air quality) + NWS.

    Open-Meteo (the default) and OpenAQ are global; NWS (``api.weather.gov``,
    open, US coverage only) is selectable via ``source="nws"``. The credentialed
    NOAA CDO source (``source="cdo"``) is added by the server when configured
    (it reads ``NOAA_CDO_TOKEN``) and is not part of the open default set.
    """
    return [OpenMeteoSource(), OpenAQSource(), NwsSource()]


async def observations(
    *,
    location: Any,
    time_range: Sequence[str],
    http: HttpClient,
    sources: Sequence[WeatherSource],
    source: str = DEFAULT_SOURCE,
) -> List[Dict[str, Any]]:
    """Return observations for ``location`` over ``time_range`` (Req 7.6).

    Validates ``location`` and ``time_range`` first (Requirement 7.12): a
    malformed coordinate, an unparseable timestamp, or a ``start`` later than
    its ``end`` raises a :class:`~geo_common.errors.ValidationError` before any
    source is queried. The named ``source`` is then resolved against the
    configured set (an unknown name is itself a validation error) and queried
    through the shared :class:`HttpClient`, inheriting its 30-second
    per-request timeout (Requirement 7.6). A source/transport failure is
    re-tagged with the logical source name and raised as a taxonomy-classified
    availability error so no partial list is returned (Requirement 7.11).
    """
    validated_location = validate_location(location)
    start, end = validate_time_range(time_range)
    chosen = _resolve_source(sources, source)

    try:
        records = await chosen.fetch(http, validated_location, start, end)
    except GeoError as exc:
        # Re-tag an availability failure with the logical source name so the
        # error identifies the unavailable source, and raise before returning
        # so no partial list escapes (Requirement 7.11). Other taxonomy
        # categories pass through unchanged (Requirement 11.2).
        raise _identify_unavailable_source(exc, source_name=chosen.name)
    return [record.as_dict() for record in records]


def _resolve_source(sources: Sequence[WeatherSource], name: str) -> WeatherSource:
    """Resolve a source by name, or raise a validation error naming ``source``."""
    if not isinstance(name, str) or not name.strip():
        raise ValidationError(
            "source must be a non-empty source name",
            source=_SERVER_NAME,
            detail={"parameter": "source"},
        )
    for candidate in sources:
        if candidate.name == name:
            return candidate
    available = ", ".join(sorted(candidate.name for candidate in sources))
    raise ValidationError(
        "unknown source %r; configured sources: %s" % (name, available or "none"),
        source=_SERVER_NAME,
        detail={"parameter": "source", "available": available},
    )


def _identify_unavailable_source(exc: GeoError, *, source_name: str) -> GeoError:
    """Ensure an availability failure identifies the unavailable source (Req 7.11).

    The shared :class:`HttpClient` tags a timeout/connection failure
    (``NETWORK``) or an upstream 5xx/server rate-limit (``UPSTREAM``) with the
    request *host*. For an availability error this re-tags the error's
    ``source`` with the configured source's logical name (e.g. ``"open-meteo"``)
    while preserving the category, message, structured ``detail``, retained
    ``original`` detail, and any ``retry_after``. Errors in any other taxonomy
    category pass through unchanged.
    """
    if exc.category not in _AVAILABILITY_CATEGORIES:
        return exc
    if exc.source == source_name:
        return exc
    cls = NetworkError if exc.category is ErrorCategory.NETWORK else UpstreamError
    return cls(
        exc.message,
        source=source_name,
        detail=exc.detail,
        original=exc.original,
        retry_after=exc.retry_after,
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _to_rfc3339(value: str, *, end: bool = False) -> str:
    """Normalize a timestamp to an RFC3339 instant with an explicit timezone.

    The NWS observations endpoint requires full RFC3339 ``start``/``end`` values
    (a date-only string like ``2024-06-01`` is rejected with HTTP 400). This
    expands a date-only bound to the start (``T00:00:00Z``) or, for an ``end``
    bound, the end (``T23:59:59Z``) of that UTC day, and stamps a naive
    timestamp as UTC; a value that already carries a ``Z``/offset is returned
    in canonical form. A value that cannot be parsed is returned unchanged so
    NWS performs the final validation.
    """
    text = str(value).strip()
    if len(text) == 10 and text[4:5] == "-" and text[7:8] == "-":
        return text + ("T23:59:59Z" if end else "T00:00:00Z")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat().replace("+00:00", "Z")


def _date_part(timestamp: str) -> str:
    """Return the ``YYYY-MM-DD`` date portion of an ISO-8601 timestamp.

    The Open-Meteo archive API takes whole-day ``start_date``/``end_date``
    bounds, so the time-of-day component (if any) is dropped.
    """
    text = timestamp.strip()
    for sep in ("T", " "):
        if sep in text:
            return text.split(sep, 1)[0]
    return text


def _parse_open_meteo(payload: Any, location: Coordinate) -> List[Observation]:
    """Convert an Open-Meteo ``hourly`` payload into observation records.

    Open-Meteo returns parallel arrays under ``hourly`` (a ``time`` array plus
    one array per requested variable). Each index becomes one observation whose
    ``values`` hold that hour's variables.
    """
    if not isinstance(payload, dict):
        raise UpstreamError(
            "Open-Meteo payload was not a JSON object",
            source="open-meteo",
        )
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        return []
    times = hourly.get("time")
    if not isinstance(times, list):
        return []

    variable_keys = [k for k in hourly.keys() if k != "time" and isinstance(hourly.get(k), list)]
    observations_out: List[Observation] = []
    for index, when in enumerate(times):
        if not isinstance(when, str):
            continue
        values: Dict[str, Any] = {}
        for key in variable_keys:
            series = hourly[key]
            if index < len(series):
                values[key] = series[index]
        observations_out.append(
            Observation(
                time=when,
                source="open-meteo",
                values=values,
                lon=location.lon,
                lat=location.lat,
            )
        )
    return observations_out


def _parse_openaq(payload: Any, location: Coordinate) -> List[Observation]:
    """Convert an OpenAQ ``measurements`` payload into observation records.

    Each OpenAQ measurement (``{parameter, value, unit, date: {utc}}``) becomes
    one observation whose ``values`` carry the pollutant reading.
    """
    if not isinstance(payload, dict):
        raise UpstreamError(
            "OpenAQ payload was not a JSON object",
            source="openaq",
        )
    results = payload.get("results")
    if not isinstance(results, list):
        return []

    observations_out: List[Observation] = []
    for measurement in results:
        if not isinstance(measurement, dict):
            continue
        date = measurement.get("date")
        when = date.get("utc") if isinstance(date, dict) else None
        if not isinstance(when, str) or not when:
            continue
        parameter = measurement.get("parameter")
        values: Dict[str, Any] = {}
        if isinstance(parameter, str) and parameter:
            values[parameter] = measurement.get("value")
            if measurement.get("unit") is not None:
                values["unit"] = measurement.get("unit")
        coords = measurement.get("coordinates")
        lon = location.lon
        lat = location.lat
        if isinstance(coords, dict):
            if isinstance(coords.get("longitude"), (int, float)):
                lon = float(coords["longitude"])
            if isinstance(coords.get("latitude"), (int, float)):
                lat = float(coords["latitude"])
        observations_out.append(
            Observation(
                time=when,
                source="openaq",
                values=values,
                lon=lon,
                lat=lat,
            )
        )
    return observations_out


#: NWS observation properties (each a ``{"value", "unitCode"}`` object) mapped
#: into an observation's ``values``.
_NWS_FIELDS = (
    "temperature",
    "dewpoint",
    "relativeHumidity",
    "windSpeed",
    "windDirection",
    "barometricPressure",
    "precipitationLastHour",
)


def _first_station_id(stations: Any) -> Optional[str]:
    """Return the first station identifier from an NWS stations response."""
    if not isinstance(stations, dict):
        return None
    features = stations.get("features")
    if isinstance(features, list):
        for feature in features:
            if not isinstance(feature, dict):
                continue
            props = feature.get("properties")
            if isinstance(props, dict) and props.get("stationIdentifier"):
                return str(props["stationIdentifier"])
            ident = feature.get("id")
            if isinstance(ident, str) and ident:
                return ident.rstrip("/").rsplit("/", 1)[-1]
    return None


def _parse_nws(payload: Any, location: Coordinate) -> List[Observation]:
    """Convert an NWS station-observations GeoJSON payload into records.

    Each ``features[].properties`` carries a ``timestamp`` plus a set of
    ``{"value", "unitCode"}`` measurement objects; the numeric values (with
    their unit codes) are flattened into the observation's ``values``.
    """
    if payload is None:
        return []
    if not isinstance(payload, dict):
        raise UpstreamError(
            "NWS payload was not a JSON object", source="nws"
        )
    features = payload.get("features")
    if not isinstance(features, list):
        return []

    out: List[Observation] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties")
        if not isinstance(props, dict):
            continue
        when = props.get("timestamp")
        if not isinstance(when, str) or not when:
            continue
        values: Dict[str, Any] = {}
        for field in _NWS_FIELDS:
            entry = props.get(field)
            if isinstance(entry, dict) and entry.get("value") is not None:
                values[field] = entry.get("value")
                unit = entry.get("unitCode")
                if unit is not None:
                    values["%s_unit" % field] = unit
        out.append(
            Observation(
                time=when,
                source="nws",
                values=values,
                lon=location.lon,
                lat=location.lat,
            )
        )
    return out


def _first_cdo_station_id(stations: Any) -> Optional[str]:
    """Return the first station id from a NOAA CDO ``/stations`` response."""
    if not isinstance(stations, dict):
        return None
    results = stations.get("results")
    if isinstance(results, list):
        for entry in results:
            if isinstance(entry, dict) and entry.get("id"):
                return str(entry["id"])
    return None


def _parse_cdo(
    payload: Any, location: Coordinate, station_id: str
) -> List[Observation]:
    """Convert a NOAA CDO ``/data`` payload into observation records.

    CDO returns ``{"results": [{"date","datatype","station","value", ...}]}``.
    Rows are grouped by ``date`` so each timestamp yields one observation whose
    ``values`` carry the per-datatype readings (e.g. ``TMAX``, ``TMIN``,
    ``PRCP``) for the resolved station.
    """
    if payload is None:
        return []
    if not isinstance(payload, dict):
        raise UpstreamError(
            "NOAA CDO payload was not a JSON object", source="cdo"
        )
    results = payload.get("results")
    if not isinstance(results, list):
        return []

    by_date: "Dict[str, Dict[str, Any]]" = {}
    order: List[str] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        when = row.get("date")
        datatype = row.get("datatype")
        if not isinstance(when, str) or not when or not isinstance(datatype, str):
            continue
        if when not in by_date:
            by_date[when] = {"station": station_id}
            order.append(when)
        by_date[when][datatype] = row.get("value")

    return [
        Observation(
            time=when,
            source="cdo",
            values=by_date[when],
            lon=location.lon,
            lat=location.lat,
        )
        for when in order
    ]
