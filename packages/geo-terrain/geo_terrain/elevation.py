"""Elevation queries for ``geo-terrain`` (Requirements 7.5, 7.12).

This module implements ``elevation``: given either a single location
(:class:`~geo_terrain.models.Coordinate`) or an extent
(:class:`~geo_terrain.models.GeoWindow`), it validates the request, samples the
selected terrain dataset, and returns a scalar elevation (metres) for a point
or a :class:`~geo_terrain.models.RasterArray` for an extent (Requirement 7.5).

Two datasets ship by default, selected by the ``source`` parameter
(Requirement 7.5's "configured terrain source"):

* ``srtm`` - SRTM (the default; served by the public OpenTopoData endpoint); and
* ``3dep`` - USGS 3DEP (NED; served by the public endpoint).

Both datasets (``srtm30m`` and ``ned10m``) are hosted by the default public
endpoint (``api.opentopodata.org``), so elevation works out of the box with no
configuration or credential. The base URL and dataset key are overridable per
source, so a deployment can point at a self-hosted OpenTopoData (or compatible)
instance - for example to serve a higher-resolution or regional dataset.

Each source is a :class:`TerrainSource` that samples elevations for a list of
points through the shared :class:`~geo_common.http.HttpClient`, so every query
inherits the same retry/backoff, rate-limit handling, and 30-second
per-request timeout (Requirement 5) - which is how the 30-second response bound
of Requirement 7.5 is enforced. The request/response shape follows the
OpenTopoData point-query convention (``/v1/{dataset}?locations=lat,lng|...``),
and both the base URL and the dataset key are overridable per source so a
deployment (or a test) can point at a mirror or a mock transport.

All parameters are validated *before* any network call and a malformed request
raises an ``Error_Taxonomy`` :class:`~geo_common.errors.ValidationError` naming
the offending parameter, producing no partial result (Requirement 7.12).
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from geo_common.errors import (
    ErrorCategory,
    GeoError,
    NetworkError,
    UpstreamError,
    ValidationError,
)
from geo_common.http import HttpClient

from geo_terrain.models import BBox, Coordinate, GeoWindow, RasterArray

__all__ = [
    "DEFAULT_TERRAIN_API_URL",
    "DEFAULT_SOURCE",
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_MAX_LOCATIONS_PER_REQUEST",
    "TerrainSource",
    "default_sources",
    "parse_location",
    "elevation",
]

#: The secret-free source identifier used on errors raised from this module.
_SERVER_NAME = "geo-terrain"

#: Taxonomy categories that mean a configured source was unavailable -
#: unreachable or not responding within the 30-second per-request timeout
#: (Requirement 7.11). The shared :class:`HttpClient` maps a timeout/connection
#: failure onto ``NETWORK`` and an upstream 5xx / server rate-limit onto
#: ``UPSTREAM`` tagged with the request *host*; :func:`elevation` re-tags such
#: an error with the logical terrain-source name so the availability error
#: identifies the unavailable source.
_AVAILABILITY_CATEGORIES = (ErrorCategory.NETWORK, ErrorCategory.UPSTREAM)

#: Default public elevation point-query API root. Overridable per source so a
#: deployment or test can point at a mirror or an ``httpx.MockTransport``.
DEFAULT_TERRAIN_API_URL = "https://api.opentopodata.org"

#: The default terrain dataset selected when ``source`` is not given
#: (Requirement 7.5). ``srtm`` is used because its OpenTopoData dataset
#: (``srtm30m``) is hosted by the default public endpoint, so it works without
#: any configuration.
DEFAULT_SOURCE = "srtm"

#: The default maximum number of grid samples a single ``GeoWindow`` request
#: may resolve. Bounds request size so an extent query stays within the
#: 30-second response bound (Requirement 7.5); configurable per server/call.
DEFAULT_MAX_SAMPLES = 10_000

#: OpenTopoData's public API caps a single point-query at 100 locations
#: (a request with more returns HTTP 400). Windowed reads are therefore split
#: into batches of at most this many locations and the results concatenated in
#: order. Overridable per source for a self-hosted instance with a higher cap.
DEFAULT_MAX_LOCATIONS_PER_REQUEST = 100


class TerrainSource:
    """A named terrain dataset sampled via an elevation point-query API.

    A source maps a logical :attr:`name` (e.g. ``"srtm"``) onto an
    API ``dataset`` key and samples elevations for a list of ``(lon, lat)``
    points through the shared :class:`HttpClient`. All I/O goes through that
    client, so the source inherits the 30-second per-request timeout and
    retry/backoff (Requirement 5); failures surface as taxonomy-classified
    :class:`~geo_common.errors.GeoError`\\ s.
    """

    def __init__(
        self,
        name: str,
        dataset: str,
        *,
        url: str = DEFAULT_TERRAIN_API_URL,
        max_locations_per_request: int = DEFAULT_MAX_LOCATIONS_PER_REQUEST,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.url = url
        self.max_locations_per_request = max(1, int(max_locations_per_request))

    def _sample_url(self) -> str:
        """Build the dataset point-query URL from the API root."""
        return "%s/v1/%s" % (self.url.rstrip("/"), self.dataset)

    async def sample(
        self, http: HttpClient, points: Sequence[Tuple[float, float]]
    ) -> List[Optional[float]]:
        """Return elevations (metres) for ``points`` in order.

        ``points`` is a sequence of ``(lon, lat)`` pairs. Because the public
        endpoint caps a single request at
        :data:`DEFAULT_MAX_LOCATIONS_PER_REQUEST` locations (a larger request is
        rejected with HTTP 400), the points are split into batches of at most
        :attr:`max_locations_per_request` and the per-batch results are
        concatenated in order, so a windowed read of any size resolves. A
        transport/source failure on any batch propagates as a
        taxonomy-classified :class:`GeoError` (no partial result is returned).
        """
        ordered = list(points)
        if not ordered:
            return []
        elevations: List[Optional[float]] = []
        step = self.max_locations_per_request
        for start in range(0, len(ordered), step):
            elevations.extend(await self._sample_batch(http, ordered[start : start + step]))
        return elevations

    async def _sample_batch(
        self, http: HttpClient, points: Sequence[Tuple[float, float]]
    ) -> List[Optional[float]]:
        """Sample one batch of ``points`` (already bounded to the request cap)."""
        locations = "|".join("%g,%g" % (lat, lon) for (lon, lat) in points)
        try:
            response = await http.get(
                self._sample_url(), params={"locations": locations}
            )
        except GeoError:
            # Already taxonomy-classified (timeout/network/rate-limit/etc.).
            raise
        if response.status_code >= 400:
            raise UpstreamError(
                "%s returned HTTP %d" % (self.name, response.status_code),
                source=self.name,
                detail={"status_code": response.status_code},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "%s returned a non-JSON response" % self.name,
                source=self.name,
                original=str(exc),
            )
        return _parse_results(payload, expected=len(points), source=self.name)


def default_sources() -> Dict[str, TerrainSource]:
    """The default terrain sources keyed by their logical name (Req 7.5).

    SRTM (the default) and USGS 3DEP. The dataset keys follow the OpenTopoData
    naming (``srtm30m``, ``ned10m``); both are hosted by the default public
    endpoint, so they work out of the box. Both the endpoint and dataset are
    overridable per source (e.g. to point at a self-hosted instance serving a
    different or higher-resolution dataset).
    """
    return {
        "srtm": TerrainSource("srtm", "srtm30m"),
        "3dep": TerrainSource("3dep", "ned10m"),
    }


# ---------------------------------------------------------------------------
# Validation (Requirement 7.12)
# ---------------------------------------------------------------------------


def parse_location(location: Any) -> Union[Coordinate, GeoWindow]:
    """Normalize ``location`` to a :class:`Coordinate` or :class:`GeoWindow`.

    Accepts, and validates (Requirement 7.12):

    * a :class:`Coordinate` / :class:`GeoWindow` instance (returned as-is after
      revalidation);
    * a mapping ``{"lon": .., "lat": ..}`` → :class:`Coordinate`;
    * a mapping ``{"bbox": [..], "width": .., "height": ..}`` → :class:`GeoWindow`;
    * a 2-element ``(lon, lat)`` sequence → :class:`Coordinate`; or
    * a 4-element ``(min_lon, min_lat, max_lon, max_lat)`` sequence → a
      :class:`GeoWindow` with the default grid.

    Raises :class:`~geo_common.errors.ValidationError` (taxonomy ``validation``)
    naming ``location`` for any other shape, and delegates coordinate/extent
    range checks to :func:`_validate_coordinate` / :func:`_validate_window`.
    """
    if isinstance(location, Coordinate):
        return _validate_coordinate(location.lon, location.lat)
    if isinstance(location, GeoWindow):
        return _validate_window(location.bbox, location.width, location.height)

    if isinstance(location, Mapping):
        if "bbox" in location:
            return _validate_window(
                location.get("bbox"),
                location.get("width", 16),
                location.get("height", 16),
            )
        if "lon" in location and "lat" in location:
            return _validate_coordinate(location.get("lon"), location.get("lat"))
        raise ValidationError(
            "location mapping must contain either 'lon'/'lat' (a point) or "
            "'bbox' (an extent)",
            source=_SERVER_NAME,
            detail={"parameter": "location"},
        )

    if isinstance(location, (str, bytes)) or not isinstance(location, Sequence):
        raise ValidationError(
            "location must be a Coordinate, a GeoWindow, a (lon, lat) pair, or "
            "a (min_lon, min_lat, max_lon, max_lat) extent",
            source=_SERVER_NAME,
            detail={"parameter": "location"},
        )

    values = list(location)
    if len(values) == 2:
        return _validate_coordinate(values[0], values[1])
    if len(values) == 4:
        return _validate_window(values, 16, 16)
    raise ValidationError(
        "location sequence must have 2 values (a point) or 4 values (an "
        "extent), got %d" % len(values),
        source=_SERVER_NAME,
        detail={"parameter": "location"},
    )


def _coerce_number(value: Any, *, parameter: str, label: str) -> float:
    """Return ``value`` as a finite float or raise a validation error."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(
            "%s must be a number, got %r" % (label, value),
            source=_SERVER_NAME,
            detail={"parameter": parameter},
        )
    fvalue = float(value)
    if not math.isfinite(fvalue):
        raise ValidationError(
            "%s must be finite" % label,
            source=_SERVER_NAME,
            detail={"parameter": parameter},
        )
    return fvalue


def _validate_coordinate(lon: Any, lat: Any) -> Coordinate:
    """Validate a ``(lon, lat)`` point (Requirement 7.12)."""
    flon = _coerce_number(lon, parameter="location", label="longitude")
    flat = _coerce_number(lat, parameter="location", label="latitude")
    if not (-180.0 <= flon <= 180.0):
        raise ValidationError(
            "longitude must be within [-180, 180], got %g" % flon,
            source=_SERVER_NAME,
            detail={"parameter": "location"},
        )
    if not (-90.0 <= flat <= 90.0):
        raise ValidationError(
            "latitude must be within [-90, 90], got %g" % flat,
            source=_SERVER_NAME,
            detail={"parameter": "location"},
        )
    return Coordinate(lon=flon, lat=flat)


def _validate_window(
    bbox: Any, width: Any, height: Any, *, max_samples: int = DEFAULT_MAX_SAMPLES
) -> GeoWindow:
    """Validate an extent and its sample grid (Requirement 7.12)."""
    if isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence):
        raise ValidationError(
            "bbox must be a sequence of four numbers "
            "(min_lon, min_lat, max_lon, max_lat)",
            source=_SERVER_NAME,
            detail={"parameter": "location"},
        )
    coords = list(bbox)
    if len(coords) != 4:
        raise ValidationError(
            "bbox must contain exactly four numbers "
            "(min_lon, min_lat, max_lon, max_lat), got %d" % len(coords),
            source=_SERVER_NAME,
            detail={"parameter": "location"},
        )
    min_lon, min_lat, max_lon, max_lat = (
        _coerce_number(coords[0], parameter="location", label="min_lon"),
        _coerce_number(coords[1], parameter="location", label="min_lat"),
        _coerce_number(coords[2], parameter="location", label="max_lon"),
        _coerce_number(coords[3], parameter="location", label="max_lat"),
    )
    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise ValidationError(
            "bbox longitudes must be within [-180, 180]",
            source=_SERVER_NAME,
            detail={"parameter": "location"},
        )
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise ValidationError(
            "bbox latitudes must be within [-90, 90]",
            source=_SERVER_NAME,
            detail={"parameter": "location"},
        )
    if min_lon > max_lon:
        raise ValidationError(
            "bbox min_lon (%g) must not exceed max_lon (%g)" % (min_lon, max_lon),
            source=_SERVER_NAME,
            detail={"parameter": "location"},
        )
    if min_lat > max_lat:
        raise ValidationError(
            "bbox min_lat (%g) must not exceed max_lat (%g)" % (min_lat, max_lat),
            source=_SERVER_NAME,
            detail={"parameter": "location"},
        )

    iwidth = _validate_grid_dimension(width, label="width")
    iheight = _validate_grid_dimension(height, label="height")
    if iwidth * iheight > max_samples:
        raise ValidationError(
            "sample grid %d×%d = %d exceeds the configured maximum of %d samples"
            % (iwidth, iheight, iwidth * iheight, max_samples),
            source=_SERVER_NAME,
            detail={
                "parameter": "location",
                "width": iwidth,
                "height": iheight,
                "max_samples": max_samples,
            },
        )
    return GeoWindow(
        bbox=(min_lon, min_lat, max_lon, max_lat), width=iwidth, height=iheight
    )


def _validate_grid_dimension(value: Any, *, label: str) -> int:
    """Validate a positive-integer grid dimension (Requirement 7.12)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(
            "%s must be a positive integer" % label,
            source=_SERVER_NAME,
            detail={"parameter": "location"},
        )
    if value < 1:
        raise ValidationError(
            "%s must be a positive integer, got %d" % (label, value),
            source=_SERVER_NAME,
            detail={"parameter": "location"},
        )
    return value


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _grid_points(window: GeoWindow) -> List[Tuple[float, float]]:
    """Generate the ``(lon, lat)`` sample points for ``window``, row-major.

    Rows run north→south (so ``values[0]`` is the northernmost row) and columns
    west→east. A 1-wide/1-high axis samples the corresponding bound rather than
    dividing by zero.
    """
    min_lon, min_lat, max_lon, max_lat = window.bbox
    w, h = window.width, window.height
    lon_step = (max_lon - min_lon) / (w - 1) if w > 1 else 0.0
    lat_step = (max_lat - min_lat) / (h - 1) if h > 1 else 0.0
    points: List[Tuple[float, float]] = []
    for row in range(h):
        # North (max_lat) first.
        lat = max_lat - lat_step * row if h > 1 else max_lat
        for col in range(w):
            lon = min_lon + lon_step * col if w > 1 else min_lon
            points.append((lon, lat))
    return points


def _parse_results(
    payload: Any, *, expected: int, source: str
) -> List[Optional[float]]:
    """Parse an OpenTopoData-style point-query payload into elevations.

    Expects ``{"results": [{"elevation": <num|null>, ...}, ...]}`` and returns
    the elevations in order. A ``null`` elevation becomes ``None`` (no data).
    A structurally unusable payload raises an :class:`UpstreamError`.
    """
    if not isinstance(payload, dict):
        raise UpstreamError(
            "%s payload was not a JSON object" % source, source=source
        )
    results = payload.get("results")
    if not isinstance(results, list):
        raise UpstreamError(
            "%s payload had no 'results' list" % source, source=source
        )
    elevations: List[Optional[float]] = []
    for entry in results:
        if not isinstance(entry, dict) or "elevation" not in entry:
            elevations.append(None)
            continue
        value = entry.get("elevation")
        if value is None:
            elevations.append(None)
        elif isinstance(value, bool) or not isinstance(value, Real):
            elevations.append(None)
        else:
            elevations.append(float(value))
    return elevations


async def _sample(
    terrain: TerrainSource,
    http: HttpClient,
    points: Sequence[Tuple[float, float]],
) -> List[Optional[float]]:
    """Sample ``terrain`` for ``points``, re-tagging availability errors (Req 7.11).

    Delegates to :meth:`TerrainSource.sample`. An availability failure
    (``NETWORK``/``UPSTREAM``) that the shared :class:`HttpClient` tagged with
    the request *host* is re-tagged with the logical terrain-source name so the
    raised error identifies the unavailable source; the call raises before any
    value is returned, so no partial result escapes. Other taxonomy categories
    (e.g. an authentication failure) pass through unchanged.
    """
    try:
        return await terrain.sample(http, points)
    except GeoError as exc:
        raise _identify_unavailable_source(exc, source_name=terrain.name)


def _identify_unavailable_source(exc: GeoError, *, source_name: str) -> GeoError:
    """Re-tag an availability error with the logical source name (Req 7.11)."""
    if exc.category not in _AVAILABILITY_CATEGORIES or exc.source == source_name:
        return exc
    cls = NetworkError if exc.category is ErrorCategory.NETWORK else UpstreamError
    return cls(
        exc.message,
        source=source_name,
        detail=exc.detail,
        original=exc.original,
        retry_after=exc.retry_after,
    )


async def elevation(
    *,
    location: Any,
    http: HttpClient,
    sources: Mapping[str, TerrainSource],
    source: str = DEFAULT_SOURCE,
) -> Union[float, RasterArray]:
    """Return elevation for a location or extent (Requirement 7.5).

    Validates ``location`` and ``source`` first (Requirement 7.12): an unknown
    ``source`` or a malformed location raises a
    :class:`~geo_common.errors.ValidationError` before any source is queried,
    so no partial result is produced. On a valid request the selected terrain
    source is sampled through the shared :class:`HttpClient` (inheriting its
    30-second per-request timeout, Requirement 7.5):

    * a :class:`Coordinate` yields a single scalar elevation in metres; and
    * a :class:`GeoWindow` yields a :class:`RasterArray` of sampled elevations.

    A source/transport failure propagates as a taxonomy-classified
    :class:`~geo_common.errors.GeoError` so no partial result is returned.
    """
    if not isinstance(source, str) or source not in sources:
        available = ", ".join(sorted(sources)) or "(none configured)"
        raise ValidationError(
            "unknown terrain source %r; configured sources: %s"
            % (source, available),
            source=_SERVER_NAME,
            detail={"parameter": "source", "available": sorted(sources)},
        )
    terrain = sources[source]
    parsed = parse_location(location)

    if isinstance(parsed, Coordinate):
        samples = await _sample(terrain, http, [(parsed.lon, parsed.lat)])
        value = samples[0] if samples else None
        if value is None:
            raise UpstreamError(
                "%s returned no elevation for the requested location" % source,
                source=source,
            )
        return value

    points = _grid_points(parsed)
    flat = await _sample(terrain, http, points)
    # Reshape the flat, row-major samples into a height × width grid.
    grid: List[List[Optional[float]]] = []
    for row in range(parsed.height):
        start = row * parsed.width
        grid.append(list(flat[start : start + parsed.width]))
    return RasterArray(
        values=grid,
        width=parsed.width,
        height=parsed.height,
        bbox=parsed.bbox,
        source=source,
    )
