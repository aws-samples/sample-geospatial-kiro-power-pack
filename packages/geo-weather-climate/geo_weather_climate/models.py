"""Data models for the ``geo-weather-climate`` server (Pillar A, expansion).

``observations`` (Requirement 7.6) returns a list of observation records for a
point :class:`Coordinate` and a ``(start, end)`` time range. The design types
the return value as ``list[dict]`` so each record faithfully carries whatever
variables a configured source reports; :class:`Observation` is a thin, typed
view used internally and is serialized back to a plain mapping at the tool
boundary via :meth:`Observation.as_dict`.

Python 3.10+: uses ``from __future__ import annotations`` together with
``typing`` generics so the pydantic models resolve their annotations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

__all__ = ["Coordinate", "Observation"]


class Coordinate(BaseModel):
    """A geographic point in EPSG:4326 decimal degrees.

    ``lon`` (longitude) must lie in ``[-180, 180]`` and ``lat`` (latitude) in
    ``[-90, 90]``; out-of-range or non-finite values are rejected by
    :func:`geo_weather_climate.timerange.validate_location` with an
    ``Error_Taxonomy`` validation error (Requirement 7.12).
    """

    lon: float
    lat: float


class Observation(BaseModel):
    """A single weather/climate observation at a timestamp (Requirement 7.6).

    ``time`` is the observation's ISO-8601 timestamp, ``source`` names the
    originating dataset (e.g. ``"open-meteo"``), and ``values`` carries the
    reported variables (temperature, precipitation, PM2.5, ...). Returned to
    callers as a plain mapping via :meth:`as_dict` so the tool's response
    matches the design's ``list[dict]`` contract.
    """

    time: str
    source: str
    values: Dict[str, Any] = Field(default_factory=dict)
    lon: Optional[float] = None
    lat: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        """Flatten the observation into a single secret-free mapping."""
        record: Dict[str, Any] = {"time": self.time, "source": self.source}
        if self.lon is not None:
            record["lon"] = self.lon
        if self.lat is not None:
            record["lat"] = self.lat
        record.update(self.values)
        return record
