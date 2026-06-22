"""Data models for the ``geo-geocode-route`` server (Pillar A, expansion).

Two tools are exposed (design.md "Pillar A — Data Connectors"):

* ``geocode`` turns an address into a :class:`Coordinate` (Requirement 7.4).
* ``route`` turns an origin/destination :class:`Coordinate` pair into a
  :class:`Route` (Requirement 7.10).

:class:`Coordinate` is the shared currency between the two tools - ``geocode``
returns one and ``route`` consumes two - so it is deliberately minimal:
longitude/latitude in EPSG:4326 decimal degrees, the same order GeoJSON uses
for positions (``[lon, lat]``, RFC 7946 §3.1.1).

Python 3.9 compatibility: this module uses ``from __future__ import
annotations`` together with ``typing`` generics so the pydantic models resolve
their annotations on 3.9+.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

__all__ = ["Coordinate", "Route"]


class Coordinate(BaseModel):
    """A geographic point in EPSG:4326 decimal degrees.

    ``lon`` is the longitude in ``[-180, 180]`` and ``lat`` the latitude in
    ``[-90, 90]``. Returned by ``geocode`` (Requirement 7.4) and consumed as
    the ``origin``/``destination`` of ``route`` (Requirement 7.10). The
    ``label`` carries the human-readable place name the geocoder matched (when
    a source provides one) so a geocode result stays attributable without
    changing the point's geometry.
    """

    lon: float
    lat: float
    label: Optional[str] = None
    source: Optional[str] = None

    def as_position(self) -> List[float]:
        """Return the point as a GeoJSON ``[lon, lat]`` position."""
        return [self.lon, self.lat]


class Route(BaseModel):
    """A computed route between two coordinates (Requirement 7.10).

    ``geometry`` is a GeoJSON ``LineString`` mapping
    (``{"type": "LineString", "coordinates": [[lon, lat], ...]}``) tracing the
    route; ``distance_m`` and ``duration_s`` are the route's total length in
    metres and estimated travel time in seconds; ``profile`` echoes the
    requested travel mode; and ``source`` names the routing engine that
    produced the result so a route stays attributable.
    """

    distance_m: float
    duration_s: float
    geometry: Dict[str, Any] = Field(default_factory=dict)
    profile: str = "car"
    source: Optional[str] = None
    origin: Optional[Coordinate] = None
    destination: Optional[Coordinate] = None
