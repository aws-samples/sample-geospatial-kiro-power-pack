"""Data models for the ``geo-biodiversity`` server (Pillar A, expansion).

``species_occurrences`` returns a list of occurrence records (Requirement 7.7).
The design signature types each record as a plain ``dict`` so the connector can
faithfully pass through whatever attributes a biodiversity source returns. This
module adds :class:`OccurrenceRecord` as a small, RFC-friendly *shape* helper:
it normalizes the handful of fields the connector always sets (a stable id, the
scientific name, the point coordinate, the observation date, and the source
label) while keeping the source's remaining fields under ``properties``.

The connector returns ``record.as_dict()`` for each record, honoring the
design's ``list[dict]`` return type while giving the parsing code one typed
place to build a record.

Python 3.10+ : uses ``from __future__ import annotations`` together with
``typing`` generics so the pydantic models resolve their annotations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

__all__ = ["BBox", "OccurrenceRecord"]

#: A geographic bounding box as ``(min_lon, min_lat, max_lon, max_lat)`` in
#: EPSG:4326 decimal degrees - the order used throughout the connector.
BBox = Tuple[float, float, float, float]


class OccurrenceRecord(BaseModel):
    """A single species-occurrence record (Requirement 7.7).

    ``longitude`` / ``latitude`` are the record's point location in EPSG:4326
    decimal degrees when the source provides one (``None`` for a record with no
    coordinate). ``properties`` carries the source's remaining attributes; the
    connector adds a ``"source"`` key identifying the originating dataset
    (``"gbif"`` / ``"inaturalist"`` / ...) so merged results stay attributable.
    """

    id: Optional[str] = None
    scientific_name: Optional[str] = None
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    event_date: Optional[str] = None
    source: str
    properties: Dict[str, Any] = Field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return the record as a plain ``dict`` (the connector's wire type)."""
        return self.model_dump()
