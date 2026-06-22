"""Data models for the ``geo-ogc`` server (Pillar A).

``geo-ogc`` connects to **OGC API - Features** services (pygeoapi, GeoServer's
OGC API, ldproxy, …) and returns the features for a spatial extent. The single
result type is :class:`FeatureCollectionResult`, a thin envelope over the
GeoJSON ``FeatureCollection`` the service returns, plus provenance (which
endpoint/collection produced it) and the OGC paging counts.

Python 3.10+: uses ``from __future__ import annotations`` with ``typing``
generics so the pydantic model resolves its annotations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

__all__ = ["FeatureCollectionResult"]


class FeatureCollectionResult(BaseModel):
    """A GeoJSON ``FeatureCollection`` returned by an OGC API - Features query.

    ``features`` are the raw GeoJSON features (each with ``geometry`` and
    ``properties``) as the service returned them. ``number_returned`` is how
    many features this response carries and ``number_matched`` is the total the
    service reports matching the query (``None`` when the service does not
    advertise it). ``endpoint`` and ``collection`` record the provenance so a
    caller knows which OGC service and collection produced the result.
    """

    type: str = "FeatureCollection"
    features: List[Dict[str, Any]] = Field(default_factory=list)
    number_returned: int = Field(default=0, ge=0)
    number_matched: Optional[int] = None
    endpoint: str
    collection: str
