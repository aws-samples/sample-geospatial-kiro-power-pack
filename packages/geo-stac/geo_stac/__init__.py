"""geo-stac: the Pillar A (MVP) STAC catalog-discovery MCP server.

``geo-stac`` wraps mature STAC API sources (Earth Search / Element 84, Microsoft
Planetary Computer, NASA CMR-STAC, Copernicus Data Space, USGS) behind the
Geospatial Power Pack conventions: it uses ``geo-common``'s single async
:class:`~geo_common.http.HttpClient` for outbound requests and the shared
``Error_Taxonomy`` for failures (Requirement 11).

The public surface mirrors design.md "Pillar A — Data Connectors":

* :class:`StacItem` - a returned catalog item with its asset references and
  spatio-temporal metadata (Requirement 7.1).
* :func:`stac_search` - search a STAC catalog by spatial + temporal parameters,
  returning matching items capped at the configured maximum of 1,000 and
  raising :class:`~geo_common.errors.ValidationError` on malformed parameters
  (Requirements 7.1, 7.12).
* :class:`GeoStacServer` - the :class:`~geo_common.server.BaseGeoServer`
  subclass that owns the shared HTTP client and registers ``stac_search`` as an
  MCP tool.
"""

from __future__ import annotations

from geo_stac.search import (
    DEFAULT_STAC_API_URL,
    KNOWN_STAC_ENDPOINTS,
    MAX_ITEMS,
    StacItem,
    StacSearchResult,
    StacSourceStatus,
    stac_search,
    stac_search_multi,
)
from geo_stac.server import GeoStacServer, main

__all__ = [
    "StacItem",
    "stac_search",
    "stac_search_multi",
    "StacSearchResult",
    "StacSourceStatus",
    "KNOWN_STAC_ENDPOINTS",
    "DEFAULT_STAC_API_URL",
    "MAX_ITEMS",
    "GeoStacServer",
    "main",
]
