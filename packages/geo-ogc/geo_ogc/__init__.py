"""geo-ogc: Pillar A MCP server bridging OGC API - Features services.

Exposes ``ogc_features``, which fetches the GeoJSON features for a spatial
extent from any conformant OGC API - Features service (pygeoapi, GeoServer's
OGC API, ldproxy, …) named per call, and rejects malformed parameters with an
``Error_Taxonomy`` validation error before any network call.
"""

from __future__ import annotations

from geo_ogc.features import (
    DEFAULT_FEATURES,
    MAX_FEATURES,
    ogc_features,
)
from geo_ogc.models import FeatureCollectionResult
from geo_ogc.server import GeoOgcServer, INSTALL_COMMAND, main

__all__ = [
    "FeatureCollectionResult",
    "ogc_features",
    "DEFAULT_FEATURES",
    "MAX_FEATURES",
    "GeoOgcServer",
    "INSTALL_COMMAND",
    "main",
]
