"""geo-biodiversity: Pillar A (expansion) species-occurrence MCP server.

Exposes ``species_occurrences``, which returns species-occurrence records from
biodiversity sources (GBIF by default) for a bounding box, optionally filtered
by taxon, capped at a configurable maximum (default 10,000 records), and rejects
malformed parameters with an ``Error_Taxonomy`` validation error (Requirements
7.7, 7.12).
"""

from __future__ import annotations

from geo_biodiversity.models import BBox, OccurrenceRecord
from geo_biodiversity.occurrences import (
    BiodiversitySource,
    GBIFSource,
    INaturalistSource,
    IUCNSource,
    DEFAULT_GBIF_URL,
    DEFAULT_INATURALIST_URL,
    DEFAULT_IUCN_URL,
    default_sources,
    species_occurrences,
)
from geo_biodiversity.server import GeoBiodiversityServer, INSTALL_COMMAND
from geo_biodiversity.validation import (
    DEFAULT_MAX_RECORDS,
    validate_and_cap_limit,
    validate_bbox,
    validate_taxon,
)

__all__ = [
    # Data models
    "BBox",
    "OccurrenceRecord",
    # Validation
    "DEFAULT_MAX_RECORDS",
    "validate_bbox",
    "validate_and_cap_limit",
    "validate_taxon",
    # Sources + connector
    "BiodiversitySource",
    "GBIFSource",
    "INaturalistSource",
    "IUCNSource",
    "DEFAULT_GBIF_URL",
    "DEFAULT_INATURALIST_URL",
    "DEFAULT_IUCN_URL",
    "default_sources",
    "species_occurrences",
    # Server
    "GeoBiodiversityServer",
    "INSTALL_COMMAND",
]
