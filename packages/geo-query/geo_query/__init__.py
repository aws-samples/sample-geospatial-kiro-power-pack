"""geo-query: spatial SQL execution over data in place (Pillar B, expansion).

Exposes ``spatial_sql`` (Requirement 8.6), executing a spatial SQL query
against one of the configured query engines - DuckDB Spatial (open, in-process)
or Amazon Athena (ad-hoc SQL over data in S3) - behind an injectable
:class:`~geo_query.engine.QueryEngine` interface, and returning the result set.

Each engine is described by a :class:`~geo_query.engine.EngineSpec` carrying its
openness tier and (for credentialed engines such as Athena) the ``mcp.json``
key it reads. Engines register a Resource Catalog entry and any failure maps
onto the shared ``Error_Taxonomy``.
"""

from __future__ import annotations

from geo_query.engine import (
    SUPPORTED_ENGINES,
    DEFAULT_ENGINE,
    EngineSpec,
    QueryEngine,
    supported_engine_names,
)
from geo_query.athena_engine import (
    ATHENA_S3_STAGING_DIR_KEY,
    AthenaEngine,
    default_athena_engines,
)
from geo_query.duckdb_engine import DuckDBEngine, default_query_engines
from geo_query.models import ResultSet
from geo_query.server import INSTALL_COMMAND, GeoQueryServer, main

__all__ = [
    "ResultSet",
    "QueryEngine",
    "EngineSpec",
    "SUPPORTED_ENGINES",
    "DEFAULT_ENGINE",
    "supported_engine_names",
    "DuckDBEngine",
    "default_query_engines",
    "AthenaEngine",
    "ATHENA_S3_STAGING_DIR_KEY",
    "default_athena_engines",
    "GeoQueryServer",
    "INSTALL_COMMAND",
    "main",
]
