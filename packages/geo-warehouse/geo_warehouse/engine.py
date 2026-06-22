"""The injectable warehouse-engine interface and the proprietary engine catalog.

``geo-warehouse`` executes spatial SQL against one of four proprietary
warehouse engines - BigQuery, Snowflake, Redshift, and Databricks
(Requirement 8.6; design "geo-warehouse - first expansion"). The real engines
require proprietary client drivers and licensed accounts, so this module keeps
the *execution* concern behind a small, injectable interface:

* :class:`WarehouseEngine` - the abstract contract a concrete engine
  implements. It carries a stable :attr:`~WarehouseEngine.name` and a single
  :meth:`~WarehouseEngine.execute` method returning a
  :class:`~geo_warehouse.models.ResultSet`. Making this an injectable
  abstraction means the server is exercisable end-to-end with a mock engine in
  tests, with no proprietary driver installed.
* :class:`EngineSpec` - the static description of a supported engine: its
  taxonomy ``key`` (the engine selector), a human ``source`` label, the
  ``mcp.json`` credential key it reads, and the ``license_reference`` that
  identifies its licensing obligation (proprietary / License-Needed - Req 3.7,
  10.4).
* :data:`SUPPORTED_ENGINES` - the registry mapping each engine key to its
  :class:`EngineSpec`, the single source of truth used for both validation
  (which engine names are accepted) and credential declaration.

Python 3.10+: uses ``from __future__ import annotations`` together with
``typing`` generics.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, List

from geo_warehouse.models import ResultSet

__all__ = [
    "WarehouseEngine",
    "EngineSpec",
    "SUPPORTED_ENGINES",
    "supported_engine_names",
]


@dataclass(frozen=True)
class EngineSpec:
    """Static description of a supported proprietary warehouse engine.

    ``key`` is the value callers pass as ``engine`` to ``warehouse_spatial_sql``;
    ``source`` is the human-readable provider; ``mcp_json_key`` is the single
    ``mcp.json`` configuration key holding that engine's connection credential;
    and ``license_reference`` identifies the licensing requirement reported for
    this proprietary source (Req 3.7, 10.4).
    """

    key: str
    source: str
    mcp_json_key: str
    license_reference: str


#: The four proprietary warehouse engines geo-warehouse can target (Req 8.6).
#: This registry is the single source of truth for accepted engine names and
#: for the per-engine License-Needed credential declarations.
SUPPORTED_ENGINES: "Dict[str, EngineSpec]" = {
    spec.key: spec
    for spec in (
        EngineSpec(
            key="bigquery",
            source="Google BigQuery (GIS)",
            mcp_json_key="BIGQUERY_CREDENTIALS",
            license_reference="Google Cloud Platform Terms of Service (BigQuery)",
        ),
        EngineSpec(
            key="snowflake",
            source="Snowflake (Geospatial)",
            mcp_json_key="SNOWFLAKE_CONNECTION",
            license_reference="Snowflake Master Subscription Agreement",
        ),
        EngineSpec(
            key="redshift",
            source="Amazon Redshift (Spatial)",
            mcp_json_key="REDSHIFT_CONNECTION",
            license_reference="AWS Customer Agreement (Amazon Redshift)",
        ),
        EngineSpec(
            key="databricks",
            source="Databricks (Mosaic / Spatial SQL)",
            mcp_json_key="DATABRICKS_CONNECTION",
            license_reference="Databricks Master Cloud Services Agreement",
        ),
    )
}


def supported_engine_names() -> "List[str]":
    """Return the accepted engine keys, in declaration order."""
    return list(SUPPORTED_ENGINES.keys())


class WarehouseEngine(abc.ABC):
    """The injectable contract a concrete warehouse engine implements.

    Concrete subclasses wrap a proprietary client driver (BigQuery, Snowflake,
    Redshift, Databricks) and translate a spatial SQL ``query`` executed
    over a ``connection`` into a :class:`~geo_warehouse.models.ResultSet`. The
    server depends only on this abstraction, so tests inject a mock engine and
    no proprietary driver is required to exercise the full path.

    Implementations should raise the most specific
    :class:`~geo_common.errors.GeoError` they can (e.g. ``AuthenticationError``
    for a rejected credential); any other exception is mapped onto the
    ``Error_Taxonomy`` by the server's :meth:`~geo_common.server.BaseGeoServer.map_error`.
    """

    #: The engine key this implementation handles (must be a SUPPORTED_ENGINES key).
    name: str = ""

    @abc.abstractmethod
    def execute(self, *, query: str, connection: str) -> ResultSet:
        """Execute ``query`` over ``connection`` and return its result set."""
        raise NotImplementedError
