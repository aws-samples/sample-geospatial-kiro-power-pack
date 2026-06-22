"""Auto-wiring of the concrete warehouse engines from the environment.

:func:`default_warehouse_engines` is what the server's ``main()`` uses to decide
which engines are configured. Each concrete engine module exposes a
``*_engine_if_configured()`` gate that returns its engine when its ``mcp.json``
credential is present (or, for BigQuery, when ADC is requested) and ``None``
otherwise. This module merges those gates into the keyed engine set the
:class:`~geo_warehouse.server.GeoWarehouseServer` injects.

An engine that is not configured is simply absent, so the server denies
``warehouse_spatial_sql`` for it with an ``AuthenticationError`` naming its
``mcp.json`` key - the documented healthy refusal. Drivers are imported lazily
at query time, so a configured-but-driver-missing engine surfaces an actionable
``UpstreamError`` pointing at the relevant ``geo-warehouse[...]`` extra.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from geo_warehouse.bigquery_engine import bigquery_engine_if_configured
from geo_warehouse.databricks_engine import databricks_engine_if_configured
from geo_warehouse.engine import WarehouseEngine
from geo_warehouse.redshift_engine import redshift_engine_if_configured
from geo_warehouse.snowflake_engine import snowflake_engine_if_configured

__all__ = ["default_warehouse_engines"]

#: The per-engine gates, in declaration order. Each returns an engine when its
#: credential is configured, else ``None``.
_ENGINE_GATES: "tuple[Callable[[], Optional[WarehouseEngine]], ...]" = (
    bigquery_engine_if_configured,
    redshift_engine_if_configured,
    snowflake_engine_if_configured,
    databricks_engine_if_configured,
)


def default_warehouse_engines() -> "Dict[str, WarehouseEngine]":
    """Return the keyed set of engines configured via the environment.

    Built engines: BigQuery, Redshift, Snowflake, Databricks (each wired when its
    credential is set). The result is keyed by each engine's ``name`` (a
    ``SUPPORTED_ENGINES`` key).
    """
    engines: "Dict[str, WarehouseEngine]" = {}
    for gate in _ENGINE_GATES:
        engine = gate()
        if engine is not None:
            engines[engine.name] = engine
    return engines
