"""geo-warehouse: credentialed warehouse-scale spatial SQL (first expansion).

Exposes ``warehouse_spatial_sql`` (Requirement 8.6), executing spatial SQL
against one of four proprietary warehouse engines - BigQuery, Snowflake,
Redshift, Databricks - behind an injectable
:class:`~geo_warehouse.engine.WarehouseEngine` interface. Each engine declares a
proprietary (License-Needed) credential spec and registers a Proprietary
Resource Catalog entry; failures map onto the shared ``Error_Taxonomy``.
"""

from __future__ import annotations

from geo_warehouse.bigquery_engine import (
    BIGQUERY_CREDENTIALS_KEY,
    BIGQUERY_USE_ADC_KEY,
    BigQueryEngine,
)
from geo_warehouse.databricks_engine import (
    DATABRICKS_CONNECTION_KEY,
    DatabricksEngine,
)
from geo_warehouse.redshift_engine import REDSHIFT_CONNECTION_KEY, RedshiftEngine
from geo_warehouse.snowflake_engine import (
    SNOWFLAKE_CONNECTION_KEY,
    SnowflakeEngine,
)
from geo_warehouse.wiring import default_warehouse_engines
from geo_warehouse.engine import (
    SUPPORTED_ENGINES,
    EngineSpec,
    WarehouseEngine,
    supported_engine_names,
)
from geo_warehouse.models import ResultSet
from geo_warehouse.server import INSTALL_COMMAND, GeoWarehouseServer, main

__all__ = [
    "ResultSet",
    "WarehouseEngine",
    "EngineSpec",
    "SUPPORTED_ENGINES",
    "supported_engine_names",
    "BigQueryEngine",
    "BIGQUERY_CREDENTIALS_KEY",
    "BIGQUERY_USE_ADC_KEY",
    "RedshiftEngine",
    "REDSHIFT_CONNECTION_KEY",
    "SnowflakeEngine",
    "SNOWFLAKE_CONNECTION_KEY",
    "DatabricksEngine",
    "DATABRICKS_CONNECTION_KEY",
    "default_warehouse_engines",
    "GeoWarehouseServer",
    "INSTALL_COMMAND",
    "main",
]
