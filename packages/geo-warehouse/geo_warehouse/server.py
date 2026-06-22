"""The ``geo-warehouse`` MCP server (first expansion, Pillar B-class).

``geo-warehouse`` exposes one tool, ``warehouse_spatial_sql``, that runs a
spatial SQL query against a configured proprietary warehouse engine - BigQuery,
Snowflake, Redshift, or Databricks - and returns the result set
(Requirement 8.6). It wires the injectable :mod:`geo_warehouse.engine`
abstraction into the shared :class:`~geo_common.server.BaseGeoServer` contract
so that:

* execution is delegated to an injected :class:`~geo_warehouse.engine.WarehouseEngine`,
  keeping the server testable with a mock and free of proprietary drivers;
* each supported engine declares a proprietary, **License-Needed** credential
  spec naming its ``mcp.json`` key and licensing reference (Req 3.7, 10.4,
  16.1); License-Needed keys never block startup (Req 16.5), so installing the
  server and using a single engine needs only that engine's credential;
* each engine registers a Resource Catalog entry under the Proprietary/Licensed
  openness tier, naming ``geo-warehouse`` as the provider (Req 2.1, 11.3); and
* every library/driver/transport failure maps onto exactly one
  ``Error_Taxonomy`` category via the inherited
  :meth:`~geo_common.server.BaseGeoServer.map_error` (Req 11.2, 11.5).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from geo_common.errors import AuthenticationError, ValidationError
from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from geo_common.server import BaseGeoServer

from geo_warehouse.engine import SUPPORTED_ENGINES, WarehouseEngine
from geo_warehouse.models import ResultSet

__all__ = ["GeoWarehouseServer", "INSTALL_COMMAND", "main"]

#: ``uvx`` command that installs this server (Req 2.6, 16.8; bundle-manifest).
INSTALL_COMMAND = "uvx geo-warehouse"


class GeoWarehouseServer(BaseGeoServer):
    """Credentialed warehouse-scale spatial SQL server (first expansion, Req 8.6).

    Holds the set of injectable :class:`~geo_warehouse.engine.WarehouseEngine`
    implementations keyed by engine name and registers ``warehouse_spatial_sql``
    as an MCP tool. Engines are injected (``engines=...``) so the full path is
    testable with a mock and no proprietary driver is needed; an engine that is
    not injected is treated as unconfigured at call time.
    """

    pillar = "B"
    server_name = "geo-warehouse"
    version = "0.2.0"

    def __init__(
        self,
        *,
        engines: Optional[Mapping[str, WarehouseEngine]] = None,
        http=None,
    ) -> None:
        super().__init__(http=http)
        # Only engines that are actually supported can be registered; this
        # guards against typo'd keys leaking into the configured set.
        self.engines: "Dict[str, WarehouseEngine]" = {}
        for name, engine in (engines or {}).items():
            if name not in SUPPORTED_ENGINES:
                raise ValueError(
                    "unknown warehouse engine %r; supported: %s"
                    % (name, ", ".join(SUPPORTED_ENGINES))
                )
            self.engines[name] = engine
        self.register_tool("warehouse_spatial_sql", self.warehouse_spatial_sql)

    # ------------------------------------------------------------------
    # The tool (Requirement 8.6)
    # ------------------------------------------------------------------

    async def warehouse_spatial_sql(
        self, *, query: str, engine: str, connection: str
    ) -> ResultSet:
        """Execute a spatial SQL ``query`` against ``engine`` and return its rows.

        Validates inputs before doing any work: a blank ``query`` or
        ``connection``, or an ``engine`` that is not one of the supported
        warehouses (BigQuery, Snowflake, Redshift, Databricks), raises an
        ``Error_Taxonomy`` ``ValidationError`` and runs nothing (Req 8.6).

        The selected engine is a proprietary, credentialed source: if its
        connection credential is not configured (no engine injected for that
        key), the invocation is denied with an ``AuthenticationError`` naming
        the missing ``mcp.json`` key, never the secret value itself (Req 3.6,
        3.8, 10.5).

        Any failure raised by the underlying engine driver is routed through
        :meth:`map_error`, so it surfaces as exactly one ``Error_Taxonomy``
        category (an already-classified :class:`~geo_common.errors.GeoError`
        passes through unchanged; an unmapped driver error becomes ``UPSTREAM``
        with its original detail retained - Req 11.2, 11.5).
        """
        spec = self._validate_request(query=query, engine=engine, connection=connection)

        engine_impl = self.engines.get(engine)
        if engine_impl is None:
            # Known engine, but no credential/driver configured for it: deny the
            # invocation and name the mcp.json key (secret value never exposed).
            raise AuthenticationError(
                "%s is not configured: set its credential in mcp.json"
                % spec.source,
                source=self.server_name,
                detail={"engine": spec.key, "mcp_json_key": spec.mcp_json_key},
            )

        try:
            return engine_impl.execute(query=query, connection=connection)
        except Exception as exc:  # noqa: BLE001 - mapped onto the taxonomy
            raise self.map_error(exc, source="%s:%s" % (self.server_name, spec.key))

    def _validate_request(self, *, query: str, engine: str, connection: str):
        """Validate inputs and return the resolved :class:`EngineSpec`.

        Raises an ``Error_Taxonomy`` ``ValidationError`` identifying the invalid
        parameter when the query/connection is blank or the engine is unknown.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValidationError(
                "query must be a non-empty spatial SQL string",
                source=self.server_name,
                detail={"parameter": "query"},
            )
        if not isinstance(connection, str) or not connection.strip():
            raise ValidationError(
                "connection must be a non-empty connection identifier",
                source=self.server_name,
                detail={"parameter": "connection"},
            )
        spec = SUPPORTED_ENGINES.get(engine)
        if spec is None:
            raise ValidationError(
                "unsupported engine %r; supported engines are: %s"
                % (engine, ", ".join(SUPPORTED_ENGINES)),
                source=self.server_name,
                detail={"parameter": "engine", "supported": list(SUPPORTED_ENGINES)},
            )
        return spec

    # ------------------------------------------------------------------
    # Catalog + credential declaration (Req 2.1, 11.3, 16.1)
    # ------------------------------------------------------------------

    def catalog_entries(self) -> "List[CatalogEntry]":
        """Register one Proprietary catalog entry per supported engine (Req 2.1, 11.3).

        Each supported warehouse engine is a distinct, separately-licensed
        capability, so each gets its own
        :class:`~geo_common.models.CatalogEntry` under the
        :attr:`~geo_common.models.OpennessTier.PROPRIETARY` tier, naming
        ``geo-warehouse`` as the provider (Req 11.3). ``installed`` reflects
        whether that engine currently has an injected/configured driver;
        entries that are not yet configured carry the install command (Req 2.6).
        """
        entries: "List[CatalogEntry]" = []
        for spec in SUPPORTED_ENGINES.values():
            configured = spec.key in self.engines
            entries.append(
                CatalogEntry(
                    name="warehouse_spatial_sql:%s" % spec.key,
                    pillar=self.pillar,
                    capability_description=(
                        "Execute warehouse-scale spatial SQL against %s and "
                        "return the result set." % spec.source
                    ),
                    openness_tier=OpennessTier.PROPRIETARY,
                    provider_server=self.server_name,
                    installed=configured,
                    install_command=None if configured else INSTALL_COMMAND,
                )
            )
        return entries

    def required_credentials(self) -> "List[CredentialSpec]":
        """Declare each engine's proprietary connection credential (Req 16.1).

        One :class:`~geo_common.models.CredentialSpec` per supported engine,
        classified :attr:`~geo_common.models.CredentialClassification.LICENSE_NEEDED`
        and carrying the licensing reference for that proprietary source (Req
        3.7, 10.4). Because they are License-Needed rather than Required, an
        absent key never blocks startup (Req 16.5): a user installs the server
        and configures only the engine(s) they license.
        """
        return [
            CredentialSpec(
                source=spec.source,
                mcp_json_key=spec.mcp_json_key,
                classification=CredentialClassification.LICENSE_NEEDED,
                license_reference=spec.license_reference,
            )
            for spec in SUPPORTED_ENGINES.values()
        ]


def main() -> None:
    """Console entry point: serve geo-warehouse over MCP stdio.

    Wires the auto-configured warehouse engines (BigQuery when
    ``BIGQUERY_CREDENTIALS`` is set; the other engines remain inject-only), runs
    the startup credential guard, then serves this server's registered tools
    over stdin/stdout via the shared geo-common MCP runtime until the client
    disconnects.
    """
    from geo_warehouse.wiring import default_warehouse_engines

    GeoWarehouseServer(engines=default_warehouse_engines()).run()
