"""The ``geo-query`` MCP server (Pillar B, expansion).

``geo-query`` exposes one tool, ``spatial_sql``, that runs a spatial SQL query
against a configured engine - DuckDB Spatial (open, in-process) or Amazon
Athena (ad-hoc SQL over data in S3) - *over data in place* and returns the
result set (Requirement 8.6). It wires the injectable :mod:`geo_query.engine`
abstraction into the shared :class:`~geo_common.server.BaseGeoServer` contract
so that:

* execution is delegated to an injected :class:`~geo_query.engine.QueryEngine`,
  keeping the server testable with a mock and free of engine drivers;
* the open DuckDB engine carries no credential, while the credentialed Athena
  engine declares an Optional ``mcp.json`` key (so it never blocks startup -
  Req 16.5) and its catalog entry is classified at its openness tier;
* each engine registers a Resource Catalog entry naming ``geo-query`` as the
  provider (Req 2.1, 11.3); and
* every library/driver/transport failure maps onto exactly one
  ``Error_Taxonomy`` category via the inherited
  :meth:`~geo_common.server.BaseGeoServer.map_error` (Req 11.2, 11.5).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from geo_common.errors import AuthenticationError, UpstreamError, ValidationError
from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
)
from geo_common.server import BaseGeoServer

from geo_query.engine import DEFAULT_ENGINE, SUPPORTED_ENGINES, QueryEngine
from geo_query.models import ResultSet

__all__ = ["GeoQueryServer", "INSTALL_COMMAND", "main"]

#: ``uvx`` command that installs this server (Req 2.6; bundle-manifest).
INSTALL_COMMAND = "uvx geo-query"


class GeoQueryServer(BaseGeoServer):
    """Spatial SQL execution server (Pillar B expansion, Req 8.6).

    Holds the set of injectable :class:`~geo_query.engine.QueryEngine`
    implementations keyed by engine name and registers ``spatial_sql`` as an MCP
    tool. Engines are injected (``engines=...``) so the full path is testable
    with a mock and no driver is needed; an engine that is not injected is
    treated as unconfigured at call time.
    """

    pillar = "B"
    server_name = "geo-query"
    version = "0.2.0"

    def __init__(
        self,
        *,
        engines: Optional[Mapping[str, QueryEngine]] = None,
        http=None,
    ) -> None:
        super().__init__(http=http)
        # Only engines that are actually supported can be registered; this
        # guards against typo'd keys leaking into the configured set.
        self.engines: "Dict[str, QueryEngine]" = {}
        for name, engine in (engines or {}).items():
            if name not in SUPPORTED_ENGINES:
                raise ValueError(
                    "unknown query engine %r; supported: %s"
                    % (name, ", ".join(SUPPORTED_ENGINES))
                )
            self.engines[name] = engine
        self.register_tool("spatial_sql", self.spatial_sql)

    # ------------------------------------------------------------------
    # The tool (Requirement 8.6)
    # ------------------------------------------------------------------

    async def spatial_sql(
        self,
        *,
        query: str,
        engine: str = DEFAULT_ENGINE,
        sources: Optional[Mapping[str, str]] = None,
    ) -> ResultSet:
        """Execute a spatial SQL ``query`` against ``engine`` and return its rows.

        Validates inputs before doing any work: a blank ``query``, an ``engine``
        that is not one of the supported engines (DuckDB Spatial, Amazon
        Athena), or a malformed ``sources`` mapping (a non-string/blank logical
        name or data location) raises an ``Error_Taxonomy`` ``ValidationError``
        and runs nothing (Req 8.6). When ``engine`` is omitted the default
        engine (DuckDB) is used.

        If the selected engine is not configured, the invocation is denied:
        a credentialed engine (Athena) raises an ``AuthenticationError`` naming
        the missing ``mcp.json`` key (never the secret value - Req 3.8), while
        the open DuckDB engine raises an ``UpstreamError`` reporting that its
        runtime is not configured.

        Any failure raised by the underlying engine driver is routed through
        :meth:`map_error`, so it surfaces as exactly one ``Error_Taxonomy``
        category (an already-classified :class:`~geo_common.errors.GeoError`
        passes through unchanged; an unmapped driver error becomes ``UPSTREAM``
        with its original detail retained - Req 11.2, 11.5).
        """
        spec = self._validate_request(query=query, engine=engine, sources=sources)

        engine_impl = self.engines.get(engine)
        if engine_impl is None:
            if spec.requires_credential:
                # Credentialed engine, not configured: deny the invocation and
                # name the mcp.json key (the secret value is never exposed).
                raise AuthenticationError(
                    "%s is not configured: set %s in mcp.json"
                    % (spec.source, spec.mcp_json_key),
                    source=self.server_name,
                    detail={"engine": spec.key, "mcp_json_key": spec.mcp_json_key},
                )
            # Open engine (DuckDB) whose runtime is not wired in this deployment.
            raise UpstreamError(
                "%s engine is not configured/available" % spec.source,
                source=self.server_name,
                detail={"engine": spec.key},
            )

        try:
            return engine_impl.execute(query=query, sources=sources)
        except Exception as exc:  # noqa: BLE001 - mapped onto the taxonomy
            raise self.map_error(exc, source="%s:%s" % (self.server_name, spec.key))

    def _validate_request(
        self, *, query: str, engine: str, sources: Optional[Mapping[str, str]]
    ):
        """Validate inputs and return the resolved :class:`EngineSpec`.

        Raises an ``Error_Taxonomy`` ``ValidationError`` identifying the invalid
        parameter when the query is blank, the engine is unknown, or the
        ``sources`` mapping contains a blank/non-string name or location.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValidationError(
                "query must be a non-empty spatial SQL string",
                source=self.server_name,
                detail={"parameter": "query"},
            )
        spec = SUPPORTED_ENGINES.get(engine)
        if spec is None:
            raise ValidationError(
                "unsupported engine %r; supported engines are: %s"
                % (engine, ", ".join(SUPPORTED_ENGINES)),
                source=self.server_name,
                detail={"parameter": "engine", "supported": list(SUPPORTED_ENGINES)},
            )
        if sources is not None:
            if not isinstance(sources, Mapping):
                raise ValidationError(
                    "sources must be a mapping of logical name to data location",
                    source=self.server_name,
                    detail={"parameter": "sources"},
                )
            for name, location in sources.items():
                if not isinstance(name, str) or not name.strip():
                    raise ValidationError(
                        "each sources key must be a non-empty logical name",
                        source=self.server_name,
                        detail={"parameter": "sources"},
                    )
                if not isinstance(location, str) or not location.strip():
                    raise ValidationError(
                        "each sources value must be a non-empty data location",
                        source=self.server_name,
                        detail={"parameter": "sources", "name": name},
                    )
        return spec

    # ------------------------------------------------------------------
    # Catalog + credential declaration (Req 2.1, 11.3, 16.1)
    # ------------------------------------------------------------------

    def catalog_entries(self) -> "List[CatalogEntry]":
        """Register one catalog entry per supported engine (Req 2.1, 11.3).

        Each engine is a distinct way to run spatial SQL, so each gets its own
        :class:`~geo_common.models.CatalogEntry` at the engine's openness tier
        (DuckDB Open, Athena Free-Tier), naming ``geo-query`` as the provider
        (Req 11.3). ``installed`` reflects whether that engine currently has an
        injected/configured driver; entries that are not yet configured carry
        the install command (Req 2.6).
        """
        entries: "List[CatalogEntry]" = []
        for spec in SUPPORTED_ENGINES.values():
            configured = spec.key in self.engines
            entries.append(
                CatalogEntry(
                    name="spatial_sql:%s" % spec.key,
                    pillar=self.pillar,
                    capability_description=(
                        "Execute spatial SQL over data in place against %s and "
                        "return the result set." % spec.source
                    ),
                    openness_tier=spec.openness_tier,
                    provider_server=self.server_name,
                    installed=configured,
                    install_command=None if configured else INSTALL_COMMAND,
                )
            )
        return entries

    def required_credentials(self) -> "List[CredentialSpec]":
        """Declare every credential the credentialed engines read (Req 16.1).

        Only the credentialed engine (Athena) declares
        :class:`~geo_common.models.CredentialSpec` entries; the open, in-process
        DuckDB engine needs none. Each Athena ``mcp.json`` key is classified
        :attr:`~geo_common.models.CredentialClassification.OPTIONAL` so an
        absent key never blocks startup (Req 16.5): a user installs the server
        and runs DuckDB queries with no configuration, configuring AWS only when
        they want Athena. The declared keys agree exactly with the ``geo-query``
        credential block in ``bundle-manifest.json``.
        """
        return [
            CredentialSpec(
                source=source,
                mcp_json_key=key,
                classification=CredentialClassification.OPTIONAL,
            )
            for spec in SUPPORTED_ENGINES.values()
            if spec.requires_credential
            for source, key in spec.credentials
        ]


def main() -> None:
    """Console entry point: serve geo-query over MCP stdio.

    Wires the open, in-process DuckDB Spatial engine when the ``[duckdb]`` extra
    is installed (via :func:`~geo_query.duckdb_engine.default_query_engines`) and
    the Amazon Athena engine when ``AWS_ACCESS_KEY_ID`` is configured (via
    :func:`~geo_query.athena_engine.default_athena_engines`), so a deployed
    ``geo-query`` runs DuckDB out of the box with no credential and offers Athena
    once AWS is set up. Then runs the startup credential guard and serves this
    server's registered tools over stdin/stdout via the shared geo-common MCP
    runtime until the client disconnects.
    """
    from geo_query.athena_engine import default_athena_engines
    from geo_query.duckdb_engine import default_query_engines

    engines = {**default_query_engines(), **default_athena_engines()}
    GeoQueryServer(engines=engines).run()
