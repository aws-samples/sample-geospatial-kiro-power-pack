"""The injectable query-engine interface and the supported-engine registry.

``geo-query`` executes spatial SQL *over data in place* against one of two
configured engines (Requirement 8.6; design "geo-query - DuckDB Spatial /
Athena"):

* **DuckDB Spatial** - an open, in-process analytical engine; no credential.
* **Amazon Athena** - ad-hoc SQL over data in S3; reads AWS credentials from
  the single ``mcp.json`` surface.

Both real engines require driver libraries (and, for Athena, AWS access), so
this module keeps the *execution* concern behind a small, injectable interface:

* :class:`QueryEngine` - the abstract contract a concrete engine implements. It
  carries a stable :attr:`~QueryEngine.name` and a single
  :meth:`~QueryEngine.execute` method (taking the ``query`` and an optional
  ``sources`` mapping of logical name -> data location) returning a
  :class:`~geo_query.models.ResultSet`. Making this injectable means the server
  is exercisable end-to-end with a mock engine in tests, with no driver
  installed.
* :class:`EngineSpec` - the static description of a supported engine: its
  taxonomy ``key`` (the engine selector), a human ``source`` label, its
  :class:`~geo_common.models.OpennessTier`, whether it ``requires_credential``,
  and (for credentialed engines) the ``mcp_json_key`` it reads.
* :data:`SUPPORTED_ENGINES` - the registry mapping each engine key to its
  :class:`EngineSpec`; the single source of truth for which engine names are
  accepted, the catalog openness tier, and credential declaration.
* :data:`DEFAULT_ENGINE` - the engine used when the caller does not name one
  (``"duckdb"``; design default).

Python 3.9+: uses ``from __future__ import annotations`` together with
``typing`` generics.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from geo_common.models import OpennessTier

from geo_query.models import ResultSet

__all__ = [
    "QueryEngine",
    "EngineSpec",
    "SUPPORTED_ENGINES",
    "DEFAULT_ENGINE",
    "supported_engine_names",
]


@dataclass(frozen=True)
class EngineSpec:
    """Static description of a supported query engine.

    ``key`` is the value callers pass as ``engine`` to ``spatial_sql``;
    ``source`` is the human-readable provider; ``openness_tier`` classifies how
    open the engine is for the Resource Catalog; ``requires_credential`` marks a
    credentialed engine (Athena). ``mcp_json_key`` is the *primary* ``mcp.json``
    key named in the credential-deny message for an unconfigured credentialed
    engine, and ``credentials`` is the full set of ``(source, key)`` pairs the
    engine reads from ``mcp.json`` - declared in the Resource Catalog /
    credential manager and kept in exact agreement with the ``geo-query``
    credential block in ``bundle-manifest.json`` (Req 16.1). The open,
    in-process DuckDB engine reads no credential, so both are empty/``None``.
    """

    key: str
    source: str
    openness_tier: OpennessTier
    requires_credential: bool = False
    mcp_json_key: Optional[str] = None
    license_reference: Optional[str] = None
    credentials: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)


#: The query engines geo-query can target (Req 8.6). This registry is the single
#: source of truth for accepted engine names and per-engine catalog/credential
#: declarations.
SUPPORTED_ENGINES: "Dict[str, EngineSpec]" = {
    spec.key: spec
    for spec in (
        EngineSpec(
            key="duckdb",
            source="DuckDB Spatial",
            openness_tier=OpennessTier.OPEN,
            requires_credential=False,
        ),
        EngineSpec(
            key="athena",
            source="Amazon Athena",
            openness_tier=OpennessTier.FREE_TIER,
            requires_credential=True,
            # Primary key named when an unconfigured Athena engine is invoked.
            mcp_json_key="AWS_ACCESS_KEY_ID",
            # Full credential set (source, mcp.json key), matching the
            # ``geo-query`` block in bundle-manifest.json exactly (all Optional).
            credentials=(
                ("Amazon Athena / S3", "AWS_ACCESS_KEY_ID"),
                ("Amazon Athena / S3", "AWS_SECRET_ACCESS_KEY"),
                ("Amazon Athena", "ATHENA_S3_STAGING_DIR"),
            ),
        ),
    )
}

#: The engine used when the caller does not name one (design default).
DEFAULT_ENGINE = "duckdb"


def supported_engine_names() -> "List[str]":
    """Return the accepted engine keys, in declaration order."""
    return list(SUPPORTED_ENGINES.keys())


class QueryEngine(abc.ABC):
    """The injectable contract a concrete query engine implements.

    Concrete subclasses wrap a driver (DuckDB Spatial, an Athena client) and
    translate a spatial SQL ``query`` - optionally referencing the data
    locations supplied in ``sources`` (logical name -> href/path) - into a
    :class:`~geo_query.models.ResultSet`. The server depends only on this
    abstraction, so tests inject a mock engine and no driver is required to
    exercise the full path.

    Implementations should raise the most specific
    :class:`~geo_common.errors.GeoError` they can (e.g. ``AuthenticationError``
    for rejected AWS credentials); any other exception is mapped onto the
    ``Error_Taxonomy`` by the server's
    :meth:`~geo_common.server.BaseGeoServer.map_error`.
    """

    #: The engine key this implementation handles (must be a SUPPORTED_ENGINES key).
    name: str = ""

    @abc.abstractmethod
    def execute(
        self, *, query: str, sources: Optional[Mapping[str, str]] = None
    ) -> ResultSet:
        """Execute ``query`` (over the data in ``sources``) and return its rows."""
        raise NotImplementedError
