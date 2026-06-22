"""Hub assembly / MVP wiring (Requirements 2.1, 4.2, 16.2, 16.3).

This module is the **Power Hub assembly point**: it takes the MVP MCP servers
(``geo-stac``, ``geo-vector``, ``geo-ops``, ``geo-foundation-models``) and wires
them into the Hub so that, at startup, every MVP capability is discoverable in
the Resource Catalog, every MVP credential spec is registered with the
Credential Manager, and the Orchestration Router can route a single
natural-language request through the **discover -> process -> analyze** pillars
across those servers (design.md "Power Hub", "Request flow").

Concretely :func:`assemble_mvp_hub` builds a fully wired :class:`MvpHub`:

* **Catalog registration (Requirement 2.1).** Every bound server's
  :meth:`~geo_common.server.BaseGeoServer.catalog_entries` is registered in the
  :class:`~kiro_geospatial.catalog.ResourceCatalog`. Because a bound server is,
  by definition, installed in this Hub process, each registered entry is marked
  ``installed=True`` and its not-installed ``install_command`` is cleared
  (Requirement 2.6).
* **Credential registration.** Every bound server's
  :meth:`~geo_common.server.BaseGeoServer.required_credentials` is registered
  with the :class:`~kiro_geospatial.credentials.CredentialManager` over the
  single ``mcp.json`` surface (Requirement 3.1).
* **Router wiring (Requirement 4.2).** A :class:`~kiro_geospatial.orchestration.CatalogCapabilityResolver`
  built from the registered entries resolves each capability to its providing
  server, and a :class:`BoundServerInvoker` dispatches a planned step to that
  server's MCP tool. The router is seeded with the MVP capability vocabulary
  (:data:`MVP_CAPABILITY_SPECS`) mapping ``stac_search`` / ``vector_features``
  to *discover*, ``transform_crs`` to *process*, and ``embed_tile`` to
  *analyze*, so a request mentioning all three pillars yields a single ordered
  discover -> process -> analyze plan (Requirement 4.1).

**Decoupled, testable wiring.** The MVP servers live in separate ``uvx``
packages that are not guaranteed to be importable together. The assembly
therefore takes **injectable bindings**: any object satisfying the small
:class:`ServerBinding` protocol (``server_name`` + ``catalog_entries`` +
``required_credentials`` + ``get_tool``, i.e. exactly what
:class:`~geo_common.server.BaseGeoServer` already provides) can be wired in.
Tests inject lightweight fakes; production uses :func:`default_mvp_bindings`,
which imports the real servers from a ``PYTHONPATH`` that includes every MVP
package root. Either way the Hub never hard-depends on the concrete server
packages at import time.

Python 3.9-compatible annotations (``from __future__ import annotations`` plus
``typing`` aliases), matching the rest of the package.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from pydantic import BaseModel, Field

from geo_common.errors import AuthenticationError, GeoError, NotFoundError
from geo_common.models import CatalogEntry, CredentialSpec

from kiro_geospatial.catalog import ResourceCatalog
from kiro_geospatial.credentials import CredentialManager
from kiro_geospatial.orchestration import (
    CapabilitySpec,
    CatalogCapabilityResolver,
    OrchestrationPlan,
    OrchestrationResult,
    OrchestrationRouter,
    PlanStep,
    PlanStepKind,
    ServerInvoker,
)

__all__ = [
    "MVP_SERVER_NAMES",
    "MVP_CAPABILITY_SPECS",
    "ServerBinding",
    "BoundServerInvoker",
    "MvpStartupReport",
    "MvpHub",
    "assemble_mvp_hub",
    "default_mvp_bindings",
]

#: The five MVP modules; ``geo-common`` is the shared base (not an MCP server),
#: so the four servers below are the ones the Hub wires (Requirement 16.2).
MVP_SERVER_NAMES = (
    "geo-stac",
    "geo-vector",
    "geo-ops",
    "geo-foundation-models",
)


#: The MVP capability vocabulary the Orchestration Router plans for. Each spec
#: maps a catalog capability onto its pillar and the trigger keywords whose
#: presence in a request selects it. ``stac_search`` and ``vector_features``
#: are *discover*; ``transform_crs`` is *process*; ``embed_tile`` is *analyze*
#: (design.md "Request flow: discover -> process -> analyze").
MVP_CAPABILITY_SPECS: List[CapabilitySpec] = [
    CapabilitySpec(
        capability="stac_search",
        kind=PlanStepKind.DISCOVER,
        keywords=[
            "stac",
            "imagery",
            "scene",
            "scenes",
            "satellite",
            "catalog",
            "discover",
            "search",
        ],
    ),
    CapabilitySpec(
        capability="vector_features",
        kind=PlanStepKind.DISCOVER,
        keywords=[
            "vector",
            "features",
            "buildings",
            "roads",
            "osm",
            "openstreetmap",
            "overture",
        ],
    ),
    CapabilitySpec(
        capability="transform_crs",
        kind=PlanStepKind.PROCESS,
        keywords=[
            "reproject",
            "reprojection",
            "transform",
            "crs",
            "projection",
            "process",
        ],
    ),
    CapabilitySpec(
        capability="embed_tile",
        kind=PlanStepKind.ANALYZE,
        keywords=[
            "embed",
            "embedding",
            "analyze",
            "analysis",
            "foundation model",
            "change detection",
        ],
    ),
]


@runtime_checkable
class ServerBinding(Protocol):
    """The minimal server surface the Hub assembly needs (Req 2.1, 4.2, 16.1).

    This is exactly the subset of :class:`~geo_common.server.BaseGeoServer`
    every MCP server already implements, so every MVP server is a binding
    as-is and tests can inject a lightweight fake without importing the concrete
    server packages:

    * ``server_name`` - the provider identifier; it must equal the
      ``provider_server`` on the server's catalog entries so the
      capability->server resolution and the :class:`BoundServerInvoker` agree.
    * :meth:`catalog_entries` - the capabilities to register (Requirement 2.1).
    * :meth:`required_credentials` - the credential specs to register
      (Requirement 16.1).
    * :meth:`get_tool` - resolve a capability name to its callable; the invoker
      calls ``get_tool(capability).func`` to dispatch a planned step
      (Requirement 4.2).
    """

    server_name: str

    def catalog_entries(self) -> List[CatalogEntry]:
        ...

    def required_credentials(self) -> List[CredentialSpec]:
        ...

    def get_tool(self, name: str):  # -> RegisteredTool (geo_common.server)
        ...


class BoundServerInvoker:
    """A :class:`~kiro_geospatial.orchestration.ServerInvoker` over bound servers.

    Holds the Hub's ``server_name -> ServerBinding`` map. To invoke a planned
    step against ``source`` (Requirement 4.2) it looks up the bound server,
    resolves the tool whose name equals the step's ``capability``, and calls it
    with the step ``params`` (awaiting the result when the tool is a
    coroutine). This is the single place the Hub turns an abstract plan step
    into a concrete MCP tool call, keeping the Orchestration Router decoupled
    from the servers.

    Failures propagate as raised exceptions so the router can record per-source
    provenance and degrade gracefully (Requirements 4.5-4.8): an unknown source
    or a capability the bound server does not expose raises
    :class:`~geo_common.errors.NotFoundError`; the tool's own errors propagate
    unchanged (a :class:`~geo_common.errors.GeoError` keeps its taxonomy
    category, anything else is mapped to ``upstream`` by the router).
    """

    def __init__(self, bindings: Mapping[str, "ServerBinding"]) -> None:
        self._bindings: Dict[str, "ServerBinding"] = dict(bindings)

    @property
    def bindings(self) -> Dict[str, "ServerBinding"]:
        """A copy of the ``server_name -> binding`` map."""
        return dict(self._bindings)

    async def invoke(self, step: "PlanStep", source: str) -> object:
        """Dispatch ``step``'s capability to ``source``'s MCP tool (Req 4.2)."""
        binding = self._bindings.get(source)
        if binding is None:
            raise NotFoundError(
                "no server is bound for source %r" % source,
                source=source,
                detail={"source": source, "capability": step.capability},
            )

        try:
            tool = binding.get_tool(step.capability)
        except GeoError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise NotFoundError(
                "server %r does not expose capability %r"
                % (source, step.capability),
                source=source,
                detail={"source": source, "capability": step.capability},
            ) from exc

        result = tool.func(**dict(step.params))
        if inspect.isawaitable(result):
            result = await result
        return result


class MvpStartupReport(BaseModel):
    """The outcome of starting the wired MVP servers (Requirements 16.3-16.6).

    ``started`` lists the servers whose startup credential guard passed (able to
    start). ``blocking`` maps any server that refused to start to the Required
    ``mcp.json`` keys that were missing (Requirement 16.6); for the MVP set,
    whose credentials are all Optional, ``blocking`` is empty and every server
    starts (Requirement 16.5). ``all_started`` is ``True`` exactly when no
    server was blocked.
    """

    started: List[str] = Field(default_factory=list)
    blocking: Dict[str, List[str]] = Field(default_factory=dict)

    @property
    def all_started(self) -> bool:
        return not self.blocking


@dataclass
class MvpHub:
    """A fully wired Power Hub over the MVP servers (Requirements 2.1, 4.2).

    Produced by :func:`assemble_mvp_hub`. Bundles the registered
    :class:`~kiro_geospatial.catalog.ResourceCatalog`, the
    :class:`~kiro_geospatial.credentials.CredentialManager`, the wired
    :class:`~kiro_geospatial.orchestration.OrchestrationRouter` (and its
    :class:`BoundServerInvoker`), and the ``server_name -> binding`` map.

    :meth:`run` is the single-request entry point: it plans a natural-language
    request into an ordered discover -> process -> analyze plan and executes it
    across the bound servers, returning the
    :class:`~kiro_geospatial.orchestration.OrchestrationResult` with provenance.
    """

    catalog: ResourceCatalog
    credentials: CredentialManager
    router: OrchestrationRouter
    invoker: BoundServerInvoker
    bindings: Dict[str, "ServerBinding"] = field(default_factory=dict)

    def start(
        self, *, configured_keys: Optional[Iterable[str]] = None
    ) -> MvpStartupReport:
        """Run each bound server's startup credential guard (Req 16.3-16.6).

        Starts every bound server, applying its
        :meth:`~geo_common.server.BaseGeoServer.start` guard so a server with a
        missing Required credential is reported as blocked (Requirement 16.6)
        while servers with only Optional credentials start normally
        (Requirement 16.5). Returns an :class:`MvpStartupReport`; for the MVP
        set every server starts and ``all_started`` is ``True`` (Requirement
        16.3).
        """
        report = MvpStartupReport()
        for name, binding in self.bindings.items():
            start = getattr(binding, "start", None)
            if not callable(start):
                report.started.append(name)
                continue
            try:
                start(configured_keys=configured_keys)
                report.started.append(name)
            except AuthenticationError as exc:
                missing = []
                if exc.detail and isinstance(exc.detail, dict):
                    missing = list(exc.detail.get("missing_required_keys", []))
                report.blocking[name] = missing
        return report

    def plan(self, request: str) -> OrchestrationPlan:
        """Decompose ``request`` into an ordered discover->process->analyze plan."""
        return self.router.plan(request)

    async def run(self, request: str) -> OrchestrationResult:
        """Plan and execute ``request`` end-to-end across the MVP servers.

        A single request flows discover -> process -> analyze (Requirement 4.2):
        the router builds the ordered plan, resolves each step's capability to
        its providing server, and the :class:`BoundServerInvoker` dispatches each
        step to that server's tool, returning the aggregate result with
        provenance and partial-result labeling.
        """
        plan = self.router.plan(request)
        return await self.router.execute(plan, invoker=self.invoker)


def _normalize_installed(entry: CatalogEntry) -> CatalogEntry:
    """Mark a bound server's catalog entry as installed (Requirement 2.6).

    A bound server is, by definition, installed in this Hub process, so its
    entries are recorded ``installed=True`` and the not-installed
    ``install_command`` is cleared (the catalog surfaces an install command only
    for *not*-installed providers - Requirement 2.6).
    """
    return entry.model_copy(update={"installed": True, "install_command": None})


def assemble_mvp_hub(
    bindings: Optional[Mapping[str, "ServerBinding"]] = None,
    *,
    surface: Optional[Mapping[str, str]] = None,
    capabilities: Optional[Sequence[CapabilitySpec]] = None,
    source_timeout_s: Optional[float] = None,
) -> MvpHub:
    """Assemble and wire the MVP Power Hub (Requirements 2.1, 4.2, 16.2, 16.3).

    Registers every bound server's catalog entries (Requirement 2.1) and
    credential specs into a fresh :class:`ResourceCatalog` and
    :class:`CredentialManager`, then wires an :class:`OrchestrationRouter` with a
    catalog-backed resolver and a :class:`BoundServerInvoker` so a single request
    flows discover -> process -> analyze across the bound servers (Requirement
    4.2).

    Parameters
    ----------
    bindings:
        ``server_name -> ServerBinding`` map of the MVP servers to wire. Each
        binding's ``server_name`` must equal the ``provider_server`` on its
        catalog entries. Defaults to :func:`default_mvp_bindings` (the real MVP
        servers, requiring their packages on ``PYTHONPATH``); inject fakes in
        tests to wire without the concrete packages.
    surface:
        The single ``mcp.json`` credential surface passed to the Credential
        Manager (defaults to the process environment, Requirement 3.1).
    capabilities:
        The capability vocabulary the router plans for. Defaults to
        :data:`MVP_CAPABILITY_SPECS`.
    source_timeout_s:
        Optional per-source invocation timeout for the router; defaults to the
        router's own default (30s).
    """
    resolved_bindings: Dict[str, "ServerBinding"] = dict(
        bindings if bindings is not None else default_mvp_bindings()
    )

    # 1) Register every bound server's catalog entries (Requirement 2.1).
    catalog = ResourceCatalog()
    all_entries: List[CatalogEntry] = []
    for binding in resolved_bindings.values():
        for entry in binding.catalog_entries():
            normalized = _normalize_installed(entry)
            all_entries.append(normalized)
    catalog.register(all_entries)

    # 2) Register every bound server's credential specs (Requirement 16.1).
    credentials = CredentialManager(surface=surface)
    for binding in resolved_bindings.values():
        for spec in binding.required_credentials():
            credentials.register_spec(spec)
    # Refresh the presence snapshot now that all specs are registered.
    credentials.load()

    # 3) Wire the Orchestration Router: a catalog-backed resolver maps each
    #    capability to its providing server (Requirement 4.2), and the
    #    BoundServerInvoker dispatches a planned step to that server's tool.
    resolver = CatalogCapabilityResolver(all_entries)
    invoker = BoundServerInvoker(resolved_bindings)
    router_kwargs: Dict[str, object] = {"invoker": invoker}
    if source_timeout_s is not None:
        router_kwargs["source_timeout_s"] = source_timeout_s
    router = OrchestrationRouter(
        resolver,
        list(capabilities if capabilities is not None else MVP_CAPABILITY_SPECS),
        **router_kwargs,
    )

    return MvpHub(
        catalog=catalog,
        credentials=credentials,
        router=router,
        invoker=invoker,
        bindings=resolved_bindings,
    )


def default_mvp_bindings() -> Dict[str, "ServerBinding"]:
    """Construct the real MVP server bindings (Requirements 16.2, 16.3).

    Lazily imports and instantiates ``geo-stac``, ``geo-vector``, ``geo-ops``,
    and ``geo-foundation-models``. These live in separate ``uvx`` packages, so
    they must be importable - in production the Hub runs with a ``PYTHONPATH``
    that includes every MVP package root (or the packages installed via
    ``uvx``). When a package is not importable this raises a clear
    :class:`~geo_common.errors.NotFoundError` naming the missing module, rather
    than failing with a bare ``ImportError``; tests that want to wire without
    the concrete packages should inject bindings into :func:`assemble_mvp_hub`
    instead.
    """
    try:
        from geo_stac.server import GeoStacServer
        from geo_vector.server import GeoVectorServer
        from geo_ops.server import GeoOpsServer
        from geo_foundation_models.server import GeoFoundationModelsServer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise NotFoundError(
            "one or more MVP server packages are not importable; ensure "
            "geo-stac, geo-vector, geo-ops and geo-foundation-models are "
            "installed (uvx) or on PYTHONPATH",
            source="kiro-geospatial",
            detail={"missing_import": str(exc)},
        ) from exc

    servers = [
        GeoStacServer(),
        GeoVectorServer(),
        GeoOpsServer(),
        GeoFoundationModelsServer(),
    ]
    return {server.server_name: server for server in servers}
