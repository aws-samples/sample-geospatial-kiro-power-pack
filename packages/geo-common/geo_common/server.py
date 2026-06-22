"""The ``BaseGeoServer`` contract every MCP server inherits (Req 5, 6.3).

Every Geospatial Power Pack MCP server subclasses :class:`BaseGeoServer`. The
base class gives each server four shared behaviors so the whole pack is
uniform:

* **One shared HTTP client** (Requirement 5.1). The base owns a single
  :class:`~geo_common.http.HttpClient`; subclasses make all outbound calls
  through ``self.http`` and inherit identical retry/backoff/rate-limit
  handling.
* **Error mapping** (:meth:`BaseGeoServer.map_error`). A helper that maps any
  library/transport/server exception onto the shared ``Error_Taxonomy``,
  guaranteeing exactly one category per error and routing unmapped external
  errors onto ``UPSTREAM`` while retaining the original detail (Requirement
  11.5; design Property 8).
* **Catalog + credential declaration** (:meth:`BaseGeoServer.catalog_entries`
  and :meth:`BaseGeoServer.required_credentials`). Subclasses override these so
  the Power Hub gets a uniform view of every server's capabilities (Req 2.1,
  11.3) and credential needs (Req 16.1).
* **Tool registration** (:meth:`BaseGeoServer.register_tool`). A framework-light
  registry of MCP tool callables exposed by the server.

It also provides the **startup credential guard** (:meth:`BaseGeoServer.start`,
:meth:`BaseGeoServer.missing_required_credentials`): a server refuses to start
when any *Required* credential is absent, naming each missing ``mcp.json`` key
(Requirement 16.6), while starting normally when only *Optional* credentials
are unconfigured (Requirement 16.5). Because the guard inspects only this
server's own ``required_credentials()``, startup is independent of which other
servers are installed (Requirement 16.4).

Python 3.9 compatibility: this module uses ``from __future__ import
annotations`` and ``typing`` generics (``Optional``, ``List``, ``Dict`` ...) so
it imports cleanly on 3.9+.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

import httpx

from geo_common.errors import (
    AuthenticationError,
    AuthorizationError,
    GeoError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    UpstreamError,
    ValidationError,
)
from geo_common.http import HttpClient
from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
)

__all__ = ["BaseGeoServer", "RegisteredTool", "main"]

#: An MCP tool callable. Tools are plain (sync or async) callables; the base
#: class only stores and exposes them, leaving invocation to the MCP runtime.
ToolFn = Callable[..., Any]


@dataclass(frozen=True)
class RegisteredTool:
    """A tool a server exposes through MCP.

    Stores the public ``name``, the underlying callable ``func``, and a short
    human ``description`` (defaulting to the callable's docstring). Frozen so a
    registered tool cannot be mutated in place after registration.
    """

    name: str
    func: ToolFn
    description: str = ""


class BaseGeoServer:
    """Base class every MCP_Server inherits (Requirements 5, 6.3).

    Owns the shared :class:`HttpClient`, exposes the :meth:`map_error` taxonomy
    helper, standardizes catalog/credential declaration and MCP tool
    registration, and enforces the startup credential guard.

    Subclasses set the three class attributes below and override
    :meth:`catalog_entries` / :meth:`required_credentials` (and register their
    tools, typically in ``__init__``)::

        class GeoStacServer(BaseGeoServer):
            pillar = "A"
            server_name = "geo-stac"
            version = "0.1.0"

            def required_credentials(self) -> list[CredentialSpec]:
                return [...]
    """

    #: ``"A" | "B" | "C" | "expansion" | "peer"``. Subclasses override.
    pillar: str = ""
    #: e.g. ``"geo-stac"``. Subclasses override.
    server_name: str = "base-geo-server"
    #: Semantic version of the server. Subclasses override.
    version: str = "0.0.0"

    def __init__(self, http: Optional[HttpClient] = None) -> None:
        # One shared async HTTP client per server (Requirement 5.1). A server
        # may share a client across instances by passing it in; otherwise each
        # server owns its own.
        self.http: HttpClient = http if http is not None else HttpClient()
        self._tools: "Dict[str, RegisteredTool]" = {}
        self._started: bool = False

    # ------------------------------------------------------------------
    # Error mapping (Requirement 11.5; design Property 8)
    # ------------------------------------------------------------------

    def map_error(self, exc: Exception, *, source: str) -> GeoError:
        """Map any exception onto the shared ``Error_Taxonomy``.

        Guarantees exactly one taxonomy category per error:

        * An already-``GeoError`` passes through unchanged - it already carries
          exactly one category.
        * Known transport/timeout/connection failures map to ``NETWORK``
          (Requirements 5.4, 5.9).
        * HTTP status errors map by status code: 401 -> ``AUTHENTICATION``,
          403 -> ``AUTHORIZATION``, 404 -> ``NOT_FOUND``, 429 -> ``RATE_LIMIT``,
          400/422 -> ``VALIDATION``, 5xx -> ``UPSTREAM``; any other status
          falls through to ``UPSTREAM``.
        * Any other exception - an unmapped library or external-server error -
          maps to ``UPSTREAM`` and retains the original detail in ``original``
          (Requirement 11.5).

        ``source`` identifies the failing source on the returned error; it must
        be a secret-free identifier (e.g. a server name or host).
        """
        # Already taxonomy-classified: pass through (exactly one category).
        if isinstance(exc, GeoError):
            return exc

        # Timeouts must be checked before the broader TransportError because
        # httpx.TimeoutException is a subclass of httpx.TransportError.
        if isinstance(exc, httpx.TimeoutException):
            return NetworkError(
                "request timed out",
                source=source,
                original=self._safe_detail(exc),
            )

        if isinstance(exc, httpx.HTTPStatusError):
            return self._map_status_error(exc, source=source)

        if isinstance(exc, httpx.TransportError):
            # Connection/network-level failure (ConnectError, ReadError, ...).
            return NetworkError(
                "network transport error",
                source=source,
                original=self._safe_detail(exc),
            )

        # Unmapped external/library error -> UPSTREAM, original detail retained
        # (Requirement 11.5). This catch-all makes the mapping total.
        return UpstreamError(
            "unmapped upstream error",
            source=source,
            original=self._safe_detail(exc),
        )

    def _map_status_error(
        self, exc: "httpx.HTTPStatusError", *, source: str
    ) -> GeoError:
        """Map an :class:`httpx.HTTPStatusError` by its status code."""
        status = exc.response.status_code
        original = self._safe_detail(exc)
        detail = {"status_code": status}

        if status == 401:
            return AuthenticationError(
                "upstream rejected the request with HTTP 401",
                source=source,
                detail=detail,
                original=original,
            )
        if status == 403:
            return AuthorizationError(
                "upstream rejected the request with HTTP 403",
                source=source,
                detail=detail,
                original=original,
            )
        if status == 404:
            return NotFoundError(
                "upstream resource not found (HTTP 404)",
                source=source,
                detail=detail,
                original=original,
            )
        if status == 429:
            return RateLimitError(
                "upstream rate-limited the request (HTTP 429)",
                source=source,
                detail=detail,
                original=original,
                retry_after=self._parse_retry_after(exc.response),
            )
        if status in (400, 422):
            return ValidationError(
                "upstream rejected the request as invalid (HTTP %d)" % status,
                source=source,
                detail=detail,
                original=original,
            )
        # 5xx and any other unmapped status -> UPSTREAM, retaining detail.
        return UpstreamError(
            "upstream returned HTTP %d" % status,
            source=source,
            detail=detail,
            original=original,
        )

    # ------------------------------------------------------------------
    # Catalog + credential declaration (Req 2.1, 11.3, 16.1)
    # ------------------------------------------------------------------

    def catalog_entries(self) -> "List[CatalogEntry]":
        """Capabilities this server registers in the Resource Catalog.

        Subclasses override to declare their capabilities (Req 2.1, 11.3). The
        base returns an empty list.
        """
        return []

    def required_credentials(self) -> "List[CredentialSpec]":
        """The ``mcp.json`` keys this server needs and their classification.

        Subclasses override (Requirement 16.1). The base returns an empty list,
        meaning a server with no declared credentials always starts (its set of
        Required credentials is empty - Requirement 16.5).
        """
        return []

    # ------------------------------------------------------------------
    # MCP tool registration (framework-light registry)
    # ------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        func: Optional[ToolFn] = None,
        *,
        description: Optional[str] = None,
    ) -> Any:
        """Register an MCP tool callable under ``name``.

        Usable directly::

            server.register_tool("stac_search", stac_search)

        or as a decorator::

            @server.register_tool("stac_search")
            async def stac_search(...): ...

        Raises :class:`ValueError` if ``name`` is empty or already registered.
        Returns the registered callable so the decorator form is transparent.
        """
        if not name:
            raise ValueError("tool name must be a non-empty string")

        def _register(fn: ToolFn) -> ToolFn:
            if not callable(fn):
                raise TypeError("registered tool must be callable")
            if name in self._tools:
                raise ValueError(
                    "tool %r is already registered on %s"
                    % (name, self.server_name)
                )
            doc = description if description is not None else (inspect.getdoc(fn) or "")
            self._tools[name] = RegisteredTool(name=name, func=fn, description=doc)
            return fn

        # Direct call form: register and return the callable.
        if func is not None:
            return _register(func)
        # Decorator form: return the registrar.
        return _register

    @property
    def tools(self) -> "Mapping[str, RegisteredTool]":
        """A read-only snapshot of the registered tools keyed by name."""
        return dict(self._tools)

    def tool_names(self) -> "List[str]":
        """The names of all registered tools, in registration order."""
        return list(self._tools.keys())

    def get_tool(self, name: str) -> RegisteredTool:
        """Return the :class:`RegisteredTool` for ``name``.

        Raises :class:`~geo_common.errors.NotFoundError` (taxonomy
        ``not-found``) when no tool is registered under ``name``.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise NotFoundError(
                "tool %r is not registered on %s" % (name, self.server_name),
                source=self.server_name,
            )

    # ------------------------------------------------------------------
    # Startup credential guard (Requirements 16.4, 16.5, 16.6)
    # ------------------------------------------------------------------

    def missing_required_credentials(
        self, *, configured_keys: Optional[Iterable[str]] = None
    ) -> "List[CredentialSpec]":
        """Return the Required credential specs that are not configured.

        A credential is *configured* when its ``mcp_json_key`` appears in
        ``configured_keys``. When ``configured_keys`` is ``None`` the set is
        derived from the process environment (an ``mcp.json`` ``env`` key with
        a non-empty value), which is how MCP exposes the single credential
        surface to a running server.

        Only :attr:`CredentialClassification.REQUIRED` specs can appear in the
        result - Optional and License-Needed credentials never block startup
        (Requirement 16.5). The result is independent of any other server's
        credentials, since only ``self.required_credentials()`` is inspected
        (Requirement 16.4).
        """
        configured = self._resolve_configured_keys(configured_keys)
        return [
            spec
            for spec in self.required_credentials()
            if spec.classification is CredentialClassification.REQUIRED
            and spec.mcp_json_key not in configured
        ]

    def start(self, *, configured_keys: Optional[Iterable[str]] = None) -> None:
        """Run the startup credential guard, then mark the server started.

        Refuses to start when any Required credential is absent, raising an
        :class:`~geo_common.errors.AuthenticationError` that names **each**
        missing ``mcp.json`` key (Requirement 16.6). Starts normally when only
        Optional credentials are unconfigured (Requirement 16.5) and regardless
        of which other servers are installed (Requirement 16.4).

        ``configured_keys`` lets callers (and tests) supply the exact set of
        configured ``mcp.json`` keys; when omitted the process environment is
        used.
        """
        missing = self.missing_required_credentials(configured_keys=configured_keys)
        if missing:
            keys = [spec.mcp_json_key for spec in missing]
            raise AuthenticationError(
                "%s cannot start: missing required credential(s): %s"
                % (self.server_name, ", ".join(keys)),
                source=self.server_name,
                detail={"missing_required_keys": keys},
            )
        self._started = True

    @property
    def started(self) -> bool:
        """Whether the startup credential guard has passed for this server."""
        return self._started

    async def aclose(self) -> None:
        """Release the shared HTTP client's resources."""
        await self.http.aclose()

    # ------------------------------------------------------------------
    # MCP stdio runtime (deploy-time; lazily imports the ``mcp`` SDK)
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the startup credential guard, then serve this server over stdio.

        The single call a server's ``main()`` entry point makes to become a
        live MCP server: it runs :meth:`start` (refusing to start when a
        Required credential is missing) and then serves the registered tools
        over stdin/stdout until the client disconnects. The MCP SDK is imported
        lazily by :mod:`geo_common.runtime`, so importing or unit-testing a
        server never requires the SDK; it is only needed when actually serving.
        """
        from geo_common.runtime import run_server

        run_server(self)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_configured_keys(
        self, configured_keys: Optional[Iterable[str]]
    ) -> "set[str]":
        """Resolve the set of configured ``mcp.json`` keys.

        Explicit ``configured_keys`` win; otherwise read the environment,
        treating only non-empty values as configured.
        """
        if configured_keys is not None:
            return set(configured_keys)
        return {key for key, value in os.environ.items() if value}

    @staticmethod
    def _safe_detail(exc: Exception) -> str:
        """A secret-free description of an exception for ``original`` retention."""
        text = str(exc).strip()
        return text if text else type(exc).__name__

    @staticmethod
    def _parse_retry_after(response: "httpx.Response") -> Optional[float]:
        """Best-effort parse of a numeric ``Retry-After`` header to seconds."""
        raw = response.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return max(0.0, float(raw.strip()))
        except ValueError:
            return None


def main() -> None:
    """Console entry point declared by ``geo-common``'s ``pyproject.toml``.

    ``geo-common`` is a shared *base library*, not a standalone MCP server, so
    this entry point only reports that and returns. It exists so the declared
    ``[project.scripts]`` ``geo-common`` entry point resolves.
    """
    print(
        "geo-common is the shared base library for the Geospatial Power Pack "
        "(HttpClient, Error_Taxonomy, BaseGeoServer); it is not a standalone "
        "MCP server. Install a domain server (e.g. geo-stac) instead."
    )
