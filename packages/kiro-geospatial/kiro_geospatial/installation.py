"""The Installation Manager (Requirements 6.2, 6.5, 6.6, 6.8, 6.9, 16.2, 16.3, 16.7).

The Installation Manager is the Power Hub mechanism by which a user installs,
lists, and removes individual MCP servers via ``uvx`` (the design's "modular
installation" principle). It guarantees the modular-install contract:

* **Install one server + ``geo-common``, nothing else** (Requirements 6.2, 6.3).
  Installing a server installs that server and its dependency on ``geo-common``
  and pulls in no other MCP server.
* **Install the MVP set together** (Requirements 16.2, 16.3). ``geo-common``,
  ``geo-stac``, ``geo-vector``, ``geo-ops``, and ``geo-foundation-models`` are
  installed as one unit, and the manager reports each as installed and able to
  start.
* **List installed servers with versions** (Requirement 6.5).
* **Remove a server without touching ``geo-common`` or others** (Req 6.6).
* **Atomic rollback on a failed install** (Requirements 6.8, 16.7). If any
  package in an install operation fails, every package this operation installed
  is rolled back, leaving nothing partially installed, and an error names the
  module that failed.
* **Errors for impossible operations** (Requirements 6.9, 16.7). Removing a
  server that is not installed, or installing a module that is unknown / cannot
  be installed, raises an ``Error_Taxonomy`` error.

The actual ``uvx`` subprocess invocation is abstracted behind :class:`UvxRunner`
so it can be replaced by a fake in tests. :class:`SubprocessUvxRunner` is the
production implementation that shells out to ``uv tool``.

Python 3.9 compatibility: this module uses ``from __future__ import
annotations`` together with ``typing`` generics so it imports cleanly on 3.9+.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from pydantic import BaseModel, Field

from geo_common.errors import NotFoundError, UpstreamError, ValidationError

__all__ = [
    "GEO_COMMON",
    "HUB_DEFAULT_NAME",
    "UvxCommandError",
    "UvxRunner",
    "SubprocessUvxRunner",
    "ServerSpec",
    "InstalledServerView",
    "InstallReport",
    "InstallationManager",
    "load_server_specs",
]

#: The shared base package every MCP server depends on. It is installed as a
#: dependency of any server and is never removed by removing a server
#: (Requirements 6.3, 6.6).
GEO_COMMON = "geo-common"

#: The Power Hub package itself. It is not an à-la-carte MCP server, so it is
#: excluded from the installable MVP module set (Requirement 16.2 lists the
#: five servers, not the hub).
HUB_DEFAULT_NAME = "kiro-geospatial"


# ---------------------------------------------------------------------------
# uvx invocation abstraction (mockable in tests)
# ---------------------------------------------------------------------------
class UvxCommandError(Exception):
    """A ``uvx``/``uv tool`` command failed.

    Carries the ``package`` whose command failed so the Installation Manager
    can name the failed module when it rolls back (Requirements 6.8, 16.7).
    ``detail`` retains the secret-free command output for the taxonomy error's
    ``original`` field.
    """

    def __init__(self, package: str, message: str, *, detail: Optional[str] = None) -> None:
        super().__init__(message)
        self.package = package
        self.message = message
        self.detail = detail


class UvxRunner(ABC):
    """Abstraction over the ``uvx`` tool installer (Requirement 6.2).

    The Installation Manager talks only to this interface so the real
    subprocess calls can be replaced by a fake in tests. Implementations manage
    the actual install state of ``uvx`` tools.

    Contract:

    * :meth:`install` installs a single package and returns its installed
      version, raising :class:`UvxCommandError` on failure.
    * :meth:`uninstall` removes a single package, raising
      :class:`UvxCommandError` on failure.
    * :meth:`list_installed` returns a mapping of installed package name to
      version string.
    """

    @abstractmethod
    def install(self, package: str) -> str:
        """Install ``package`` and return its installed version string."""

    @abstractmethod
    def uninstall(self, package: str) -> None:
        """Uninstall ``package``."""

    @abstractmethod
    def list_installed(self) -> "Dict[str, str]":
        """Return a mapping of installed package name -> version."""

    # -- convenience helpers built on the three primitives ------------------

    def installed_version(self, package: str) -> Optional[str]:
        """Return the installed version of ``package`` or ``None`` if absent."""
        return self.list_installed().get(package)

    def is_installed(self, package: str) -> bool:
        """Whether ``package`` is currently installed."""
        return package in self.list_installed()


class SubprocessUvxRunner(UvxRunner):
    """Production :class:`UvxRunner` that shells out to ``uv tool`` (``uvx``).

    Uses ``uv tool install`` / ``uv tool uninstall`` / ``uv tool list`` with
    argument lists (never a shell string) so package names cannot be used for
    command injection. Each call is bounded by ``timeout`` seconds.
    """

    #: ``uv tool list`` prints one tool per line as ``name vX.Y.Z`` followed by
    #: indented entry-point lines beginning with ``-``. This matches the
    #: leading ``name version`` portion of a tool line.
    _LIST_LINE = re.compile(r"^(?P<name>\S+)\s+v?(?P<version>\S+)")

    def __init__(self, *, uv_bin: str = "uv", timeout: float = 300.0) -> None:
        self._base: List[str] = [uv_bin, "tool"]
        self._timeout = timeout

    def _run(self, args: "Sequence[str]") -> "subprocess.CompletedProcess[str]":
        try:
            return subprocess.run(
                [*self._base, *args],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except FileNotFoundError as exc:  # uv not on PATH
            raise UvxCommandError(
                args[-1] if args else "",
                "the 'uv' executable was not found on PATH",
                detail=str(exc),
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise UvxCommandError(
                args[-1] if args else "",
                "uv tool command timed out after %ss" % self._timeout,
                detail=str(exc),
            ) from exc

    def install(self, package: str) -> str:
        proc = self._run(["install", package])
        if proc.returncode != 0:
            raise UvxCommandError(
                package,
                "uv tool install %s failed" % package,
                detail=_clean(proc.stderr or proc.stdout),
            )
        return self.installed_version(package) or "unknown"

    def uninstall(self, package: str) -> None:
        proc = self._run(["uninstall", package])
        if proc.returncode != 0:
            raise UvxCommandError(
                package,
                "uv tool uninstall %s failed" % package,
                detail=_clean(proc.stderr or proc.stdout),
            )

    def list_installed(self) -> "Dict[str, str]":
        proc = self._run(["list"])
        if proc.returncode != 0:
            raise UvxCommandError(
                "",
                "uv tool list failed",
                detail=_clean(proc.stderr or proc.stdout),
            )
        return self._parse_list(proc.stdout)

    @classmethod
    def _parse_list(cls, stdout: str) -> "Dict[str, str]":
        installed: "Dict[str, str]" = {}
        for raw in stdout.splitlines():
            line = raw.rstrip()
            if not line or line.lstrip().startswith("-"):
                # Blank line or an indented entry-point line: skip.
                continue
            match = cls._LIST_LINE.match(line)
            if match:
                installed[match.group("name")] = match.group("version")
        return installed


def _clean(text: Optional[str]) -> Optional[str]:
    """Strip command output for secret-free ``original`` retention."""
    if not text:
        return None
    stripped = text.strip()
    return stripped or None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ServerSpec(BaseModel):
    """Metadata about an installable MCP server, drawn from the manifest.

    ``required_credential_keys`` lists the ``mcp.json`` keys classified
    *Required*; they drive the "able to start" determination (Requirements
    16.3, 16.6). A server with no Required keys can always start.
    """

    name: str = Field(min_length=1)
    pillar: str = Field(min_length=1)
    uvx_command: str = ""
    required_credential_keys: List[str] = Field(default_factory=list)
    is_mvp: bool = False
    first_expansion: bool = False


class InstalledServerView(BaseModel):
    """An installed server with its version and startability (Req 6.5, 16.3)."""

    name: str
    version: str
    pillar: str
    can_start: bool = True
    #: ``mcp.json`` keys for Required credentials that are absent. Empty when
    #: the server can start (Requirement 16.6).
    blocking_credentials: List[str] = Field(default_factory=list)


class InstallReport(BaseModel):
    """The outcome of an install operation (Requirements 16.2, 16.3).

    ``all_installed`` is ``True`` when every requested server is installed;
    ``all_can_start`` is ``True`` when every requested server is also able to
    start (no Required credential missing).
    """

    requested: List[str]
    installed: List[InstalledServerView]
    all_installed: bool
    all_can_start: bool


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------
def load_server_specs(
    manifest: Mapping[str, object],
) -> "tuple[Dict[str, ServerSpec], List[str], List[str], str]":
    """Build server specs and module-set lists from a ``bundle-manifest.json``.

    Returns ``(specs, mvp_set, first_expansions, hub_name)`` where ``specs``
    maps server name -> :class:`ServerSpec`, ``mvp_set`` is the manifest's MVP
    server list, ``first_expansions`` is the post-MVP install list (Requirement
    16.8), and ``hub_name`` is the Power Hub package name (excluded from the
    installable MVP set, Requirement 16.2).
    """
    specs: "Dict[str, ServerSpec]" = {}
    servers = manifest.get("servers", [])  # type: ignore[assignment]
    for entry in servers:  # type: ignore[assignment]
        name = entry["name"]
        required_keys = [
            cred["key"]
            for cred in entry.get("credentials", [])
            if cred.get("classification") == "Required"
        ]
        specs[name] = ServerSpec(
            name=name,
            pillar=str(entry.get("pillar", "")) or "unknown",
            uvx_command=str(entry.get("uvx", "")),
            required_credential_keys=required_keys,
            is_mvp=entry.get("status") == "MVP",
            first_expansion=bool(entry.get("firstExpansion", False)),
        )
    mvp_set = list(manifest.get("mvpSet", []))  # type: ignore[arg-type]
    first_expansions = list(manifest.get("firstExpansions", []))  # type: ignore[arg-type]
    hub_name = str(manifest.get("power", HUB_DEFAULT_NAME))
    return specs, mvp_set, first_expansions, hub_name


# ---------------------------------------------------------------------------
# Installation Manager
# ---------------------------------------------------------------------------
class InstallationManager:
    """Installs, lists, and removes MCP servers via ``uvx`` (Req 6, 16).

    Construct directly with explicit ``servers`` metadata (handy for tests) or
    via :meth:`from_manifest` to load from ``bundle-manifest.json``. The
    ``uvx`` calls go through an injectable :class:`UvxRunner`.

    ``configured_keys`` is the set of configured ``mcp.json`` keys used to
    decide whether an installed server can start; when ``None`` the process
    environment is consulted (mirroring ``BaseGeoServer``'s startup guard).
    """

    def __init__(
        self,
        *,
        servers: "Mapping[str, ServerSpec]",
        mvp_set: "Sequence[str]" = (),
        first_expansions: "Sequence[str]" = (),
        hub_name: str = HUB_DEFAULT_NAME,
        runner: Optional[UvxRunner] = None,
        configured_keys: Optional[Iterable[str]] = None,
    ) -> None:
        self._servers: "Dict[str, ServerSpec]" = dict(servers)
        self._mvp_set: List[str] = list(mvp_set)
        self._first_expansions: List[str] = list(first_expansions)
        self._hub_name = hub_name
        self._runner: UvxRunner = runner if runner is not None else SubprocessUvxRunner()
        self._configured_keys: Optional[set] = (
            set(configured_keys) if configured_keys is not None else None
        )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_manifest(
        cls,
        manifest_path: "Union[str, Path]",
        *,
        runner: Optional[UvxRunner] = None,
        configured_keys: Optional[Iterable[str]] = None,
    ) -> "InstallationManager":
        """Build a manager from a ``bundle-manifest.json`` file path."""
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        specs, mvp_set, first_expansions, hub_name = load_server_specs(data)
        return cls(
            servers=specs,
            mvp_set=mvp_set,
            first_expansions=first_expansions,
            hub_name=hub_name,
            runner=runner,
            configured_keys=configured_keys,
        )

    # -- introspection ------------------------------------------------------

    @property
    def runner(self) -> UvxRunner:
        return self._runner

    def known_servers(self) -> "List[str]":
        """All server names the manager knows about (from the manifest)."""
        return list(self._servers.keys())

    def mvp_install_set(self) -> "List[str]":
        """The installable MVP server set (Requirement 16.2).

        The five MCP servers ``geo-common``, ``geo-stac``, ``geo-vector``,
        ``geo-ops``, ``geo-foundation-models`` - the Power Hub package itself is
        excluded because it is not an à-la-carte server.
        """
        return [
            name
            for name in self._mvp_set
            if name != self._hub_name and name in self._servers
        ]

    def first_expansions(self) -> "List[str]":
        """The first expansion modules to install after the MVP set (Req 16.8)."""
        return [name for name in self._first_expansions if name in self._servers]

    # -- install (Requirements 6.2, 6.3, 16.2, 16.3, 6.8, 16.7) -------------

    def install_server(self, name: str) -> InstallReport:
        """Install a single server plus ``geo-common`` (Requirements 6.2, 6.3).

        Installs ``name`` and its dependency on ``geo-common`` and **no** other
        MCP server. Unknown ``name`` raises :class:`ValidationError`. On install
        failure the operation rolls back atomically and raises
        :class:`UpstreamError` naming the failed module (Requirement 6.8).
        """
        self._require_known([name])
        return self._install_atomic([name])

    def install_set(self, names: "Sequence[str]") -> InstallReport:
        """Install a set of servers together, atomically (Req 16.2, 16.7).

        Every requested module must be known/installable; otherwise the request
        is rejected with a :class:`ValidationError` identifying the offending
        modules and **nothing is installed**. If any install fails, the whole
        set is rolled back and an :class:`UpstreamError` names the failed
        module(s) (Requirement 16.7).
        """
        self._require_known(list(names))
        return self._install_atomic(list(names))

    def install_mvp_set(self) -> InstallReport:
        """Install the MVP module set together (Requirements 16.2, 16.3).

        Installs ``geo-common``, ``geo-stac``, ``geo-vector``, ``geo-ops``, and
        ``geo-foundation-models`` as one unit and reports each as installed and
        able to start.
        """
        return self._install_atomic(self.mvp_install_set())

    def _install_atomic(self, requested: "Sequence[str]") -> InstallReport:
        """Install ``requested`` (+ ``geo-common``) atomically with rollback.

        ``geo-common`` is always ensured first so every server has its shared
        base (Requirement 6.3). Already-installed packages are left untouched
        and are **not** part of this operation's rollback set, so installing one
        server never disturbs others (Requirements 6.3, 6.6). Any failure rolls
        back exactly what this operation installed, leaving nothing partially
        installed (Requirements 6.8, 16.7).
        """
        ordered = self._ordered_packages(requested)
        newly_installed: "List[str]" = []
        try:
            for package in ordered:
                if self._runner.is_installed(package):
                    # Pre-existing install: not part of this operation.
                    continue
                self._runner.install(package)
                newly_installed.append(package)
        except UvxCommandError as exc:
            rolled_back = self._rollback(newly_installed)
            raise UpstreamError(
                "installation failed for %s; rolled back, nothing was left "
                "partially installed" % exc.package,
                source="installation-manager",
                detail={
                    "failed_module": exc.package,
                    "requested": list(requested),
                    "rolled_back": rolled_back,
                },
                original=exc.detail,
            ) from exc

        installed_now = self._runner.list_installed()
        views = [self._installed_view(name) for name in requested]
        all_installed = all(name in installed_now for name in requested)
        return InstallReport(
            requested=list(requested),
            installed=views,
            all_installed=all_installed,
            all_can_start=all_installed and all(v.can_start for v in views),
        )

    def _ordered_packages(self, requested: "Sequence[str]") -> "List[str]":
        """``geo-common`` first, then the requested servers (de-duplicated)."""
        ordered: "List[str]" = [GEO_COMMON]
        for name in requested:
            if name not in ordered:
                ordered.append(name)
        return ordered

    def _rollback(self, newly_installed: "Sequence[str]") -> "List[str]":
        """Uninstall, in reverse order, everything this operation installed.

        Best-effort: a failure to uninstall one package does not prevent
        attempting the others. Returns the packages that were rolled back.
        """
        rolled_back: "List[str]" = []
        for package in reversed(list(newly_installed)):
            try:
                self._runner.uninstall(package)
                rolled_back.append(package)
            except UvxCommandError:
                # Keep rolling back the rest; the package may already be gone.
                continue
        return rolled_back

    # -- list (Requirement 6.5) ---------------------------------------------

    def list_installed(self) -> "List[InstalledServerView]":
        """List installed servers with their versions (Requirement 6.5).

        Returns a view per installed, known package - including ``geo-common`` -
        with its version, pillar, and whether it can start.
        """
        installed = self._runner.list_installed()
        views: "List[InstalledServerView]" = []
        for name, version in installed.items():
            if name not in self._servers:
                continue
            views.append(self._view(name, version))
        views.sort(key=lambda v: v.name)
        return views

    # -- remove (Requirements 6.6, 6.9) -------------------------------------

    def remove_server(self, name: str) -> InstalledServerView:
        """Remove a single server, leaving ``geo-common`` and others intact.

        Refuses to remove ``geo-common`` itself (it is the shared base every
        server depends on) with a :class:`ValidationError`. Removing a server
        that is not installed raises a :class:`NotFoundError` and leaves all
        installed components unchanged (Requirement 6.9). On success only the
        named server is uninstalled (Requirement 6.6).
        """
        if name == GEO_COMMON:
            raise ValidationError(
                "%s is the shared base package and cannot be removed; removing "
                "a server never removes %s" % (GEO_COMMON, GEO_COMMON),
                source="installation-manager",
                detail={"server": name},
            )
        self._require_known([name])

        version = self._runner.installed_version(name)
        if version is None:
            raise NotFoundError(
                "cannot remove %r: it is not installed" % name,
                source="installation-manager",
                detail={"server": name},
            )

        self._runner.uninstall(name)
        # Report the server that was removed (with the version it had).
        return self._view(name, version)

    # -- internal helpers ---------------------------------------------------

    def _require_known(self, names: "Sequence[str]") -> None:
        """Reject unknown / non-installable module names (Req 16.7, validation).

        ``geo-common`` is accepted (it is installable as the shared base). Any
        name not present in the server manifest is unknown and cannot be
        installed, so the whole request is rejected before anything is
        installed.
        """
        unknown = [
            name
            for name in names
            if name != GEO_COMMON and name not in self._servers
        ]
        if unknown:
            raise ValidationError(
                "unknown / non-installable module(s): %s" % ", ".join(unknown),
                source="installation-manager",
                detail={"unknown_modules": unknown, "known": self.known_servers()},
            )

    def _installed_view(self, name: str) -> InstalledServerView:
        """Build a view for a server expected to be installed after an op."""
        version = self._runner.installed_version(name) or "unknown"
        return self._view(name, version)

    def _view(self, name: str, version: str) -> InstalledServerView:
        """Build an :class:`InstalledServerView`, computing startability."""
        spec = self._servers.get(name)
        pillar = spec.pillar if spec is not None else "unknown"
        blocking = self._missing_required_keys(spec)
        return InstalledServerView(
            name=name,
            version=version,
            pillar=pillar,
            can_start=not blocking,
            blocking_credentials=blocking,
        )

    def _missing_required_keys(self, spec: Optional[ServerSpec]) -> "List[str]":
        """Required ``mcp.json`` keys that are not configured (Req 16.3, 16.6)."""
        if spec is None or not spec.required_credential_keys:
            return []
        configured = self._resolve_configured_keys()
        return [key for key in spec.required_credential_keys if key not in configured]

    def _resolve_configured_keys(self) -> "set":
        """Resolve configured ``mcp.json`` keys (explicit set or environment)."""
        if self._configured_keys is not None:
            return self._configured_keys
        return {key for key, value in os.environ.items() if value}
