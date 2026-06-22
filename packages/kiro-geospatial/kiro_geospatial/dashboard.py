"""The Power Hub Onboarding Dashboard (Requirement 1).

The Onboarding Dashboard is the first thing a developer new to the Geospatial
Power Pack sees. It answers three questions at a glance - *what is installed*,
*what is available*, and *how do I get started* - and tells the user whether
each installed server is actually ready to use. Its contract (see design.md
"Power Hub -> Onboarding Dashboard"):

* Display each **installed** MCP server's name, pillar, and version
  (Requirement 1.1).
* Display each **available but not-installed** MCP server's name, pillar, and
  :class:`~geo_common.models.OpennessTier` (Requirement 1.2).
* Display the :class:`~kiro_geospatial.credentials.CredentialStatus` of every
  credential for each installed server (Requirement 1.3).
* Present getting-started guidance: one ``uvx`` install command per MVP module
  (Requirement 1.4).
* Mark an installed server **not ready** when any of its *Required* credentials
  is not ``Configured`` (i.e. Missing / Invalid / Unverifiable), and name each
  such credential (Requirement 1.5).
* When **no** servers are installed, return an empty installed list and signal
  that nothing is installed (Requirement 1.6).
* Mark an installed server **ready** only when *every* Required credential has
  status ``Configured`` (Requirement 1.7).

The dashboard is a **pure projection** over three sources it does not own: the
server metadata (from ``bundle-manifest.json``), the
:class:`~kiro_geospatial.installation.InstallationManager` (what is installed
and at which version), and the
:class:`~kiro_geospatial.credentials.CredentialManager` (per-credential status).
It reuses those components' models rather than redefining them, so there is one
definition of an installed server, a credential spec, and a credential status
across the Hub.

``render()`` is synchronous (per the design) and reflects the Credential
Manager's *current* knowledge. Call :meth:`OnboardingDashboard.render_live`
first when you want the dashboard to validate present credentials against their
sources before projecting readiness.

Python 3.9+ (``from __future__ import annotations`` keeps the model annotations
resolvable on 3.9).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from geo_common.models import (
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)

from kiro_geospatial.credentials import (
    CredentialManager,
    CredentialStatus,
    CredentialStatusView,
)
from kiro_geospatial.installation import GEO_COMMON, HUB_DEFAULT_NAME

__all__ = [
    "DashboardServerSpec",
    "DashboardServerView",
    "GettingStartedItem",
    "DashboardView",
    "OnboardingDashboard",
    "load_dashboard_specs",
]


# ---------------------------------------------------------------------------
# Input metadata
# ---------------------------------------------------------------------------
class DashboardServerSpec(BaseModel):
    """Static metadata about one MCP server, drawn from the bundle manifest.

    This is the dashboard's view of a server *before* it knows whether the
    server is installed. ``credentials`` carries the full
    :class:`~geo_common.models.CredentialSpec` for each of the server's
    ``mcp.json`` keys (including the classification used to decide readiness),
    while ``openness_tier`` and ``uvx_command`` drive the available-server and
    getting-started sections respectively.
    """

    name: str = Field(min_length=1)
    pillar: str = Field(min_length=1)
    openness_tier: Optional[OpennessTier] = None
    uvx_command: str = ""
    credentials: List[CredentialSpec] = Field(default_factory=list)
    is_mvp: bool = False


# ---------------------------------------------------------------------------
# Output view models (match design.md "Onboarding Dashboard")
# ---------------------------------------------------------------------------
class DashboardServerView(BaseModel):
    """A single server's row in the dashboard (Requirements 1.1-1.3, 1.5, 1.7).

    For an **installed** server ``version`` is set (Requirement 1.1),
    ``credential_status`` lists every credential's status (Requirement 1.3),
    and ``ready`` / ``not_ready_reasons`` report readiness (Requirements 1.5,
    1.7). For an **available** server ``openness_tier`` is set (Requirement
    1.2) and the credential / readiness fields are empty because status is only
    reported for installed servers (Requirement 1.3).
    """

    name: str
    pillar: str
    version: Optional[str] = None  # present for installed servers (Req 1.1)
    openness_tier: Optional[OpennessTier] = None  # present for available (Req 1.2)
    installed: bool
    credential_status: List[CredentialStatusView] = Field(default_factory=list)  # Req 1.3
    ready: bool = False  # Req 1.5 / 1.7
    not_ready_reasons: List[str] = Field(default_factory=list)  # names creds (Req 1.5)


class GettingStartedItem(BaseModel):
    """One MVP module and the ``uvx`` command that installs it (Req 1.4 / 16.1)."""

    server_name: str
    uvx_command: str


class DashboardView(BaseModel):
    """The full dashboard projection returned by :meth:`OnboardingDashboard.render`.

    ``installed`` is empty and ``no_servers_installed`` is ``True`` when nothing
    is installed (Requirement 1.6).
    """

    installed: List[DashboardServerView] = Field(default_factory=list)  # Req 1.1, 1.6
    available: List[DashboardServerView] = Field(default_factory=list)  # Req 1.2
    getting_started: List[GettingStartedItem] = Field(default_factory=list)  # Req 1.4
    no_servers_installed: bool = True  # Req 1.6


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------
#: Manifest openness-tier strings map onto the three canonical
#: :class:`~geo_common.models.OpennessTier` values. The manifest uses a few
#: composite labels ("Open / Free-Tier", "Open (AWS account)") that we fold
#: onto the closest canonical tier for display (Requirement 1.2).
_TIER_ALIASES: Dict[str, OpennessTier] = {
    "open": OpennessTier.OPEN,
    "open / free-tier": OpennessTier.OPEN,
    "open (aws account)": OpennessTier.OPEN,
    "free-tier": OpennessTier.FREE_TIER,
    "proprietary/licensed": OpennessTier.PROPRIETARY,
}


def _parse_tier(raw: object) -> Optional[OpennessTier]:
    """Map a manifest tier string onto a canonical :class:`OpennessTier`.

    Unknown / missing strings map to ``None`` so the dashboard still renders
    the server (just without a tier) rather than failing to load.
    """
    if not raw:
        return None
    return _TIER_ALIASES.get(str(raw).strip().lower())


def load_dashboard_specs(
    manifest: Mapping[str, object],
) -> "tuple[Dict[str, DashboardServerSpec], str]":
    """Build dashboard server specs from a ``bundle-manifest.json`` mapping.

    Returns ``(specs, hub_name)`` where ``specs`` maps server name ->
    :class:`DashboardServerSpec` and ``hub_name`` is the Power Hub package name
    (excluded from the installed/available server lists by default - it is the
    component doing the rendering, not an à-la-carte MCP server).
    """
    specs: "Dict[str, DashboardServerSpec]" = {}
    servers = manifest.get("servers", [])  # type: ignore[assignment]
    for entry in servers:  # type: ignore[assignment]
        credentials: List[CredentialSpec] = []
        for cred in entry.get("credentials", []):  # type: ignore[union-attr]
            credentials.append(
                CredentialSpec(
                    source=str(cred.get("source") or cred["key"]),
                    mcp_json_key=str(cred["key"]),
                    classification=CredentialClassification(cred["classification"]),
                    license_reference=cred.get("license_reference"),
                )
            )
        name = str(entry["name"])
        specs[name] = DashboardServerSpec(
            name=name,
            pillar=str(entry.get("pillar", "")) or "unknown",
            openness_tier=_parse_tier(entry.get("tier")),
            uvx_command=str(entry.get("uvx", "")),
            credentials=credentials,
            is_mvp=entry.get("status") == "MVP",
        )
    hub_name = str(manifest.get("power", HUB_DEFAULT_NAME))
    return specs, hub_name


# ---------------------------------------------------------------------------
# Onboarding Dashboard
# ---------------------------------------------------------------------------
class OnboardingDashboard:
    """Projects installed/available servers, credential status, and readiness (Req 1).

    Parameters
    ----------
    servers:
        Server metadata keyed by name (typically from
        :func:`load_dashboard_specs`).
    installation_manager:
        The Hub's :class:`~kiro_geospatial.installation.InstallationManager`,
        consulted for which servers are installed and at which version
        (Requirement 1.1).
    credential_manager:
        Optional :class:`~kiro_geospatial.credentials.CredentialManager`,
        consulted for each credential's status (Requirements 1.3, 1.5, 1.7).
        When ``None``, installed servers report their credentials as
        ``Unverifiable`` (no manager to ask), so a server with Required
        credentials is reported not ready.
    hub_name:
        The Power Hub package name; excluded from the installed/available lists.
    excluded:
        Server names that are not à-la-carte MCP servers and so never appear in
        the installed/available lists. Defaults to the hub package and
        ``geo-common`` (the shared base installed as every server's dependency).
    """

    def __init__(
        self,
        *,
        servers: "Mapping[str, DashboardServerSpec]",
        installation_manager: object,
        credential_manager: Optional[CredentialManager] = None,
        hub_name: str = HUB_DEFAULT_NAME,
        excluded: Optional[Iterable[str]] = None,
    ) -> None:
        self._servers: "Dict[str, DashboardServerSpec]" = dict(servers)
        self._install = installation_manager
        self._credentials = credential_manager
        self._hub_name = hub_name
        self._excluded = (
            set(excluded) if excluded is not None else {hub_name, GEO_COMMON}
        )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_manifest_data(
        cls,
        manifest: Mapping[str, object],
        *,
        installation_manager: object,
        credential_manager: Optional[CredentialManager] = None,
        excluded: Optional[Iterable[str]] = None,
    ) -> "OnboardingDashboard":
        """Build a dashboard from an already-parsed ``bundle-manifest.json`` mapping."""
        specs, hub_name = load_dashboard_specs(manifest)
        return cls(
            servers=specs,
            installation_manager=installation_manager,
            credential_manager=credential_manager,
            hub_name=hub_name,
            excluded=excluded,
        )

    # -- rendering ----------------------------------------------------------

    def render(self) -> DashboardView:
        """Project the current state into a :class:`DashboardView` (Requirement 1).

        Synchronous: it reflects the Credential Manager's *current* knowledge
        without performing network validation. Use :meth:`render_live` to
        validate present credentials first.
        """
        return self._render(self._credential_status_lookup())

    async def render_live(self) -> DashboardView:
        """Validate present credentials against their sources, then render.

        Awaits the Credential Manager's live report (Requirements 3.3/3.4) so
        readiness reflects validated ``Configured`` statuses (Requirement 1.7),
        then projects exactly as :meth:`render` does.
        """
        if self._credentials is not None:
            await self._credentials.report()
        return self._render(self._credential_status_lookup())

    # -- internal -----------------------------------------------------------

    def _render(self, status_by_source: "Dict[str, CredentialStatusView]") -> DashboardView:
        installed_versions = self._installed_versions()

        installed: List[DashboardServerView] = []
        available: List[DashboardServerView] = []

        for name, spec in self._servers.items():
            if name in self._excluded:
                continue
            if name in installed_versions:
                installed.append(
                    self._installed_view(
                        spec, installed_versions[name], status_by_source
                    )
                )
            else:
                available.append(self._available_view(spec))

        return DashboardView(
            installed=installed,
            available=available,
            getting_started=self._getting_started(),
            no_servers_installed=not installed,  # Req 1.6
        )

    def _installed_view(
        self,
        spec: DashboardServerSpec,
        version: str,
        status_by_source: "Dict[str, CredentialStatusView]",
    ) -> DashboardServerView:
        """Build the row for an installed server (Req 1.1, 1.3, 1.5, 1.7)."""
        credential_status: List[CredentialStatusView] = []
        not_ready_reasons: List[str] = []

        for cred in spec.credentials:
            view = self._status_for(cred, status_by_source)
            credential_status.append(view)
            # Readiness is decided by Required credentials only (Req 1.5/1.7).
            if (
                cred.classification is CredentialClassification.REQUIRED
                and view.status is not CredentialStatus.CONFIGURED
            ):
                not_ready_reasons.append(
                    "required credential %r (source %r) is %s"
                    % (cred.mcp_json_key, cred.source, view.status.value)
                )

        # Ready iff every Required credential is Configured. A server with no
        # Required credentials is trivially ready (Requirement 1.7).
        ready = not not_ready_reasons
        return DashboardServerView(
            name=spec.name,
            pillar=spec.pillar,
            version=version,
            installed=True,
            credential_status=credential_status,
            ready=ready,
            not_ready_reasons=not_ready_reasons,
        )

    def _available_view(self, spec: DashboardServerSpec) -> DashboardServerView:
        """Build the row for an available, not-installed server (Req 1.2)."""
        return DashboardServerView(
            name=spec.name,
            pillar=spec.pillar,
            openness_tier=spec.openness_tier,
            installed=False,
            credential_status=[],
            ready=False,
            not_ready_reasons=[],
        )

    def _status_for(
        self,
        cred: CredentialSpec,
        status_by_source: "Dict[str, CredentialStatusView]",
    ) -> CredentialStatusView:
        """Resolve this credential's status view (Requirement 1.3).

        The view's *identity* (key, classification, license reference) always
        comes from the credential's own spec, so every credential is labelled
        with its own ``mcp.json`` key even when several credentials share one
        source string. The *status* (and any error category) is taken from the
        Credential Manager's report for that source. When the manager does not
        cover the source (or there is no manager) the credential cannot be
        verified, so it is reported ``Unverifiable`` - which keeps a
        Required-but-unknown credential from being mistaken for ``Configured``
        (Requirement 1.7).
        """
        record = status_by_source.get(cred.source)
        status = record.status if record is not None else CredentialStatus.UNVERIFIABLE
        error_category = record.error_category if record is not None else None
        return CredentialStatusView(
            source=cred.source,
            mcp_json_key=cred.mcp_json_key,
            classification=cred.classification,
            status=status,
            license_reference=(
                cred.license_reference
                if cred.classification is CredentialClassification.LICENSE_NEEDED
                else None
            ),
            error_category=error_category,
        )

    def _getting_started(self) -> List[GettingStartedItem]:
        """One getting-started item per MVP module with a ``uvx`` command (Req 1.4).

        Modules whose manifest ``uvx`` entry is not an actual install command
        (e.g. ``geo-common``, which is installed as a dependency) are skipped so
        the guidance only lists commands a user can run.
        """
        items: List[GettingStartedItem] = []
        for spec in self._servers.values():
            if not spec.is_mvp:
                continue
            command = spec.uvx_command.strip()
            if not command.startswith("uvx"):
                continue
            items.append(
                GettingStartedItem(server_name=spec.name, uvx_command=command)
            )
        return items

    def _installed_versions(self) -> "Dict[str, str]":
        """Map installed server name -> version via the Installation Manager (Req 1.1)."""
        try:
            views = self._install.list_installed()
        except Exception:
            return {}
        return {view.name: view.version for view in views}

    def _credential_status_lookup(self) -> "Dict[str, CredentialStatusView]":
        """Build a source -> status view lookup from the Credential Manager."""
        if self._credentials is None:
            return {}
        return {view.source: view for view in self._credentials.status_report()}
