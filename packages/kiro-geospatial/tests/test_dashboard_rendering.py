"""Snapshot/example tests for the Onboarding Dashboard projection (Req 1.5-1.7).

The Onboarding Dashboard is a *pure projection* over three sources it does not
own: server metadata, the Installation Manager (what is installed + version),
and the Credential Manager (per-credential status). These tests pin the exact
shape of that projection - a "snapshot" of ``DashboardView.model_dump()`` - for
the three states the task calls out:

* **ready** - an installed server whose every Required credential is
  ``Configured`` is marked ready with no not-ready reasons (Requirement 1.7);
* **not ready** - an installed server with a Required credential that is
  Missing and another that is Invalid is marked not ready, and each such
  credential is named in ``not_ready_reasons`` (Requirement 1.5);
* **empty-installed** - when nothing is installed, the installed list is empty
  and ``no_servers_installed`` is ``True`` (Requirement 1.6).

The dashboard talks to the Installation Manager only through ``list_installed``,
so we drive it with a tiny ``FakeInstallationManager`` returning
``InstalledServerView`` rows. Credential status comes from a *real*
``CredentialManager`` over a mock ``mcp.json`` surface (a plain dict) plus
in-memory validators, so the Configured/Invalid/Missing statuses are produced
by the real status pipeline rather than stubbed views. ``render_live`` is used
where a present credential must be validated first (Configured / Invalid);
``render`` is used for the empty state where no credential is consulted.

Example-based (per the design's "dashboard rendering snapshots"); the repo's
``asyncio_mode = "auto"`` lets the ``async def`` tests run directly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from geo_common.models import (
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from kiro_geospatial.credentials import CredentialManager
from kiro_geospatial.dashboard import DashboardServerSpec, OnboardingDashboard
from kiro_geospatial.installation import InstalledServerView

REQUIRED = CredentialClassification.REQUIRED


# --------------------------------------------------------------------------- #
# Fakes / fixtures
# --------------------------------------------------------------------------- #
class FakeInstallationManager:
    """A minimal stand-in exposing only what the dashboard uses.

    The dashboard reads installed servers exclusively through
    ``list_installed()`` (``name`` + ``version`` per row), so a fake returning
    :class:`InstalledServerView` rows is enough to exercise the projection.
    """

    def __init__(self, installed: Optional[Dict[str, str]] = None) -> None:
        self._installed = dict(installed or {})

    def list_installed(self) -> List[InstalledServerView]:
        return [
            InstalledServerView(name=name, version=version, pillar="ignored")
            for name, version in self._installed.items()
        ]


def _servers(fm_credentials: List[CredentialSpec]) -> Dict[str, DashboardServerSpec]:
    """A compact two-server vocabulary shared by every snapshot.

    ``geo-stac`` (an MVP, Open, credential-free Pillar A connector) exercises
    the available-server and getting-started sections, while
    ``geo-foundation-models`` (an MVP Pillar C server) carries the Required
    credentials whose status drives readiness. Varying only what is installed
    and what is on the credential surface yields the three target states.
    """
    return {
        "geo-stac": DashboardServerSpec(
            name="geo-stac",
            pillar="Pillar A",
            openness_tier=OpennessTier.OPEN,
            uvx_command="uvx geo-stac",
            credentials=[],
            is_mvp=True,
        ),
        "geo-foundation-models": DashboardServerSpec(
            name="geo-foundation-models",
            pillar="Pillar C",
            openness_tier=OpennessTier.PROPRIETARY,
            uvx_command="uvx geo-foundation-models",
            credentials=fm_credentials,
            is_mvp=True,
        ),
    }


# A geo-stac row as it appears in the *available* list across snapshots.
_GEO_STAC_AVAILABLE = {
    "name": "geo-stac",
    "pillar": "Pillar A",
    "version": None,
    "openness_tier": "Open",
    "installed": False,
    "credential_status": [],
    "ready": False,
    "not_ready_reasons": [],
}

# Getting-started guidance: one item per MVP module with a uvx command (Req 1.4).
_GETTING_STARTED = [
    {"server_name": "geo-stac", "uvx_command": "uvx geo-stac"},
    {"server_name": "geo-foundation-models", "uvx_command": "uvx geo-foundation-models"},
]


# --------------------------------------------------------------------------- #
# Ready state (Requirement 1.7)
# --------------------------------------------------------------------------- #
async def test_ready_state_snapshot():
    """Installed server with every Required credential Configured -> ready."""
    fm_cred = CredentialSpec(
        source="fm-key", mcp_json_key="FM_API_KEY", classification=REQUIRED
    )
    credentials = CredentialManager(
        [fm_cred],
        surface={"FM_API_KEY": "a-present-secret"},
        validators={"fm-key": lambda secret: True},  # source accepts -> Configured
    )
    dashboard = OnboardingDashboard(
        servers=_servers([fm_cred]),
        installation_manager=FakeInstallationManager(
            {"geo-foundation-models": "0.1.0"}
        ),
        credential_manager=credentials,
    )

    # render_live validates the present credential before projecting (Req 1.7).
    view = await dashboard.render_live()

    assert view.model_dump() == {
        "installed": [
            {
                "name": "geo-foundation-models",
                "pillar": "Pillar C",
                "version": "0.1.0",
                "openness_tier": None,
                "installed": True,
                "credential_status": [
                    {
                        "source": "fm-key",
                        "mcp_json_key": "FM_API_KEY",
                        "classification": "Required",
                        "status": "Configured",
                        "license_reference": None,
                        "error_category": None,
                    }
                ],
                "ready": True,
                "not_ready_reasons": [],
            }
        ],
        "available": [_GEO_STAC_AVAILABLE],
        "getting_started": _GETTING_STARTED,
        "no_servers_installed": False,
    }


# --------------------------------------------------------------------------- #
# Not-ready state, naming the missing/invalid credential (Requirement 1.5)
# --------------------------------------------------------------------------- #
async def test_not_ready_state_names_missing_and_invalid_credentials_snapshot():
    """A Missing and an Invalid Required credential -> not ready, both named."""
    fm_missing = CredentialSpec(
        source="fm-key", mcp_json_key="FM_API_KEY", classification=REQUIRED
    )
    fm_invalid = CredentialSpec(
        source="fm-secret", mcp_json_key="FM_SECRET", classification=REQUIRED
    )
    credentials = CredentialManager(
        [fm_missing, fm_invalid],
        # FM_API_KEY absent -> Missing; FM_SECRET present but rejected -> Invalid.
        surface={"FM_SECRET": "a-rejected-secret"},
        validators={"fm-secret": lambda secret: False},
    )
    dashboard = OnboardingDashboard(
        servers=_servers([fm_missing, fm_invalid]),
        installation_manager=FakeInstallationManager(
            {"geo-foundation-models": "0.1.0"}
        ),
        credential_manager=credentials,
    )

    view = await dashboard.render_live()

    assert view.model_dump() == {
        "installed": [
            {
                "name": "geo-foundation-models",
                "pillar": "Pillar C",
                "version": "0.1.0",
                "openness_tier": None,
                "installed": True,
                "credential_status": [
                    {
                        "source": "fm-key",
                        "mcp_json_key": "FM_API_KEY",
                        "classification": "Required",
                        "status": "Missing",
                        "license_reference": None,
                        "error_category": None,
                    },
                    {
                        "source": "fm-secret",
                        "mcp_json_key": "FM_SECRET",
                        "classification": "Required",
                        "status": "Invalid",
                        "license_reference": None,
                        "error_category": None,
                    },
                ],
                "ready": False,
                "not_ready_reasons": [
                    "required credential 'FM_API_KEY' (source 'fm-key') is Missing",
                    "required credential 'FM_SECRET' (source 'fm-secret') is Invalid",
                ],
            }
        ],
        "available": [_GEO_STAC_AVAILABLE],
        "getting_started": _GETTING_STARTED,
        "no_servers_installed": False,
    }


# --------------------------------------------------------------------------- #
# Empty-installed state (Requirement 1.6)
# --------------------------------------------------------------------------- #
def test_empty_installed_state_snapshot():
    """No servers installed -> empty installed list, no_servers_installed True."""
    fm_cred = CredentialSpec(
        source="fm-key", mcp_json_key="FM_API_KEY", classification=REQUIRED
    )
    credentials = CredentialManager([fm_cred], surface={})
    dashboard = OnboardingDashboard(
        servers=_servers([fm_cred]),
        installation_manager=FakeInstallationManager({}),  # nothing installed
        credential_manager=credentials,
    )

    # Synchronous render: with nothing installed no credential status is read.
    view = dashboard.render()

    assert view.model_dump() == {
        "installed": [],
        "available": [
            _GEO_STAC_AVAILABLE,
            {
                "name": "geo-foundation-models",
                "pillar": "Pillar C",
                "version": None,
                "openness_tier": "Proprietary/Licensed",
                "installed": False,
                "credential_status": [],
                "ready": False,
                "not_ready_reasons": [],
            },
        ],
        "getting_started": _GETTING_STARTED,
        "no_servers_installed": True,
    }


# --------------------------------------------------------------------------- #
# Targeted assertions backing the snapshots (clarify intent of Req 1.5-1.7)
# --------------------------------------------------------------------------- #
def test_no_credential_manager_marks_required_server_not_ready():
    """No manager -> Required credentials are Unverifiable, so server not ready."""
    fm_cred = CredentialSpec(
        source="fm-key", mcp_json_key="FM_API_KEY", classification=REQUIRED
    )
    dashboard = OnboardingDashboard(
        servers=_servers([fm_cred]),
        installation_manager=FakeInstallationManager(
            {"geo-foundation-models": "0.1.0"}
        ),
        credential_manager=None,
    )

    view = dashboard.render()

    [installed] = view.installed
    assert installed.ready is False
    [cred] = installed.credential_status
    assert cred.status.value == "Unverifiable"
    assert installed.not_ready_reasons == [
        "required credential 'FM_API_KEY' (source 'fm-key') is Unverifiable"
    ]
