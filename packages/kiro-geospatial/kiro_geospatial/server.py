"""Console entry point for the Geospatial Power Pack Hub (``kiro-geospatial``).

Unlike the domain MCP servers (``geo-stac``, ``geo-ops``, ...), the Hub is the
**Power** itself: Kiro consumes it through ``POWER.md``, ``bundle-manifest.json``,
``skills/``, and ``steering/`` (copied into ``~/.kiro/powers/``), and it
coordinates the domain servers via the Onboarding Dashboard, Resource Catalog,
Credential Manager, Orchestration Router, and Installation Manager exposed as
Python APIs from :mod:`kiro_geospatial`.

It therefore is not registered as a tool MCP server in ``mcp.json``; this
``main()`` is the resolvable target for the ``[project.scripts]``
``kiro-geospatial`` entry point and reports that the Hub is installed and which
coordinator components are importable, so ``uvx kiro-geospatial`` (or the
installed console script) verifies a working install.
"""

from __future__ import annotations

import importlib

__all__ = ["main"]

#: The Hub coordinator components, by module and the primary class each exposes.
#: Used only to report which components import cleanly in the current install.
_HUB_COMPONENTS = (
    ("Credential Manager", "kiro_geospatial.credentials", "CredentialManager"),
    ("Resource Catalog", "kiro_geospatial.catalog", "ResourceCatalog"),
    ("Onboarding Dashboard", "kiro_geospatial.dashboard", "OnboardingDashboard"),
    ("Orchestration Router", "kiro_geospatial.orchestration", "OrchestrationRouter"),
    ("Installation Manager", "kiro_geospatial.installation", "InstallationManager"),
    ("Steering Engine", "kiro_geospatial.steering", "SteeringEngine"),
)


def _available_components() -> "list[str]":
    """Return the names of Hub components that import cleanly in this install."""
    available = []
    for label, module_name, attr in _HUB_COMPONENTS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, attr):
            available.append(label)
    return available


def main() -> None:
    """Console entry point declared by ``kiro-geospatial``'s ``pyproject.toml``.

    Reports the Hub identity and the coordinator components available in the
    current install. The Hub is the Power Kiro activates (via ``POWER.md`` /
    ``bundle-manifest.json`` / ``skills`` / ``steering``); the per-domain MCP
    servers are what get registered in ``mcp.json``.
    """
    components = _available_components()
    print(
        "kiro-geospatial: the Geospatial Power Pack Hub is installed.\n"
        "Coordinator components available: %s\n"
        "Activate the 'kiro-geospatial' Power in Kiro, then register the domain "
        "MCP servers you installed (geo-stac, geo-ops, ...) in mcp.json. See "
        "POWER.md and bundle-manifest.json."
        % (", ".join(components) if components else "(none importable)")
    )


if __name__ == "__main__":  # pragma: no cover - module CLI shim
    main()
