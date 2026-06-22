"""kiro-geospatial: the Power Hub for the Geospatial Power Pack.

The Power Hub is the central coordinator (Requirements 1-4, 6, 16). It provides:

* an **Onboarding Dashboard** (Requirement 1),
* a searchable **Resource Catalog** (Requirement 2),
* a **Credential Manager** over the single ``mcp.json`` surface (Requirement 3),
* a discover -> process -> analyze **Orchestration Router** (Requirement 4),
* and the **Installation Manager** that installs, lists, and removes individual
  MCP servers via ``uvx`` (Requirements 6, 16).

This module re-exports each Hub component's public surface. Component modules
land independently, so optional imports are guarded to keep the package
importable while the Hub is being assembled.
"""

from __future__ import annotations

# Credential Manager (Requirement 3) - always available.
from kiro_geospatial.credentials import (
    CredentialClassification,
    CredentialManager,
    CredentialRecord,
    CredentialRedactionFilter,
    CredentialSpec,
    CredentialStatus,
    CredentialStatusView,
    Validator,
)

__all__ = [
    # Credential Manager (Requirement 3)
    "CredentialClassification",
    "CredentialStatus",
    "CredentialSpec",
    "CredentialStatusView",
    "CredentialRecord",
    "CredentialManager",
    "CredentialRedactionFilter",
    "Validator",
]

# Installation Manager (Requirements 6, 16) - re-exported when its module is
# present. Guarded so the Hub package imports cleanly even before that module
# lands during incremental assembly.
try:  # pragma: no cover - import wiring
    from kiro_geospatial.installation import (
        GEO_COMMON,
        HUB_DEFAULT_NAME,
        InstallationManager,
        InstallReport,
        InstalledServerView,
        ServerSpec,
        SubprocessUvxRunner,
        UvxCommandError,
        UvxRunner,
        load_server_specs,
    )
except ImportError:
    pass
else:
    __all__ += [
        "InstallationManager",
        "UvxRunner",
        "SubprocessUvxRunner",
        "UvxCommandError",
        "ServerSpec",
        "InstalledServerView",
        "InstallReport",
        "load_server_specs",
        "GEO_COMMON",
        "HUB_DEFAULT_NAME",
    ]

# Onboarding Dashboard (Requirement 1) - re-exported when its module is
# present. Guarded for the same incremental-assembly reason as below.
try:  # pragma: no cover - import wiring
    from kiro_geospatial.dashboard import (
        DashboardServerSpec,
        DashboardServerView,
        DashboardView,
        GettingStartedItem,
        OnboardingDashboard,
        load_dashboard_specs,
    )
except ImportError:
    pass
else:
    __all__ += [
        "OnboardingDashboard",
        "DashboardView",
        "DashboardServerView",
        "DashboardServerSpec",
        "GettingStartedItem",
        "load_dashboard_specs",
    ]

# Orchestration Router (Requirement 4) - re-exported when its module is
# present. Guarded for the same incremental-assembly reason as above; the
# planning portion lands before execution/provenance.
try:  # pragma: no cover - import wiring
    from kiro_geospatial.orchestration import (
        KIND_ORDER,
        CapabilityResolver,
        CapabilitySpec,
        CatalogCapabilityResolver,
        DEFAULT_SOURCE_TIMEOUT_S,
        MappingCapabilityResolver,
        OrchestrationPlan,
        OrchestrationResult,
        OrchestrationRouter,
        PlanStep,
        PlanStepKind,
        ProvenanceRecord,
        ServerInvoker,
        SourceOutcome,
        StepProvenance,
    )
except ImportError:
    pass
else:
    __all__ += [
        "PlanStepKind",
        "PlanStep",
        "OrchestrationPlan",
        "KIND_ORDER",
        "CapabilityResolver",
        "MappingCapabilityResolver",
        "CatalogCapabilityResolver",
        "CapabilitySpec",
        "ServerInvoker",
        "SourceOutcome",
        "StepProvenance",
        "ProvenanceRecord",
        "OrchestrationResult",
        "OrchestrationRouter",
        "DEFAULT_SOURCE_TIMEOUT_S",
    ]

# MVP wiring / Hub assembly (Requirements 2.1, 4.2, 16.2, 16.3) - re-exported
# when its module is present. Guarded for the same incremental-assembly reason
# as above; it depends on the catalog, credential, and orchestration modules.
try:  # pragma: no cover - import wiring
    from kiro_geospatial.wiring import (
        MVP_CAPABILITY_SPECS,
        MVP_SERVER_NAMES,
        BoundServerInvoker,
        MvpHub,
        MvpStartupReport,
        ServerBinding,
        assemble_mvp_hub,
        default_mvp_bindings,
    )
except ImportError:
    pass
else:
    __all__ += [
        "assemble_mvp_hub",
        "default_mvp_bindings",
        "MvpHub",
        "MvpStartupReport",
        "BoundServerInvoker",
        "ServerBinding",
        "MVP_CAPABILITY_SPECS",
        "MVP_SERVER_NAMES",
    ]

# Steering activation engine (Requirement 14) - re-exported when its module is
# present. Guarded for the same incremental-assembly reason as above.
try:  # pragma: no cover - import wiring
    from kiro_geospatial.steering import (
        DeactivationReport,
        SteeringEngine,
        SteeringUpdate,
        SteeringWorkflow,
        StepPresentation,
        StepPresentationError,
        load_workflows_from_directory,
    )
except ImportError:
    pass
else:
    __all__ += [
        "SteeringEngine",
        "SteeringWorkflow",
        "load_workflows_from_directory",
        "StepPresentation",
        "DeactivationReport",
        "SteeringUpdate",
        "StepPresentationError",
    ]
