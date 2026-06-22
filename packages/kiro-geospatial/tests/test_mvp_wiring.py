"""Tests for the Hub MVP wiring / assembly (Requirements 2.1, 4.2, 16.2, 16.3).

Covers :mod:`kiro_geospatial.wiring`:

* **Catalog registration (Requirement 2.1).** Every bound MVP server's catalog
  entries are registered and discoverable, marked installed with no leftover
  install command (Requirement 2.6).
* **Credential registration (Requirement 16.1).** Every bound server's
  credential specs are registered with the Credential Manager and appear in its
  status report over the single ``mcp.json`` surface.
* **Router wiring (Requirement 4.2).** A single natural-language request is
  decomposed into an ordered discover -> process -> analyze plan that resolves
  each capability to its providing server, and executes across the bound
  servers via the :class:`BoundServerInvoker`, returning provenance that honors
  the pillar-ordering invariant (Requirement 4.1).
* **Startup (Requirements 16.3-16.6).** The MVP set (credentials all Optional)
  all starts; a server with a missing Required credential is reported blocked.
* **Decoupled bindings.** The assembly works with injected fakes (no concrete
  server-package imports). A guarded integration check wires the real MVP
  servers when their packages are importable.

The fake servers subclass :class:`~geo_common.server.BaseGeoServer` - the exact
contract the wiring expects - so these tests exercise the real assembly path
without importing the separate MVP server packages.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from geo_common.errors import GeoError, NotFoundError
from geo_common.models import (
    CatalogEntry,
    CredentialClassification,
    CredentialSpec,
    OpennessTier,
)
from geo_common.server import BaseGeoServer

from kiro_geospatial.catalog import CatalogQuery
from kiro_geospatial.orchestration import KIND_ORDER, PlanStep, PlanStepKind
from kiro_geospatial.wiring import (
    MVP_SERVER_NAMES,
    BoundServerInvoker,
    MvpHub,
    assemble_mvp_hub,
    default_mvp_bindings,
)


# --------------------------------------------------------------------------- #
# Fake MVP servers - BaseGeoServer subclasses mirroring the real MVP surface.
# --------------------------------------------------------------------------- #
class FakeStac(BaseGeoServer):
    pillar = "A"
    server_name = "geo-stac"
    version = "0.1.0"

    def __init__(self) -> None:
        super().__init__()
        self.register_tool("stac_search", self.stac_search)

    async def stac_search(self, **params) -> dict:
        return {"items": [{"id": "scene-1"}], "params": params}

    def catalog_entries(self) -> List[CatalogEntry]:
        return [
            CatalogEntry(
                name="stac_search",
                pillar=self.pillar,
                capability_description="Search STAC catalogs for imagery items.",
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                # Not-installed defaults: the Hub should normalize these.
                installed=False,
                install_command="uvx geo-stac",
            )
        ]

    def required_credentials(self) -> List[CredentialSpec]:
        return [
            CredentialSpec(
                source="Microsoft Planetary Computer",
                mcp_json_key="PC_SDK_SUBSCRIPTION_KEY",
                classification=CredentialClassification.OPTIONAL,
            )
        ]


class FakeVector(BaseGeoServer):
    pillar = "A"
    server_name = "geo-vector"
    version = "0.1.0"

    def __init__(self) -> None:
        super().__init__()
        self.register_tool("vector_features", self.vector_features)

    async def vector_features(self, **params) -> dict:
        return {"features": [{"type": "Feature"}], "params": params}

    def catalog_entries(self) -> List[CatalogEntry]:
        return [
            CatalogEntry(
                name="vector_features",
                pillar=self.pillar,
                capability_description="Fetch OSM/Overture vector features.",
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
            )
        ]

    def required_credentials(self) -> List[CredentialSpec]:
        return []


class FakeOps(BaseGeoServer):
    pillar = "B"
    server_name = "geo-ops"
    version = "0.1.0"

    def __init__(self) -> None:
        super().__init__()
        self.register_tool("transform_crs", self.transform_crs)

    async def transform_crs(self, **params) -> dict:
        return {"reprojected": True, "params": params}

    def catalog_entries(self) -> List[CatalogEntry]:
        return [
            CatalogEntry(
                name="transform_crs",
                pillar=self.pillar,
                capability_description="Reproject a geometry to a target CRS.",
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                installed=True,
            )
        ]

    def required_credentials(self) -> List[CredentialSpec]:
        return []


class FakeFoundation(BaseGeoServer):
    pillar = "C"
    server_name = "geo-foundation-models"
    version = "0.1.0"

    def __init__(self) -> None:
        super().__init__()
        self.register_tool("embed_tile", self.embed_tile)

    async def embed_tile(self, **params) -> dict:
        return {"embedding": [0.1, 0.2, 0.3], "params": params}

    def catalog_entries(self) -> List[CatalogEntry]:
        return [
            CatalogEntry(
                name="embed_tile",
                pillar=self.pillar,
                capability_description="Foundation-model embedding for a tile.",
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                installed=True,
            )
        ]

    def required_credentials(self) -> List[CredentialSpec]:
        return [
            CredentialSpec(
                source="Hugging Face (model weights)",
                mcp_json_key="HF_TOKEN",
                classification=CredentialClassification.OPTIONAL,
            )
        ]


def _fake_bindings() -> dict:
    servers = [FakeStac(), FakeVector(), FakeOps(), FakeFoundation()]
    return {s.server_name: s for s in servers}


def _hub(surface: Optional[dict] = None) -> MvpHub:
    return assemble_mvp_hub(_fake_bindings(), surface=surface or {})


# --------------------------------------------------------------------------- #
# Catalog registration (Requirement 2.1)
# --------------------------------------------------------------------------- #
def test_assembly_registers_all_mvp_catalog_entries():
    hub = _hub()
    names = {e.name for e in hub.catalog.entries}
    assert names == {"stac_search", "vector_features", "transform_crs", "embed_tile"}
    # Each provider is one of the four MVP servers.
    providers = {e.provider_server for e in hub.catalog.entries}
    assert providers == set(MVP_SERVER_NAMES)


def test_registered_entries_are_marked_installed_with_no_install_command():
    # A bound server is installed in this Hub; its entries must reflect that
    # (Requirement 2.6), even if the server declared a not-installed default.
    hub = _hub()
    for entry in hub.catalog.entries:
        assert entry.installed is True
        assert entry.install_command is None


def test_catalog_is_searchable_after_assembly():
    hub = _hub()
    result = hub.catalog.search(CatalogQuery(keyword="imagery"))
    assert [e.name for e in result.entries] == ["stac_search"]

    pillar_c = hub.catalog.search(CatalogQuery(pillar="C"))
    assert [e.name for e in pillar_c.entries] == ["embed_tile"]


# --------------------------------------------------------------------------- #
# Credential registration (Requirement 16.1 / 3.1)
# --------------------------------------------------------------------------- #
def test_assembly_registers_all_mvp_credential_specs():
    hub = _hub(surface={})
    sources = set(hub.credentials.sources())
    assert {"Microsoft Planetary Computer", "Hugging Face (model weights)"} <= sources
    report = hub.credentials.status_report()
    keys = {view.mcp_json_key for view in report}
    assert {"PC_SDK_SUBSCRIPTION_KEY", "HF_TOKEN"} <= keys


def test_status_report_reflects_configured_surface():
    hub = _hub(surface={"PC_SDK_SUBSCRIPTION_KEY": "abc123"})
    by_key = {v.mcp_json_key: v for v in hub.credentials.status_report()}
    # Present (configured) -> not Missing; absent -> Missing.
    assert by_key["PC_SDK_SUBSCRIPTION_KEY"].status.value != "Missing"
    assert by_key["HF_TOKEN"].status.value == "Missing"


# --------------------------------------------------------------------------- #
# Router wiring: discover -> process -> analyze (Requirements 4.1, 4.2)
# --------------------------------------------------------------------------- #
def test_plan_flows_discover_process_analyze_across_servers():
    hub = _hub()
    plan = hub.plan(
        "Discover imagery scenes and vector buildings, reproject the geometry, "
        "then analyze with a foundation-model embedding"
    )

    # Every MVP capability is planned, resolved to its providing server (Req 4.2).
    capability_to_source = {
        step.capability: step.candidate_sources for step in plan.steps
    }
    assert capability_to_source["stac_search"] == ["geo-stac"]
    assert capability_to_source["vector_features"] == ["geo-vector"]
    assert capability_to_source["transform_crs"] == ["geo-ops"]
    assert capability_to_source["embed_tile"] == ["geo-foundation-models"]

    # Pillar-ordering invariant: ranks are non-decreasing (Requirement 4.1).
    ranks = [KIND_ORDER[step.kind] for step in plan.steps]
    assert ranks == sorted(ranks)
    assert ranks[0] == KIND_ORDER[PlanStepKind.DISCOVER]
    assert ranks[-1] == KIND_ORDER[PlanStepKind.ANALYZE]


async def test_run_executes_single_request_end_to_end():
    hub = _hub()
    result = await hub.run(
        "discover imagery and vector features, reproject, then embed for analysis"
    )

    assert result.partial is False
    assert result.failed_sources == []
    # Every MVP server contributed (Requirement 4.4).
    assert set(result.contributing_sources) == set(MVP_SERVER_NAMES)
    # The analysis is the analyze step's (embed_tile) result.
    assert result.analysis["embedding"] == [0.1, 0.2, 0.3]
    # Provenance honors discover < process < analyze ordering (Requirement 4.1).
    prov_ranks = [KIND_ORDER[p.kind] for p in result.provenance.steps]
    assert prov_ranks == sorted(prov_ranks)


async def test_run_discover_only_request_has_no_analysis():
    hub = _hub()
    result = await hub.run("discover imagery scenes for this area")
    assert result.analysis is None
    assert result.partial is False
    assert result.contributing_sources == ["geo-stac"]


# --------------------------------------------------------------------------- #
# BoundServerInvoker dispatch (Requirement 4.2)
# --------------------------------------------------------------------------- #
async def test_invoker_dispatches_capability_to_bound_server_tool_with_params():
    bindings = _fake_bindings()
    invoker = BoundServerInvoker(bindings)
    step = PlanStep(
        step_id="step-1",
        kind=PlanStepKind.DISCOVER,
        capability="stac_search",
        candidate_sources=["geo-stac"],
        params={"bbox": [0, 0, 1, 1]},
    )
    out = await invoker.invoke(step, "geo-stac")
    assert out["items"] == [{"id": "scene-1"}]
    # Step params are forwarded to the tool call.
    assert out["params"] == {"bbox": [0, 0, 1, 1]}


async def test_invoker_unknown_source_raises_not_found():
    invoker = BoundServerInvoker(_fake_bindings())
    step = PlanStep(
        step_id="step-1",
        kind=PlanStepKind.PROCESS,
        capability="transform_crs",
        candidate_sources=["nope"],
    )
    with pytest.raises(NotFoundError):
        await invoker.invoke(step, "nope")


async def test_invoker_unknown_capability_raises_geo_error():
    invoker = BoundServerInvoker(_fake_bindings())
    step = PlanStep(
        step_id="step-1",
        kind=PlanStepKind.PROCESS,
        capability="does_not_exist",
        candidate_sources=["geo-ops"],
    )
    with pytest.raises(GeoError):
        await invoker.invoke(step, "geo-ops")


# --------------------------------------------------------------------------- #
# Startup credential guard (Requirements 16.3-16.6)
# --------------------------------------------------------------------------- #
def test_mvp_servers_all_start_with_no_credentials_configured():
    # MVP credentials are all Optional, so the whole set starts (Req 16.3/16.5).
    hub = _hub(surface={})
    report = hub.start(configured_keys=[])
    assert report.all_started is True
    assert set(report.started) == set(MVP_SERVER_NAMES)
    assert report.blocking == {}


def test_server_with_missing_required_credential_is_reported_blocked():
    # Make geo-stac's credential Required so an empty surface blocks it (Req 16.6).
    class RequiredStac(FakeStac):
        def required_credentials(self) -> List[CredentialSpec]:
            return [
                CredentialSpec(
                    source="Microsoft Planetary Computer",
                    mcp_json_key="PC_SDK_SUBSCRIPTION_KEY",
                    classification=CredentialClassification.REQUIRED,
                )
            ]

    bindings = _fake_bindings()
    bindings["geo-stac"] = RequiredStac()
    hub = assemble_mvp_hub(bindings, surface={})

    report = hub.start(configured_keys=[])
    assert report.all_started is False
    assert report.blocking["geo-stac"] == ["PC_SDK_SUBSCRIPTION_KEY"]
    # The other three still start (independent of geo-stac, Requirement 16.4).
    assert set(report.started) == {"geo-vector", "geo-ops", "geo-foundation-models"}


# --------------------------------------------------------------------------- #
# Integration: wire the real MVP servers when their packages are importable.
# --------------------------------------------------------------------------- #
def test_default_mvp_bindings_wire_real_servers_when_importable():
    try:
        bindings = default_mvp_bindings()
    except NotFoundError:
        pytest.skip("MVP server packages are not importable in this environment")

    assert set(bindings) == set(MVP_SERVER_NAMES)
    hub = assemble_mvp_hub(bindings, surface={})

    # All four real capabilities are registered and resolve to their servers.
    names = {e.name for e in hub.catalog.entries}
    assert {"stac_search", "vector_features", "transform_crs", "embed_tile"} <= names

    plan = hub.plan(
        "discover imagery and vector features, reproject, then embed for analysis"
    )
    cap_to_source = {s.capability: s.candidate_sources for s in plan.steps}
    assert cap_to_source["stac_search"] == ["geo-stac"]
    assert cap_to_source["vector_features"] == ["geo-vector"]
    assert cap_to_source["transform_crs"] == ["geo-ops"]
    assert cap_to_source["embed_tile"] == ["geo-foundation-models"]
    # discover < process < analyze ordering holds for the real wiring (Req 4.1).
    ranks = [KIND_ORDER[s.kind] for s in plan.steps]
    assert ranks == sorted(ranks)
