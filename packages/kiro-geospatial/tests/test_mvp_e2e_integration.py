"""End-to-end MVP integration test (task 8.3; Requirements 4.4, 4.7, 4.8).

Where ``test_mvp_wiring.py`` exercises the Hub assembly with *fake* servers and
``test_orchestration_execution.py`` exercises the router with a *programmable*
invoker, this module drives a single **discover -> process -> analyze** request
across the **real** MVP servers - ``geo-stac``, ``geo-vector``, ``geo-ops`` and
``geo-foundation-models`` - assembled into the real Power Hub
(:func:`~kiro_geospatial.wiring.assemble_mvp_hub`). The only thing mocked is the
outermost boundary: each data-connector server's shared
:class:`~geo_common.http.HttpClient` is backed by an ``httpx.MockTransport`` so
the STAC and vector "sources" return deterministic payloads with no real
network I/O, exactly as the connector smoke tests do. The processing
(``transform_crs``) and analysis (``embed_tile``) steps run their real local
PyProj/foundation-model logic.

Two scenarios are covered:

* **Happy path (Requirement 4.4).** Every step succeeds; the request returns an
  analysis result together with provenance that identifies, by source id, every
  source used in each discover/process/analyze step, and the result is not
  partial.
* **Graceful degradation (Requirements 4.5, 4.7, 4.8).** A discovery step has
  two redundant STAC sources; one is unreachable (its mock transport raises a
  connection error) while the other succeeds. The request still completes
  through process and analyze, the result is labeled **partial**, and provenance
  enumerates both the contributing sources and the failed source together with
  its ``Error_Taxonomy`` category.

The MVP server packages are imported via ``PYTHONPATH`` (they are not installed
as wheels in the test environment); run with every MVP package root on
``PYTHONPATH`` - see the module-level skip below, which keeps the suite green in
an environment where the optional connector packages are unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx
import pytest

from geo_common.errors import ErrorCategory
from geo_common.http import HttpClient
from geo_common.retry import RetryPolicy

# The MVP server packages live in separate uvx packages; skip cleanly when any
# of them (or their scientific deps) are not importable in this environment.
geo_stac_server = pytest.importorskip("geo_stac.server")
geo_vector_server = pytest.importorskip("geo_vector.server")
geo_ops_server = pytest.importorskip("geo_ops.server")
geo_fm_server = pytest.importorskip("geo_foundation_models.server")

from geo_foundation_models.models import RasterTile  # noqa: E402
from geo_ops.models import GeoJSONGeometry  # noqa: E402

from kiro_geospatial.orchestration import (  # noqa: E402
    KIND_ORDER,
    OrchestrationPlan,
    OrchestrationRouter,
    PlanStep,
    PlanStepKind,
    MappingCapabilityResolver,
)
from kiro_geospatial.wiring import (  # noqa: E402
    MVP_SERVER_NAMES,
    BoundServerInvoker,
    assemble_mvp_hub,
)

GeoStacServer = geo_stac_server.GeoStacServer
GeoVectorServer = geo_vector_server.GeoVectorServer
GeoOpsServer = geo_ops_server.GeoOpsServer
GeoFoundationModelsServer = geo_fm_server.GeoFoundationModelsServer


# --------------------------------------------------------------------------- #
# Mocked source payloads (deterministic, no network).
# --------------------------------------------------------------------------- #
# A small extent (well under geo-vector's 2,500 km² default cap) and a valid
# datetime window used for the discovery steps.
AOI_BBOX = (13.40, 52.50, 13.45, 52.55)  # central Berlin, ~19 km²
AOI_RANGE = ("2023-01-01T00:00:00Z", "2023-02-01T00:00:00Z")

OVERPASS_HOST = "overpass-api.de"
OVERTURE_HOST = "api.overturemaps.org"


def _stac_feature_collection() -> Dict[str, Any]:
    """A STAC ItemCollection with two Sentinel-2 items (assets + datetime)."""
    def feature(item_id: str) -> Dict[str, Any]:
        return {
            "type": "Feature",
            "id": item_id,
            "collection": "sentinel-2-l2a",
            "bbox": list(AOI_BBOX),
            "properties": {"datetime": "2023-01-15T10:30:00Z", "eo:cloud_cover": 4.0},
            "assets": {
                "visual": {
                    "href": "https://example.com/%s/visual.tif" % item_id,
                    "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                }
            },
        }

    return {"type": "FeatureCollection", "features": [feature("S2-A"), feature("S2-B")]}


_OVERPASS_PAYLOAD = {
    "version": 0.6,
    "generator": "Overpass API",
    "elements": [
        {
            "type": "node",
            "id": 1,
            "lon": 13.41,
            "lat": 52.51,
            "tags": {"amenity": "cafe", "name": "Cafe Kiro"},
        }
    ],
}

_OVERTURE_PAYLOAD = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "id": "overture-1",
            "geometry": {"type": "Point", "coordinates": [13.42, 52.52]},
            "properties": {"class": "restaurant", "name": "Overture Bistro"},
        }
    ],
}


def _stac_handler(payload: Dict[str, Any]):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return handler


def _vector_handler():
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == OVERPASS_HOST:
            return httpx.Response(200, json=_OVERPASS_PAYLOAD)
        if host == OVERTURE_HOST:
            return httpx.Response(200, json=_OVERTURE_PAYLOAD)
        return httpx.Response(404)  # pragma: no cover - unexpected host

    return handler


def _unreachable_handler():
    """A transport that always fails to connect (an unreachable source)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return handler


def _client(handler) -> HttpClient:
    """A no-retry HttpClient wired to a mock transport (fast + deterministic)."""
    return HttpClient(RetryPolicy(max_attempts=1), transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------- #
# Server construction with mocked sources.
# --------------------------------------------------------------------------- #
def _stac_server(handler=None) -> GeoStacServer:
    return GeoStacServer(http=_client(handler or _stac_handler(_stac_feature_collection())))


def _vector_server() -> GeoVectorServer:
    return GeoVectorServer(http=_client(_vector_handler()))


def _mvp_servers():
    """The four real MVP servers, each backed by a mocked source where it has one."""
    return [
        _stac_server(),
        _vector_server(),
        GeoOpsServer(),  # local PyProj/Shapely; no HTTP
        GeoFoundationModelsServer(),  # local foundation-model backend; no HTTP
    ]


# Per-capability params that make each real tool succeed against its source.
def _params_for(capability: str) -> Dict[str, Any]:
    return {
        "stac_search": {"bbox": AOI_BBOX, "datetime_range": AOI_RANGE},
        "vector_features": {"bbox": AOI_BBOX},
        "transform_crs": {
            "geometry": GeoJSONGeometry(type="Point", coordinates=[13.42, 52.52]),
            "src_crs": "EPSG:4326",
            "dst_crs": "EPSG:3857",
        },
        "embed_tile": {
            "tile": RasterTile(width=64, height=64, bands=3, format="GTiff"),
            "model": "Clay",
        },
    }[capability]


def _with_params(plan: OrchestrationPlan) -> OrchestrationPlan:
    """Return a copy of ``plan`` with realistic params attached to each step."""
    steps = [
        step.model_copy(update={"params": _params_for(step.capability)})
        for step in plan.steps
    ]
    return OrchestrationPlan(request=plan.request, steps=steps)


REQUEST = (
    "Discover imagery scenes and vector buildings over the AOI, reproject the "
    "geometry, then analyze it with a foundation-model embedding"
)


# --------------------------------------------------------------------------- #
# Happy path: discover -> process -> analyze with full provenance (Req 4.4)
# --------------------------------------------------------------------------- #
async def test_e2e_happy_path_returns_analysis_with_full_provenance():
    servers = _mvp_servers()
    hub = assemble_mvp_hub({s.server_name: s for s in servers}, surface={})
    try:
        # Real planning resolves each capability to its providing MVP server,
        # then we attach realistic params and execute across the real servers.
        plan = _with_params(hub.plan(REQUEST))
        result = await hub.router.execute(plan, invoker=hub.invoker)
    finally:
        for s in servers:
            await s.aclose()

    # The request flowed across all four MVP servers in pillar order (Req 4.1).
    capabilities = [step.capability for step in plan.steps]
    assert capabilities == [
        "stac_search",
        "vector_features",
        "transform_crs",
        "embed_tile",
    ]
    ranks = [KIND_ORDER[step.kind] for step in plan.steps]
    assert ranks == sorted(ranks)

    # An analysis result is returned and the run is not degraded (Req 4.4).
    assert result.partial is False
    assert result.failed_sources == []
    assert result.analysis is not None
    # The analysis is the analyze step's (embed_tile) real embedding result.
    assert result.analysis["model"] == "Clay"
    assert result.analysis["dimension"] == 768
    assert len(result.analysis["vector"]) == 768

    # Provenance identifies, by source id, every source used in every step
    # (Requirement 4.4): one StepProvenance per step, each naming its server.
    used_by_capability = {
        sp.capability: sp.sources_used for sp in result.provenance.steps
    }
    assert used_by_capability == {
        "stac_search": ["geo-stac"],
        "vector_features": ["geo-vector"],
        "transform_crs": ["geo-ops"],
        "embed_tile": ["geo-foundation-models"],
    }
    # Every MVP server contributed; none failed.
    assert set(result.contributing_sources) == set(MVP_SERVER_NAMES)
    assert all(sp.failed_sources == [] for sp in result.provenance.steps)


# --------------------------------------------------------------------------- #
# Graceful degradation: one discovery source fails (Req 4.5, 4.7, 4.8)
# --------------------------------------------------------------------------- #
async def test_e2e_partial_result_preserves_provenance_for_failed_source():
    # Two redundant STAC discovery sources: one unreachable, one healthy.
    failing_stac = _stac_server(_unreachable_handler())
    healthy_stac = _stac_server()  # default: returns the mocked ItemCollection
    vector = _vector_server()
    ops = GeoOpsServer()
    fm = GeoFoundationModelsServer()

    bindings = {
        "geo-stac": failing_stac,          # primary STAC source - unreachable
        "geo-stac-pc": healthy_stac,       # redundant STAC source - healthy
        "geo-vector": vector,
        "geo-ops": ops,
        "geo-foundation-models": fm,
    }
    invoker = BoundServerInvoker(bindings)
    router = OrchestrationRouter(MappingCapabilityResolver({}), [], invoker=invoker)

    # A plan whose discovery step queries BOTH STAC sources for the same
    # capability (Requirement 4.3), then vector discovery, process, analyze.
    plan = OrchestrationPlan(
        request=REQUEST,
        steps=[
            PlanStep(
                step_id="step-1",
                kind=PlanStepKind.DISCOVER,
                capability="stac_search",
                candidate_sources=["geo-stac", "geo-stac-pc"],
                params=_params_for("stac_search"),
            ),
            PlanStep(
                step_id="step-2",
                kind=PlanStepKind.DISCOVER,
                capability="vector_features",
                candidate_sources=["geo-vector"],
                params=_params_for("vector_features"),
            ),
            PlanStep(
                step_id="step-3",
                kind=PlanStepKind.PROCESS,
                capability="transform_crs",
                candidate_sources=["geo-ops"],
                params=_params_for("transform_crs"),
            ),
            PlanStep(
                step_id="step-4",
                kind=PlanStepKind.ANALYZE,
                capability="embed_tile",
                candidate_sources=["geo-foundation-models"],
                params=_params_for("embed_tile"),
            ),
        ],
    )

    try:
        result = await router.execute(plan)
    finally:
        for s in bindings.values():
            await s.aclose()

    # The request still completed end to end: an analysis was produced even
    # though a discovery source failed (Requirement 4.5).
    assert result.analysis is not None
    assert result.analysis["model"] == "Clay"

    # The result is labeled partial under graceful degradation (Requirement 4.7).
    assert result.partial is True

    # Provenance enumerates the contributing sources AND the failed source with
    # its Error_Taxonomy category (Requirement 4.8). The unreachable STAC source
    # is recorded as a network/availability failure; the redundant source
    # carried the discovery step.
    assert len(result.failed_sources) == 1
    failed = result.failed_sources[0]
    assert failed.source == "geo-stac"
    assert failed.ok is False
    assert failed.error_category == ErrorCategory.NETWORK

    # The healthy STAC source plus vector/ops/foundation-models all contributed.
    assert set(result.contributing_sources) == {
        "geo-stac-pc",
        "geo-vector",
        "geo-ops",
        "geo-foundation-models",
    }

    # The discovery step's provenance accounts for both sources: one used, one
    # failed-with-category (Requirement 4.8).
    discover_prov = next(
        sp for sp in result.provenance.steps if sp.capability == "stac_search"
    )
    assert discover_prov.sources_used == ["geo-stac-pc"]
    assert [o.source for o in discover_prov.failed_sources] == ["geo-stac"]
    assert discover_prov.failed_sources[0].error_category == ErrorCategory.NETWORK

    # Across the whole plan, every source - contributing or failed - is
    # accounted for in provenance (no provenance is dropped under degradation).
    accounted: set = set()
    for sp in result.provenance.steps:
        accounted.update(sp.sources_used)
        accounted.update(o.source for o in sp.failed_sources)
    assert accounted == {
        "geo-stac",
        "geo-stac-pc",
        "geo-vector",
        "geo-ops",
        "geo-foundation-models",
    }
