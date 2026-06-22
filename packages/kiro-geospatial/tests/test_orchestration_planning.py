"""Tests for the Orchestration Router planning portion (Requirements 4.1, 4.2).

Covers:

* the discover < process < analyze pillar-ordering invariant, both as produced
  by ``OrchestrationRouter.plan`` and as enforced by ``OrchestrationPlan``
  construction (Requirement 4.1);
* capability -> providing-server resolution via the catalog (Requirement 4.2);
* edge cases: no matching capabilities, capabilities the catalog cannot
  resolve, and duplicate providers.

A light Hypothesis property check asserts the ordering invariant holds for any
selection of the known capabilities, complementing the named graceful-
degradation property test that lands with execution.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError as PydanticValidationError

from geo_common.models import CatalogEntry, OpennessTier
from kiro_geospatial.orchestration import (
    KIND_ORDER,
    CapabilitySpec,
    CatalogCapabilityResolver,
    MappingCapabilityResolver,
    OrchestrationPlan,
    OrchestrationRouter,
    PlanStep,
    PlanStepKind,
)


# --------------------------------------------------------------------------- #
# Fixtures: a small but representative MVP capability vocabulary + catalog.
# --------------------------------------------------------------------------- #
def _catalog_entries() -> list[CatalogEntry]:
    return [
        CatalogEntry(
            name="stac_search",
            pillar="A",
            capability_description="Search STAC catalogs for imagery items.",
            openness_tier=OpennessTier.OPEN,
            provider_server="geo-stac",
        ),
        CatalogEntry(
            name="vector_features",
            pillar="A",
            capability_description="Fetch OSM/Overture vector features for an extent.",
            openness_tier=OpennessTier.OPEN,
            provider_server="geo-vector",
        ),
        CatalogEntry(
            name="transform_crs",
            pillar="B",
            capability_description="Reproject a geometry to a target CRS.",
            openness_tier=OpennessTier.OPEN,
            provider_server="geo-ops",
        ),
        CatalogEntry(
            name="embed_tile",
            pillar="C",
            capability_description="Produce a foundation-model embedding for a tile.",
            openness_tier=OpennessTier.OPEN,
            provider_server="geo-foundation-models",
        ),
    ]


def _capability_specs() -> list[CapabilitySpec]:
    return [
        CapabilitySpec(
            capability="embed_tile",
            kind=PlanStepKind.ANALYZE,
            keywords=["embed", "embedding", "analyze", "change detection"],
        ),
        CapabilitySpec(
            capability="transform_crs",
            kind=PlanStepKind.PROCESS,
            keywords=["reproject", "transform", "crs", "process"],
        ),
        CapabilitySpec(
            capability="stac_search",
            kind=PlanStepKind.DISCOVER,
            keywords=["imagery", "stac", "scenes", "discover"],
        ),
        CapabilitySpec(
            capability="vector_features",
            kind=PlanStepKind.DISCOVER,
            keywords=["vector", "buildings", "roads", "features"],
        ),
    ]


def _router() -> OrchestrationRouter:
    resolver = CatalogCapabilityResolver(_catalog_entries())
    return OrchestrationRouter(resolver, _capability_specs())


# --------------------------------------------------------------------------- #
# Planning: ordering invariant and resolution (Requirements 4.1, 4.2)
# --------------------------------------------------------------------------- #
def test_plan_orders_discover_before_process_before_analyze():
    plan = _router().plan(
        "Discover imagery scenes and vector buildings, reproject the geometry, "
        "then analyze with a foundation-model embedding"
    )

    kinds = [step.kind for step in plan.steps]
    # Every capability mentioned is present.
    assert {step.capability for step in plan.steps} == {
        "stac_search",
        "vector_features",
        "transform_crs",
        "embed_tile",
    }
    # Pillar-ordering invariant: ranks are non-decreasing (Requirement 4.1).
    ranks = [KIND_ORDER[k] for k in kinds]
    assert ranks == sorted(ranks)
    # Concretely: both discover steps come first, then process, then analyze.
    assert kinds[:2] == [PlanStepKind.DISCOVER, PlanStepKind.DISCOVER]
    assert kinds[2] == PlanStepKind.PROCESS
    assert kinds[3] == PlanStepKind.ANALYZE


def test_plan_resolves_capability_to_providing_server():
    plan = _router().plan("reproject this geometry")

    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.capability == "transform_crs"
    assert step.candidate_sources == ["geo-ops"]  # resolved via the catalog (Req 4.2)


def test_plan_step_ids_are_unique_and_sequential():
    plan = _router().plan("discover imagery, reproject, embed")
    ids = [step.step_id for step in plan.steps]
    assert ids == [f"step-{i + 1}" for i in range(len(ids))]
    assert len(set(ids)) == len(ids)


def test_plan_skips_capability_with_no_providing_server():
    # The router knows a capability the catalog does not provide.
    resolver = CatalogCapabilityResolver(_catalog_entries())
    specs = _capability_specs() + [
        CapabilitySpec(
            capability="unprovided_capability",
            kind=PlanStepKind.PROCESS,
            keywords=["unprovided"],
        )
    ]
    router = OrchestrationRouter(resolver, specs)

    plan = router.plan("please run the unprovided step and reproject")
    capabilities = {step.capability for step in plan.steps}
    assert "unprovided_capability" not in capabilities  # no server -> no step (Req 4.2)
    assert "transform_crs" in capabilities


def test_plan_with_no_matching_capabilities_is_empty_but_valid():
    plan = _router().plan("tell me a joke about maps")
    assert plan.steps == []
    assert plan.request == "tell me a joke about maps"


def test_plan_matches_keywords_case_insensitively():
    plan = _router().plan("DISCOVER IMAGERY AND EMBED THE RESULT")
    capabilities = [step.capability for step in plan.steps]
    assert "stac_search" in capabilities
    assert "embed_tile" in capabilities


def test_multiple_providers_for_one_capability_are_all_returned():
    entries = _catalog_entries() + [
        CatalogEntry(
            name="stac_search",
            pillar="A",
            capability_description="Alternate STAC provider.",
            openness_tier=OpennessTier.FREE_TIER,
            provider_server="geo-stac-mirror",
        )
    ]
    resolver = CatalogCapabilityResolver(entries)
    assert resolver.resolve("STAC_SEARCH") == ["geo-stac", "geo-stac-mirror"]


def test_mapping_resolver_dedups_and_is_case_insensitive():
    resolver = MappingCapabilityResolver(
        {"Cap": ["s1", "s2", "s1"]}
    )
    assert resolver.resolve("cap") == ["s1", "s2"]
    assert resolver.resolve("missing") == []


# --------------------------------------------------------------------------- #
# OrchestrationPlan validation enforces the invariant on construction (Req 4.1)
# --------------------------------------------------------------------------- #
def _step(step_id: str, kind: PlanStepKind, capability: str) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        kind=kind,
        capability=capability,
        candidate_sources=["srv"],
    )


def test_plan_construction_rejects_out_of_order_steps():
    with pytest.raises(PydanticValidationError):
        OrchestrationPlan(
            request="x",
            steps=[
                _step("step-1", PlanStepKind.ANALYZE, "embed_tile"),
                _step("step-2", PlanStepKind.DISCOVER, "stac_search"),
            ],
        )


def test_plan_construction_rejects_process_before_discover():
    with pytest.raises(PydanticValidationError):
        OrchestrationPlan(
            request="x",
            steps=[
                _step("step-1", PlanStepKind.PROCESS, "transform_crs"),
                _step("step-2", PlanStepKind.DISCOVER, "stac_search"),
            ],
        )


def test_plan_construction_rejects_duplicate_step_ids():
    with pytest.raises(PydanticValidationError):
        OrchestrationPlan(
            request="x",
            steps=[
                _step("dup", PlanStepKind.DISCOVER, "stac_search"),
                _step("dup", PlanStepKind.PROCESS, "transform_crs"),
            ],
        )


def test_plan_step_requires_at_least_one_candidate_source():
    with pytest.raises(PydanticValidationError):
        PlanStep(
            step_id="step-1",
            kind=PlanStepKind.DISCOVER,
            capability="stac_search",
            candidate_sources=[],
        )


def test_well_ordered_plan_constructs_and_filters_by_kind():
    plan = OrchestrationPlan(
        request="x",
        steps=[
            _step("step-1", PlanStepKind.DISCOVER, "stac_search"),
            _step("step-2", PlanStepKind.DISCOVER, "vector_features"),
            _step("step-3", PlanStepKind.PROCESS, "transform_crs"),
            _step("step-4", PlanStepKind.ANALYZE, "embed_tile"),
        ],
    )
    assert len(plan.steps_of_kind(PlanStepKind.DISCOVER)) == 2
    assert len(plan.steps_of_kind(PlanStepKind.PROCESS)) == 1
    assert len(plan.steps_of_kind(PlanStepKind.ANALYZE)) == 1


# --------------------------------------------------------------------------- #
# Property: any selection of known capabilities yields a well-ordered plan.
# --------------------------------------------------------------------------- #
_KEYWORD_BY_CAPABILITY = {
    "embed_tile": "embedding",
    "transform_crs": "reproject",
    "stac_search": "imagery",
    "vector_features": "vector",
}


@given(
    selected=st.lists(
        st.sampled_from(sorted(_KEYWORD_BY_CAPABILITY)),
        unique=True,
    )
)
def test_planning_invariant_holds_for_any_capability_subset(selected):
    request = " ".join(_KEYWORD_BY_CAPABILITY[c] for c in selected)
    plan = _router().plan(request)

    ranks = [KIND_ORDER[step.kind] for step in plan.steps]
    # discover < process < analyze: ranks must be non-decreasing (Requirement 4.1).
    assert ranks == sorted(ranks)
    # Every selected, catalog-resolvable capability appears exactly once.
    planned = [step.capability for step in plan.steps]
    assert sorted(planned) == sorted(set(selected))
    # Every step resolved to at least one providing server (Requirement 4.2).
    assert all(step.candidate_sources for step in plan.steps)
