"""Property test for Orchestration Router graceful-degradation provenance.

Feature: geospatial-power-pack, Property 6: Graceful degradation never drops
provenance.

**Validates: Requirements 4.1, 4.4, 4.5, 4.7, 4.8**

Property 6 (design.md): *For any* orchestration plan and any subset of its
sources that fail, the result's provenance accounts for every source — each
contributing source and each failed source (with its ``Error_Taxonomy``
category) appears — and whenever any source failed the result is labeled
partial.

This module drives :meth:`OrchestrationRouter.execute` with a programmable fake
:class:`ServerInvoker` over Hypothesis-generated plans. Each generated step is
guaranteed at least one succeeding source (so no step fails outright via
Requirement 4.6, which is exercised by the example-based execution tests); an
arbitrary subset of the remaining sources fail with an arbitrary
``Error_Taxonomy`` category. The test then asserts:

* every contributing source and every failed source (with category) is
  accounted for in the result and in per-step provenance (Requirements 4.4,
  4.5, 4.8);
* the result is labeled ``partial`` iff at least one source failed
  (Requirement 4.7);
* the plan honors the discover < process < analyze ordering invariant
  (Requirement 4.1).

The router invokes discovery sources concurrently (all attempted) and
process/analyze sources sequentially with first-success fallback (sources after
the first success are never invoked), so the expected provenance is computed
per-pillar to match that contract exactly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import asyncio

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from geo_common.errors import (
    AuthenticationError,
    AuthorizationError,
    ErrorCategory,
    NetworkError,
    NotFoundError,
    RateLimitError,
    UpstreamError,
    ValidationError,
)
from kiro_geospatial.orchestration import (
    KIND_ORDER,
    MappingCapabilityResolver,
    OrchestrationPlan,
    OrchestrationRouter,
    PlanStep,
    PlanStepKind,
    SourceOutcome,
)


# Map each taxonomy category onto a GeoError subclass that fixes that category,
# so a raised failure is recorded under exactly the intended category
# (Requirements 4.5, 4.8). Every category is reachable as a source failure.
CATEGORY_TO_ERROR = {
    ErrorCategory.AUTHENTICATION: AuthenticationError,
    ErrorCategory.AUTHORIZATION: AuthorizationError,
    ErrorCategory.RATE_LIMIT: RateLimitError,
    ErrorCategory.NOT_FOUND: NotFoundError,
    ErrorCategory.VALIDATION: ValidationError,
    ErrorCategory.UPSTREAM: UpstreamError,
    ErrorCategory.NETWORK: NetworkError,
}
CATEGORIES = list(CATEGORY_TO_ERROR)

# Pillars in canonical discover -> process -> analyze order.
_KINDS = [PlanStepKind.DISCOVER, PlanStepKind.PROCESS, PlanStepKind.ANALYZE]


class _ProgrammableInvoker:
    """A ``ServerInvoker`` whose per-source behavior is programmed up front.

    ``behaviors`` maps a (globally unique) source identifier to either a result
    value to return or a ``GeoError`` instance to raise. The router records a
    raised ``GeoError`` under its own category, so failures map deterministically
    to the category chosen by the generator.
    """

    def __init__(self, behaviors: Dict[str, object]) -> None:
        self._behaviors = behaviors

    async def invoke(self, step: PlanStep, source: str) -> object:
        behavior = self._behaviors[source]
        if isinstance(behavior, BaseException):
            raise behavior
        return behavior


@st.composite
def degradable_plans(
    draw,
) -> Tuple[OrchestrationPlan, Dict[str, object], List[str], List[Tuple[str, ErrorCategory]]]:
    """Generate a (plan, behaviors, expected_contributing, expected_failed) case.

    Every source identifier is globally unique across the plan so provenance can
    key on it unambiguously. Every step has at least one succeeding source. The
    expected contributing/failed sequences are computed to match the router's
    execution contract: discovery steps attempt every candidate concurrently;
    process/analyze steps attempt candidates in order and stop at the first
    success (later candidates are never invoked).
    """
    counts = {
        PlanStepKind.DISCOVER: draw(st.integers(min_value=0, max_value=3)),
        PlanStepKind.PROCESS: draw(st.integers(min_value=0, max_value=3)),
        PlanStepKind.ANALYZE: draw(st.integers(min_value=0, max_value=2)),
    }
    assume(sum(counts.values()) > 0)

    behaviors: Dict[str, object] = {}
    steps: List[PlanStep] = []
    expected_contributing: List[str] = []
    expected_failed: List[Tuple[str, ErrorCategory]] = []

    step_counter = 0
    source_counter = 0

    for kind in _KINDS:
        for _ in range(counts[kind]):
            step_counter += 1

            # Per-candidate outcome: None == success, otherwise a failure
            # category. Force at least one success so the step never fails
            # outright (Requirement 4.6 is covered by the example tests).
            outcomes: List[Optional[ErrorCategory]] = draw(
                st.lists(
                    st.one_of(st.none(), st.sampled_from(CATEGORIES)),
                    min_size=1,
                    max_size=4,
                )
            )
            if all(outcome is not None for outcome in outcomes):
                success_idx = draw(st.integers(min_value=0, max_value=len(outcomes) - 1))
                outcomes[success_idx] = None

            candidate_sources: List[str] = []
            local: List[Tuple[str, Optional[ErrorCategory]]] = []
            for category in outcomes:
                source_counter += 1
                source = f"src-{source_counter}"
                candidate_sources.append(source)
                local.append((source, category))
                if category is None:
                    behaviors[source] = {"source": source}
                else:
                    behaviors[source] = CATEGORY_TO_ERROR[category](
                        f"{category.value} failure", source=source
                    )

            steps.append(
                PlanStep(
                    step_id=f"step-{step_counter}",
                    kind=kind,
                    capability=f"cap-{step_counter}",
                    candidate_sources=candidate_sources,
                )
            )

            if kind == PlanStepKind.DISCOVER:
                # All candidates are invoked concurrently (Requirement 4.3).
                for source, category in local:
                    if category is None:
                        expected_contributing.append(source)
                    else:
                        expected_failed.append((source, category))
            else:
                # Sequential with first-success fallback: candidates after the
                # first success are never invoked, so they never appear.
                for source, category in local:
                    if category is None:
                        expected_contributing.append(source)
                        break
                    expected_failed.append((source, category))

    plan = OrchestrationPlan(request="generated request", steps=steps)
    return plan, behaviors, expected_contributing, expected_failed


@settings(max_examples=100, deadline=None)
@given(degradable_plans())
def test_graceful_degradation_never_drops_provenance(case):
    """Feature: geospatial-power-pack, Property 6: Graceful degradation never
    drops provenance.

    **Validates: Requirements 4.1, 4.4, 4.5, 4.7, 4.8**
    """
    plan, behaviors, expected_contributing, expected_failed = case
    router = OrchestrationRouter(
        MappingCapabilityResolver({}),
        [],
        invoker=_ProgrammableInvoker(behaviors),
    )

    result = asyncio.run(router.execute(plan))

    # Requirement 4.1: the plan honors discover < process < analyze ordering.
    ranks = [KIND_ORDER[step.kind] for step in plan.steps]
    assert ranks == sorted(ranks)

    # Requirement 4.4: every contributing source appears, in order.
    assert result.contributing_sources == expected_contributing

    # Requirements 4.5/4.8: every failed source appears with its category.
    expected_failed_outcomes = [
        SourceOutcome(source=source, ok=False, error_category=category)
        for source, category in expected_failed
    ]
    assert result.failed_sources == expected_failed_outcomes

    # Requirement 4.7: partial iff any source failed.
    assert result.partial is bool(expected_failed)

    # Requirements 4.4/4.8: per-step provenance accounts for every source, and
    # the union of provenance equals the union of contributing + failed sources.
    prov_contributing: List[str] = []
    prov_failed: List[Tuple[str, ErrorCategory]] = []
    for step_prov in result.provenance.steps:
        prov_contributing.extend(step_prov.sources_used)
        prov_failed.extend(
            (outcome.source, outcome.error_category)
            for outcome in step_prov.failed_sources
        )
        # Each failed outcome in provenance carries a category (Requirement 4.8).
        assert all(
            outcome.error_category is not None for outcome in step_prov.failed_sources
        )

    assert prov_contributing == expected_contributing
    assert prov_failed == expected_failed

    # No source is dropped: provenance covers exactly the attempted sources.
    accounted = set(prov_contributing) | {source for source, _ in prov_failed}
    expected_accounted = set(expected_contributing) | {
        source for source, _ in expected_failed
    }
    assert accounted == expected_accounted

    # One provenance record per executed step, in plan order (Requirement 4.4).
    assert [p.step_id for p in result.provenance.steps] == [
        step.step_id for step in plan.steps
    ]
