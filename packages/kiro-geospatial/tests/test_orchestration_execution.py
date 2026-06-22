"""Tests for the Orchestration Router execution portion (Requirements 4.2-4.8).

Covers:

* invoking each step against its providing server via a pluggable
  ``ServerInvoker`` (Requirement 4.2);
* querying a discovery step's candidate sources concurrently so their
  invocation periods overlap (Requirement 4.3);
* returning provenance that names every source used in every step on full
  success (Requirement 4.4);
* continuing past a per-source error or 30s timeout while recording the failed
  source and its taxonomy category (Requirement 4.5);
* failing a step (with an error naming the step and the attempted sources) only
  when every source for that step fails (Requirement 4.6);
* labeling a degraded result ``partial`` and enumerating contributing and
  failed sources with categories (Requirements 4.7, 4.8).

These are example-based unit tests; the named Hypothesis property test for
graceful-degradation provenance (Property 6) lands with task 3.10.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Dict, List, Optional, Tuple

import pytest

from geo_common.errors import (
    ErrorCategory,
    GeoError,
    NetworkError,
    RateLimitError,
)
from kiro_geospatial.orchestration import (
    OrchestrationPlan,
    OrchestrationResult,
    PlanStep,
    PlanStepKind,
    SourceOutcome,
)


# --------------------------------------------------------------------------- #
# Fakes: a programmable ServerInvoker that maps (capability, source) to a
# behavior - return a value, raise a GeoError, or sleep (to exercise timeout
# and concurrency).
# --------------------------------------------------------------------------- #
class FakeInvoker:
    """A ``ServerInvoker`` whose per-source behavior is programmed up front.

    ``behaviors`` maps a ``source`` identifier to a coroutine-producing callable
    ``(step) -> result`` or to an exception instance to raise. Records the start
    and end time of each invocation so tests can assert overlap (Req 4.3).
    """

    def __init__(
        self,
        behaviors: Dict[str, object],
    ) -> None:
        self._behaviors = behaviors
        self.calls: List[Tuple[str, str]] = []  # (capability, source)
        self.intervals: Dict[str, Tuple[float, float]] = {}

    async def invoke(self, step: PlanStep, source: str) -> object:
        self.calls.append((step.capability, source))
        start = asyncio.get_event_loop().time()
        try:
            behavior = self._behaviors[source]
            if isinstance(behavior, BaseException):
                raise behavior
            if callable(behavior):
                return await behavior(step)
            return behavior
        finally:
            end = asyncio.get_event_loop().time()
            self.intervals[source] = (start, end)


def _sleep_then(value: object, delay: float) -> Callable:
    async def _behavior(step: PlanStep) -> object:
        await asyncio.sleep(delay)
        return value

    return _behavior


def _discover_step(step_id: str, capability: str, sources: List[str]) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        kind=PlanStepKind.DISCOVER,
        capability=capability,
        candidate_sources=sources,
    )


def _process_step(step_id: str, capability: str, sources: List[str]) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        kind=PlanStepKind.PROCESS,
        capability=capability,
        candidate_sources=sources,
    )


def _analyze_step(step_id: str, capability: str, sources: List[str]) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        kind=PlanStepKind.ANALYZE,
        capability=capability,
        candidate_sources=sources,
    )


def _router(invoker: Optional[FakeInvoker] = None, *, source_timeout_s: float = 30.0):
    # The resolver/capabilities are unused by execute(); a trivial resolver is
    # enough to construct the router.
    from kiro_geospatial.orchestration import (
        MappingCapabilityResolver,
        OrchestrationRouter,
    )

    return OrchestrationRouter(
        MappingCapabilityResolver({}),
        [],
        invoker=invoker,
        source_timeout_s=source_timeout_s,
    )


# --------------------------------------------------------------------------- #
# Happy path: every source succeeds -> full provenance, not partial (Req 4.2/4.4)
# --------------------------------------------------------------------------- #
async def test_execute_all_success_returns_full_provenance_not_partial():
    invoker = FakeInvoker(
        {
            "geo-stac": {"items": 3},
            "geo-vector": {"features": 5},
            "geo-ops": {"reprojected": True},
            "geo-foundation-models": {"embedding": [0.1, 0.2]},
        }
    )
    plan = OrchestrationPlan(
        request="discover, process, analyze",
        steps=[
            _discover_step("step-1", "stac_search", ["geo-stac"]),
            _discover_step("step-2", "vector_features", ["geo-vector"]),
            _process_step("step-3", "transform_crs", ["geo-ops"]),
            _analyze_step("step-4", "embed_tile", ["geo-foundation-models"]),
        ],
    )

    result = await _router(invoker).execute(plan)

    assert isinstance(result, OrchestrationResult)
    assert result.partial is False
    assert result.failed_sources == []
    # Every source contributed (Requirement 4.4).
    assert result.contributing_sources == [
        "geo-stac",
        "geo-vector",
        "geo-ops",
        "geo-foundation-models",
    ]
    # Provenance has one record per step naming the source used.
    assert [p.step_id for p in result.provenance.steps] == [
        "step-1",
        "step-2",
        "step-3",
        "step-4",
    ]
    assert all(p.sources_used for p in result.provenance.steps)
    # Analysis comes from the analyze step's result.
    assert result.analysis == {"embedding": [0.1, 0.2]}


async def test_execute_uses_invoker_passed_to_execute():
    invoker = FakeInvoker({"geo-ops": {"ok": True}})
    plan = OrchestrationPlan(
        request="process",
        steps=[_process_step("step-1", "transform_crs", ["geo-ops"])],
    )
    # No invoker at construction; supply it to execute().
    result = await _router(None).execute(plan, invoker=invoker)
    assert result.partial is False
    assert invoker.calls == [("transform_crs", "geo-ops")]


async def test_execute_without_any_invoker_raises():
    plan = OrchestrationPlan(request="x", steps=[])
    with pytest.raises(ValueError):
        await _router(None).execute(plan)


# --------------------------------------------------------------------------- #
# Concurrency: discovery sources overlap in time (Requirement 4.3)
# --------------------------------------------------------------------------- #
async def test_discovery_sources_are_queried_concurrently():
    delay = 0.2
    invoker = FakeInvoker(
        {
            "geo-stac": _sleep_then({"a": 1}, delay),
            "geo-vector": _sleep_then({"b": 2}, delay),
        }
    )
    plan = OrchestrationPlan(
        request="discover",
        steps=[_discover_step("step-1", "discover_all", ["geo-stac", "geo-vector"])],
    )

    loop_start = asyncio.get_event_loop().time()
    result = await _router(invoker).execute(plan)
    elapsed = asyncio.get_event_loop().time() - loop_start

    assert result.partial is False
    # Concurrent: total time ~= one delay, well under the sequential 2*delay.
    assert elapsed < delay * 1.8
    # Invocation periods overlap: each source started before the other ended.
    s_start, s_end = invoker.intervals["geo-stac"]
    v_start, v_end = invoker.intervals["geo-vector"]
    assert s_start < v_end and v_start < s_end


# --------------------------------------------------------------------------- #
# Per-source failure: continue, record source + category (Requirements 4.5/4.8)
# --------------------------------------------------------------------------- #
async def test_discovery_continues_past_failed_source_and_records_category():
    invoker = FakeInvoker(
        {
            "geo-stac": RateLimitError("429 from stac", source="geo-stac"),
            "geo-vector": {"features": 5},
        }
    )
    plan = OrchestrationPlan(
        request="discover",
        steps=[_discover_step("step-1", "discover_all", ["geo-stac", "geo-vector"])],
    )

    result = await _router(invoker).execute(plan)

    # Degraded but succeeded: the surviving source carried the step (Req 4.5).
    assert result.partial is True
    assert result.contributing_sources == ["geo-vector"]
    assert result.failed_sources == [
        SourceOutcome(
            source="geo-stac",
            ok=False,
            error_category=ErrorCategory.RATE_LIMIT,
        )
    ]
    # Provenance accounts for both the used and the failed source (Req 4.8).
    prov = result.provenance.steps[0]
    assert prov.sources_used == ["geo-vector"]
    assert [o.source for o in prov.failed_sources] == ["geo-stac"]
    assert prov.failed_sources[0].error_category == ErrorCategory.RATE_LIMIT


async def test_timeout_is_recorded_as_network_failure_and_execution_continues():
    invoker = FakeInvoker(
        {
            "slow-source": _sleep_then({"late": True}, 0.5),
            "fast-source": {"features": 1},
        }
    )
    plan = OrchestrationPlan(
        request="discover",
        steps=[
            _discover_step("step-1", "discover_all", ["slow-source", "fast-source"])
        ],
    )

    # Timeout shorter than the slow source's delay.
    result = await _router(invoker, source_timeout_s=0.1).execute(plan)

    assert result.partial is True
    assert result.contributing_sources == ["fast-source"]
    assert result.failed_sources == [
        SourceOutcome(
            source="slow-source",
            ok=False,
            error_category=ErrorCategory.NETWORK,
        )
    ]


async def test_unexpected_exception_maps_to_upstream_category():
    invoker = FakeInvoker(
        {
            "broken": ValueError("boom"),
            "ok-source": {"x": 1},
        }
    )
    plan = OrchestrationPlan(
        request="discover",
        steps=[_discover_step("step-1", "discover_all", ["broken", "ok-source"])],
    )

    result = await _router(invoker).execute(plan)

    assert result.partial is True
    assert result.failed_sources[0].source == "broken"
    assert result.failed_sources[0].error_category == ErrorCategory.UPSTREAM


# --------------------------------------------------------------------------- #
# Process/analyze fallback: try next source on failure (Requirements 4.2/4.5)
# --------------------------------------------------------------------------- #
async def test_process_step_falls_back_to_next_source_on_failure():
    invoker = FakeInvoker(
        {
            "primary": NetworkError("down", source="primary"),
            "backup": {"reprojected": True},
        }
    )
    plan = OrchestrationPlan(
        request="process",
        steps=[_process_step("step-1", "transform_crs", ["primary", "backup"])],
    )

    result = await _router(invoker).execute(plan)

    assert result.partial is True
    assert result.contributing_sources == ["backup"]
    assert result.failed_sources[0].source == "primary"
    # Once a source succeeds the step stops; both were attempted in order.
    assert invoker.calls == [
        ("transform_crs", "primary"),
        ("transform_crs", "backup"),
    ]


async def test_process_step_stops_at_first_success():
    invoker = FakeInvoker(
        {
            "primary": {"reprojected": True},
            "backup": {"reprojected": False},
        }
    )
    plan = OrchestrationPlan(
        request="process",
        steps=[_process_step("step-1", "transform_crs", ["primary", "backup"])],
    )

    result = await _router(invoker).execute(plan)

    assert result.partial is False
    # The backup is never invoked because the primary succeeded.
    assert invoker.calls == [("transform_crs", "primary")]


# --------------------------------------------------------------------------- #
# All sources fail -> step fails with a naming error (Requirement 4.6)
# --------------------------------------------------------------------------- #
async def test_step_fails_when_all_sources_fail_naming_step_and_sources():
    invoker = FakeInvoker(
        {
            "geo-stac": NetworkError("timeout", source="geo-stac"),
            "geo-vector": NetworkError("timeout", source="geo-vector"),
        }
    )
    plan = OrchestrationPlan(
        request="discover",
        steps=[_discover_step("step-1", "discover_all", ["geo-stac", "geo-vector"])],
    )

    with pytest.raises(GeoError) as excinfo:
        await _router(invoker).execute(plan)

    err = excinfo.value
    # Names the step/capability and lists every attempted source (Req 4.6).
    assert "step-1" in str(err)
    assert "discover_all" in str(err)
    assert "geo-stac" in str(err) and "geo-vector" in str(err)
    assert err.detail is not None
    assert err.detail["attempted_sources"] == ["geo-stac", "geo-vector"]
    # All failures were network -> the step error is a network error.
    assert err.category == ErrorCategory.NETWORK


async def test_step_failure_with_mixed_categories_is_upstream():
    invoker = FakeInvoker(
        {
            "a": NetworkError("timeout", source="a"),
            "b": RateLimitError("429", source="b"),
        }
    )
    plan = OrchestrationPlan(
        request="discover",
        steps=[_discover_step("step-1", "discover_all", ["a", "b"])],
    )

    with pytest.raises(GeoError) as excinfo:
        await _router(invoker).execute(plan)

    assert excinfo.value.category == ErrorCategory.UPSTREAM


# --------------------------------------------------------------------------- #
# Partial labeling + enumeration across multiple steps (Requirements 4.7/4.8)
# --------------------------------------------------------------------------- #
async def test_partial_result_enumerates_all_contributing_and_failed_sources():
    invoker = FakeInvoker(
        {
            "geo-stac": {"items": 1},
            "geo-vector": RateLimitError("429", source="geo-vector"),
            "geo-ops": {"reprojected": True},
            "geo-foundation-models": {"embedding": [0.0]},
        }
    )
    plan = OrchestrationPlan(
        request="full pipeline",
        steps=[
            _discover_step("step-1", "stac_search", ["geo-stac", "geo-vector"]),
            _process_step("step-2", "transform_crs", ["geo-ops"]),
            _analyze_step("step-3", "embed_tile", ["geo-foundation-models"]),
        ],
    )

    result = await _router(invoker).execute(plan)

    assert result.partial is True
    assert result.contributing_sources == [
        "geo-stac",
        "geo-ops",
        "geo-foundation-models",
    ]
    assert result.failed_sources == [
        SourceOutcome(
            source="geo-vector", ok=False, error_category=ErrorCategory.RATE_LIMIT
        )
    ]
    assert result.analysis == {"embedding": [0.0]}

    # Every source - contributing or failed - is accounted for in provenance.
    accounted = set()
    for step_prov in result.provenance.steps:
        accounted.update(step_prov.sources_used)
        accounted.update(o.source for o in step_prov.failed_sources)
    assert accounted == {
        "geo-stac",
        "geo-vector",
        "geo-ops",
        "geo-foundation-models",
    }


async def test_plan_with_no_analyze_step_has_none_analysis():
    invoker = FakeInvoker({"geo-stac": {"items": 1}})
    plan = OrchestrationPlan(
        request="discover only",
        steps=[_discover_step("step-1", "stac_search", ["geo-stac"])],
    )

    result = await _router(invoker).execute(plan)

    assert result.analysis is None
    assert result.partial is False
    assert result.contributing_sources == ["geo-stac"]


# --------------------------------------------------------------------------- #
# SourceOutcome model invariants
# --------------------------------------------------------------------------- #
def test_source_outcome_rejects_inconsistent_ok_and_category():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        SourceOutcome(source="s", ok=True, error_category=ErrorCategory.NETWORK)
    with pytest.raises(pydantic.ValidationError):
        SourceOutcome(source="s", ok=False, error_category=None)
