"""The Power Hub Orchestration Router (Requirement 4).

The Orchestration Router decomposes a single natural-language request into an
ordered **discover -> process -> analyze** plan and executes it across multiple
MCP servers with graceful degradation (see design.md "Power Hub -> Orchestration
Router"). This module implements both halves of that contract: planning and
execution-with-provenance.

Planning contract:

* Decompose a natural-language request into an :class:`OrchestrationPlan` - an
  ordered list of :class:`PlanStep` - that honors the **pillar-ordering
  invariant** (Requirement 4.1): every ``DISCOVER`` step precedes every
  ``PROCESS`` step, and every ``PROCESS`` step precedes every ``ANALYZE`` step.
* Resolve each step's ``capability`` to the MCP server(s) that provide it via
  the Resource Catalog (Requirement 4.2). Each emitted step carries at least
  one candidate source; a capability the catalog cannot resolve produces no
  step (there is nothing to invoke it against).

The invariant is enforced two ways: :meth:`OrchestrationRouter.plan` *produces*
plans in pillar order, and :class:`OrchestrationPlan` *validates* the order on
construction, so a hand-built or deserialized plan that violates it is rejected.

Execution contract (see :meth:`OrchestrationRouter.execute`):

* Invoke each step against the MCP server(s) that provide its capability
  (Requirement 4.2) through a pluggable :class:`ServerInvoker`, so the router
  stays decoupled from concrete servers and tests can inject fakes.
* Query a discovery step's candidate sources **concurrently** so their
  invocation periods overlap rather than running sequentially (Requirement 4.3).
* On a per-source error or a 30s timeout, continue with the remaining sources
  and record the failed source identifier plus its :class:`~geo_common.errors.ErrorCategory`
  (Requirement 4.5).
* Fail a step only when **every** one of its sources fails, raising a
  :class:`~geo_common.errors.GeoError` that names the step and lists the
  attempted sources (Requirement 4.6).
* Return an :class:`OrchestrationResult` carrying full provenance for every
  source used in every step (Requirement 4.4); label the result ``partial`` and
  enumerate contributing and failed sources (with categories) whenever any
  source failed (Requirements 4.7, 4.8).

Capability -> server resolution is delegated to a :class:`CapabilityResolver`.
The Resource Catalog (``kiro_geospatial.catalog``) lands concurrently; this
module stays decoupled from it by depending only on the small, shared
:class:`~geo_common.models.CatalogEntry` shape. :class:`CatalogCapabilityResolver`
resolves a capability to its providing servers from a collection of catalog
entries, so the same catalog entries the Hub already indexes drive planning
without this module importing the catalog component directly.

Python 3.9-compatible annotations (``from __future__ import annotations`` plus
``typing`` aliases), matching the rest of the package.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from pydantic import BaseModel, Field, field_validator, model_validator

from geo_common.errors import ErrorCategory, GeoError, NetworkError, UpstreamError
from geo_common.models import CatalogEntry

__all__ = [
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


class PlanStepKind(str, Enum):
    """The pillar a plan step belongs to (Requirement 4.1).

    A ``str`` enum so the value serializes directly to its wire string. The
    three kinds map onto the discover -> process -> analyze pillars and define
    the only legal ordering of a plan.
    """

    DISCOVER = "discover"
    PROCESS = "process"
    ANALYZE = "analyze"


# The canonical pillar order. A plan is well-ordered iff the kinds, read in
# step order, are non-decreasing under this ranking (Requirement 4.1).
KIND_ORDER: Dict[PlanStepKind, int] = {
    PlanStepKind.DISCOVER: 0,
    PlanStepKind.PROCESS: 1,
    PlanStepKind.ANALYZE: 2,
}


class PlanStep(BaseModel):
    """One step of an :class:`OrchestrationPlan` (Requirements 4.1, 4.2).

    ``capability`` is resolved against the Resource Catalog to the server(s)
    that provide it; ``candidate_sources`` holds those resolved server
    identifiers and must contain at least one (a step with nothing to invoke is
    not a valid step). For a ``DISCOVER`` step the candidates are queried
    concurrently at execution time (Requirement 4.3).
    """

    step_id: str = Field(min_length=1)
    kind: PlanStepKind
    capability: str = Field(min_length=1)
    candidate_sources: List[str] = Field(min_length=1)
    params: Dict[str, object] = Field(default_factory=dict)

    @field_validator("candidate_sources")
    @classmethod
    def _sources_non_empty(cls, value: List[str]) -> List[str]:
        """Every candidate source identifier must itself be a non-empty string."""
        if any(not isinstance(s, str) or not s.strip() for s in value):
            raise ValueError("candidate_sources must be non-empty source identifiers")
        return value


class OrchestrationPlan(BaseModel):
    """An ordered discover -> process -> analyze plan (Requirement 4.1).

    The pillar-ordering invariant is validated on construction: the steps, read
    in order, must be non-decreasing under :data:`KIND_ORDER`. This guarantees
    every ``DISCOVER`` step precedes every ``PROCESS`` step which precedes every
    ``ANALYZE`` step, whether the plan was produced by
    :meth:`OrchestrationRouter.plan` or built/deserialized by hand.
    """

    request: str
    steps: List[PlanStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_pillar_order(self) -> "OrchestrationPlan":
        ranks = [KIND_ORDER[step.kind] for step in self.steps]
        for earlier, later in zip(ranks, ranks[1:]):
            if earlier > later:
                raise ValueError(
                    "plan steps violate the discover < process < analyze "
                    "ordering invariant (Requirement 4.1)"
                )
        # Step ids must be unique so provenance can key on them later.
        ids = [step.step_id for step in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("plan step ids must be unique")
        return self

    def steps_of_kind(self, kind: PlanStepKind) -> List[PlanStep]:
        """Return the steps of a given pillar, in plan order."""
        return [step for step in self.steps if step.kind == kind]


@runtime_checkable
class CapabilityResolver(Protocol):
    """Resolves a capability name to the server(s) that provide it (Req 4.2).

    The Resource Catalog is the source of truth for this mapping. Any object
    that can answer "which servers provide this capability?" satisfies the
    protocol, so planning never has to import the catalog component directly.
    """

    def resolve(self, capability: str) -> List[str]:
        """Return the provider server identifiers for ``capability``.

        Returns an empty list when no known server provides the capability.
        """
        ...


class MappingCapabilityResolver:
    """A :class:`CapabilityResolver` backed by an explicit mapping.

    Useful for wiring a fixed capability -> servers table and for tests. Lookup
    is case-insensitive on the capability name; the order of the configured
    server list is preserved.
    """

    def __init__(self, mapping: Mapping[str, Sequence[str]]) -> None:
        self._mapping: Dict[str, List[str]] = {
            key.casefold(): list(dict.fromkeys(servers))
            for key, servers in mapping.items()
        }

    def resolve(self, capability: str) -> List[str]:
        return list(self._mapping.get(capability.casefold(), []))


class CatalogCapabilityResolver:
    """A :class:`CapabilityResolver` backed by Resource Catalog entries.

    Resolves a capability to its providing servers from a collection of
    :class:`~geo_common.models.CatalogEntry` - the same entries the Power Hub's
    Resource Catalog indexes (Requirements 2.1, 4.2). A capability matches a
    catalog entry by case-insensitive equality on the entry ``name``. When
    several entries share a name (e.g. multiple servers expose the same
    capability), every distinct provider is returned, preserving first-seen
    order so planning is deterministic.
    """

    def __init__(self, entries: Iterable[CatalogEntry]) -> None:
        index: Dict[str, List[str]] = {}
        for entry in entries:
            providers = index.setdefault(entry.name.casefold(), [])
            if entry.provider_server not in providers:
                providers.append(entry.provider_server)
        self._index = index

    def resolve(self, capability: str) -> List[str]:
        return list(self._index.get(capability.casefold(), []))


class CapabilitySpec(BaseModel):
    """A capability the router knows how to plan for.

    Maps a catalog ``capability`` name onto its pillar :class:`PlanStepKind` and
    the trigger ``keywords`` whose presence in a natural-language request means
    the request asks for that capability. Keyword matching is a deliberately
    simple, deterministic substring scan: the router's job here is to build a
    correctly *ordered* plan from the capabilities a request mentions, not to do
    deep language understanding.
    """

    capability: str = Field(min_length=1)
    kind: PlanStepKind
    keywords: List[str] = Field(min_length=1)

    @field_validator("keywords")
    @classmethod
    def _keywords_non_empty(cls, value: List[str]) -> List[str]:
        cleaned = [k for k in value if isinstance(k, str) and k.strip()]
        if not cleaned:
            raise ValueError("a capability must declare at least one non-empty keyword")
        return cleaned


#: The per-source invocation timeout. A source that does not respond within
#: this window is treated as failed (Error_Taxonomy ``network``) and the router
#: moves on to the remaining sources (Requirements 4.5, 4.6).
DEFAULT_SOURCE_TIMEOUT_S: float = 30.0


@runtime_checkable
class ServerInvoker(Protocol):
    """Invokes a plan step against one providing server (Requirement 4.2).

    The router never talks to concrete MCP servers directly; it delegates to a
    ``ServerInvoker`` so the same execution logic works for real servers and
    for injected fakes in tests. An implementation invokes the capability of
    ``step`` against ``source`` and returns the server's result on success, or
    raises on failure - a :class:`~geo_common.errors.GeoError` to signal a
    specific taxonomy category, or any other exception (mapped to ``upstream``).
    A source that does not return within the router's per-source timeout is
    recorded as a ``network`` failure (Requirement 4.5).
    """

    async def invoke(self, step: "PlanStep", source: str) -> object:
        """Invoke ``step``'s capability against ``source`` and return its result."""
        ...


class SourceOutcome(BaseModel):
    """The outcome of invoking one source for a step (Requirements 4.5, 4.8).

    ``ok`` is ``False`` for any source that errored or timed out; in that case
    ``error_category`` carries its :class:`~geo_common.errors.ErrorCategory` so
    a degraded result can enumerate each failed source with its category
    (Requirement 4.8). For a contributing source ``ok`` is ``True`` and
    ``error_category`` is ``None``.
    """

    source: str = Field(min_length=1)
    ok: bool
    error_category: Optional[ErrorCategory] = None

    @model_validator(mode="after")
    def _category_consistent_with_ok(self) -> "SourceOutcome":
        if self.ok and self.error_category is not None:
            raise ValueError("a successful SourceOutcome must not carry an error_category")
        if not self.ok and self.error_category is None:
            raise ValueError("a failed SourceOutcome must carry an error_category")
        return self


class StepProvenance(BaseModel):
    """Provenance for a single executed step (Requirements 4.4, 4.8).

    ``sources_used`` lists, by source identifier, every source that contributed
    to the step (Requirement 4.4); ``failed_sources`` lists every source that
    failed, each with its taxonomy category (Requirement 4.8). Together they
    account for every source the step attempted.
    """

    step_id: str = Field(min_length=1)
    kind: PlanStepKind
    capability: str = Field(min_length=1)
    sources_used: List[str] = Field(default_factory=list)
    failed_sources: List[SourceOutcome] = Field(default_factory=list)


class ProvenanceRecord(BaseModel):
    """Provenance for an entire executed plan (Requirement 4.4).

    Records the originating ``request`` and one :class:`StepProvenance` per
    executed step, so every source used in every discover/process/analyze step
    is identifiable by source id.
    """

    request: str
    steps: List[StepProvenance] = Field(default_factory=list)


class OrchestrationResult(BaseModel):
    """The result of executing an :class:`OrchestrationPlan` (Requirement 4).

    ``analysis`` is the analyze step's result (``None`` when the plan has no
    analyze step). ``provenance`` identifies every source used in every step
    (Requirement 4.4). ``partial`` is ``True`` whenever any source failed under
    graceful degradation (Requirement 4.7), in which case ``contributing_sources``
    lists every source that contributed and ``failed_sources`` lists every
    failed source with its taxonomy category (Requirement 4.8).
    """

    analysis: Optional[Dict[str, object]] = None
    provenance: ProvenanceRecord
    partial: bool = False
    contributing_sources: List[str] = Field(default_factory=list)
    failed_sources: List[SourceOutcome] = Field(default_factory=list)


class OrchestrationRouter:
    """Decomposes a natural-language request into an ordered plan (Req 4.1, 4.2).

    Constructed with the capability vocabulary it can plan for and a
    :class:`CapabilityResolver` that maps each capability to its providing
    server(s) via the catalog. :meth:`plan` scans the request for the keywords
    of each known capability, resolves the matched capabilities to servers, and
    emits a pillar-ordered :class:`OrchestrationPlan`.

    To :meth:`execute` a plan, supply a :class:`ServerInvoker` (the object that
    actually calls the providing servers). Execution runs discovery sources
    concurrently, degrades gracefully on per-source failure, and returns full
    provenance; see :meth:`execute`.
    """

    def __init__(
        self,
        resolver: CapabilityResolver,
        capabilities: Iterable[CapabilitySpec],
        invoker: Optional["ServerInvoker"] = None,
        *,
        source_timeout_s: float = DEFAULT_SOURCE_TIMEOUT_S,
    ) -> None:
        self._resolver = resolver
        # Preserve registration order; it breaks ties between steps of the same
        # pillar so planning is deterministic.
        self._capabilities: List[CapabilitySpec] = list(capabilities)
        self._invoker = invoker
        if source_timeout_s <= 0:
            raise ValueError("source_timeout_s must be positive")
        self._source_timeout_s = source_timeout_s

    def plan(self, nl_request: str) -> OrchestrationPlan:
        """Decompose ``nl_request`` into an ordered discover->process->analyze plan.

        A capability is selected when any of its keywords appears (case-
        insensitively) in the request. Each selected capability is resolved to
        its providing server(s) via the catalog; a capability the catalog cannot
        resolve yields no step. The resulting steps are ordered by pillar
        (Requirement 4.1), ties broken by capability registration order, and
        assigned stable sequential ``step_id``s.
        """
        haystack = nl_request.casefold()

        matched: List[CapabilitySpec] = []
        seen_capabilities = set()
        for spec in self._capabilities:
            if spec.capability in seen_capabilities:
                continue
            if any(keyword.casefold() in haystack for keyword in spec.keywords):
                matched.append(spec)
                seen_capabilities.add(spec.capability)

        # Order by pillar first (Req 4.1), then by the order capabilities were
        # registered (a stable sort keeps that secondary order intact).
        matched.sort(key=lambda spec: KIND_ORDER[spec.kind])

        steps: List[PlanStep] = []
        position = 0
        for spec in matched:
            candidate_sources = self._resolver.resolve(spec.capability)
            if not candidate_sources:
                # No server provides this capability; nothing to invoke (Req 4.2).
                continue
            position += 1
            steps.append(
                PlanStep(
                    step_id=f"step-{position}",
                    kind=spec.kind,
                    capability=spec.capability,
                    candidate_sources=candidate_sources,
                    params={},
                )
            )

        return OrchestrationPlan(request=nl_request, steps=steps)

    async def execute(
        self,
        plan: OrchestrationPlan,
        invoker: Optional["ServerInvoker"] = None,
    ) -> OrchestrationResult:
        """Execute ``plan`` step by step with graceful degradation (Req 4.2-4.8).

        Each step is invoked against the server(s) that provide its capability
        through the :class:`ServerInvoker` supplied here or at construction
        (Requirement 4.2). A discover step's candidate sources are queried
        **concurrently** so their invocation periods overlap (Requirement 4.3);
        process and analyze steps try their candidate sources in order, falling
        back to the next on failure. A source that errors or does not respond
        within the per-source timeout is recorded and skipped, and execution
        continues with the remaining sources (Requirement 4.5).

        A step fails only when **every** one of its sources fails; that raises a
        :class:`~geo_common.errors.GeoError` naming the step and listing the
        attempted sources (Requirement 4.6). The returned
        :class:`OrchestrationResult` carries provenance for every source used in
        every step (Requirement 4.4); whenever any source failed it is labeled
        ``partial`` with contributing and failed sources (and their categories)
        enumerated (Requirements 4.7, 4.8).
        """
        active_invoker = invoker if invoker is not None else self._invoker
        if active_invoker is None:
            raise ValueError(
                "execute() requires a ServerInvoker, supplied either to "
                "OrchestrationRouter(...) or to execute(...)"
            )

        step_provenances: List[StepProvenance] = []
        contributing_sources: List[str] = []
        all_failed: List[SourceOutcome] = []
        analysis: Optional[Dict[str, object]] = None

        for step in plan.steps:
            outcomes = await self._invoke_step(step, active_invoker)

            sources_used = [source for source, result, cat in outcomes if cat is None]
            failed = [
                SourceOutcome(source=source, ok=False, error_category=cat)
                for source, result, cat in outcomes
                if cat is not None
            ]

            step_provenances.append(
                StepProvenance(
                    step_id=step.step_id,
                    kind=step.kind,
                    capability=step.capability,
                    sources_used=sources_used,
                    failed_sources=failed,
                )
            )
            contributing_sources.extend(sources_used)
            all_failed.extend(failed)

            if not sources_used:
                # Every source for a required step failed (Requirement 4.6).
                raise self._step_failed_error(step, failed)

            if step.kind == PlanStepKind.ANALYZE:
                # The analysis result is produced by the analyze step; the last
                # analyze step in plan order wins.
                first_success = next(
                    result for source, result, cat in outcomes if cat is None
                )
                analysis = _as_analysis(first_success)

        return OrchestrationResult(
            analysis=analysis,
            provenance=ProvenanceRecord(request=plan.request, steps=step_provenances),
            partial=bool(all_failed),
            contributing_sources=contributing_sources,
            failed_sources=all_failed,
        )

    async def _invoke_step(
        self, step: PlanStep, invoker: "ServerInvoker"
    ) -> List[Tuple[str, object, Optional[ErrorCategory]]]:
        """Invoke a step's candidate sources, returning per-source outcomes.

        Discovery sources run **concurrently** (overlapping invocation periods,
        Requirement 4.3); process/analyze sources run sequentially and stop at
        the first success (fallback). Each outcome is a
        ``(source, result, error_category)`` triple where ``error_category`` is
        ``None`` on success.
        """
        if step.kind == PlanStepKind.DISCOVER:
            return list(
                await asyncio.gather(
                    *(
                        self._invoke_source(step, source, invoker)
                        for source in step.candidate_sources
                    )
                )
            )

        outcomes: List[Tuple[str, object, Optional[ErrorCategory]]] = []
        for source in step.candidate_sources:
            outcome = await self._invoke_source(step, source, invoker)
            outcomes.append(outcome)
            if outcome[2] is None:
                # First success satisfies a process/analyze step; stop here.
                break
        return outcomes

    async def _invoke_source(
        self, step: PlanStep, source: str, invoker: "ServerInvoker"
    ) -> Tuple[str, object, Optional[ErrorCategory]]:
        """Invoke one source with the per-source timeout, never raising.

        Returns ``(source, result, None)`` on success or
        ``(source, None, category)`` on error/timeout (Requirement 4.5). A
        timeout maps to ``network``; a :class:`~geo_common.errors.GeoError`
        keeps its own category; any other exception maps to ``upstream``.
        """
        try:
            result = await asyncio.wait_for(
                invoker.invoke(step, source),
                timeout=self._source_timeout_s,
            )
            return source, result, None
        except asyncio.TimeoutError:
            return source, None, ErrorCategory.NETWORK
        except GeoError as exc:
            return source, None, exc.category
        except Exception:  # noqa: BLE001 - any source failure is recorded, not raised
            return source, None, ErrorCategory.UPSTREAM

    @staticmethod
    def _step_failed_error(
        step: PlanStep, failed: List[SourceOutcome]
    ) -> GeoError:
        """Build the GeoError for a step whose every source failed (Req 4.6).

        The error names the step and lists the attempted sources. The category
        is ``network`` when every failure was a network/timeout failure,
        otherwise ``upstream``.
        """
        attempted = [outcome.source for outcome in failed]
        detail: Dict[str, object] = {
            "step_id": step.step_id,
            "capability": step.capability,
            "kind": step.kind.value,
            "attempted_sources": attempted,
            "outcomes": [
                {
                    "source": outcome.source,
                    "error_category": outcome.error_category.value
                    if outcome.error_category is not None
                    else None,
                }
                for outcome in failed
            ],
        }
        message = (
            f"Orchestration step {step.step_id!r} (capability {step.capability!r}, "
            f"{step.kind.value}) could not be completed: all sources failed "
            f"({', '.join(attempted)})"
        )
        all_network = bool(failed) and all(
            outcome.error_category == ErrorCategory.NETWORK for outcome in failed
        )
        if all_network:
            return NetworkError(message, detail=detail)
        return UpstreamError(message, detail=detail)


def _as_analysis(result: object) -> Optional[Dict[str, object]]:
    """Coerce a step result into the ``analysis`` dict shape (or ``None``).

    A dict result passes through; a pydantic model is dumped; ``None`` stays
    ``None``; anything else is wrapped as ``{"result": value}`` so the analysis
    field is always a mapping or ``None``.
    """
    if result is None:
        return None
    if isinstance(result, dict):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump()
    return {"result": result}
