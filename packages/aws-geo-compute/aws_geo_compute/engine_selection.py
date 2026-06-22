"""Engine-selection decision tree and the in-process-vs-delegate threshold.

This module implements the **documented engine-selection decision tree**
(Requirement 12.4) and the **in-process-vs-delegate threshold** (Requirement
12.5) for the ``aws-geo-compute`` peer Power. Both are pure, deterministic
functions over the two task characteristics the design's decision tree consults
- the **input dataset size in bytes** and the **access pattern** - plus the
**required memory** that the delegation threshold evaluates.

The decision tree (design.md "Engine-selection decision tree", Requirement
12.4)::

    dataset size < 1 GB                         -> GeoPandas / Shapely (in-process)
    1 GB <= dataset size <= 100 GB              -> by access pattern:
        single-user analytical                  -> DuckDB Spatial
        multi-user / transactional              -> PostGIS
        ad-hoc over S3                           -> Athena
    dataset size > 100 GB / distributed          -> Apache Sedona on EMR (peer)

The in-process-vs-delegate threshold (design.md, Requirement 12.5): a task is
delegated to the ``aws-geo-compute`` peer Power **if and only if** it requires
more memory than the in-process execution memory limit (default **4 GB**) *or*
operates on an input dataset larger than **5 GB**. This is exactly the predicate
Property 17 ("Engine and delegation selection are deterministic") validates, and
it makes the ``> 100 GB`` EMR-Sedona branch always delegate (its dataset is, by
construction, larger than 5 GB).

Keeping selection pure (no I/O, no side effects) makes it deterministic and
trivially testable, and keeps the actual delegation act - which can fail or time
out - in :mod:`aws_geo_compute.delegation`.

Python 3.10+ (matching the package's ``requires-python``): uses
``from __future__ import annotations`` together with ``typing`` generics.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict

from pydantic import BaseModel, Field

__all__ = [
    "Engine",
    "AccessPattern",
    "GB",
    "SMALL_DATASET_LIMIT_BYTES",
    "DISTRIBUTED_DATASET_LIMIT_BYTES",
    "DEFAULT_IN_PROCESS_MEMORY_LIMIT_BYTES",
    "MAX_IN_PROCESS_DATASET_BYTES",
    "ExecutionPlan",
    "select_engine",
    "should_delegate",
    "select_execution_plan",
]

#: One gibibyte in bytes. The decision-tree and threshold sizes are expressed in
#: binary GB (GiB) consistently, so "1 GB", "4 GB", "5 GB", and "100 GB" are all
#: exact byte multiples of this constant.
GB = 1024 ** 3

#: Datasets strictly smaller than this run in-process on GeoPandas/Shapely
#: (decision-tree first branch, "< 1 GB").
SMALL_DATASET_LIMIT_BYTES = 1 * GB

#: Datasets strictly larger than this go to distributed Apache Sedona on EMR
#: (decision-tree last branch, "> 100 GB / distributed").
DISTRIBUTED_DATASET_LIMIT_BYTES = 100 * GB

#: The default in-process execution memory limit (Requirement 12.5, "default
#: 4 GB"). A task needing strictly more than this is delegated.
DEFAULT_IN_PROCESS_MEMORY_LIMIT_BYTES = 4 * GB

#: The maximum input dataset size that may run in-process (Requirement 12.5,
#: "larger than 5 GB"). A dataset strictly larger than this is delegated.
MAX_IN_PROCESS_DATASET_BYTES = 5 * GB


class AccessPattern(str, Enum):
    """How a processing task accesses its input data (Requirement 12.4).

    The decision tree consults the access pattern only in the middle
    (1 GB - 100 GB) tier, where it selects among the analytical engines. A
    ``str`` enum so the value serializes directly to its wire string.
    """

    #: A single analyst scanning/aggregating data: DuckDB Spatial (in-process,
    #: open). Covers full-scan analytical reads.
    SINGLE_USER_ANALYTICAL = "single-user-analytical"
    #: Concurrent / transactional access by many clients: PostGIS backend.
    MULTI_USER_TRANSACTIONAL = "multi-user-transactional"
    #: Ad-hoc, windowed/scan queries over data left in S3: Amazon Athena.
    AD_HOC_OVER_S3 = "ad-hoc-over-s3"


class Engine(str, Enum):
    """A processing engine the decision tree can select (Requirement 12.4).

    Exactly one of these is returned for any task (Property 17). The first four
    are in-process / managed engines; :attr:`EMR_SEDONA` is the distributed
    engine run on the ``aws-geo-compute`` peer Power.
    """

    GEOPANDAS = "geopandas"  # in-process GeoPandas/Shapely (geo-ops)
    DUCKDB = "duckdb"  # DuckDB Spatial (geo-query)
    POSTGIS = "postgis"  # PostGIS backend (geo-query)
    ATHENA = "athena"  # Amazon Athena, ad-hoc over S3 (geo-query)
    EMR_SEDONA = "emr-sedona"  # Apache Sedona on EMR (aws-geo-compute peer)


#: The access-pattern -> engine mapping for the middle (1 GB - 100 GB) tier. A
#: module-level table makes the branch exhaustive over :class:`AccessPattern`
#: and the single source of truth for that tier.
_MIDDLE_TIER_BY_ACCESS_PATTERN: "Dict[AccessPattern, Engine]" = {
    AccessPattern.SINGLE_USER_ANALYTICAL: Engine.DUCKDB,
    AccessPattern.MULTI_USER_TRANSACTIONAL: Engine.POSTGIS,
    AccessPattern.AD_HOC_OVER_S3: Engine.ATHENA,
}


class ExecutionPlan(BaseModel):
    """The deterministic outcome of engine selection + the delegation gate.

    Records the selected :class:`Engine`, whether the task is delegated to the
    ``aws-geo-compute`` peer Power, the inputs the decision consulted, and a
    short human ``reason``. ``delegate`` is ``True`` exactly when
    :func:`should_delegate` holds (Requirement 12.5; Property 17).
    """

    engine: Engine
    delegate: bool
    dataset_size_bytes: int = Field(ge=0)
    required_memory_bytes: int = Field(ge=0)
    access_pattern: AccessPattern
    memory_limit_bytes: int = Field(gt=0)
    reason: str = Field(min_length=1)


def select_engine(
    *, dataset_size_bytes: int, access_pattern: AccessPattern
) -> Engine:
    """Select exactly one processing engine from the documented tree (Req 12.4).

    Evaluates the input ``dataset_size_bytes`` first, then - only in the
    1 GB - 100 GB tier - the ``access_pattern``:

    * ``< 1 GB``                 -> :attr:`Engine.GEOPANDAS`
    * ``1 GB .. 100 GB`` (incl.) -> :attr:`Engine.DUCKDB` /
      :attr:`Engine.POSTGIS` / :attr:`Engine.ATHENA` by access pattern
    * ``> 100 GB``               -> :attr:`Engine.EMR_SEDONA`

    The function is total over a valid (non-negative size, valid access pattern)
    input and deterministic: the same input always yields the same engine
    (Property 17).
    """
    if not isinstance(access_pattern, AccessPattern):
        raise TypeError(
            "access_pattern must be an AccessPattern, got %r"
            % type(access_pattern).__name__
        )
    if dataset_size_bytes < 0:
        raise ValueError("dataset_size_bytes must be non-negative")

    if dataset_size_bytes < SMALL_DATASET_LIMIT_BYTES:
        return Engine.GEOPANDAS
    if dataset_size_bytes <= DISTRIBUTED_DATASET_LIMIT_BYTES:
        # The mapping is exhaustive over AccessPattern, so this is total.
        return _MIDDLE_TIER_BY_ACCESS_PATTERN[access_pattern]
    return Engine.EMR_SEDONA


def should_delegate(
    *,
    dataset_size_bytes: int,
    required_memory_bytes: int,
    memory_limit_bytes: int = DEFAULT_IN_PROCESS_MEMORY_LIMIT_BYTES,
) -> bool:
    """Decide whether a task is delegated to the peer Power (Requirement 12.5).

    Returns ``True`` **if and only if** the task requires strictly more memory
    than ``memory_limit_bytes`` (default 4 GB) *or* operates on an input dataset
    strictly larger than 5 GB - the exact predicate of Property 17. Both
    comparisons are strict (">"), matching "more memory than the limit" and
    "larger than 5 GB".
    """
    if dataset_size_bytes < 0:
        raise ValueError("dataset_size_bytes must be non-negative")
    if required_memory_bytes < 0:
        raise ValueError("required_memory_bytes must be non-negative")
    if memory_limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be positive")

    return (
        required_memory_bytes > memory_limit_bytes
        or dataset_size_bytes > MAX_IN_PROCESS_DATASET_BYTES
    )


def select_execution_plan(
    *,
    dataset_size_bytes: int,
    access_pattern: AccessPattern,
    required_memory_bytes: int = 0,
    memory_limit_bytes: int = DEFAULT_IN_PROCESS_MEMORY_LIMIT_BYTES,
) -> ExecutionPlan:
    """Run the decision tree and the delegation gate, returning an ExecutionPlan.

    Combines :func:`select_engine` (Requirement 12.4) and :func:`should_delegate`
    (Requirement 12.5) into a single deterministic :class:`ExecutionPlan`. The
    result names the selected engine, whether the task is delegated, and the
    inputs and threshold that produced the decision.
    """
    engine = select_engine(
        dataset_size_bytes=dataset_size_bytes, access_pattern=access_pattern
    )
    delegate = should_delegate(
        dataset_size_bytes=dataset_size_bytes,
        required_memory_bytes=required_memory_bytes,
        memory_limit_bytes=memory_limit_bytes,
    )
    if delegate:
        reason = (
            "delegate to aws-geo-compute: required memory (%d bytes) exceeds the "
            "%d-byte limit or dataset (%d bytes) exceeds the %d-byte in-process "
            "maximum" % (
                required_memory_bytes,
                memory_limit_bytes,
                dataset_size_bytes,
                MAX_IN_PROCESS_DATASET_BYTES,
            )
        )
    else:
        reason = (
            "run in-process on %s: within the %d-byte memory limit and the "
            "%d-byte in-process dataset maximum"
            % (engine.value, memory_limit_bytes, MAX_IN_PROCESS_DATASET_BYTES)
        )
    return ExecutionPlan(
        engine=engine,
        delegate=delegate,
        dataset_size_bytes=dataset_size_bytes,
        required_memory_bytes=required_memory_bytes,
        access_pattern=access_pattern,
        memory_limit_bytes=memory_limit_bytes,
        reason=reason,
    )
