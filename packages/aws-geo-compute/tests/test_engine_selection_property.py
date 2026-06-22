"""Property-based tests for engine selection and the delegation gate.

**Feature: geospatial-power-pack, Property 17: Engine and delegation selection
are deterministic**

*For any* processing task characterized by input dataset size and access
pattern (and required memory), the engine-selection decision tree returns
exactly one engine deterministically, and the task is delegated to the
``aws-geo-compute`` peer Power if and only if required memory exceeds 4 GB or
the input dataset exceeds 5 GB.

**Validates: Requirements 12.4, 12.5**

These tests use Hypothesis (>= 100 generated cases per property, enforced by the
``geospatial-power-pack`` profile registered in the root ``conftest.py``) over
smart generators that deliberately straddle every threshold in the decision
tree (1 GB, 100 GB) and the delegation gate (4 GB memory, 5 GB dataset), so the
strict boundary behavior is exercised, not just the bulk of the input space.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from aws_geo_compute.engine_selection import (
    DEFAULT_IN_PROCESS_MEMORY_LIMIT_BYTES,
    DISTRIBUTED_DATASET_LIMIT_BYTES,
    GB,
    MAX_IN_PROCESS_DATASET_BYTES,
    SMALL_DATASET_LIMIT_BYTES,
    AccessPattern,
    Engine,
    select_engine,
    select_execution_plan,
    should_delegate,
)

# --- Smart generators -------------------------------------------------------
# Generate sizes that cluster around the decision-tree and gate thresholds so
# the strict "<", "<=", ">" boundaries are hit repeatedly, while still covering
# the wider space (up to well past the 100 GB distributed cutoff).
_THRESHOLDS = (
    SMALL_DATASET_LIMIT_BYTES,  # 1 GB
    MAX_IN_PROCESS_DATASET_BYTES,  # 5 GB
    DISTRIBUTED_DATASET_LIMIT_BYTES,  # 100 GB
    DEFAULT_IN_PROCESS_MEMORY_LIMIT_BYTES,  # 4 GB
)

# Non-negative byte counts: a broad uniform range plus values within +/-2 bytes
# of every threshold to nail the boundaries.
_boundary_bytes = st.sampled_from(
    [
        max(0, t + delta)
        for t in _THRESHOLDS
        for delta in (-2, -1, 0, 1, 2)
    ]
)
sizes = st.one_of(
    st.integers(min_value=0, max_value=300 * GB),
    _boundary_bytes,
)
access_patterns = st.sampled_from(list(AccessPattern))
# Positive memory limits, again clustered around the 4 GB default.
memory_limits = st.one_of(
    st.integers(min_value=1, max_value=64 * GB),
    st.sampled_from([DEFAULT_IN_PROCESS_MEMORY_LIMIT_BYTES + d for d in (-2, -1, 0, 1, 2) if DEFAULT_IN_PROCESS_MEMORY_LIMIT_BYTES + d > 0]),
)


# --- Engine selection is single-valued and deterministic (Req 12.4) ---------


@given(dataset_size_bytes=sizes, access_pattern=access_patterns)
def test_select_engine_returns_exactly_one_known_engine(
    dataset_size_bytes: int, access_pattern: AccessPattern
) -> None:
    """The tree returns exactly one member of the Engine enum for any input."""
    engine = select_engine(
        dataset_size_bytes=dataset_size_bytes, access_pattern=access_pattern
    )
    assert isinstance(engine, Engine)
    assert engine in set(Engine)


@given(dataset_size_bytes=sizes, access_pattern=access_patterns)
def test_select_engine_is_deterministic(
    dataset_size_bytes: int, access_pattern: AccessPattern
) -> None:
    """Repeated calls on identical input yield the identical engine."""
    first = select_engine(
        dataset_size_bytes=dataset_size_bytes, access_pattern=access_pattern
    )
    second = select_engine(
        dataset_size_bytes=dataset_size_bytes, access_pattern=access_pattern
    )
    assert first is second


@given(dataset_size_bytes=sizes, access_pattern=access_patterns)
def test_select_engine_matches_documented_tree(
    dataset_size_bytes: int, access_pattern: AccessPattern
) -> None:
    """The selected engine matches the documented size/access-pattern tree."""
    engine = select_engine(
        dataset_size_bytes=dataset_size_bytes, access_pattern=access_pattern
    )
    if dataset_size_bytes < SMALL_DATASET_LIMIT_BYTES:
        assert engine is Engine.GEOPANDAS
    elif dataset_size_bytes <= DISTRIBUTED_DATASET_LIMIT_BYTES:
        expected = {
            AccessPattern.SINGLE_USER_ANALYTICAL: Engine.DUCKDB,
            AccessPattern.MULTI_USER_TRANSACTIONAL: Engine.POSTGIS,
            AccessPattern.AD_HOC_OVER_S3: Engine.ATHENA,
        }[access_pattern]
        assert engine is expected
    else:
        assert engine is Engine.EMR_SEDONA


# --- Delegation gate is the exact iff predicate (Req 12.5) ------------------


@given(
    dataset_size_bytes=sizes,
    required_memory_bytes=sizes,
)
def test_delegate_iff_memory_over_4gb_or_dataset_over_5gb(
    dataset_size_bytes: int, required_memory_bytes: int
) -> None:
    """With the default limit: delegate iff memory > 4 GB OR dataset > 5 GB."""
    expected = (
        required_memory_bytes > DEFAULT_IN_PROCESS_MEMORY_LIMIT_BYTES
        or dataset_size_bytes > MAX_IN_PROCESS_DATASET_BYTES
    )
    assert (
        should_delegate(
            dataset_size_bytes=dataset_size_bytes,
            required_memory_bytes=required_memory_bytes,
        )
        is expected
    )


@given(
    dataset_size_bytes=sizes,
    required_memory_bytes=sizes,
    memory_limit_bytes=memory_limits,
)
def test_delegate_iff_predicate_with_configurable_limit(
    dataset_size_bytes: int,
    required_memory_bytes: int,
    memory_limit_bytes: int,
) -> None:
    """The iff predicate uses the supplied memory limit and the fixed 5 GB cap."""
    expected = (
        required_memory_bytes > memory_limit_bytes
        or dataset_size_bytes > MAX_IN_PROCESS_DATASET_BYTES
    )
    assert (
        should_delegate(
            dataset_size_bytes=dataset_size_bytes,
            required_memory_bytes=required_memory_bytes,
            memory_limit_bytes=memory_limit_bytes,
        )
        is expected
    )


@given(
    dataset_size_bytes=sizes,
    required_memory_bytes=sizes,
)
def test_should_delegate_is_deterministic(
    dataset_size_bytes: int, required_memory_bytes: int
) -> None:
    """The gate is a pure function: identical input -> identical decision."""
    first = should_delegate(
        dataset_size_bytes=dataset_size_bytes,
        required_memory_bytes=required_memory_bytes,
    )
    second = should_delegate(
        dataset_size_bytes=dataset_size_bytes,
        required_memory_bytes=required_memory_bytes,
    )
    assert first is second


# --- Combined plan ties the two decisions together --------------------------


@given(
    dataset_size_bytes=sizes,
    required_memory_bytes=sizes,
    access_pattern=access_patterns,
    memory_limit_bytes=memory_limits,
)
def test_execution_plan_is_consistent_and_deterministic(
    dataset_size_bytes: int,
    required_memory_bytes: int,
    access_pattern: AccessPattern,
    memory_limit_bytes: int,
) -> None:
    """The plan's engine and delegate flag agree with the component functions,
    and the whole plan is reproduced exactly on a second call."""
    kwargs = dict(
        dataset_size_bytes=dataset_size_bytes,
        access_pattern=access_pattern,
        required_memory_bytes=required_memory_bytes,
        memory_limit_bytes=memory_limit_bytes,
    )
    plan = select_execution_plan(**kwargs)

    assert plan.engine is select_engine(
        dataset_size_bytes=dataset_size_bytes, access_pattern=access_pattern
    )
    assert plan.delegate is should_delegate(
        dataset_size_bytes=dataset_size_bytes,
        required_memory_bytes=required_memory_bytes,
        memory_limit_bytes=memory_limit_bytes,
    )
    # A distributed (> 100 GB) dataset is always delegated (it exceeds 5 GB).
    if dataset_size_bytes > DISTRIBUTED_DATASET_LIMIT_BYTES:
        assert plan.engine is Engine.EMR_SEDONA
        assert plan.delegate is True

    # Determinism of the full structured result.
    again = select_execution_plan(**kwargs)
    assert plan == again
