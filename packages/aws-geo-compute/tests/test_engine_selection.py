"""Unit tests for engine selection and the delegation threshold (task 17.1).

Covers the documented decision tree (Requirement 12.4), the
in-process-vs-delegate threshold (Requirement 12.5), and the determinism the
later Property 17 relies on - all as pure-function tests requiring no AWS.
"""

from __future__ import annotations

import pytest

from aws_geo_compute.engine_selection import (
    DEFAULT_IN_PROCESS_MEMORY_LIMIT_BYTES,
    GB,
    MAX_IN_PROCESS_DATASET_BYTES,
    AccessPattern,
    Engine,
    select_engine,
    select_execution_plan,
    should_delegate,
)


# --- Decision tree by dataset size (Req 12.4) ------------------------------


@pytest.mark.parametrize("pattern", list(AccessPattern))
def test_small_dataset_runs_in_process_geopandas(pattern: AccessPattern) -> None:
    """< 1 GB -> GeoPandas, regardless of access pattern."""
    assert (
        select_engine(dataset_size_bytes=512 * 1024 * 1024, access_pattern=pattern)
        is Engine.GEOPANDAS
    )


def test_just_below_one_gb_is_geopandas_and_one_gb_is_middle_tier() -> None:
    """Boundary: < 1 GB is GeoPandas; exactly 1 GB enters the middle tier."""
    assert (
        select_engine(
            dataset_size_bytes=1 * GB - 1,
            access_pattern=AccessPattern.SINGLE_USER_ANALYTICAL,
        )
        is Engine.GEOPANDAS
    )
    assert (
        select_engine(
            dataset_size_bytes=1 * GB,
            access_pattern=AccessPattern.SINGLE_USER_ANALYTICAL,
        )
        is Engine.DUCKDB
    )


@pytest.mark.parametrize(
    "pattern, expected",
    [
        (AccessPattern.SINGLE_USER_ANALYTICAL, Engine.DUCKDB),
        (AccessPattern.MULTI_USER_TRANSACTIONAL, Engine.POSTGIS),
        (AccessPattern.AD_HOC_OVER_S3, Engine.ATHENA),
    ],
)
def test_middle_tier_branches_on_access_pattern(
    pattern: AccessPattern, expected: Engine
) -> None:
    """1 GB - 100 GB selects the engine for the access pattern."""
    assert (
        select_engine(dataset_size_bytes=10 * GB, access_pattern=pattern) is expected
    )


def test_hundred_gb_is_middle_tier_and_above_is_emr() -> None:
    """Boundary: exactly 100 GB stays in-tier; > 100 GB goes to EMR Sedona."""
    assert (
        select_engine(
            dataset_size_bytes=100 * GB,
            access_pattern=AccessPattern.AD_HOC_OVER_S3,
        )
        is Engine.ATHENA
    )
    assert (
        select_engine(
            dataset_size_bytes=100 * GB + 1,
            access_pattern=AccessPattern.AD_HOC_OVER_S3,
        )
        is Engine.EMR_SEDONA
    )


def test_select_engine_is_deterministic() -> None:
    """The same input always yields the same engine (Property 17 basis)."""
    kwargs = dict(
        dataset_size_bytes=42 * GB,
        access_pattern=AccessPattern.MULTI_USER_TRANSACTIONAL,
    )
    assert select_engine(**kwargs) is select_engine(**kwargs) is Engine.POSTGIS


def test_select_engine_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        select_engine(dataset_size_bytes=-1, access_pattern=AccessPattern.AD_HOC_OVER_S3)
    with pytest.raises(TypeError):
        select_engine(dataset_size_bytes=1, access_pattern="not-a-pattern")  # type: ignore[arg-type]


# --- Delegation threshold (Req 12.5; Property 17 predicate) ----------------


def test_delegate_iff_memory_or_dataset_over_limit() -> None:
    """Delegate iff required memory > 4 GB OR dataset > 5 GB (strict)."""
    # Neither exceeded -> in-process.
    assert (
        should_delegate(dataset_size_bytes=5 * GB, required_memory_bytes=4 * GB)
        is False
    )
    # Memory just over the limit -> delegate.
    assert (
        should_delegate(dataset_size_bytes=1 * GB, required_memory_bytes=4 * GB + 1)
        is True
    )
    # Dataset just over the limit -> delegate.
    assert (
        should_delegate(dataset_size_bytes=5 * GB + 1, required_memory_bytes=0)
        is True
    )


def test_threshold_boundaries_are_strict() -> None:
    """Exactly at the limit does not delegate; one byte over does."""
    assert should_delegate(
        dataset_size_bytes=MAX_IN_PROCESS_DATASET_BYTES,
        required_memory_bytes=DEFAULT_IN_PROCESS_MEMORY_LIMIT_BYTES,
    ) is False


def test_configurable_memory_limit() -> None:
    """A non-default memory limit moves the delegation boundary."""
    assert (
        should_delegate(
            dataset_size_bytes=0,
            required_memory_bytes=2 * GB + 1,
            memory_limit_bytes=2 * GB,
        )
        is True
    )


def test_should_delegate_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        should_delegate(dataset_size_bytes=-1, required_memory_bytes=0)
    with pytest.raises(ValueError):
        should_delegate(dataset_size_bytes=0, required_memory_bytes=-1)
    with pytest.raises(ValueError):
        should_delegate(dataset_size_bytes=0, required_memory_bytes=0, memory_limit_bytes=0)


# --- Combined execution plan -----------------------------------------------


def test_distributed_dataset_always_delegates() -> None:
    """> 100 GB selects EMR Sedona and is delegated (dataset > 5 GB)."""
    plan = select_execution_plan(
        dataset_size_bytes=200 * GB,
        access_pattern=AccessPattern.SINGLE_USER_ANALYTICAL,
    )
    assert plan.engine is Engine.EMR_SEDONA
    assert plan.delegate is True


def test_small_in_process_plan() -> None:
    plan = select_execution_plan(
        dataset_size_bytes=100 * 1024 * 1024,
        access_pattern=AccessPattern.SINGLE_USER_ANALYTICAL,
        required_memory_bytes=1 * GB,
    )
    assert plan.engine is Engine.GEOPANDAS
    assert plan.delegate is False
    assert plan.reason


def test_memory_pressure_delegates_even_for_small_dataset() -> None:
    """A small dataset that needs > 4 GB memory is still delegated (Req 12.5)."""
    plan = select_execution_plan(
        dataset_size_bytes=100 * 1024 * 1024,
        access_pattern=AccessPattern.SINGLE_USER_ANALYTICAL,
        required_memory_bytes=8 * GB,
    )
    assert plan.engine is Engine.GEOPANDAS  # tree still picks the size-based engine
    assert plan.delegate is True  # but the threshold forces delegation
