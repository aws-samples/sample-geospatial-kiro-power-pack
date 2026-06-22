"""Property-based test for the compute task-type -> target mapping.

**Feature: geospatial-power-pack, Property 18: Compute task-type mapping is
deterministic and single-valued**

*For any* supported task type, ``select_target`` returns exactly one execution
target (AWS Batch, ECS Fargate, EMR with Sedona, or SageMaker), and the same
task type always maps to the same target.

**Validates: Requirements 13.1**

These tests use Hypothesis (>= 100 generated cases per property, enforced by the
``geospatial-power-pack`` profile registered in the root ``conftest.py``). The
generator draws task types directly from the supported set
(:data:`TASK_TARGET_MAP` keys), the single source of truth for what is
supported, so every generated case exercises the deterministic, single-valued
mapping rather than the unsupported-type rejection path.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from aws_geo_compute.jobs import (
    TASK_TARGET_MAP,
    ExecutionTarget,
    select_target,
)

# --- Smart generator --------------------------------------------------------
# Draw only from the supported task types: a task type is "supported" iff it is
# a key in TASK_TARGET_MAP, so sampling the keys keeps generation inside the
# input space the property is stated over (supported task types).
supported_task_types = st.sampled_from(sorted(TASK_TARGET_MAP))


# --- Property 18: deterministic and single-valued mapping (Req 13.1) --------


@given(task_type=supported_task_types)
def test_select_target_returns_exactly_one_known_target(task_type: str) -> None:
    """``select_target`` returns exactly one member of the ExecutionTarget enum."""
    target = select_target(task_type)
    assert isinstance(target, ExecutionTarget)
    assert target in set(ExecutionTarget)


@given(task_type=supported_task_types)
def test_select_target_is_deterministic(task_type: str) -> None:
    """The same task type always maps to the same target across repeated calls."""
    first = select_target(task_type)
    second = select_target(task_type)
    assert first is second


@given(task_type=supported_task_types)
def test_select_target_is_single_valued(task_type: str) -> None:
    """Each supported task type resolves to exactly the one target the table
    records: the mapping is single-valued (one target, never a set)."""
    target = select_target(task_type)
    assert target is TASK_TARGET_MAP[task_type]
