"""Property test for the deterministic exponential-backoff schedule.

Feature: geospatial-power-pack, Property 9: Exponential backoff schedule is correct

This module validates :func:`geo_common.retry.backoff_schedule` against
Property 9 of the design (Validates: Requirements 5.2, 5.3, 5.8):

*For any* retry policy with ``max_attempts`` in 1-10, the computed backoff
schedule equals ``min(initial_delay_s * multiplier**n, max_delay_s)`` for each
retry index ``n``, is non-decreasing, never exceeds the 30-second cap (under
the default ``max_delay_s``), and contains exactly ``max_attempts - 1`` waits.

The symbols under test are imported directly from the submodule (rather than
from the ``geo_common`` package root) to avoid coupling to ``__init__.py``.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_common.retry import RetryPolicy, backoff_schedule

# Default cap from RetryPolicy (Req 5.3): each individual wait is capped at 30s.
DEFAULT_MAX_DELAY_S = 30.0


# A strategy producing valid RetryPolicy instances across the documented input
# space. Bounds respect the model's field constraints (max_attempts 1-10,
# initial_delay_s > 0, multiplier >= 1.0, max_delay_s > 0) while staying in a
# numeric range where ``multiplier ** n`` is exact enough to reason about.
retry_policies = st.builds(
    RetryPolicy,
    max_attempts=st.integers(min_value=1, max_value=10),
    initial_delay_s=st.floats(
        min_value=1e-3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
    ),
    multiplier=st.floats(
        min_value=1.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    max_delay_s=st.floats(
        min_value=1e-3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
    ),
)


@pytest.mark.property
@settings(deadline=None)  # max_examples (>=100) is inherited from the loaded profile
@given(policy=retry_policies)
def test_backoff_schedule_is_correct(policy: RetryPolicy) -> None:
    """Feature: geospatial-power-pack, Property 9: Exponential backoff schedule is correct.

    Validates: Requirements 5.2, 5.3, 5.8
    """
    schedule = backoff_schedule(policy)

    # (a) Exactly max_attempts - 1 waits (the gaps between attempts). Req 5.2.
    assert len(schedule) == policy.max_attempts - 1

    for n, wait in enumerate(schedule):
        expected = min(
            policy.initial_delay_s * policy.multiplier**n,
            policy.max_delay_s,
        )
        # (b) Each wait equals the capped exponential term. Req 5.3.
        assert math.isclose(wait, expected, rel_tol=1e-9, abs_tol=0.0)

        # (d) No entry exceeds the policy's own cap. Req 5.3.
        assert wait <= policy.max_delay_s or math.isclose(wait, policy.max_delay_s)

    # (c) The schedule is non-decreasing (multiplier >= 1 => monotonic; the cap
    # preserves monotonicity). Req 5.3 / 5.8.
    for earlier, later in zip(schedule, schedule[1:]):
        assert later >= earlier or math.isclose(earlier, later, rel_tol=1e-9)


@pytest.mark.property
@settings(deadline=None)
@given(
    max_attempts=st.integers(min_value=1, max_value=10),
    initial_delay_s=st.floats(
        min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
    multiplier=st.floats(
        min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
)
def test_default_cap_never_exceeds_30_seconds(
    max_attempts: int, initial_delay_s: float, multiplier: float
) -> None:
    """With the default ``max_delay_s`` (30s), no wait ever exceeds 30 seconds.

    Feature: geospatial-power-pack, Property 9: Exponential backoff schedule is correct
    Validates: Requirements 5.3, 5.8
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        initial_delay_s=initial_delay_s,
        multiplier=multiplier,
        # max_delay_s left at its 30.0 default.
    )
    assert policy.max_delay_s == DEFAULT_MAX_DELAY_S

    schedule = backoff_schedule(policy)
    assert len(schedule) == max_attempts - 1
    for wait in schedule:
        assert wait <= DEFAULT_MAX_DELAY_S or math.isclose(wait, DEFAULT_MAX_DELAY_S)
