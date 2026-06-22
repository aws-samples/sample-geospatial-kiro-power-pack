"""Property test for HTTP ``Retry-After`` honoring within the cap.

Feature: geospatial-power-pack, Property 10: Retry-after indications are honored within the cap

This module validates :class:`geo_common.http.HttpClient` against Property 10 of
the design (Validates: Requirements 5.6, 5.7):

*For any* rate-limit (HTTP 429) response carrying a ``Retry-After`` value, the
client waits **exactly** the indicated duration before retrying when the value
is 120 seconds or less (Requirement 5.6), and **stops** retrying — raising an
``Error_Taxonomy`` rate-limit error without waiting — when the value exceeds
120 seconds (Requirement 5.7).

Testability notes
-----------------
The client is exercised end-to-end through the public ``request`` path with two
injected collaborators, exactly as designed for testability:

* an ``httpx.MockTransport`` that returns deterministic 429/200 responses
  without real network I/O, and
* an async ``sleep`` recorder that appends each requested wait to a list
  instead of actually sleeping, so the honored duration can be asserted
  exactly (``math.isclose``) and the no-wait behavior can be confirmed.

Each generated example is driven via ``asyncio.run`` so the async client gets a
fresh event loop per case; the repo's ``asyncio_mode = "auto"`` configuration
governs the suite's example-based async tests. The ``@settings(deadline=None)``
decorator inherits the loaded profile's >=100-example minimum (Requirement 15.1).
"""

from __future__ import annotations

import asyncio
import math
from typing import List

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_common.errors import ErrorCategory, RateLimitError
from geo_common.http import HttpClient
from geo_common.retry import RetryPolicy

# The default Retry-After cap from RetryPolicy (Req 5.6 / 5.7): the client
# honors values at or below this and refuses values above it.
RETRY_AFTER_CAP_S = RetryPolicy().retry_after_cap_s  # 120.0

_URL = "https://api.example.test/resource"


def _format_retry_after(seconds: float) -> str:
    """Render a numeric-seconds ``Retry-After`` header value.

    ``str(float)`` is the repr round-trip in Python 3, so ``float(str(x)) == x``
    exactly for any finite float — the client therefore parses back the precise
    generated duration, letting us assert the honored wait exactly.
    """
    return str(seconds)


async def _honored_wait_for(retry_after: float) -> List[float]:
    """Drive a 429-then-200 exchange and return the recorded sleep waits.

    The first response is a 429 carrying ``Retry-After: <retry_after>``; the
    second is a 200. With ``retry_after`` within the cap the client must sleep
    exactly once for the indicated duration, then succeed on retry.
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429, headers={"Retry-After": _format_retry_after(retry_after)}
            )
        return httpx.Response(200, json={"ok": True})

    recorded: List[float] = []

    async def sleep_recorder(seconds: float) -> None:
        recorded.append(seconds)

    client = HttpClient(
        transport=httpx.MockTransport(handler),
        sleep=sleep_recorder,
    )
    try:
        response = await client.request("GET", _URL)
    finally:
        await client.aclose()

    # The retry must have ultimately succeeded (Req 5.6: wait, then retry).
    assert response.status_code == 200
    assert calls["n"] == 2
    return recorded


async def _waits_before_over_cap_error(retry_after: float) -> List[float]:
    """Drive a 429 with an over-cap ``Retry-After`` and return recorded waits.

    The transport always returns 429 with the over-cap header. The client must
    raise a rate-limit ``GeoError`` immediately, carrying the indicated
    ``retry_after``, and must not have recorded any sleep (Req 5.7).
    """
    recorded: List[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, headers={"Retry-After": _format_retry_after(retry_after)}
        )

    async def sleep_recorder(seconds: float) -> None:
        recorded.append(seconds)

    client = HttpClient(
        transport=httpx.MockTransport(handler),
        sleep=sleep_recorder,
    )
    try:
        with pytest.raises(RateLimitError) as excinfo:
            await client.request("GET", _URL)
    finally:
        await client.aclose()

    err = excinfo.value
    # Exactly one taxonomy category, and it is rate-limit (Req 5.7).
    assert err.category is ErrorCategory.RATE_LIMIT
    # The error carries the indicated retry-after duration.
    assert err.retry_after is not None
    assert math.isclose(err.retry_after, retry_after, rel_tol=1e-9, abs_tol=0.0)
    return recorded


# Values at or below the 120s cap. The lower bound stays strictly positive so a
# Retry-After is actually present; the upper bound includes the 120.0 boundary,
# which the cap check treats as honored (``retry_after > cap`` is False at 120).
retry_after_within_cap = st.floats(
    min_value=1e-3,
    max_value=RETRY_AFTER_CAP_S,
    allow_nan=False,
    allow_infinity=False,
)

# Values strictly above the cap. ``float(str(x)) == x`` guarantees a value
# generated above 120.0 parses back above 120.0, so the over-cap branch fires.
retry_after_over_cap = st.floats(
    min_value=math.nextafter(RETRY_AFTER_CAP_S, math.inf),
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


@pytest.mark.property
@settings(deadline=None)  # max_examples (>=100) is inherited from the loaded profile
@given(retry_after=retry_after_within_cap)
def test_retry_after_within_cap_is_honored_exactly(retry_after: float) -> None:
    """Feature: geospatial-power-pack, Property 10: Retry-after indications are honored within the cap.

    For a Retry-After value of 120s or less, the client waits EXACTLY the
    indicated duration before retrying.

    **Validates: Requirements 5.6**
    """
    recorded = asyncio.run(_honored_wait_for(retry_after))

    # Exactly one wait was performed — the honored Retry-After — before the
    # successful retry; no exponential-backoff wait was substituted.
    assert len(recorded) == 1
    assert math.isclose(recorded[0], retry_after, rel_tol=1e-9, abs_tol=0.0)


@pytest.mark.property
@settings(deadline=None)
@given(retry_after=retry_after_over_cap)
def test_retry_after_over_cap_stops_with_rate_limit_error(retry_after: float) -> None:
    """Feature: geospatial-power-pack, Property 10: Retry-after indications are honored within the cap.

    For a Retry-After value greater than 120s, the client stops retrying and
    returns a rate-limit error without waiting.

    **Validates: Requirements 5.7**
    """
    recorded = asyncio.run(_waits_before_over_cap_error(retry_after))

    # The client stopped immediately: it did not record any sleep for the
    # over-cap value (it neither honored it nor backed off).
    assert recorded == []
