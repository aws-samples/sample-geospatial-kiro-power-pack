"""Example-based unit tests for ``HttpClient`` transport behavior.

These tests exercise :class:`geo_common.http.HttpClient` over an
``httpx.MockTransport`` so the retry/backoff machinery runs deterministically
with no real network I/O and no real delays. An injected async ``sleep``
recorder captures the exact wait durations the client requests between
attempts, which lets us assert backoff timing precisely.

Coverage maps onto Requirement 5:

* Req 5.2 - retry on rate-limit (429), upstream 5xx, and transient network
  failures, using exponential backoff up to ``max_attempts``.
* Req 5.4 - exhausted retries map to a NETWORK or UPSTREAM taxonomy error
  (network/timeout causes -> NETWORK; 5xx / server rate-limit -> UPSTREAM).
* Req 5.8 - when no ``Retry-After`` is present, exponential backoff is applied;
  the recorded waits equal ``backoff_schedule(policy)`` (or a prefix of it).
* Req 5.9 - a per-request timeout aborts the attempt; exhausting retries on
  timeouts yields a NETWORK error.

The companion Property 10 (``Retry-After`` honoring) lives in its own module
(task 2.5); this file deliberately keeps to no-``Retry-After`` transport,
backoff, timeout, and exhaustion behavior.
"""

from __future__ import annotations

from typing import Any, Callable, List, Tuple, Union

import httpx
import pytest

from geo_common.errors import (
    AuthenticationError,
    AuthorizationError,
    ErrorCategory,
    NetworkError,
    UpstreamError,
)
from geo_common.http import HttpClient
from geo_common.retry import RetryPolicy, backoff_schedule

# These are example-based tests driven through a mocked transport.
pytestmark = pytest.mark.integration

URL = "https://example.com/data"
HOST = "example.com"

# A single "action" the mock transport takes for one attempt: either an HTTP
# status code (int), a ready ``httpx.Response``, or an exception instance to
# raise (simulating a transport-level failure).
Action = Union[int, httpx.Response, BaseException]


class SleepRecorder:
    """An async ``sleep`` stand-in that records requested waits without delay.

    Injected as ``HttpClient(..., sleep=recorder)`` so tests can assert the
    exact backoff durations the client requested (Requirements 5.2, 5.8) while
    the suite runs instantly.
    """

    def __init__(self) -> None:
        self.waits: List[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def _sequence_handler(
    actions: List[Action],
) -> Tuple[Callable[[httpx.Request], httpx.Response], List[int]]:
    """Build a MockTransport handler that walks ``actions`` per attempt.

    The Nth request consumes ``actions[N]``: an ``int`` becomes an
    ``httpx.Response`` with that status, an ``httpx.Response`` is returned as
    is, and an exception instance is raised to simulate a transport failure.
    Requests beyond the list default to a ``200`` response. The returned
    ``calls`` list (single counter) lets a test assert how many attempts ran.
    """
    calls: List[int] = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        index = calls[0]
        calls[0] += 1
        action: Action = actions[index] if index < len(actions) else 200
        if isinstance(action, BaseException):
            raise action
        if isinstance(action, httpx.Response):
            return action
        return httpx.Response(action)

    return handler, calls


def _make_client(
    actions: List[Action],
    *,
    policy: RetryPolicy,
    recorder: SleepRecorder,
) -> Tuple[HttpClient, List[int]]:
    """Wire an ``HttpClient`` to a sequence-driven mock transport + recorder."""
    handler, calls = _sequence_handler(actions)
    client = HttpClient(
        policy,
        transport=httpx.MockTransport(handler),
        sleep=recorder,
    )
    return client, calls


# ---------------------------------------------------------------------------
# Retry-then-success: backoff applied between attempts (Req 5.2, 5.8)
# ---------------------------------------------------------------------------
async def test_retry_then_success_on_429_uses_exponential_backoff() -> None:
    """A 429 with no ``Retry-After`` is retried with one backoff wait, then 200.

    Validates: Requirements 5.2, 5.8
    """
    policy = RetryPolicy()  # max_attempts=3, schedule == [1.0, 2.0]
    recorder = SleepRecorder()
    client, calls = _make_client([429, 200], policy=policy, recorder=recorder)

    async with client:
        response = await client.get(URL)

    assert response.status_code == 200
    assert calls[0] == 2  # one failed attempt, then success
    # No Retry-After header -> exponential backoff prefix of length 1 (Req 5.8).
    assert recorder.waits == backoff_schedule(policy)[:1] == [1.0]


async def test_retry_then_success_on_5xx_uses_exponential_backoff() -> None:
    """An upstream 503 is retried with one backoff wait, then succeeds.

    Validates: Requirements 5.2, 5.8
    """
    policy = RetryPolicy()
    recorder = SleepRecorder()
    client, calls = _make_client([503, 200], policy=policy, recorder=recorder)

    async with client:
        response = await client.get(URL)

    assert response.status_code == 200
    assert calls[0] == 2
    assert recorder.waits == backoff_schedule(policy)[:1] == [1.0]


async def test_retry_then_success_on_transient_network_failure() -> None:
    """A transient connection failure is retried, then the request succeeds.

    Validates: Requirements 5.2, 5.8
    """
    policy = RetryPolicy()
    recorder = SleepRecorder()
    client, calls = _make_client(
        [httpx.ConnectError("simulated connection failure"), 200],
        policy=policy,
        recorder=recorder,
    )

    async with client:
        response = await client.get(URL)

    assert response.status_code == 200
    assert calls[0] == 2
    assert recorder.waits == backoff_schedule(policy)[:1] == [1.0]


# ---------------------------------------------------------------------------
# Per-request timeout abort -> NETWORK on exhaustion (Req 5.9)
# ---------------------------------------------------------------------------
async def test_timeout_on_every_attempt_exhausts_to_network_error() -> None:
    """A request that times out on every attempt aborts and yields NetworkError.

    Validates: Requirements 5.9, 5.4
    """
    policy = RetryPolicy()  # 3 attempts -> schedule [1.0, 2.0]
    recorder = SleepRecorder()
    client, calls = _make_client(
        [
            httpx.ReadTimeout("simulated read timeout"),
            httpx.ReadTimeout("simulated read timeout"),
            httpx.ReadTimeout("simulated read timeout"),
        ],
        policy=policy,
        recorder=recorder,
    )

    async with client:
        with pytest.raises(NetworkError) as exc_info:
            await client.get(URL)

    err = exc_info.value
    assert err.category is ErrorCategory.NETWORK
    assert err.source == HOST
    assert calls[0] == policy.max_attempts  # every attempt was made
    # Backoff between the three attempts == the full schedule (no wait after
    # the final attempt). Req 5.2 / 5.8.
    assert recorder.waits == backoff_schedule(policy) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Retry exhaustion category mapping (Req 5.4)
# ---------------------------------------------------------------------------
async def test_exhausted_retries_all_5xx_map_to_upstream_error() -> None:
    """All-5xx exhaustion maps to an UPSTREAM taxonomy error.

    Validates: Requirements 5.4
    """
    policy = RetryPolicy()
    recorder = SleepRecorder()
    client, calls = _make_client([503, 503, 503], policy=policy, recorder=recorder)

    async with client:
        with pytest.raises(UpstreamError) as exc_info:
            await client.get(URL)

    err = exc_info.value
    assert err.category is ErrorCategory.UPSTREAM
    assert err.source == HOST
    assert calls[0] == policy.max_attempts
    assert recorder.waits == backoff_schedule(policy) == [1.0, 2.0]


async def test_exhausted_retries_all_network_map_to_network_error() -> None:
    """All transient-network exhaustion maps to a NETWORK taxonomy error.

    Validates: Requirements 5.4
    """
    policy = RetryPolicy()
    recorder = SleepRecorder()
    client, calls = _make_client(
        [
            httpx.ConnectError("simulated connection failure"),
            httpx.ConnectError("simulated connection failure"),
            httpx.ConnectError("simulated connection failure"),
        ],
        policy=policy,
        recorder=recorder,
    )

    async with client:
        with pytest.raises(NetworkError) as exc_info:
            await client.get(URL)

    err = exc_info.value
    assert err.category is ErrorCategory.NETWORK
    assert err.source == HOST
    assert calls[0] == policy.max_attempts
    assert recorder.waits == backoff_schedule(policy) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Backoff timing: exact recorded sequence equals the policy's schedule (Req 5.8)
# ---------------------------------------------------------------------------
async def test_backoff_timing_matches_schedule_for_custom_policy() -> None:
    """Recorded waits equal ``backoff_schedule`` exactly for a custom policy.

    Validates: Requirements 5.2, 5.8
    """
    policy = RetryPolicy(max_attempts=4, initial_delay_s=0.5, multiplier=3.0)
    expected = backoff_schedule(policy)
    assert expected == [0.5, 1.5, 4.5]  # sanity: documented growth

    recorder = SleepRecorder()
    client, calls = _make_client(
        [503, 503, 503, 503], policy=policy, recorder=recorder
    )

    async with client:
        with pytest.raises(UpstreamError):
            await client.get(URL)

    assert calls[0] == policy.max_attempts
    assert recorder.waits == expected


async def test_backoff_timing_respects_30s_cap() -> None:
    """Each individual wait is capped at ``max_delay_s`` (30s default).

    Validates: Requirements 5.3, 5.8
    """
    policy = RetryPolicy(max_attempts=5, initial_delay_s=10.0, multiplier=2.0)
    expected = backoff_schedule(policy)
    assert expected == [10.0, 20.0, 30.0, 30.0]  # 40 and 80 are capped to 30

    recorder = SleepRecorder()
    client, _ = _make_client(
        [503, 503, 503, 503, 503], policy=policy, recorder=recorder
    )

    async with client:
        with pytest.raises(UpstreamError):
            await client.get(URL)

    assert recorder.waits == expected
    assert all(wait <= policy.max_delay_s for wait in recorder.waits)


# ---------------------------------------------------------------------------
# Auth statuses map directly and are not retried (transport status mapping)
# ---------------------------------------------------------------------------
async def test_401_maps_to_authentication_error_without_retry() -> None:
    """A 401 raises AuthenticationError immediately with no backoff.

    Validates: Requirements 5.10
    """
    policy = RetryPolicy()
    recorder = SleepRecorder()
    client, calls = _make_client([401], policy=policy, recorder=recorder)

    async with client:
        with pytest.raises(AuthenticationError) as exc_info:
            await client.get(URL)

    assert exc_info.value.category is ErrorCategory.AUTHENTICATION
    assert calls[0] == 1  # not retried
    assert recorder.waits == []


async def test_403_maps_to_authorization_error_without_retry() -> None:
    """A 403 raises AuthorizationError immediately with no backoff.

    Validates: Requirements 5.10
    """
    policy = RetryPolicy()
    recorder = SleepRecorder()
    client, calls = _make_client([403], policy=policy, recorder=recorder)

    async with client:
        with pytest.raises(AuthorizationError) as exc_info:
            await client.get(URL)

    assert exc_info.value.category is ErrorCategory.AUTHORIZATION
    assert calls[0] == 1
    assert recorder.waits == []
