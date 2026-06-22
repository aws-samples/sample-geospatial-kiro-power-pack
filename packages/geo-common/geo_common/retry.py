"""Retry and exponential-backoff policy for the shared HTTP client.

This module defines the :class:`RetryPolicy` configuration model and the
deterministic :func:`backoff_schedule` used by ``geo-common``'s async HTTP
client (Requirement 5.2, 5.3, 5.8).

The schedule is intentionally *deterministic* and ignores any server-provided
``Retry-After`` indication; retry-after handling lives in the HTTP client
itself. This keeps the schedule a pure function of the policy, which the
property test for Property 9 relies on.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["RetryPolicy", "backoff_schedule"]


class RetryPolicy(BaseModel):
    """Configuration for HTTP retry behavior and exponential backoff.

    Field constraints encode the acceptance criteria for Requirement 5:

    * ``max_attempts`` is bounded to 1–10 and defaults to 3 (Req 5.2).
    * Backoff starts at ``initial_delay_s`` and grows by ``multiplier`` each
      attempt, with each individual wait capped at ``max_delay_s`` (Req 5.3).
    * ``request_timeout_s`` bounds any single HTTP request (Req 5.9).
    * ``retry_after_cap_s`` is the largest server-provided ``Retry-After`` the
      client will honor before raising a rate-limit error (Req 5.6 / 5.7).
    """

    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_delay_s: float = Field(default=1.0, gt=0.0)
    multiplier: float = Field(default=2.0, ge=1.0)
    max_delay_s: float = Field(default=30.0, gt=0.0)
    request_timeout_s: float = Field(default=30.0, gt=0.0)
    retry_after_cap_s: float = Field(default=120.0, gt=0.0)


def backoff_schedule(policy: RetryPolicy) -> list[float]:
    """Return the deterministic per-attempt wait schedule for ``policy``.

    Each wait is ``Wait_n = min(initial_delay_s * multiplier**n, max_delay_s)``
    for ``n`` in ``0 .. max_attempts - 2``, yielding exactly
    ``max_attempts - 1`` waits (the number of gaps between attempts). The
    server-provided ``Retry-After`` indication is deliberately ignored here.

    The returned list is non-decreasing and every entry is capped at
    ``max_delay_s`` (Property 9; Requirements 5.2, 5.3, 5.8). When
    ``max_attempts`` is 1 there are no retries and the list is empty.
    """

    return [
        min(policy.initial_delay_s * policy.multiplier**n, policy.max_delay_s)
        for n in range(policy.max_attempts - 1)
    ]
