"""The single async HTTP client shared by every MCP server (Requirement 5.1).

This module implements :class:`HttpClient`, the one async HTTP client built on
``httpx.AsyncClient`` that all Geospatial Power Pack MCP servers use for
outbound requests. It centralizes the retry/backoff and rate-limit handling
defined by Requirement 5 so each server inherits identical behavior:

* Retries on rate-limit (HTTP 429), upstream server errors (HTTP 5xx), and
  transient network/connection failures, up to :attr:`RetryPolicy.max_attempts`
  (1-10, default 3) using the deterministic exponential backoff produced by
  :func:`geo_common.retry.backoff_schedule` (Requirements 5.2, 5.3).
* Honors a ``Retry-After`` indication of 120 seconds or less *exactly*
  (Requirement 5.6); stops and raises :class:`RateLimitError` when
  ``Retry-After`` exceeds 120 seconds (Requirement 5.7); applies exponential
  backoff when no ``Retry-After`` is present (Requirement 5.8).
* Aborts any single request that does not complete within
  :attr:`RetryPolicy.request_timeout_s` (30s) and raises
  :class:`NetworkError` (Requirement 5.9).
* Maps exhausted retries onto a ``NETWORK`` or ``UPSTREAM`` taxonomy error
  (Requirement 5.4): network/timeout causes -> ``NETWORK``; upstream 5xx or
  server rate-limiting causes -> ``UPSTREAM``.
* Maps an authentication/authorization status (401/403) directly onto the
  taxonomy (see design.md "Error Handling").

The backoff schedule is a pure function of the policy; the ``Retry-After``
logic lives here in the client. To keep the client testable without real
delays or network access, the constructor accepts an injectable async ``sleep``
callable and an ``httpx`` ``transport`` (e.g. ``httpx.MockTransport``).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable, Dict, Optional

import httpx

from geo_common.errors import (
    AuthenticationError,
    AuthorizationError,
    ErrorCategory,
    NetworkError,
    RateLimitError,
    UpstreamError,
)
from geo_common.retry import RetryPolicy, backoff_schedule

__all__ = ["HttpClient", "DEFAULT_USER_AGENT"]

#: A descriptive default ``User-Agent`` sent on every outbound request. Several
#: open geospatial APIs require a meaningful, non-default User-Agent and their
#: WAFs reject the stock ``python-httpx/<ver>`` (and browser-spoofing) agents
#: with HTTP 406 — notably OpenStreetMap's Overpass and Nominatim usage
#: policies. Identifying the client by name (overridable per server/request)
#: keeps the pack compliant and avoids those blocks.
DEFAULT_USER_AGENT = (
    "geospatial-kiro-power-pack/0.1 "
    "(+https://github.com/aws-samples/sample-geospatial-kiro-power-pack; geo-common HttpClient)"
)

#: Async sleep callable signature: ``await sleep(seconds)``.
SleepFn = Callable[[float], Awaitable[None]]


class HttpClient:
    """The single async HTTP client used by all servers (Requirement 5.1).

    Built on :class:`httpx.AsyncClient`; applies a :class:`RetryPolicy` with
    exponential backoff and rate-limit handling, and maps transport/status
    failures onto the shared ``Error_Taxonomy``.

    Parameters
    ----------
    policy:
        The :class:`RetryPolicy` governing attempts, backoff, per-request
        timeout, and the ``Retry-After`` cap. Defaults to ``RetryPolicy()``.
    transport:
        Optional ``httpx`` transport. Tests inject an ``httpx.MockTransport``
        here to drive deterministic responses without real network I/O.
    sleep:
        Optional async sleep callable used between retries. Defaults to
        :func:`asyncio.sleep`. Tests inject a recorder to assert exact waits
        (Requirement 5.6) without delaying.
    user_agent:
        The ``User-Agent`` sent on every request. Defaults to
        :data:`DEFAULT_USER_AGENT`; a server may override it (and any
        per-request ``headers={"User-Agent": ...}`` still takes precedence).
    """

    def __init__(
        self,
        policy: Optional[RetryPolicy] = None,
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        sleep: Optional[SleepFn] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        self._policy: RetryPolicy = policy if policy is not None else RetryPolicy()
        self._sleep: SleepFn = sleep if sleep is not None else asyncio.sleep
        # A descriptive default User-Agent is applied to every request (open
        # geospatial APIs such as Overpass/Nominatim reject the stock httpx
        # agent with HTTP 406). Per-request headers still override it.
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            transport=transport,
            headers={"User-Agent": user_agent or DEFAULT_USER_AGENT},
        )

    @property
    def policy(self) -> RetryPolicy:
        """The active retry/backoff policy."""
        return self._policy

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        content: Optional[bytes] = None,
        timeout_s: Optional[float] = None,
    ) -> httpx.Response:
        """Send ``method`` ``url`` with retry/backoff and rate-limit handling.

        Returns the :class:`httpx.Response` on success (and for non-retryable,
        non-auth statuses such as 2xx/3xx/4xx other than 401/403/429).

        Raises a :class:`~geo_common.errors.GeoError` subclass when retries are
        exhausted (``NETWORK``/``UPSTREAM``, Requirement 5.4), when a single
        request exceeds the timeout with no attempts left (``NETWORK``,
        Requirement 5.9), when a ``Retry-After`` exceeds the cap
        (``RATE_LIMIT``, Requirement 5.7), or on a 401/403 auth failure.
        """
        timeout = self._policy.request_timeout_s if timeout_s is None else timeout_s
        schedule = backoff_schedule(self._policy)
        source = self._host(url)

        # Terminal classification for the exhausted-retry path (Requirement 5.4).
        last_category = ErrorCategory.NETWORK
        last_original: Optional[str] = None

        for attempt in range(self._policy.max_attempts):
            try:
                response = await self._client.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    json=json,
                    content=content,
                    timeout=timeout,
                )
            except httpx.TimeoutException as exc:
                # Single request exceeded the timeout: abort this attempt and
                # treat it as a transient network failure (Requirements 5.9, 5.2).
                last_category = ErrorCategory.NETWORK
                last_original = self._safe_detail(exc)
                await self._backoff(attempt, schedule)
                continue
            except httpx.TransportError as exc:
                # Connection/network failure: transient and retryable (Req 5.2).
                last_category = ErrorCategory.NETWORK
                last_original = self._safe_detail(exc)
                await self._backoff(attempt, schedule)
                continue

            status = response.status_code

            if status == 429:
                retry_after = self._parse_retry_after(response)
                if (
                    retry_after is not None
                    and retry_after > self._policy.retry_after_cap_s
                ):
                    # Retry-After over the cap: stop and raise (Requirement 5.7).
                    raise RateLimitError(
                        "rate-limit retry-after exceeds the "
                        f"{self._policy.retry_after_cap_s:g}s cap",
                        source=source,
                        retry_after=retry_after,
                        detail={"status_code": status},
                    )
                # A server rate-limit is an upstream condition for the
                # exhausted-retry classification (Requirement 5.4).
                last_category = ErrorCategory.UPSTREAM
                last_original = self._status_detail(response)
                if self._can_retry(attempt):
                    if retry_after is not None:
                        # Honor the indicated duration exactly (Requirement 5.6).
                        await self._sleep(retry_after)
                    else:
                        # No Retry-After: exponential backoff (Requirement 5.8).
                        await self._sleep(schedule[attempt])
                continue

            if 500 <= status < 600:
                # Upstream server error: retryable (Requirement 5.2).
                last_category = ErrorCategory.UPSTREAM
                last_original = self._status_detail(response)
                await self._backoff(attempt, schedule)
                continue

            if status == 401:
                raise AuthenticationError(
                    "upstream rejected the request with HTTP 401",
                    source=source,
                    detail={"status_code": status},
                )
            if status == 403:
                raise AuthorizationError(
                    "upstream rejected the request with HTTP 403",
                    source=source,
                    detail={"status_code": status},
                )

            # Success or a non-retryable status the caller/server maps itself.
            return response

        # Every attempt failed: map to NETWORK or UPSTREAM (Requirement 5.4).
        message = (
            f"request to {source or url!r} failed after "
            f"{self._policy.max_attempts} attempt(s)"
        )
        detail = {"attempts": self._policy.max_attempts}
        if last_category is ErrorCategory.UPSTREAM:
            raise UpstreamError(
                message, source=source, detail=detail, original=last_original
            )
        raise NetworkError(
            message, source=source, detail=detail, original=last_original
        )

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Convenience wrapper for ``request("GET", url, ...)``."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """Convenience wrapper for ``request("POST", url, ...)``."""
        return await self.request("POST", url, **kwargs)

    async def aclose(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` and release resources."""
        await self._client.aclose()

    async def __aenter__(self) -> "HttpClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _can_retry(self, attempt: int) -> bool:
        """True when another attempt remains after the given 0-based attempt."""
        return attempt < self._policy.max_attempts - 1

    async def _backoff(self, attempt: int, schedule: "list[float]") -> None:
        """Sleep the scheduled exponential-backoff wait if a retry remains.

        ``schedule`` has exactly ``max_attempts - 1`` entries, so ``attempt``
        indexes it safely whenever another attempt remains.
        """
        if self._can_retry(attempt):
            await self._sleep(schedule[attempt])

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> Optional[float]:
        """Parse a ``Retry-After`` header to seconds, or ``None`` if absent.

        Supports both the numeric-seconds form and the HTTP-date form. A
        past/negative duration is clamped to ``0.0``.
        """
        raw = response.headers.get("Retry-After")
        if raw is None:
            return None
        value = raw.strip()
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        delta = (when - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)

    @staticmethod
    def _status_detail(response: httpx.Response) -> str:
        """A short, secret-free description of a failing HTTP response."""
        reason = response.reason_phrase or ""
        return f"HTTP {response.status_code} {reason}".strip()

    @staticmethod
    def _safe_detail(exc: Exception) -> str:
        """A secret-free description of a transport exception."""
        text = str(exc).strip()
        return text if text else type(exc).__name__

    @staticmethod
    def _host(url: str) -> Optional[str]:
        """Return the host of ``url`` for use as a secret-free ``source`` id."""
        try:
            return httpx.URL(url).host or None
        except Exception:  # pragma: no cover - defensive; malformed URL
            return None
