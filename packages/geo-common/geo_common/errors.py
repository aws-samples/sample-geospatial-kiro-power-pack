"""Error_Taxonomy for the Geospatial Power Pack (Requirements 5.5, 5.10).

This module defines the single, structured set of error categories every
MCP server in the Power Pack maps its source-specific errors onto:

    authentication, authorization, rate-limit, not-found,
    validation, upstream, network

It provides three things:

* :class:`ErrorCategory` - the enum of the seven taxonomy categories
  (Requirement 5.5). It is exported so each server can map source-specific
  errors onto the shared categories (Requirement 5.10).
* :class:`GeoError` - the base exception. Every instance carries **exactly
  one** :class:`ErrorCategory` and never leaks secret values. Seven concrete
  subclasses (one per category) make raising and catching ergonomic.
* :class:`ErrorObject` - a pydantic model that serializes an error for
  transport across tool/MCP boundaries, retaining the upstream ``original``
  detail (Requirement 11.5) without exposing secrets.

Design contract (see design.md "geo-common - shared base package"):

    detail   -> structured, secret-free detail
    original -> retained upstream detail (Req 11.5)
    retry_after / retry_after_s -> seconds, only meaningful for RATE_LIMIT
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Dict, Optional

from pydantic import BaseModel

__all__ = [
    "ErrorCategory",
    "GeoError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "NotFoundError",
    "ValidationError",
    "UpstreamError",
    "NetworkError",
    "ErrorObject",
]


class ErrorCategory(str, Enum):
    """The seven shared error categories of the Error_Taxonomy (Req 5.5).

    A ``str`` enum so the value serializes directly to its wire string
    (e.g. ``"rate-limit"``) in JSON and pydantic output, and so every server
    can map onto a stable, exposed category set (Req 5.10).
    """

    AUTHENTICATION = "authentication"  # missing/invalid credential (Req 5.5, 7.8, 10.5)
    AUTHORIZATION = "authorization"  # credential valid but not permitted
    RATE_LIMIT = "rate-limit"  # 429 / quota (Req 5.7)
    NOT_FOUND = "not-found"  # resource/job does not exist (Req 13.8)
    VALIDATION = "validation"  # malformed input (Req 7.12, 8.10, 9.7, 15.5)
    UPSTREAM = "upstream"  # source 5xx / unmapped external error (Req 11.5)
    NETWORK = "network"  # timeout / connection failure (Req 5.4, 5.9)


class GeoError(Exception):
    """Base error carrying exactly one taxonomy category. Never leaks secrets.

    The base constructor takes the ``category`` explicitly (matching the
    design contract) so callers can raise a taxonomy error directly. The seven
    concrete subclasses fix their category, so callers normally raise e.g.
    ``raise NotFoundError("item 42 not found", source="stac")`` without
    repeating the category.

    Secret safety: ``__str__`` / ``__repr__`` only ever expose the
    human-authored ``message``, the ``category``, the ``source`` identifier,
    and ``retry_after``. The ``detail`` and ``original`` fields are retained on
    the instance for programmatic use but are, by contract, secret-free and are
    deliberately kept out of the default string representations so that an
    accidental secret placed there can never leak through logging.
    """

    #: Subclasses set this to fix their category. ``None`` on the base means a
    #: category must be supplied explicitly at construction time.
    default_category: ClassVar[Optional[ErrorCategory]] = None

    def __init__(
        self,
        category: ErrorCategory,
        message: str,
        *,
        source: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
        original: Optional[str] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        if not isinstance(category, ErrorCategory):
            raise TypeError(
                "GeoError.category must be an ErrorCategory, "
                f"got {type(category).__name__!r}"
            )
        super().__init__(message)
        # Exactly one category per instance.
        self.category: ErrorCategory = category
        self.message: str = message
        self.source: Optional[str] = source
        self.detail: Optional[Dict[str, Any]] = detail
        self.original: Optional[str] = original
        self.retry_after: Optional[float] = retry_after

    def __str__(self) -> str:
        # Only the author-controlled message, never detail/original.
        return self.message

    def __repr__(self) -> str:
        parts = [f"category={self.category.value!r}", f"message={self.message!r}"]
        if self.source is not None:
            parts.append(f"source={self.source!r}")
        if self.retry_after is not None:
            parts.append(f"retry_after={self.retry_after!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def to_error_object(self) -> "ErrorObject":
        """Serialize this error to an :class:`ErrorObject` for transport.

        Retains the upstream ``original`` detail (Req 11.5). ``retry_after`` is
        mapped onto the model's ``retry_after_s`` field.
        """
        return ErrorObject(
            category=self.category,
            message=self.message,
            source=self.source,
            detail=self.detail,
            original=self.original,
            retry_after_s=self.retry_after,
        )


class _FixedCategoryError(GeoError):
    """Internal base for the concrete, single-category subclasses.

    Fixes the taxonomy category via :attr:`default_category` so subclasses are
    constructed as ``SubclassError(message, ...)`` without re-supplying the
    category, while still guaranteeing each instance carries exactly one.
    """

    def __init__(
        self,
        message: str,
        *,
        source: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
        original: Optional[str] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        category = type(self).default_category
        if category is None:  # pragma: no cover - guards against misdefined subclasses
            raise TypeError(
                f"{type(self).__name__} must define a default_category"
            )
        super().__init__(
            category,
            message,
            source=source,
            detail=detail,
            original=original,
            retry_after=retry_after,
        )


class AuthenticationError(_FixedCategoryError):
    """Missing or invalid credential (Req 5.5, 7.8, 10.5)."""

    default_category = ErrorCategory.AUTHENTICATION


class AuthorizationError(_FixedCategoryError):
    """Credential valid but the action is not permitted."""

    default_category = ErrorCategory.AUTHORIZATION


class RateLimitError(_FixedCategoryError):
    """Rate-limit / quota exceeded (Req 5.7).

    Set ``retry_after`` (seconds) when the source indicates one.
    """

    default_category = ErrorCategory.RATE_LIMIT


class NotFoundError(_FixedCategoryError):
    """Requested resource or job does not exist (Req 13.8)."""

    default_category = ErrorCategory.NOT_FOUND


class ValidationError(_FixedCategoryError):
    """Malformed or out-of-range input (Req 7.12, 8.10, 9.7, 15.5)."""

    default_category = ErrorCategory.VALIDATION


class UpstreamError(_FixedCategoryError):
    """Source 5xx or an unmapped external error (Req 11.5).

    Set ``original`` to retain the external server's error detail.
    """

    default_category = ErrorCategory.UPSTREAM


class NetworkError(_FixedCategoryError):
    """Timeout or connection failure (Req 5.4, 5.9)."""

    default_category = ErrorCategory.NETWORK


class ErrorObject(BaseModel):
    """Serializable error returned across tool/MCP boundaries (Req 5.5, 11.5).

    Mirrors :class:`GeoError` but as a pydantic model so it can be returned in
    tool responses and provenance. ``original`` retains the upstream detail
    for unmapped/wrapped errors (Req 11.5); ``detail`` carries structured,
    secret-free context; ``retry_after_s`` applies to rate-limit errors
    (Req 5.6 / 5.7).
    """

    category: ErrorCategory
    message: str
    source: Optional[str] = None
    detail: Optional[Dict[str, Any]] = None  # secret-free structured detail
    original: Optional[str] = None  # retained upstream detail (Req 11.5)
    retry_after_s: Optional[float] = None  # for rate-limit (Req 5.6 / 5.7)

    def to_exception(self) -> GeoError:
        """Reconstruct a :class:`GeoError` from this serialized form."""
        return GeoError(
            self.category,
            self.message,
            source=self.source,
            detail=self.detail,
            original=self.original,
            retry_after=self.retry_after_s,
        )
