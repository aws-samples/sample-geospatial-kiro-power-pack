"""Property test for taxonomy mapping totality.

Feature: geospatial-power-pack, Property 8: Every error maps to exactly one taxonomy category

This module validates :meth:`geo_common.server.BaseGeoServer.map_error` against
Property 8 of the design (Validates: Requirements 5.5, 5.10, 11.2, 11.5):

*For any* error encountered — a transport/HTTP condition, a server-raised
``GeoError``, or an error returned by a wrapped external server — the mapping
yields **exactly one** ``Error_Taxonomy`` category; an external error with no
corresponding category maps to ``upstream`` and **retains** the external
server's original error detail.

Testability notes
-----------------
``map_error`` is a pure, total function from an arbitrary ``Exception`` onto a
``GeoError``. The properties are therefore exercised directly, with no mocks:

* a heterogeneous strategy generates the full variety of inputs the mapping
  must handle — generic library/external exceptions, ``httpx`` HTTP status
  errors across every status code, ``httpx`` timeout/transport failures, and
  already-classified ``GeoError`` instances, and
* an independent oracle (:func:`_expected_category`) re-derives the category
  the design specifies, so each generated case asserts the mapping agrees.

The ``@settings(deadline=None)`` decorator inherits the loaded profile's
>=100-example minimum (Requirement 15.1).
"""

from __future__ import annotations

from typing import Optional

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_common.errors import (
    AuthenticationError,
    AuthorizationError,
    ErrorCategory,
    GeoError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    UpstreamError,
    ValidationError,
)
from geo_common.server import BaseGeoServer

# A secret-free source identifier supplied to every mapping call.
_SOURCE = "geo-test-source"

# The full set of taxonomy categories the mapping may ever produce (Req 5.5).
_ALL_CATEGORIES = set(ErrorCategory)

# Status-code -> taxonomy category, mirroring the design's documented mapping
# (Req 11.2). Any status not listed here (notably 5xx) falls through to UPSTREAM.
_STATUS_CATEGORY = {
    401: ErrorCategory.AUTHENTICATION,
    403: ErrorCategory.AUTHORIZATION,
    404: ErrorCategory.NOT_FOUND,
    429: ErrorCategory.RATE_LIMIT,
    400: ErrorCategory.VALIDATION,
    422: ErrorCategory.VALIDATION,
}


def _expected_category(exc: Exception) -> ErrorCategory:
    """Independent oracle for the category the design requires for ``exc``.

    Re-derives the expected taxonomy category from the exception type and (for
    HTTP status errors) the status code, without consulting the implementation
    under test.
    """
    if isinstance(exc, GeoError):
        return exc.category
    # Timeouts are a subclass of TransportError and must be NETWORK (Req 5.9).
    if isinstance(exc, httpx.TimeoutException):
        return ErrorCategory.NETWORK
    if isinstance(exc, httpx.HTTPStatusError):
        return _STATUS_CATEGORY.get(
            exc.response.status_code, ErrorCategory.UPSTREAM
        )
    if isinstance(exc, httpx.TransportError):
        return ErrorCategory.NETWORK
    # Any other exception is an unmapped external/library error (Req 11.5).
    return ErrorCategory.UPSTREAM


def _safe_detail(exc: Exception) -> str:
    """The secret-free ``original`` detail the mapping retains for an exception."""
    text = str(exc).strip()
    return text if text else type(exc).__name__


# --- Strategies over the variety of exceptions the mapping must handle -------

# Generic, non-httpx exceptions stand in for unmapped library/external-server
# errors (the catch-all that must route to UPSTREAM, Req 11.5).
_GENERIC_EXC_TYPES = [
    Exception,
    ValueError,
    KeyError,
    RuntimeError,
    TypeError,
    OSError,
    ArithmeticError,
    LookupError,
    NotImplementedError,
]


@st.composite
def generic_exceptions(draw: st.DrawFn) -> Exception:
    """Arbitrary non-httpx exceptions with arbitrary (incl. empty) messages."""
    exc_type = draw(st.sampled_from(_GENERIC_EXC_TYPES))
    message = draw(st.text(max_size=200))
    return exc_type(message)


@st.composite
def http_status_errors(draw: st.DrawFn) -> httpx.HTTPStatusError:
    """``httpx.HTTPStatusError`` across the full HTTP status-code range."""
    status = draw(st.integers(min_value=100, max_value=599))
    request = httpx.Request("GET", "https://api.example.test/resource")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status}", request=request, response=response
    )


# Timeout exceptions (all subclasses of httpx.TimeoutException -> NETWORK).
_TIMEOUT_EXC_TYPES = [
    httpx.TimeoutException,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
]

# Non-timeout transport failures (connection/protocol level -> NETWORK).
_TRANSPORT_EXC_TYPES = [
    httpx.TransportError,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.NetworkError,
    httpx.ProtocolError,
    httpx.LocalProtocolError,
    httpx.RemoteProtocolError,
]


@st.composite
def transport_exceptions(draw: st.DrawFn) -> httpx.TransportError:
    """``httpx`` timeout and transport-level failures (all map to NETWORK)."""
    exc_type = draw(st.sampled_from(_TIMEOUT_EXC_TYPES + _TRANSPORT_EXC_TYPES))
    message = draw(st.text(max_size=200))
    return exc_type(message)


# Already-classified taxonomy errors that must pass through unchanged.
_GEOERROR_SUBCLASSES = [
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    NotFoundError,
    ValidationError,
    UpstreamError,
    NetworkError,
]


@st.composite
def geo_errors(draw: st.DrawFn) -> GeoError:
    """Pre-classified ``GeoError`` instances (subclasses and the base class)."""
    message = draw(st.text(max_size=200))
    original: Optional[str] = draw(st.none() | st.text(max_size=200))
    use_base = draw(st.booleans())
    if use_base:
        category = draw(st.sampled_from(list(ErrorCategory)))
        return GeoError(category, message, source="prior", original=original)
    cls = draw(st.sampled_from(_GEOERROR_SUBCLASSES))
    return cls(message, source="prior", original=original)


# The full heterogeneous input space the mapping must total over.
any_exception = st.one_of(
    generic_exceptions(),
    http_status_errors(),
    transport_exceptions(),
    geo_errors(),
)


@pytest.fixture(scope="module")
def server() -> BaseGeoServer:
    """A bare BaseGeoServer; ``map_error`` needs no subclass behavior."""
    return BaseGeoServer()


@pytest.mark.property
@settings(deadline=None)  # max_examples (>=100) inherited from the loaded profile
@given(exc=any_exception)
def test_map_error_yields_exactly_one_taxonomy_category(
    exc: Exception, server: BaseGeoServer
) -> None:
    """Feature: geospatial-power-pack, Property 8: Every error maps to exactly one taxonomy category.

    Any transport/server/external error maps — totally, without raising — to a
    ``GeoError`` carrying exactly one of the seven taxonomy categories, matching
    the category the design specifies.

    **Validates: Requirements 5.5, 5.10, 11.2**
    """
    result = server.map_error(exc, source=_SOURCE)

    # The mapping is total: every input yields a taxonomy error.
    assert isinstance(result, GeoError)

    # Exactly one category: the result carries a single ErrorCategory member,
    # and it is one of exactly the seven shared categories (Req 5.5).
    assert isinstance(result.category, ErrorCategory)
    assert len([c for c in _ALL_CATEGORIES if c is result.category]) == 1

    # And it is the category the design mandates for this kind of error.
    assert result.category is _expected_category(exc)


@pytest.mark.property
@settings(deadline=None)
@given(exc=generic_exceptions())
def test_unmapped_external_errors_map_to_upstream_retaining_detail(
    exc: Exception, server: BaseGeoServer
) -> None:
    """Feature: geospatial-power-pack, Property 8: Every error maps to exactly one taxonomy category.

    An external/library error with no corresponding taxonomy category maps to
    ``upstream`` and retains the external server's original error detail.

    **Validates: Requirements 11.2, 11.5**
    """
    result = server.map_error(exc, source=_SOURCE)

    # Unmapped external errors land on exactly the UPSTREAM category (Req 11.5).
    assert result.category is ErrorCategory.UPSTREAM
    assert isinstance(result, UpstreamError)

    # The original upstream detail is retained, secret-free (Req 11.5).
    assert result.original is not None
    assert result.original == _safe_detail(exc)

    # The failing source is recorded on the returned error.
    assert result.source == _SOURCE


@pytest.mark.property
@settings(deadline=None)
@given(exc=http_status_errors())
def test_5xx_and_unmapped_status_codes_retain_original_detail(
    exc: httpx.HTTPStatusError, server: BaseGeoServer
) -> None:
    """Feature: geospatial-power-pack, Property 8: Every error maps to exactly one taxonomy category.

    HTTP status errors that do not map to a specific category (5xx and any
    other unmapped status) fall through to ``upstream`` while still retaining
    the upstream detail (Req 11.2 / 11.5).

    **Validates: Requirements 11.2, 11.5**
    """
    status = exc.response.status_code
    result = server.map_error(exc, source=_SOURCE)

    if status not in _STATUS_CATEGORY:
        assert result.category is ErrorCategory.UPSTREAM
    assert result.category is _expected_category(exc)
    # Every mapped HTTP status error retains the upstream detail.
    assert result.original is not None
