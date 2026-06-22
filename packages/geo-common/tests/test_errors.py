"""Unit tests for the geo-common Error_Taxonomy (Requirements 5.5, 5.10, 11.5)."""

from __future__ import annotations

import pytest

from geo_common import (
    AuthenticationError,
    AuthorizationError,
    ErrorCategory,
    ErrorObject,
    GeoError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    UpstreamError,
    ValidationError,
)

# Each concrete subclass paired with the single category it must carry.
SUBCLASS_CATEGORY = [
    (AuthenticationError, ErrorCategory.AUTHENTICATION),
    (AuthorizationError, ErrorCategory.AUTHORIZATION),
    (RateLimitError, ErrorCategory.RATE_LIMIT),
    (NotFoundError, ErrorCategory.NOT_FOUND),
    (ValidationError, ErrorCategory.VALIDATION),
    (UpstreamError, ErrorCategory.UPSTREAM),
    (NetworkError, ErrorCategory.NETWORK),
]


def test_taxonomy_has_exactly_seven_categories():
    """Req 5.5: the taxonomy defines exactly the seven shared categories."""
    assert {c.value for c in ErrorCategory} == {
        "authentication",
        "authorization",
        "rate-limit",
        "not-found",
        "validation",
        "upstream",
        "network",
    }


@pytest.mark.parametrize("cls,category", SUBCLASS_CATEGORY)
def test_subclass_carries_exactly_one_fixed_category(cls, category):
    """Each concrete subclass carries exactly its one taxonomy category."""
    err = cls("something went wrong")
    assert isinstance(err, GeoError)
    assert err.category is category
    assert err.message == "something went wrong"


def test_base_geoerror_requires_a_valid_category():
    """The base GeoError must be given a real ErrorCategory."""
    err = GeoError(ErrorCategory.UPSTREAM, "boom")
    assert err.category is ErrorCategory.UPSTREAM

    with pytest.raises(TypeError):
        GeoError("not-a-category", "boom")  # type: ignore[arg-type]


def test_str_and_repr_do_not_leak_detail_or_original():
    """str/repr expose only message/category/source/retry_after - never detail/original."""
    secretish_detail = {"api_key": "SUPER_SECRET_VALUE"}
    err = AuthenticationError(
        "missing credential BIGQUERY_CREDENTIALS",
        source="geo-warehouse",
        detail=secretish_detail,
        original="upstream said: token=SECRET_TOKEN",
    )

    text = str(err)
    representation = repr(err)

    # The author-controlled message is shown; the message itself names the key,
    # not its value, per the secret-free contract.
    assert text == "missing credential BIGQUERY_CREDENTIALS"
    # Neither the structured detail value nor the original upstream string leak.
    assert "SUPER_SECRET_VALUE" not in text
    assert "SUPER_SECRET_VALUE" not in representation
    assert "SECRET_TOKEN" not in text
    assert "SECRET_TOKEN" not in representation
    # repr still carries useful, non-secret diagnostics.
    assert "authentication" in representation
    assert "geo-warehouse" in representation


def test_retry_after_is_carried_on_rate_limit_error():
    err = RateLimitError("slow down", source="overpass", retry_after=42.0)
    assert err.category is ErrorCategory.RATE_LIMIT
    assert err.retry_after == 42.0


def test_to_error_object_round_trip_retains_original_detail():
    """Req 11.5: serialization retains the upstream original detail."""
    err = UpstreamError(
        "wrapped server failed",
        source="gis-mcp",
        detail={"status": 503},
        original="raw upstream traceback text",
        retry_after=None,
    )
    obj = err.to_error_object()

    assert isinstance(obj, ErrorObject)
    assert obj.category is ErrorCategory.UPSTREAM
    assert obj.message == "wrapped server failed"
    assert obj.source == "gis-mcp"
    assert obj.detail == {"status": 503}
    assert obj.original == "raw upstream traceback text"  # retained (Req 11.5)
    assert obj.retry_after_s is None

    # And back to an exception, preserving the category and original detail.
    rebuilt = obj.to_exception()
    assert isinstance(rebuilt, GeoError)
    assert rebuilt.category is ErrorCategory.UPSTREAM
    assert rebuilt.original == "raw upstream traceback text"


def test_error_object_serializes_category_to_wire_string():
    obj = ErrorObject(category=ErrorCategory.RATE_LIMIT, message="429", retry_after_s=5.0)
    dumped = obj.model_dump()
    assert dumped["category"] == "rate-limit"
    assert dumped["retry_after_s"] == 5.0

    # JSON round-trip preserves the category.
    reparsed = ErrorObject.model_validate_json(obj.model_dump_json())
    assert reparsed.category is ErrorCategory.RATE_LIMIT


def test_error_object_defaults_are_none():
    obj = ErrorObject(category=ErrorCategory.VALIDATION, message="bad input")
    assert obj.source is None
    assert obj.detail is None
    assert obj.original is None
    assert obj.retry_after_s is None
