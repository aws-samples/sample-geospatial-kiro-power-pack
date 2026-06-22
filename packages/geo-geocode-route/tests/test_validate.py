"""Unit tests for ``geo-geocode-route`` input validation (Requirement 7.12).

These cover the rejection paths that fire *before* any source is queried: an
unparseable address, a malformed origin/destination coordinate, and an
unsupported travel profile. Each must raise an ``Error_Taxonomy``
``ValidationError`` naming the offending parameter.
"""

from __future__ import annotations

import pytest

from geo_common.errors import ErrorCategory, ValidationError

from geo_geocode_route.models import Coordinate
from geo_geocode_route.validate import (
    MAX_ADDRESS_LENGTH,
    normalize_profile,
    validate_address,
    validate_coordinate,
)


# --- validate_address -------------------------------------------------------


def test_valid_address_is_normalized():
    assert validate_address("  1600  Pennsylvania   Ave  ") == "1600 Pennsylvania Ave"


@pytest.mark.parametrize(
    "address",
    [
        "",            # empty
        "   ",         # whitespace only
        "!!!",         # punctuation only — unparseable
        "--- ... ---",  # symbols only — unparseable
        123,            # not a string
        None,
    ],
)
def test_unparseable_address_is_rejected(address):
    with pytest.raises(ValidationError) as exc:
        validate_address(address)
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail == {"parameter": "address"}


def test_overlong_address_is_rejected():
    with pytest.raises(ValidationError) as exc:
        validate_address("a" * (MAX_ADDRESS_LENGTH + 1))
    assert exc.value.detail == {"parameter": "address"}


# --- validate_coordinate ----------------------------------------------------


def test_valid_coordinate_shapes():
    expected = Coordinate(lon=13.4, lat=52.5)
    assert validate_coordinate((13.4, 52.5), parameter="origin") == expected
    assert validate_coordinate([13.4, 52.5], parameter="origin") == expected
    assert validate_coordinate({"lon": 13.4, "lat": 52.5}, parameter="origin") == expected
    assert (
        validate_coordinate({"longitude": 13.4, "latitude": 52.5}, parameter="origin")
        == expected
    )
    assert validate_coordinate(expected, parameter="origin") == expected


@pytest.mark.parametrize(
    "value",
    [
        (200.0, 0.0),     # longitude out of range
        (0.0, 91.0),      # latitude out of range
        (0.0,),           # wrong arity
        (0.0, 0.0, 0.0),  # wrong arity
        ("a", 0.0),       # non-numeric
        (float("nan"), 0.0),  # non-finite
        "13.4,52.5",      # string is not a coordinate
        {"lon": 13.4},    # missing lat
    ],
)
def test_malformed_coordinate_is_rejected(value):
    with pytest.raises(ValidationError) as exc:
        validate_coordinate(value, parameter="destination")
    assert exc.value.category is ErrorCategory.VALIDATION
    assert exc.value.detail["parameter"] == "destination"


# --- normalize_profile ------------------------------------------------------


@pytest.mark.parametrize(
    "alias, canonical",
    [
        ("car", "car"),
        ("driving", "car"),
        ("BICYCLE", "bike"),
        ("walking", "foot"),
        ("pedestrian", "foot"),
    ],
)
def test_profile_aliases_normalize(alias, canonical):
    assert normalize_profile(alias) == canonical


@pytest.mark.parametrize("profile", ["boat", "", 7, None])
def test_unsupported_profile_is_rejected(profile):
    with pytest.raises(ValidationError) as exc:
        normalize_profile(profile)
    assert exc.value.detail == {"parameter": "profile"}
