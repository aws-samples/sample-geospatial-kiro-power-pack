"""Property test for spatial-index cell determinism (design Property 15).

Feature: geospatial-power-pack, Property 15: Spatial index cells are deterministic over valid ranges

This module validates :func:`geo_index.indexing.index_cell` against Property 15
of the design (Validates: Requirements 8.9):

*For any* valid coordinate (longitude in ``[-180, 180]``, latitude in
``[-90, 90]``) and any valid resolution for the selected scheme (H3 levels
0-15, S2 levels 0-30), ``index_cell`` returns a **well-formed** cell identifier
for that scheme, and the **same input always yields the same identifier**
(determinism / referential transparency).

The H3 / S2 libraries are used only to *verify* well-formedness of the returned
identifier (valid token + correct resolution/level); the determinism check is
purely about repeated calls returning an equal value.
"""

from __future__ import annotations

import h3
import pytest
import s2sphere
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_index.indexing import (
    H3,
    H3_MAX_RESOLUTION,
    H3_MIN_RESOLUTION,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    S2,
    S2_MAX_LEVEL,
    S2_MIN_LEVEL,
    index_cell,
)

# --------------------------------------------------------------------------- #
# Strategies: valid coordinates and per-scheme resolutions
# --------------------------------------------------------------------------- #
# Longitudes/latitudes spanning the full inclusive valid range (Req 8.10's
# accepted space), finite only - the valid input space for Property 15.
_lons = st.floats(
    min_value=LON_MIN, max_value=LON_MAX, allow_nan=False, allow_infinity=False
)
_lats = st.floats(
    min_value=LAT_MIN, max_value=LAT_MAX, allow_nan=False, allow_infinity=False
)


@st.composite
def _valid_inputs(draw: st.DrawFn) -> tuple[float, float, str, int]:
    """Draw a ``(lon, lat, scheme, resolution)`` tuple in the valid space.

    The resolution range is chosen to match the drawn scheme (H3 0-15,
    S2 0-30) so every generated case is a *valid* input for Property 15.
    """
    scheme = draw(st.sampled_from((H3, S2)))
    if scheme == H3:
        resolution = draw(st.integers(min_value=H3_MIN_RESOLUTION, max_value=H3_MAX_RESOLUTION))
    else:
        resolution = draw(st.integers(min_value=S2_MIN_LEVEL, max_value=S2_MAX_LEVEL))
    lon = draw(_lons)
    lat = draw(_lats)
    return lon, lat, scheme, resolution


def _assert_wellformed(cell: str, scheme: str, resolution: int) -> None:
    """Assert ``cell`` is a well-formed identifier for ``scheme``/resolution."""
    assert isinstance(cell, str) and cell  # non-empty string id
    if scheme == H3:
        assert h3.is_valid_cell(cell)
        assert h3.get_resolution(cell) == resolution
    else:
        cell_id = s2sphere.CellId.from_token(cell)
        assert cell_id.is_valid()
        assert cell_id.level() == resolution


# --------------------------------------------------------------------------- #
# Property 15: well-formed + deterministic over the valid range
# --------------------------------------------------------------------------- #
@pytest.mark.property
@settings(deadline=None)  # max_examples (>=100) is inherited from the loaded profile
@given(inputs=_valid_inputs())
def test_index_cell_is_wellformed_and_deterministic(
    inputs: tuple[float, float, str, int],
) -> None:
    """Feature: geospatial-power-pack, Property 15: Spatial index cells are deterministic over valid ranges.

    Validates: Requirements 8.9

    For any valid coordinate/resolution: (a) a well-formed cell id is returned
    for the selected scheme, and (b) repeated lookups of the identical input
    always yield the same id.
    """
    lon, lat, scheme, resolution = inputs
    kwargs = dict(lon=lon, lat=lat, scheme=scheme, resolution=resolution)

    # (a) A well-formed cell id is returned for any valid input. Req 8.9.
    first = index_cell(**kwargs)
    _assert_wellformed(first, scheme, resolution)

    # (b) Determinism: the same input always yields the same id. Req 8.9 /
    # Property 15. Repeat several times to guard against any hidden state.
    for _ in range(5):
        assert index_cell(**kwargs) == first


@pytest.mark.property
@settings(deadline=None)
@given(inputs=_valid_inputs())
def test_index_cell_is_case_insensitive_and_deterministic(
    inputs: tuple[float, float, str, int],
) -> None:
    """Feature: geospatial-power-pack, Property 15: Spatial index cells are deterministic over valid ranges.

    Validates: Requirements 8.9

    Scheme matching is case-insensitive, and the upper-cased scheme produces
    the identical (deterministic) id as the canonical lower-cased scheme.
    """
    lon, lat, scheme, resolution = inputs
    canonical = index_cell(lon=lon, lat=lat, scheme=scheme, resolution=resolution)
    upper = index_cell(lon=lon, lat=lat, scheme=scheme.upper(), resolution=resolution)
    assert upper == canonical
