"""Property test for malformed-input validation across the MVP servers.

**Feature: geospatial-power-pack, Property 4: Malformed input always yields a
validation error**

*For any* MVP MCP server and any malformed input, the server returns an
``Error_Taxonomy`` validation error identifying the invalid input, raises no
unhandled exception, and produces no partial or persisted output.

Validates: Requirements 2.7, 7.12, 9.7, 9.9, 15.5

This module parametrizes Property 4 over every validation surface the MVP
servers expose, with one Hypothesis property per surface (each inheriting the
``>= 100`` generated cases of the ``geospatial-power-pack`` profile loaded by
the root ``conftest.py``):

* **Resource Catalog query length** (``kiro-geospatial`` - Requirement 2.7):
  a keyword outside the 1-200 character range is rejected by
  ``ResourceCatalog.search`` before any matching is performed.
* **geo-vector parameters** (Requirement 7.12): a malformed bounding box, and
  a (well-formed) bounding box whose area exceeds the configured maximum, are
  rejected by ``vector_features`` / ``validate_bbox`` before any source is
  queried.
* **geo-stac parameters** (Requirement 7.12): a malformed bounding box and a
  ``start``-after-``end`` datetime range are rejected by ``stac_search`` before
  any network call.
* **geo-foundation-models tiles and embeddings** (Requirements 9.7, 9.9): an
  empty / oversized / unsupported-format tile is rejected by ``embed_tile``
  producing no embedding, and a dimensionality-mismatched embedding pair is
  rejected by ``detect_change`` producing no change measure.

Each property asserts the three clauses of Property 4 / Requirement 15.5:

1. the raised error is an ``Error_Taxonomy`` :class:`ValidationError` whose
   category is :data:`ErrorCategory.VALIDATION` (a validation error is
   *returned*);
2. **no unhandled exception** is raised - the property uses
   ``pytest.raises(ValidationError)``, so any *other* exception type propagates
   and fails the test rather than being silently tolerated;
3. **no partial or persisted output** is produced - the return value is never
   assigned, and for the I/O-bearing surfaces a sentinel HTTP client / source
   asserts that no outbound request was ever attempted (so nothing could be
   read, written, or partially emitted).
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geo_common.errors import ErrorCategory, ValidationError
from geo_common.models import CatalogEntry, OpennessTier

from kiro_geospatial.catalog import (
    KEYWORD_MAX_LENGTH,
    CatalogQuery,
    ResourceCatalog,
)

from geo_vector.bbox import DEFAULT_MAX_AREA_KM2, bbox_area_km2, validate_bbox
from geo_vector.features import VectorSource, vector_features

from geo_stac.search import stac_search

from geo_foundation_models.change import detect_change
from geo_foundation_models.embedding import embed_tile
from geo_foundation_models.models import MAX_TILE_DIMENSION, RasterTile


# --------------------------------------------------------------------------- #
# Sentinels: prove no partial/persisted output by forbidding any I/O
# --------------------------------------------------------------------------- #
class _NetworkForbiddenSource(VectorSource):
    """A ``geo-vector`` source whose ``fetch`` must never be invoked.

    Validation runs *before* any source is queried, so on malformed input
    ``fetch`` is never awaited; if it ever were, the recorded flag (and the
    raised ``AssertionError``) would fail the property, proving no partial
    collection can be produced.
    """

    name = "network-forbidden"

    def __init__(self) -> None:
        self.fetch_called = False

    async def fetch(self, http, bbox, layers):  # noqa: ANN001 - matches base signature
        self.fetch_called = True
        raise AssertionError(
            "source.fetch must not be called when input is malformed"
        )


class _NetworkForbiddenHttp:
    """A stand-in HTTP client whose request methods must never be invoked.

    Used for both ``geo-vector`` and ``geo-stac``: validation rejects malformed
    input before any outbound request, so ``get`` / ``post`` are never called.
    """

    def __init__(self) -> None:
        self.called = False

    async def get(self, *args, **kwargs):
        self.called = True
        raise AssertionError("HttpClient.get must not be called on malformed input")

    async def post(self, *args, **kwargs):
        self.called = True
        raise AssertionError("HttpClient.post must not be called on malformed input")

    async def aclose(self):  # pragma: no cover - only called for owned clients
        return None


def _assert_validation(exc: ValidationError) -> None:
    """Assert ``exc`` is an Error_Taxonomy validation error (Property 4 / 15.5)."""
    assert isinstance(exc, ValidationError)
    assert exc.category is ErrorCategory.VALIDATION
    # The error identifies the invalid input via a human-readable message.
    assert isinstance(exc.message, str) and exc.message


# --------------------------------------------------------------------------- #
# Shared generators
# --------------------------------------------------------------------------- #
_finite = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)


@st.composite
def malformed_bbox(draw: st.DrawFn) -> List[float]:
    """Draw a bounding box that is guaranteed malformed.

    Covers every rejection path shared by ``geo-vector`` and ``geo-stac``
    bbox validation: wrong arity, a non-finite ordinate, an out-of-range
    longitude or latitude, and an inverted (min > max) extent.
    """
    kind = draw(
        st.sampled_from(
            ["arity", "nonfinite", "lon_oor", "lat_oor", "inverted"]
        )
    )

    if kind == "arity":
        # Any length other than exactly four ordinates.
        n = draw(st.integers(min_value=0, max_value=8).filter(lambda x: x != 4))
        return draw(st.lists(_finite, min_size=n, max_size=n))

    if kind == "nonfinite":
        vals: List[float] = [0.0, 0.0, 1.0, 1.0]
        idx = draw(st.integers(min_value=0, max_value=3))
        vals[idx] = draw(st.sampled_from([math.nan, math.inf, -math.inf]))
        return vals

    if kind == "lon_oor":
        # Valid latitudes (min <= max) but an out-of-range longitude as max_lon.
        bad_lon = draw(
            st.one_of(
                st.floats(min_value=180.0001, max_value=1e4, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-1e4, max_value=-180.0001, allow_nan=False, allow_infinity=False),
            )
        )
        return [0.0, 0.0, bad_lon, 1.0]

    if kind == "lat_oor":
        # Valid longitudes (min <= max) but an out-of-range latitude as max_lat.
        bad_lat = draw(
            st.one_of(
                st.floats(min_value=90.0001, max_value=1e4, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-1e4, max_value=-90.0001, allow_nan=False, allow_infinity=False),
            )
        )
        return [0.0, 0.0, 1.0, bad_lat]

    # inverted: every ordinate in range, but min > max on one axis.
    axis = draw(st.sampled_from(["lon", "lat"]))
    if axis == "lon":
        hi = draw(st.floats(min_value=-179.0, max_value=178.0, allow_nan=False, allow_infinity=False))
        lo = draw(st.floats(min_value=hi + 0.001, max_value=180.0, allow_nan=False, allow_infinity=False))
        return [lo, 0.0, hi, 1.0]  # min_lon (lo) > max_lon (hi)
    hi = draw(st.floats(min_value=-89.0, max_value=88.0, allow_nan=False, allow_infinity=False))
    lo = draw(st.floats(min_value=hi + 0.001, max_value=90.0, allow_nan=False, allow_infinity=False))
    return [0.0, lo, 1.0, hi]  # min_lat (lo) > max_lat (hi)


# --------------------------------------------------------------------------- #
# Resource Catalog: query length (Requirement 2.7)
# --------------------------------------------------------------------------- #
# A keyword outside the valid 1-200 character range: empty, or longer than 200.
_out_of_range_keyword = st.one_of(
    st.just(""),
    st.text(min_size=KEYWORD_MAX_LENGTH + 1, max_size=KEYWORD_MAX_LENGTH + 200),
)


def _seeded_catalog() -> ResourceCatalog:
    """A catalog holding one entry, so a search *could* otherwise match."""
    return ResourceCatalog(
        [
            CatalogEntry(
                name="STAC search",
                pillar="A",
                capability_description="search a STAC catalog",
                openness_tier=OpennessTier.OPEN,
                provider_server="geo-stac",
                installed=True,
            )
        ]
    )


@pytest.mark.property
@settings(deadline=None)
@given(keyword=_out_of_range_keyword)
def test_catalog_query_length_yields_validation_error(keyword: str) -> None:
    """Feature: geospatial-power-pack, Property 4: Malformed input always yields a validation error.

    Validates: Requirements 2.7, 15.5

    An out-of-range catalog keyword is rejected with an Error_Taxonomy
    validation error and no result set is produced.
    """
    catalog = _seeded_catalog()
    # Bypass CatalogQuery's construction-time length guard to drive the search
    # surface's own Error_Taxonomy validation (Requirement 2.7).
    query = CatalogQuery.model_construct(
        keyword=keyword, openness_tier=None, pillar=None
    )

    result = None
    with pytest.raises(ValidationError) as exc_info:
        result = catalog.search(query)

    _assert_validation(exc_info.value)
    # No partial output: nothing was returned from search.
    assert result is None


# --------------------------------------------------------------------------- #
# geo-vector: malformed bbox and over-maximum area (Requirement 7.12)
# --------------------------------------------------------------------------- #
@pytest.mark.property
@settings(deadline=None)
@given(bbox=malformed_bbox())
def test_vector_features_malformed_bbox_yields_validation_error(
    bbox: List[float],
) -> None:
    """Feature: geospatial-power-pack, Property 4: Malformed input always yields a validation error.

    Validates: Requirements 7.12, 15.5

    A malformed bbox is rejected by ``vector_features`` before any source is
    queried, so no features are produced.
    """
    source = _NetworkForbiddenSource()
    http = _NetworkForbiddenHttp()

    result = None
    with pytest.raises(ValidationError) as exc_info:
        result = asyncio.run(
            vector_features(bbox=bbox, http=http, sources=[source])
        )

    _assert_validation(exc_info.value)
    assert result is None
    # No partial/persisted output: no source query and no outbound request.
    assert source.fetch_called is False
    assert http.called is False


@st.composite
def oversized_vector_bbox(draw: st.DrawFn) -> List[float]:
    """Draw a well-formed bbox whose spherical area exceeds the 2,500 km² max.

    Every ordinate is in range with min < max, so the *only* reason the
    request is rejected is the area gate (Requirements 7.3, 7.12). A span of at
    least ~3 degrees on each axis at these latitudes is far above the maximum.
    """
    min_lon = draw(st.floats(min_value=-170.0, max_value=150.0, allow_nan=False, allow_infinity=False))
    lon_span = draw(st.floats(min_value=3.0, max_value=20.0, allow_nan=False, allow_infinity=False))
    max_lon = min(min_lon + lon_span, 180.0)

    min_lat = draw(st.floats(min_value=-40.0, max_value=20.0, allow_nan=False, allow_infinity=False))
    lat_span = draw(st.floats(min_value=3.0, max_value=20.0, allow_nan=False, allow_infinity=False))
    max_lat = min(min_lat + lat_span, 80.0)

    return [min_lon, min_lat, max_lon, max_lat]


@pytest.mark.property
@settings(deadline=None)
@given(bbox=oversized_vector_bbox())
def test_vector_features_oversized_area_yields_validation_error(
    bbox: List[float],
) -> None:
    """Feature: geospatial-power-pack, Property 4: Malformed input always yields a validation error.

    Validates: Requirements 7.12, 15.5

    A bbox whose area exceeds the configured maximum is rejected before any
    source is queried, producing no features.
    """
    # Guard the generator's intent: the bbox is well-formed and over-area.
    assert bbox_area_km2(validate_bbox(bbox)) > DEFAULT_MAX_AREA_KM2

    source = _NetworkForbiddenSource()
    http = _NetworkForbiddenHttp()

    result = None
    with pytest.raises(ValidationError) as exc_info:
        result = asyncio.run(
            vector_features(bbox=bbox, http=http, sources=[source])
        )

    _assert_validation(exc_info.value)
    assert exc_info.value.detail is not None
    assert exc_info.value.detail.get("parameter") == "bbox"
    assert result is None
    assert source.fetch_called is False
    assert http.called is False


# --------------------------------------------------------------------------- #
# geo-stac: malformed bbox and start-after-end range (Requirement 7.12)
# --------------------------------------------------------------------------- #
_VALID_STAC_RANGE: Tuple[str, str] = (
    "2020-01-01T00:00:00Z",
    "2020-12-31T00:00:00Z",
)
_VALID_STAC_BBOX: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


@pytest.mark.property
@settings(deadline=None)
@given(bbox=malformed_bbox())
def test_stac_search_malformed_bbox_yields_validation_error(
    bbox: List[float],
) -> None:
    """Feature: geospatial-power-pack, Property 4: Malformed input always yields a validation error.

    Validates: Requirements 7.12, 15.5

    A malformed STAC bbox is rejected before any network call, producing no
    items.
    """
    http = _NetworkForbiddenHttp()

    result = None
    with pytest.raises(ValidationError) as exc_info:
        result = asyncio.run(
            stac_search(
                bbox=bbox,
                datetime_range=_VALID_STAC_RANGE,
                http=http,
            )
        )

    _assert_validation(exc_info.value)
    assert result is None
    assert http.called is False


@st.composite
def start_after_end_range(draw: st.DrawFn) -> Tuple[str, str]:
    """Draw a (start, end) ISO-8601 pair whose start is strictly later than end."""
    pair = draw(
        st.lists(
            st.datetimes(
                min_value=datetime(1990, 1, 1),
                max_value=datetime(2035, 1, 1),
            ),
            min_size=2,
            max_size=2,
            unique=True,
        )
    )
    earlier, later = sorted(pair)
    # start (later) is strictly after end (earlier).
    return later.isoformat(), earlier.isoformat()


@pytest.mark.property
@settings(deadline=None)
@given(datetime_range=start_after_end_range())
def test_stac_search_start_after_end_yields_validation_error(
    datetime_range: Tuple[str, str],
) -> None:
    """Feature: geospatial-power-pack, Property 4: Malformed input always yields a validation error.

    Validates: Requirements 7.12, 15.5

    A datetime range whose start is later than its end is rejected before any
    network call, producing no items.
    """
    http = _NetworkForbiddenHttp()

    result = None
    with pytest.raises(ValidationError) as exc_info:
        result = asyncio.run(
            stac_search(
                bbox=_VALID_STAC_BBOX,
                datetime_range=datetime_range,
                http=http,
            )
        )

    _assert_validation(exc_info.value)
    assert exc_info.value.detail is not None
    assert exc_info.value.detail.get("parameter") == "datetime_range"
    assert result is None
    assert http.called is False


# --------------------------------------------------------------------------- #
# geo-foundation-models: tile validation (Requirement 9.7)
# --------------------------------------------------------------------------- #
@st.composite
def malformed_tile(draw: st.DrawFn) -> RasterTile:
    """Draw a tile guaranteed to fail ``embed_tile`` validation (Req 9.7).

    Covers empty (zero width/height/bands), oversized (a dimension over the
    1024 limit), unsupported raster format, empty inlined pixel data, and
    inlined data whose length disagrees with the tile geometry.
    """
    kind = draw(
        st.sampled_from(
            [
                "empty_width",
                "empty_height",
                "empty_bands",
                "oversized",
                "unsupported_format",
                "empty_data",
                "bad_data_length",
            ]
        )
    )
    small = st.integers(min_value=1, max_value=64)

    if kind == "empty_width":
        return RasterTile(width=0, height=draw(small), bands=draw(st.integers(1, 4)), format="GTiff")
    if kind == "empty_height":
        return RasterTile(width=draw(small), height=0, bands=draw(st.integers(1, 4)), format="GTiff")
    if kind == "empty_bands":
        return RasterTile(width=draw(small), height=draw(small), bands=0, format="GTiff")
    if kind == "oversized":
        big = draw(st.integers(min_value=MAX_TILE_DIMENSION + 1, max_value=MAX_TILE_DIMENSION + 4096))
        if draw(st.booleans()):
            return RasterTile(width=big, height=draw(small), bands=1, format="GTiff")
        return RasterTile(width=draw(small), height=big, bands=1, format="GTiff")
    if kind == "unsupported_format":
        bad_fmt = draw(st.sampled_from(["bmp", "exr", "tiff8", "webp", "   "]))
        return RasterTile(width=draw(small), height=draw(small), bands=1, format=bad_fmt)
    if kind == "empty_data":
        return RasterTile(
            width=draw(st.integers(1, 16)),
            height=draw(st.integers(1, 16)),
            bands=1,
            format="GTiff",
            data=[],
        )
    # bad_data_length: a non-empty buffer whose length disagrees with geometry.
    w = draw(st.integers(1, 8))
    h = draw(st.integers(1, 8))
    b = draw(st.integers(1, 3))
    expected = w * h * b
    wrong = draw(
        st.integers(min_value=1, max_value=expected + 8).filter(lambda n: n != expected)
    )
    return RasterTile(width=w, height=h, bands=b, format="GTiff", data=[0.0] * wrong)


@pytest.mark.property
@settings(deadline=None)
@given(tile=malformed_tile())
def test_embed_tile_malformed_tile_yields_validation_error(tile: RasterTile) -> None:
    """Feature: geospatial-power-pack, Property 4: Malformed input always yields a validation error.

    Validates: Requirements 9.7, 15.5

    An empty, oversized, unsupported-format, or malformed-data tile is rejected
    with a validation error and produces no embedding.
    """
    # Selection ("Clay") is valid, so the only rejection cause is the tile.
    result = None
    with pytest.raises(ValidationError) as exc_info:
        result = embed_tile(tile, "Clay")

    _assert_validation(exc_info.value)
    # No partial/persisted output: no EmbeddingResult was produced.
    assert result is None


# --------------------------------------------------------------------------- #
# geo-foundation-models: change-detection dimensionality mismatch (Req 9.9)
# --------------------------------------------------------------------------- #
_embed_component = st.floats(
    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
)


@st.composite
def mismatched_embeddings(draw: st.DrawFn) -> Tuple[List[float], List[float]]:
    """Draw two embeddings of differing dimensionality (Requirement 9.9)."""
    dim_a = draw(st.integers(min_value=0, max_value=32))
    dim_b = draw(st.integers(min_value=0, max_value=32).filter(lambda d: d != dim_a))
    a = draw(st.lists(_embed_component, min_size=dim_a, max_size=dim_a))
    b = draw(st.lists(_embed_component, min_size=dim_b, max_size=dim_b))
    return a, b


@pytest.mark.property
@settings(deadline=None)
@given(pair=mismatched_embeddings())
def test_detect_change_dimensionality_mismatch_yields_validation_error(
    pair: Tuple[List[float], List[float]],
) -> None:
    """Feature: geospatial-power-pack, Property 4: Malformed input always yields a validation error.

    Validates: Requirements 9.9, 15.5

    Two embeddings of differing dimensionality are rejected with a validation
    error and produce no change measure.
    """
    embedding_a, embedding_b = pair

    measure = None
    with pytest.raises(ValidationError) as exc_info:
        measure = detect_change(embedding_a, embedding_b)

    _assert_validation(exc_info.value)
    assert exc_info.value.detail is not None
    assert set(exc_info.value.detail) >= {"dimension_a", "dimension_b"}
    # No partial output: no scalar measure was produced.
    assert measure is None
