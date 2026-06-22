"""Robustness checks for ``geo-biodiversity`` across many generated inputs.

These Hypothesis tests exercise the two invariants the connector must hold for
*any* input, complementing the example-based tests in
``test_species_occurrences.py``:

* the per-response result is **never larger than the effective cap** - whatever
  the source returns and whatever (valid) limit is requested, the merged result
  is bounded by ``min(limit, DEFAULT_MAX_RECORDS)`` (Requirement 7.7); and
* bbox validation is **total** over well-formed numeric 4-tuples: it either
  returns a normalized float tuple for an in-range, non-inverted extent or
  raises a taxonomy ``validation`` error - it never raises anything else
  (Requirement 7.12).
"""

from __future__ import annotations

from typing import Any, Dict

import httpx
from hypothesis import given
from hypothesis import strategies as st

from geo_common.errors import ErrorCategory, ValidationError
from geo_common.http import HttpClient
from geo_common.retry import RetryPolicy

from geo_biodiversity.occurrences import GBIFSource, species_occurrences
from geo_biodiversity.validation import DEFAULT_MAX_RECORDS, validate_bbox

SMALL_BBOX = (13.40, 52.50, 13.41, 52.51)


async def _no_sleep(_seconds: float) -> None:
    return None


def _payload(n: int) -> Dict[str, Any]:
    return {
        "count": n,
        "results": [
            {
                "key": i,
                "scientificName": "Taxon sp.",
                "decimalLongitude": 13.4,
                "decimalLatitude": 52.5,
            }
            for i in range(n)
        ],
    }


@given(
    available=st.integers(min_value=0, max_value=400),
    limit=st.integers(min_value=1, max_value=DEFAULT_MAX_RECORDS),
)
async def test_result_never_exceeds_effective_cap(available: int, limit: int) -> None:
    """Req 7.7: the returned count is bounded by min(limit, max) and by supply."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_payload(available))

    client = HttpClient(
        RetryPolicy(max_attempts=1),
        transport=httpx.MockTransport(handler),
        sleep=_no_sleep,
    )
    try:
        records = await species_occurrences(
            bbox=SMALL_BBOX, http=client, sources=[GBIFSource()], limit=limit
        )
    finally:
        await client.aclose()

    cap = min(limit, DEFAULT_MAX_RECORDS)
    # The result is bounded by the effective cap and never exceeds the number
    # of records the source actually supplied.
    assert len(records) <= min(available, cap)


@given(
    coords=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=4,
        max_size=4,
    )
)
def test_validate_bbox_is_total_over_numeric_tuples(coords) -> None:
    """Req 7.12: validate_bbox returns a float tuple or raises ValidationError."""
    try:
        result = validate_bbox(tuple(coords))
    except ValidationError as exc:
        assert exc.category is ErrorCategory.VALIDATION
        assert exc.detail and exc.detail.get("parameter") == "bbox"
        return
    # When it does not raise, the extent was in range and non-inverted.
    assert len(result) == 4
    min_lon, min_lat, max_lon, max_lat = result
    assert -180.0 <= min_lon <= max_lon <= 180.0
    assert -90.0 <= min_lat <= max_lat <= 90.0
