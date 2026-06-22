"""Validation sweep for ``geo-index``'s ``index_cell`` (task 14.12).

Task 14.9 covers out-of-range resolutions, out-of-bounds / non-finite
coordinates, and unknown schemes. This module fills the remaining type-guard
validation branches the Pillar B I/O + validation sweep calls out, so every
rejection surfaces on the shared ``Error_Taxonomy`` and produces no output
(Requirement 8.10):

* a ``bool`` resolution is rejected even though ``bool`` is an ``int`` subclass
  (``True``/``False`` are not meaningful resolutions);
* a non-integer resolution (a float, a string) is a validation error;
* a non-numeric / non-string scheme is a validation error;
* a grossly out-of-range resolution never leaks a raw h3/s2 library exception -
  it is always the taxonomy ``ValidationError``.

These run against the pure function and the server tool; ``asyncio_mode =
"auto"`` runs the async server test.
"""

from __future__ import annotations

import pytest

from geo_common.errors import ErrorCategory, ValidationError
from geo_index.indexing import index_cell
from geo_index.server import GeoIndexServer


@pytest.mark.parametrize("scheme", ["h3", "s2"])
@pytest.mark.parametrize("bad_resolution", [True, False])
def test_bool_resolution_is_validation_error(scheme, bad_resolution) -> None:
    """Req 8.10: a bool resolution is rejected (bool is an int subclass)."""
    with pytest.raises(ValidationError) as exc_info:
        index_cell(lon=0.0, lat=0.0, scheme=scheme, resolution=bad_resolution)
    err = exc_info.value
    assert err.category is ErrorCategory.VALIDATION
    assert err.detail and err.detail.get("parameter") == "resolution"


@pytest.mark.parametrize("bad_resolution", [3.5, "9", None])
def test_non_integer_resolution_is_validation_error(bad_resolution) -> None:
    """Req 8.10: a non-integer resolution is a validation error (no output)."""
    with pytest.raises(ValidationError) as exc_info:
        index_cell(lon=0.0, lat=0.0, scheme="h3", resolution=bad_resolution)  # type: ignore[arg-type]
    assert exc_info.value.category is ErrorCategory.VALIDATION
    assert exc_info.value.detail.get("parameter") == "resolution"


@pytest.mark.parametrize("bad_scheme", [123, None, ["h3"]])
def test_non_string_scheme_is_validation_error(bad_scheme) -> None:
    """Req 8.10: a non-string scheme is rejected as a validation error."""
    with pytest.raises(ValidationError) as exc_info:
        index_cell(lon=0.0, lat=0.0, scheme=bad_scheme, resolution=5)  # type: ignore[arg-type]
    assert exc_info.value.category is ErrorCategory.VALIDATION
    assert exc_info.value.detail.get("parameter") == "scheme"


@pytest.mark.parametrize("scheme,resolution", [("h3", 1000), ("s2", 9999)])
def test_grossly_out_of_range_resolution_is_taxonomy_error_not_raw(
    scheme, resolution
) -> None:
    """A far-out-of-range resolution is the taxonomy error, never a raw lib error."""
    with pytest.raises(ValidationError):
        index_cell(lon=0.0, lat=0.0, scheme=scheme, resolution=resolution)


async def test_server_tool_rejects_bool_resolution() -> None:
    """The MCP tool surfaces the bool-resolution rejection as a ValidationError."""
    server = GeoIndexServer()
    with pytest.raises(ValidationError):
        await server.index_cell(lon=0.0, lat=0.0, scheme="h3", resolution=True)
