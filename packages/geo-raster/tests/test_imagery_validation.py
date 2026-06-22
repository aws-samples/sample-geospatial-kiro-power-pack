"""Parameter-validation tests for ``geo-raster``'s windowed read (Requirement 7.12).

Every malformed parameter must raise the shared ``Error_Taxonomy``
:class:`~geo_common.errors.ValidationError` (category ``validation``) *before*
any byte-range read, and must produce no partial output.
"""

from __future__ import annotations

import pytest

from _cogbuild import RecordingByteRangeReader, build_cog

from geo_common.errors import ErrorCategory, ValidationError

from geo_raster import GeoRasterServer, PixelWindow, band_math, read_window


def _small_cog() -> bytes:
    band = [float(i) for i in range(16 * 16)]
    return build_cog(
        width=16, height=16, tile_width=8, tile_height=8, bands=[band], dtype="float32"
    )


async def test_empty_asset_href_is_validation_error() -> None:
    with pytest.raises(ValidationError) as exc:
        await read_window(
            asset_href="   ",
            window=PixelWindow(col_off=0, row_off=0, width=1, height=1),
            reader=RecordingByteRangeReader(_small_cog()),
        )
    assert exc.value.category is ErrorCategory.VALIDATION


async def test_unsupported_scheme_is_validation_error() -> None:
    with pytest.raises(ValidationError):
        await read_window(
            asset_href="ftp://host/file.tif",
            window=PixelWindow(col_off=0, row_off=0, width=1, height=1),
            reader=RecordingByteRangeReader(_small_cog()),
        )


@pytest.mark.parametrize(
    "window",
    [
        {"col_off": -1, "row_off": 0, "width": 4, "height": 4},
        {"col_off": 0, "row_off": 0, "width": 0, "height": 4},
        {"col_off": 0, "row_off": 0, "width": 4, "height": -2},
    ],
)
async def test_malformed_pixel_window_is_validation_error(window) -> None:
    with pytest.raises(ValidationError):
        await read_window(
            asset_href="s3://amzn-s3-demo-bucket/k.tif",
            window=window,
            reader=RecordingByteRangeReader(_small_cog()),
        )


async def test_window_outside_extent_is_validation_error() -> None:
    """A window origin beyond the asset extent is rejected (Req 7.12)."""
    reader = RecordingByteRangeReader(_small_cog())
    with pytest.raises(ValidationError):
        await read_window(
            asset_href="s3://amzn-s3-demo-bucket/k.tif",
            window=PixelWindow(col_off=100, row_off=0, width=4, height=4),
            reader=reader,
        )


async def test_band_index_out_of_range_is_validation_error() -> None:
    reader = RecordingByteRangeReader(_small_cog())
    with pytest.raises(ValidationError):
        await read_window(
            asset_href="s3://amzn-s3-demo-bucket/k.tif",
            window=PixelWindow(col_off=0, row_off=0, width=2, height=2),
            bands=[5],  # asset has 1 band
            reader=reader,
        )


async def test_empty_band_list_is_validation_error() -> None:
    reader = RecordingByteRangeReader(_small_cog())
    with pytest.raises(ValidationError):
        await read_window(
            asset_href="s3://amzn-s3-demo-bucket/k.tif",
            window=PixelWindow(col_off=0, row_off=0, width=2, height=2),
            bands=[],
            reader=reader,
        )


async def test_geo_window_without_georeferencing_is_validation_error() -> None:
    """A GeoWindow on an asset with no geotransform is rejected (Req 7.12)."""
    reader = RecordingByteRangeReader(_small_cog())  # built without geotransform
    with pytest.raises(ValidationError):
        await read_window(
            asset_href="s3://amzn-s3-demo-bucket/k.tif",
            window={"bbox": (0.0, 0.0, 1.0, 1.0)},
            reader=reader,
        )


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "   ",
        "B1 +",          # syntax error
        "foo + B1",      # unknown identifier
        "B1 ** 2",       # unsupported operator (power)
        "__import__('os')",  # unsupported / unsafe
        "42",            # references no band
    ],
)
async def test_malformed_band_math_expression_is_validation_error(expression) -> None:
    reader = RecordingByteRangeReader(_small_cog())
    with pytest.raises(ValidationError):
        await band_math(
            asset_href="s3://amzn-s3-demo-bucket/k.tif",
            expression=expression,
            window=PixelWindow(col_off=0, row_off=0, width=2, height=2),
            reader=reader,
        )


async def test_malformed_s3_uri_is_validation_error() -> None:
    reader = RecordingByteRangeReader(_small_cog())
    with pytest.raises(ValidationError):
        await read_window(
            asset_href="s3://amzn-s3-demo-bucket-only",  # no key
            window=PixelWindow(col_off=0, row_off=0, width=2, height=2),
            reader=reader,
        )


async def test_server_maps_validation_through_tool() -> None:
    """The server surfaces validation errors from its tool entry points."""
    server = GeoRasterServer()
    try:
        with pytest.raises(ValidationError):
            await server.read_window(
                asset_href="",
                window={"col_off": 0, "row_off": 0, "width": 1, "height": 1},
            )
    finally:
        await server.aclose()
