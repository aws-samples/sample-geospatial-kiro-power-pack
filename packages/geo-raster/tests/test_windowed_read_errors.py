"""Error-taxonomy sweep for ``geo-raster``'s windowed read (Reqs 7.8, 7.11).

The windowed COG read (``read_window`` / ``band_math``, merged into
``geo-raster`` from the former ``geo-imagery`` server) must surface transport
failures cleanly:

* an unreachable / non-responding source surfaces as an ``Error_Taxonomy``
  availability error (``NETWORK``) identifying the source, with no partial data
  (Requirement 7.11);
* an authentication failure surfaces as an ``Error_Taxonomy`` authentication
  error, with no partial data (Requirement 7.8).
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from geo_common.errors import (
    AuthenticationError,
    ErrorCategory,
    NetworkError,
)
from geo_common.http import HttpClient
from geo_common.retry import RetryPolicy

from geo_raster import GeoRasterServer, PixelWindow

ASSET_HREF = "s3://amzn-s3-demo-example/scene/B04.tif"
WINDOW = PixelWindow(col_off=0, row_off=0, width=2, height=2)


async def _no_sleep(_seconds: float) -> None:
    return None


def _server(handler: Any) -> GeoRasterServer:
    client = HttpClient(
        RetryPolicy(max_attempts=1),
        transport=httpx.MockTransport(handler),
        sleep=_no_sleep,
    )
    return GeoRasterServer(http=client)


async def test_unreachable_source_yields_availability_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    server = _server(handler)
    try:
        with pytest.raises(NetworkError) as exc_info:
            await server.read_window(asset_href=ASSET_HREF, window=WINDOW)
    finally:
        await server.aclose()
    err = exc_info.value
    assert err.category is ErrorCategory.NETWORK
    # The unavailable source is identified (the asset's S3 host).
    assert err.source


async def test_timeout_yields_availability_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    server = _server(handler)
    try:
        with pytest.raises(NetworkError) as exc_info:
            await server.read_window(asset_href=ASSET_HREF, window=WINDOW)
    finally:
        await server.aclose()
    assert exc_info.value.category is ErrorCategory.NETWORK


async def test_authentication_failure_yields_authentication_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "access denied"})

    server = _server(handler)
    try:
        with pytest.raises(AuthenticationError) as exc_info:
            await server.read_window(asset_href=ASSET_HREF, window=WINDOW)
    finally:
        await server.aclose()
    assert exc_info.value.category is ErrorCategory.AUTHENTICATION
