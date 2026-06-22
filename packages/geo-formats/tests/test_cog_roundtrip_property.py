"""Property test for COG round-trip pixel-exactness (design Property 2).

Feature: geospatial-power-pack, Property 2: COG round-trip is pixel-exact

Validates: Requirements 8.4, 12.2, 15.3

*For any* raster (including nodata-flagged pixels), converting it to a
Cloud-Optimized GeoTIFF with :func:`geo_formats.cog.to_cog` and reading it back
yields, for every band, pixel values **exactly** equal to the source pixel
values — nodata-flagged pixels included.

Input space
-----------
Rasters are generated across the dimensions that matter for a pixel round-trip:

* **dtype** — a representative spread of the integer and floating-point pixel
  types rasterio/GDAL write reliably (``uint8``/``uint16``/``int16``/``int32``/
  ``float32``/``float64``). COG conversion here uses lossless DEFLATE
  compression, so every one of these must round-trip bit-for-bit.
* **shape** — 1–3 bands over small height/width grids (including 1×1 and other
  edge sizes smaller than a COG tile, which legitimately produce no overviews).
* **pixel values** — drawn over each dtype's full range; floats are constrained
  to finite values (no NaN/inf) so that exact equality is well defined.
* **nodata** — either absent, or a concrete in-range sentinel that is then
  stamped into several pixels so the round-trip is forced to carry nodata
  through unchanged.

Topology/CRS are irrelevant to a per-pixel round-trip, so a fixed WGS84 CRS and
geotransform are used for every generated raster.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import rasterio
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from rasterio.transform import from_origin

from geo_formats.cog import to_cog

# dtypes whose values DEFLATE preserves bit-for-bit and that GDAL/rasterio write
# without special build flags. Mapped to the value bounds we generate within.
_INT_DTYPES = {
    "uint8": (0, 255),
    "uint16": (0, 65535),
    "int16": (-32768, 32767),
    "int32": (-(2**31), 2**31 - 1),
}
_FLOAT_DTYPES = ("float32", "float64")

# Fixed, valid CRS + geotransform; topology does not affect a pixel round-trip.
_CRS = "EPSG:4326"
_TRANSFORM = from_origin(10.0, 50.0, 0.01, 0.01)


def _pixel_strategy(dtype: str) -> st.SearchStrategy[np.ndarray]:
    """A Hypothesis array-element strategy spanning ``dtype``'s value range."""
    if dtype in _INT_DTYPES:
        lo, hi = _INT_DTYPES[dtype]
        return st.integers(min_value=lo, max_value=hi)
    # Finite floats only, so exact (bit-for-bit) equality is well defined.
    width = 32 if dtype == "float32" else 64
    return st.floats(
        allow_nan=False, allow_infinity=False, width=width
    )


@st.composite
def _rasters(draw):
    """Generate ``(data, nodata)`` for a small multi-band raster.

    ``data`` has shape ``(bands, height, width)`` in the chosen dtype; ``nodata``
    is either ``None`` or an in-range sentinel that has been stamped into a few
    pixels of every band.
    """
    dtype = draw(st.sampled_from(list(_INT_DTYPES) + list(_FLOAT_DTYPES)))
    bands = draw(st.integers(min_value=1, max_value=3))
    height = draw(st.integers(min_value=1, max_value=12))
    width = draw(st.integers(min_value=1, max_value=12))

    data = draw(
        arrays(
            dtype=np.dtype(dtype),
            shape=(bands, height, width),
            elements=_pixel_strategy(dtype),
        )
    )

    use_nodata = draw(st.booleans())
    nodata = None
    if use_nodata:
        if dtype in _INT_DTYPES:
            lo, _ = _INT_DTYPES[dtype]
            nodata = float(lo)  # a concrete in-range integer sentinel
        else:
            nodata = -9999.0
        # Stamp the sentinel into a few cells of every band so nodata-flagged
        # pixels are actually present and must survive the round-trip.
        data = data.copy()
        data[:, 0, 0] = np.array(nodata).astype(dtype)
        if height > 3 and width > 2:
            data[:, 3, 2] = np.array(nodata).astype(dtype)

    return data, nodata


@pytest.mark.property
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(raster=_rasters())
def test_cog_round_trip_is_pixel_exact(raster) -> None:
    """Feature: geospatial-power-pack, Property 2: COG round-trip is pixel-exact.

    Validates: Requirements 8.4, 12.2, 15.3
    """
    data, nodata = raster
    bands, height, width = data.shape

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src.tif")
        dst = os.path.join(tmp, "out_cog.tif")

        profile = dict(
            driver="GTiff",
            dtype=data.dtype.name,
            count=bands,
            height=height,
            width=width,
            crs=_CRS,
            transform=_TRANSFORM,
        )
        if nodata is not None:
            profile["nodata"] = nodata
        with rasterio.open(src, "w", **profile) as ds:
            ds.write(data)

        to_cog(src, dst, overviews=True)

        with rasterio.open(dst) as ds:
            back = ds.read()
            back_nodata = ds.nodata

    # Every band, every pixel (including nodata-flagged) preserved exactly.
    assert back.shape == data.shape
    assert back.dtype == data.dtype
    assert np.array_equal(back, data)

    # The nodata flag itself survives the conversion (so the flagged pixels are
    # still recognizable as nodata, not merely equal in value).
    if nodata is not None:
        assert back_nodata == nodata
