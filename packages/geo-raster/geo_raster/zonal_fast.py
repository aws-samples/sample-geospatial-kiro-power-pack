"""Backwards-compatible re-export of the shared numpy zonal fast path.

The vectorized reducer now lives in :mod:`geo_common.zonal_fast`. Re-exported
here so existing ``from geo_raster.zonal_fast import ...`` imports keep working.
"""

from __future__ import annotations

from geo_common.zonal_fast import compute_zonal_statistics_numpy

__all__ = ["compute_zonal_statistics_numpy"]
