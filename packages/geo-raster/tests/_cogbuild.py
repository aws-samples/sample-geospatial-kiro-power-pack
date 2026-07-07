"""Backwards-compatible re-export of the shared in-memory COG test builder.

The builder + recording reader now live in :mod:`geo_common.testing` so every
server's tests can construct deterministic in-memory COGs. Re-exported here so
existing ``from _cogbuild import ...`` imports in the geo-raster tests keep
working.
"""

from geo_common.testing import RecordingByteRangeReader, build_cog

__all__ = ["build_cog", "RecordingByteRangeReader"]
