"""Production COPC backend test for ``geo-pointcloud`` (PdalCopcBackend).

Exercises the *production* read/write path that uses PDAL's ``readers.copc`` /
``writers.copc`` against a real ``.copc.laz`` file.

PDAL links its own native Apache Arrow; importing ``pdal`` into a process that
also uses pip ``pyarrow`` (as ``geo-formats`` does for GeoParquet) makes Arrow
abort with a duplicate-filesystem-registration error. In real deployment each
server is its own ``uvx`` process, so this never happens — but the test runner
loads every package in one interpreter. To stay faithful to deployment *and*
safe in the shared runner, this test runs the PDAL round-trip in a **subprocess**
and never imports ``pdal`` into the pytest process. It is skipped unless the
native PDAL bindings are installed (the ``[pdal]`` extra + the native library).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.integration

# Detect PDAL WITHOUT importing it here: find_spec does not execute the module,
# so PDAL's native Arrow is never loaded into the pytest process.
if importlib.util.find_spec("pdal") is None:  # pragma: no cover - env dependent
    pytest.skip(
        "PDAL native bindings not installed ([pdal] extra)",
        allow_module_level=True,
    )


# A self-contained program: write a sample cloud to COPC via the PDAL backend,
# read it back, and assert the point set round-trips within LAS storage
# precision (default scale 0.01). Printed sentinels report the outcome.
_PROGRAM = textwrap.dedent(
    """
    import os, tempfile, sys
    from geo_pointcloud import (
        PdalCopcBackend, PointCloudChunk, PointRecord,
        read_pointcloud, write_pointcloud, COPC_FORMAT,
    )

    chunk = PointCloudChunk(
        crs="EPSG:32631",
        points=[
            PointRecord(x=1.0, y=2.0, z=3.0, intensity=10, classification=2),
            PointRecord(x=4.5, y=5.25, z=6.125, intensity=20, classification=5),
            PointRecord(x=-7.0, y=8.0, z=-9.0, red=100, green=150, blue=200),
        ],
    )
    backend = PdalCopcBackend()
    fd, path = tempfile.mkstemp(suffix=".copc.laz"); os.close(fd)
    try:
        result = write_pointcloud(points=chunk, dst_href=path, backend=backend)
        assert result.format == COPC_FORMAT
        assert result.point_count == chunk.point_count
        back = read_pointcloud(copc_href=path, backend=backend)
        assert back.point_count == chunk.point_count
        a = sorted((p.x, p.y, p.z) for p in back.points)
        b = sorted((p.x, p.y, p.z) for p in chunk.points)
        tol = 1e-2  # LAS default scale 0.01
        for (ax, ay, az), (bx, by, bz) in zip(a, b):
            assert abs(ax - bx) <= tol and abs(ay - by) <= tol and abs(az - bz) <= tol
    finally:
        if os.path.exists(path):
            os.remove(path)
    print("PDAL_ROUNDTRIP_OK")
    """
)


def test_pdal_backend_roundtrips_point_positions() -> None:
    """Write + read a real ``.copc.laz`` via PDAL, isolated in a subprocess."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROGRAM],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=120,
    )
    assert proc.returncode == 0, (
        "PDAL round-trip subprocess failed:\nstdout:\n%s\nstderr:\n%s"
        % (proc.stdout, proc.stderr)
    )
    assert "PDAL_ROUNDTRIP_OK" in proc.stdout
