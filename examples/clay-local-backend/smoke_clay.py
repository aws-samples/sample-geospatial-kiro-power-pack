"""End-to-end smoke test: drive the pack's GeoAI tools with a REAL Clay backend.

Runs three checks against the local, in-process Clay v1.5 backend
(:mod:`clay_embed`), proving the real-weight seam works:

1. **Model loads + shape** — embed a synthetic tile; expect a 1024-d vector and
   ``backend == local:clay_embed.embed`` (calibrated, not the stand-in).
2. **Change on identical vs different windows** — with two COG hrefs, run
   ``detect_change_from_assets`` for identical windows (expect ~0.0) and
   ``change_map`` over an AOI (expect a grid with ``calibrated=true``, no caveat).

Usage
-----
    # 1) install deps + weights (see README.md), then:
    export CLAY_CKPT=/path/to/clay-v1.5.ckpt
    export CLAY_METADATA=/path/to/claymodel/configs/metadata.yaml
    export CLAY_SENSOR=naip                       # easiest: single 4-band COG

    python smoke_clay.py \
        --href-a s3://.../naip_before.tif \
        --href-b s3://.../naip_after.tif \
        --bands 1 2 3 4 \
        --aoi <min_x> <min_y> <max_x> <max_y>

Omit --href-a/--href-b to run only check (1) (no network).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

# The example lives next to clay_embed.py; import it directly.
import clay_embed

from geo_foundation_models.local_backend import LocalCallableBackend
from geo_foundation_models.embedding import ModelRegistry, embed_tile
from geo_foundation_models.models import RasterTile
from geo_foundation_models.asset_embedding import detect_change_from_assets
from geo_foundation_models.change_map import change_map

MODEL = "Clay-v1.5"  # 1024-d spec shipped in the default registry


def _backend() -> LocalCallableBackend:
    return LocalCallableBackend(embed_fn=clay_embed.embed)


def check_synthetic(bands: int) -> None:
    """Check 1: the real model loads and returns a 1024-d calibrated vector."""
    n = 64 * 64 * bands
    tile = RasterTile(
        width=64, height=64, bands=bands, format="GTiff",
        data=[float(i % 97) for i in range(n)], dtype="uint16",
    )
    result = embed_tile(tile, MODEL, backend=_backend(), registry=ModelRegistry())
    print(f"[1] embed_tile: dim={result.dimension} backend={result.backend!r} "
          f"structure_only={result.structure_only}")
    assert result.dimension == 1024, result.dimension
    # Calibrated: a real backend, not the deterministic stand-in. (The injected
    # callable reports 'local-callable'; via GEO_FM_EMBED_LOCAL_CALLABLE it would
    # report 'local:clay_embed.embed'.)
    assert result.backend != "deterministic-local", result.backend
    print("    OK — real Clay embedding, calibrated backend.\n")


async def check_change(href_a: str, href_b: str, bands, aoi) -> None:
    """Checks 2-3: identical-window change ~0 and a calibrated change_map grid."""
    backend, reg = _backend(), ModelRegistry()

    same = await detect_change_from_assets(
        raster_href_a=href_a, raster_href_b=href_a, model=MODEL,
        bands=tuple(bands), backend=backend, registry=reg,
    )
    print(f"[2] detect_change_from_assets (A vs A): change={same.change:.4f} "
          f"backend={same.backend!r} caveat={same.caveat}")
    assert same.change == 0.0, "identical windows must score 0.0"
    assert same.caveat is None, "a real backend must not carry the stand-in caveat"

    grid = await change_map(
        raster_href_a=href_a, raster_href_b=href_b, model=MODEL,
        aoi_bbox=aoi, tile_size=256, bands=tuple(bands),
        backend=backend, registry=reg, require_real_backend=True,
    )
    changes = [c.change for c in grid.cells]
    print(f"[3] change_map: {grid.rows}x{grid.cols} grid, calibrated={grid.calibrated}, "
          f"change min/max={min(changes):.4f}/{max(changes):.4f} caveat={grid.caveat}")
    assert grid.calibrated is True and grid.caveat is None
    print("    OK — calibrated per-tile change grid.\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--href-a")
    p.add_argument("--href-b")
    p.add_argument("--bands", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--aoi", type=float, nargs=4, metavar=("MINX", "MINY", "MAXX", "MAXY"))
    args = p.parse_args()

    check_synthetic(bands=len(args.bands))

    if args.href_a and args.href_b and args.aoi:
        asyncio.run(check_change(args.href_a, args.href_b, args.bands, tuple(args.aoi)))
    else:
        print("(skipping change checks — pass --href-a/--href-b/--aoi to run them)")
    print("All requested Clay checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
