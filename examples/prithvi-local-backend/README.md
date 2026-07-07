# Testing the real-weight backend with Prithvi-EO-2.0 (NASA/IBM)

A second worked real-model example (alongside Clay), proving the same
`embed(payload) -> vector` seam adapts a very different model and loader —
[Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
via [`terratorch`](https://github.com/IBM/terratorch)'s backbone registry. Kept
under `examples/` so terratorch + weights stay out of the pack's default install.

## Setup

```bash
# from the repo root, in your pack venv
uv pip install -e ./packages/geo-common -e ./packages/geo-foundation-models
pip install terratorch                    # pulls torch, torchgeo, etc.

export GEO_FM_EMBED_LOCAL_CALLABLE="prithvi_embed:embed"
export PRITHVI_BACKBONE=terratorch_prithvi_eo_v2_300   # or _600 / _100_tl / _300_tl
export PYTHONPATH="$PWD/examples/prithvi-local-backend"
```

`prithvi_embed.embed` reshapes the tile payload to `[B, H, W]`, standardizes per
band, resizes to Prithvi's 224×224 input, runs the backbone, and returns the
last-layer CLS-token embedding (1024-d — matches the `Prithvi-EO-2.0` spec in the
default registry). Prithvi-EO-2.0 is a **6-band HLS** model (Blue, Green, Red,
NIR-narrow, SWIR1, SWIR2); read those 6 bands, in order, from a single
multi-band COG.

## Verify

```bash
RUN_LIVE_EMBED=1 GEO_FM_EMBED_LOCAL_CALLABLE=prithvi_embed:embed \
  GEO_FM_LIVE_MODEL="Prithvi-EO-2.0" GEO_FM_LIVE_BANDS=6 \
  PRITHVI_BACKBONE=terratorch_prithvi_eo_v2_300 \
  PYTHONPATH=examples/prithvi-local-backend \
  .venv/bin/python -m pytest packages/geo-foundation-models/tests/test_live_backend.py -q
```

This adapter was **run end-to-end**: the live self-test passes (calibrated
1024-d backend, stable, content-sensitive).

## Caveat

Per-tile standardization is used here (not Prithvi's official reflectance
statistics), which is sufficient to validate the seam; apply the published
normalization + acquisition metadata for production-quality embeddings.
