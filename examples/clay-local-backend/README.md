# Testing the real-weight GeoAI backend with Clay v1.5

The `geo-foundation-models` server embeds tiles through a pluggable
`EmbeddingBackend`. The default is an honest **deterministic stand-in** (no
semantic structure — change scores are ~0.5 noise), so `detect_change`,
`detect_change_from_assets`, and `change_map` are only *calibrated* with a real
model. This example wires **real [Clay](https://clay-foundation.github.io/model/)
v1.5** into the **local (in-process)** backend so you can test end-to-end with no
inference endpoint to stand up.

Nothing here is installed by the pack: Clay pulls in `torch`, the `claymodel`
package, and a multi-GB checkpoint, which the pack deliberately keeps out of its
default install. This example is the bridge.

## 1. Environment

```bash
# from the repo root, in your pack venv
uv pip install -e ./packages/geo-common -e ./packages/geo-foundation-models

# Clay itself (heavy — torch + the model package)
pip install "git+https://github.com/Clay-foundation/model.git" pyyaml

# weights + sensor metadata (metadata.yaml ships in the Clay repo under configs/)
wget https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt
# grab configs/metadata.yaml from the Clay repo (or point CLAY_METADATA at your clone)
```

## 2. Point the callable at Clay

```bash
export CLAY_CKPT=$PWD/clay-v1.5.ckpt
export CLAY_METADATA=$PWD/metadata.yaml     # from the Clay repo's configs/
export CLAY_SENSOR=naip                     # see "Choosing data" below
```

`clay_embed.embed(payload)` reshapes the pack's tile payload to `[B, H, W]`,
normalizes per band with the sensor's mean/std, resizes to Clay's 256×256 chip,
builds the per-band wavelength tensor, and runs `model.encoder(...)`, returning a
1024-d vector.

## 3. Run the smoke test

```bash
cd examples/clay-local-backend

# Check 1 only (loads Clay, no network): expect dim=1024, backend='local:clay_embed.embed'
python smoke_clay.py --bands 1 2 3 4

# Full run (two co-registered COGs + an AOI in their CRS):
python smoke_clay.py \
  --href-a s3://your-bucket/naip_2021.tif \
  --href-b s3://your-bucket/naip_2023.tif \
  --bands 1 2 3 4 \
  --aoi <min_x> <min_y> <max_x> <max_y>
```

Expected: check (1) prints a 1024-d vector from the `local:clay_embed.embed`
backend; the A-vs-A change is `0.0`; and `change_map` returns a grid with
`calibrated=true` and no caveat.

## 4. Use it from the running MCP server (in Kiro)

Once the smoke passes, wire the same callable into the server via env so
`change_map` / `detect_change_from_assets` are calibrated in normal use:

```bash
export GEO_FM_EMBED_LOCAL_CALLABLE="clay_embed:embed"   # module must be importable
export CLAY_CKPT=... CLAY_METADATA=... CLAY_SENSOR=...
# start geo-foundation-models; call the tools with model="Clay-v1.5"
```

`backend_from_env()` selects the local callable (precedence: local → SageMaker →
HTTPS). Put `clay_embed.py` on `PYTHONPATH` so `clay_embed:embed` resolves.

## Choosing data (important)

Clay is sensor-agnostic but expects the **exact band set and order** of the
configured sensor, read from a **single multi-band COG**:

- **NAIP (easiest):** a single 4-band (R/G/B/NIR) COG — set `CLAY_SENSOR=naip`
  and `--bands 1 2 3 4`. Ideal for a first real test.
- **Sentinel-2 L2A:** 10 bands in a fixed order. Earth Search stores each band
  as a *separate* single-band COG, and the pack's `embed_asset`/`change_map`
  read bands from **one** href — so stack the 10 bands into one multi-band COG
  first (e.g. with the companion `gdal-mcp`/GDAL), then read `--bands 1..10`.

This single-href-per-read limit is a known edge (the same one `zonal_band_math`
addresses for indices); a future `embed_assets` mapping band→href would let Clay
read Earth Search's per-band COGs directly.

## Verified

This adapter was **run end-to-end** against real Clay v1.5 (`claymodel==1.5.0`,
`clay-v1.5.ckpt`) on CPU: `embed_tile` returns a 1024-d vector from a calibrated
backend, and the live self-test passes (stable + content-sensitive):

```bash
RUN_LIVE_EMBED=1 GEO_FM_EMBED_LOCAL_CALLABLE=clay_embed:embed \
  GEO_FM_LIVE_MODEL=Clay-v1.5 GEO_FM_LIVE_BANDS=4 \
  CLAY_CKPT=... CLAY_METADATA=... CLAY_SENSOR=naip \
  PYTHONPATH=examples/clay-local-backend \
  .venv/bin/python -m pytest packages/geo-foundation-models/tests/test_live_backend.py -q
```

Two things this shook out (already handled here):
- The v1.5 checkpoint stores a cwd-relative `metadata_path='configs/metadata.yaml'`;
  the adapter overrides it in `load_from_checkpoint(..., metadata_path=CLAY_METADATA)`.
- Real Clay uses the **datacube** encoder API (`model.model.encoder({"pixels",
  "time", "latlon", "gsd", "waves"})`, CLS token = embedding), not the simplified
  `encoder(chips, timestamps, wavelengths)` from the quickstart page. The adapter
  uses the datacube API and sets `mask_ratio=0` for a full deterministic pass.

## Caveats

- **Metadata faithfulness:** `time`/`latlon` are passed as zeros, so Clay's
  season/location conditioning is omitted. Fine for a plumbing smoke and relative
  change; extend `clay_embed.py` (and forward the window bbox + datetime from the
  pack) for a faithful run.
- Real models carry ~1e-13 run-to-run float noise, so "identical input" change is
  ~0 (not bit-exact 0); the self-test uses a near-zero tolerance.
```
