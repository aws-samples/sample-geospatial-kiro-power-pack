---
name: embedding-change-detection
description: >-
  Detect change between two dates using foundation-model embeddings:
  tile two dates -> embed_tile (same model) -> detect_change -> threshold/report.
inclusion: fileMatch
fileMatchPattern:
  - "**/*change*"
  - "**/*embedding*"
  - "**/*embed*"
---

# Steering Workflow: Embedding-Based Change Detection

Quantify change for an area of interest between two acquisition dates by
comparing geospatial foundation-model embeddings.

- Provider tools: `geo-foundation-models.embed_asset` /
  `geo-foundation-models.embed_assets` (embed a COG window server-side — one
  multi-band href, or **separate single-band COGs** one per band as on Earth
  Search) and `geo-foundation-models.detect_change_from_assets` (preferred),
  `geo-foundation-models.change_map` (a per-tile change **grid** over an AOI,
  optionally reduced per vector zone), `geo-foundation-models.embed_tile` /
  `detect_change` (when pixels are already in hand), and
  `geo-foundation-models.available_embedding_periods` /
  `lookup_embeddings` (real published Clay v1.5 vectors)
- Requirements: 9.1 (tile ≤1024×1024, supported format, embedding ≤10s), 9.2
  (exactly one model per request from Clay / Prithvi-EO-2.0 / SatCLIP / ...),
  9.5 (scalar change in [0.0, 1.0], higher = more change), 9.7 (reject
  empty/oversized/unsupported tiles, no embedding), 9.9 (reject dimensionality
  mismatch, no measure), Property 12 (normalized change measure)
- Related skills: `geoai-embedding`, `tool-selection`

## Prerequisites

- Python 3.10 or higher.
- The `geo-foundation-models` server installed and registered in `mcp.json`
  (e.g. `uvx geo-foundation-models`), providing `embed_tile` and
  `detect_change`.
- Available models (Clay, Prithvi-EO-2.0, SatCLIP) and how to select exactly
  one per request; the on-tile path uses a deterministic local stand-in unless
  a real-weight backend is configured.
- Familiarity with geospatial embeddings and change-detection concepts.

## Activation

Active when the open/edited file matches `**/*change*`, `**/*embedding*`, or
`**/*embed*`. If steps cannot be presented, deactivate and report this workflow
name plus the reason. Deactivate when the triggering file no longer matches.

## Steps (in order)

1. **Obtain two co-registered tiles.** Acquire one tile per date for the same
   area of interest. Confirm each tile shares the same footprint and CRS and is
   no larger than 1,024 × 1,024 pixels in a supported raster format. Reject
   empty, oversized, or unsupported tiles with an `Error_Taxonomy` validation
   error and produce no embedding (Requirement 9.7).
2. **Pick one model and keep it fixed.** Select exactly one foundation model
   (for example `Clay`, `Prithvi-EO-2.0`, or `SatCLIP`) and use the **same**
   model for both dates so the embeddings are comparable (Requirement 9.2).
3. **Embed date A.** Call
   `geo-foundation-models.embed_tile(tile=tile_a, model=model)` — expect an
   embedding within 10s whose dimensionality is determined by the model
   (Requirement 9.1).
4. **Embed date B.** Call
   `geo-foundation-models.embed_tile(tile=tile_b, model=model)` with the same
   model. Confirm both embeddings share the same dimensionality.
5. **Detect change.** Call
   `geo-foundation-models.detect_change(embedding_a=..., embedding_b=...)` to get
   a scalar change measure normalized to [0.0, 1.0], where higher means more
   change (Requirement 9.5). If the two embeddings differ in dimensionality,
   reject with a validation error and produce no measure (Requirement 9.9).
6. **Threshold and report.** Apply a change threshold to classify the AOI as
   changed/unchanged and report the score alongside the two dates and the model
   used. Optionally persist the embeddings (e.g. via `geo-embedding-search`) for
   later semantic/similarity queries.

## Whole-AOI change map

Steps 1–6 above quantify change for a **single** AOI/tile. To map *where* change
happened across a larger area, call
`geo-foundation-models.change_map(raster_href_a=..., raster_href_b=...,
model=..., aoi_bbox=..., tile_size=256, bands=...)`. It tiles the AOI window,
embeds both dates per tile, and reduces each tile to a `[0,1]` change measure,
returning a `rows × cols` grid of cells (each with its world `bbox` and
`change`). The tile count is bounded (`max_tiles`); for a large AOI, coarsen the
grid (larger `tile_size`) or route the job to `aws-geo-compute` for distributed
change mapping. `change_map` obeys the same honesty gate — pass
`require_real_backend=true` to make it **refuse** under the uncalibrated
stand-in instead of returning a noise grid. Pass a `zones` FeatureCollection (in
the assets' CRS) to also get a **per-zone** reduction (mean/max change over the
tiles each zone contains). When the imagery's bands live in **separate
single-band COGs** (e.g. Sentinel-2 on Earth Search), embed a date with
`embed_assets(assets=[b1_href, b2_href, ...])` — an ordered list in the model's
band order — instead of pre-stacking. Optional `latlon`/`acquired` on
`embed_asset`/`embed_assets` are forwarded to location/time-aware backends
(e.g. Clay); omit them to skip that conditioning.

## Notes

- **Prefer the one-call bridge.** When the two dates are COGs (e.g. Sentinel-2
  scenes), `detect_change_from_assets(raster_href_a=..., raster_href_b=...,
  model=..., window_bbox=..., bands=...)` reads both windows server-side and
  returns the measure in a single call — no inline pixel plumbing. Use
  `embed_asset` for a single date. Fall back to `embed_tile`/`detect_change`
  only when you already hold the pixels.
- **Honesty gate.** The default backend is a deterministic stand-in with no
  semantic structure: any two differing windows score ~0.5, and two
  structure-only (no-pixel) tiles score exactly 0.0. Treat a score as a real,
  calibrated magnitude **only** with a real-weight backend; `detect_change_from_assets`
  and `change_map` set a `caveat` (and `calibrated=false`) whenever the result
  is not calibrated, and every embedding carries `backend` and `structure_only`
  provenance. Do not threshold a stand-in score as if it measured real change.
- **Wiring a real backend.** Two options, both making change results calibrated;
  configure exactly one (the local callable wins if several are set):
  - **Local, in-process (small/interactive workflows).** Set
    `GEO_FM_EMBED_LOCAL_CALLABLE=my_module:embed` to run a real model in the
    same process with no endpoint to stand up. The function receives the tile
    payload (`{model,dimension,width,height,bands,format,dtype,data}`) and
    returns a vector; the pack ships no model weights, so you bring the callable
    (a companion package, a notebook function, a small torch wrapper). `backend`
    reads e.g. `local:my_module.embed`.
  - **Remote endpoint (offload / scale).** Set `GEO_FM_EMBED_ENDPOINT_URL` (a
    generic HTTPS inference endpoint) or `GEO_FM_SAGEMAKER_ENDPOINT` (an AWS
    SageMaker endpoint name; needs the `geo-foundation-models[remote]` extra) so
    embedding is offloaded to that endpoint — matching the pack's "bring compute
    to the data" posture. The optional `GEO_FM_EMBED_API_KEY` is sent as a
    bearer header to the HTTPS endpoint. `backend` reads e.g. `sagemaker:<name>`.
    A heavier local model can also run behind a `http://localhost:...` URL here.
  - **Declare a custom model's dimension.** For a model not in the default
    registry, set `GEO_FM_EXTRA_MODELS="Name:Dimension,..."` so the tool's
    declared dimension matches what your backend returns (`embed_tile` guards
    the match), then call the tools with `model="Name"`.
  - **Confirm it works.** Run the one-command live self-test against whatever
    backend you configured: `RUN_LIVE_EMBED=1 ... pytest -k live_backend` — it
    asserts the backend is calibrated (not the stand-in), dimension-matched,
    deterministic (identical input → change 0), and content-sensitive. See
    `examples/custom-embedding-backend/` (any model, copy-paste template) and
    `examples/clay-local-backend/` (real Clay v1.5).
- **Real published vectors.** For real Clay v1.5 embeddings use
  `lookup_embeddings`; call `available_embedding_periods` first, since coverage
  is only specific months (a before/after outside them is unusable).
- Identical embeddings must yield a change measure of 0.0 (Property 12); use
  this as a validity check.
- Comparability depends on using one model and matching tile footprints — never
  mix models or mismatched extents across the two dates.
