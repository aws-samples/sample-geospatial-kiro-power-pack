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

- Provider tools: `geo-foundation-models.embed_asset` and
  `geo-foundation-models.detect_change_from_assets` (read a COG window
  server-side — preferred), `geo-foundation-models.embed_tile` /
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
  sets a `caveat` whenever the result is not calibrated, and every embedding
  carries `backend` and `structure_only` provenance. Do not threshold a
  stand-in score as if it measured real change.
- **Real published vectors.** For real Clay v1.5 embeddings use
  `lookup_embeddings`; call `available_embedding_periods` first, since coverage
  is only specific months (a before/after outside them is unusable).
- Identical embeddings must yield a change measure of 0.0 (Property 12); use
  this as a validity check.
- Comparability depends on using one model and matching tile footprints — never
  mix models or mismatched extents across the two dates.
