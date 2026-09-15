# Lab 03 — STAC catalog search (solution)

Server: `geo-stac` · Starter data: `exercises/01-stac-search/search_config.json`

Values from live catalogs vary by date; the shapes below are what a correct result looks like.

## Exercise 1 — Find recent imagery

**Prompt:** *"Search for Sentinel-2 Level-2A imagery over Denver, Colorado from the last 30 days with less than 20% cloud cover"*

**Tool:** `geo-stac` → `stac_search` against `sentinel-2-l2a` on Element84 Earth Search.

**Expected output (shape):** a list of scenes, each with:
- `id` (e.g. `S2B_13TDE_20240615_0_L2A`)
- `datetime`
- `eo:cloud_cover` (< 20)
- asset URLs for bands (B02, B03, B04, B08, …) on `s3://sentinel-cogs/…`

A handful of scenes is normal for a 30-day window over one city. If zero results, widen the date range or raise the cloud threshold.

**Landsat variant prompt:** *"Find the most recent Landsat 8 scene covering the Amazon rainforest near Manaus, Brazil"* → one `landsat-c2-l2` item near (-60.02, -3.10).

## Exercise 2 — Understand the results

**Prompt:** *"For the first Sentinel-2 result, show me what bands are available and their spatial resolution"*

**Expected output:** B02/B03/B04/B08 at 10 m; B05/B06/B07/B8A/B11/B12 at 20 m; B01/B09 at 60 m. Key point: NDVI/NDWI use the 10 m bands; SWIR (B11/B12) is 20 m and needs resampling for pixel-aligned math.

## Exercise 3 — Cloud-Optimized GeoTIFF URLs

**Prompt:** *"Give me the S3 URL for the NIR band (B08) of the first Sentinel-2 result"*

**Expected output:** an `s3://sentinel-cogs/sentinel-s2-l2a-cogs/13/T/DE/YYYY/M/<scene-id>/B08.tif` URL. It is a COG, so downstream labs read only the needed window via HTTP byte-range requests.

## Exercise 4 — Time-series discovery

**Prompt:** *"Find all Sentinel-2 scenes over Denver from January 2024 to June 2024 with less than 30% cloud cover. How many scenes are available?"*

**Expected output:** a count (typically ~15–40 scenes over 6 months for one tile/area, depending on cloud filtering) with dates spread across the period. This is the temporal baseline the change-detection labs reuse.

## Checkpoint answers

- Searched a STAC catalog and got scene results ✅
- Identified bands + resolutions ✅
- Understood results point to S3 COGs ✅
