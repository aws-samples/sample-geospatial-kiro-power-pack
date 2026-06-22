---
name: stac-discover-analyze
description: >-
  Discover, read, and analyze geospatial assets from a STAC catalog:
  stac_search -> select assets -> windowed read_window -> analyze ->
  summarize provenance.
inclusion: fileMatch
fileMatchPattern:
  - "**/*stac*"
  - "**/*discover*"
  - "**/*.ipynb"
---

# Steering Workflow: STAC Discover -> Download -> Analyze

Drive a discover → process → analyze pipeline that finds assets via STAC, reads
only the pixels needed, analyzes them, and reports full provenance.

- Provider tools: `geo-stac.stac_search`, `geo-raster.read_window`,
  `geo-raster.band_math`, plus an analysis tool
  (`geo-foundation-models.embed_tile`, `geo-raster.zonal_statistics`, ...)
- Requirements: 7.1 (STAC search ≤30s, assets + spatio-temporal metadata, cap
  1000), 7.2 (windowed read returns only in-window pixels, no full-asset copy),
  7.12 (validation of bbox / time range), 4.4 (provenance by source id),
  4.7 / 4.8 (partial labeling under degradation)
- Related skills: `stac-metadata`, `cloud-optimized-formats`, `tool-selection`

## Prerequisites

- Python 3.10 or higher.
- The `geo-stac` and `geo-raster` servers installed and registered in
  `mcp.json` (e.g. `uvx geo-stac`, `uvx geo-raster`), plus any analysis server
  used (`geo-foundation-models`). These provide `stac_search`, `read_window`,
  `band_math`, and the analysis tool.
- No credentials required for the open Earth Search STAC catalog and public COG
  reads; configure optional keys only for restricted catalogs.
- Familiarity with STAC metadata, bounding boxes, and datetime ranges.

## Activation

Active when the open/edited file matches `**/*stac*`, `**/*discover*`, or a
discovery notebook (`**/*.ipynb`). If steps cannot be presented, deactivate and
report this workflow name plus the reason. Deactivate when the triggering file
no longer matches.

## Steps (in order)

1. **Define the query.** Establish the spatial extent (`bbox`), the temporal
   range (`datetime_range`), and target `collections`. Validate the bbox and
   that the range start is not later than its end; reject malformed parameters
   with an `Error_Taxonomy` validation error before searching (Requirement 7.12).
2. **Search the catalog.** Call
   `geo-stac.stac_search(bbox=..., datetime_range=..., collections=...,
   limit<=1000)`. Expect results within 30s, each item carrying its asset
   references and spatio-temporal metadata; the response is capped at 1,000
   items (Requirement 7.1).
3. **Select assets.** Filter the returned items to the assets you need (e.g. by
   band, cloud cover, or platform) using their STAC metadata. Record which
   item/asset ids you selected.
4. **Windowed read.** For each selected asset call
   `geo-raster.read_window(asset_href=..., window=AOI, bands=...)` so only the
   pixels inside the area of interest are transferred — never copy the full
   asset to local storage (Requirements 7.2, 12.1). Apply
   `geo-raster.band_math` for indices (NDVI/NDWI/NBR) when needed.
5. **Analyze.** Run the analysis step on the windowed data — for example
   `geo-foundation-models.embed_tile` for embeddings or
   `geo-raster.zonal_statistics` for summaries — keeping discover → process →
   analyze ordering.
6. **Summarize provenance.** Report every source used in the discover, process,
   and analyze steps by source identifier (Requirement 4.4). If any source
   degraded or timed out, label the result partial and enumerate contributing
   and failed sources with their `Error_Taxonomy` categories
   (Requirements 4.7, 4.8).

## Notes

- Keep STAC metadata intact (datetime, bbox, collections) so downstream steps
  and provenance stay accurate (`stac-metadata` skill).
- Prefer cloud-optimized assets and windowed reads to honor "bring compute to
  the data."
