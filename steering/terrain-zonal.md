---
name: terrain-zonal
description: >-
  Per-zone elevation/slope/aspect over a DEM COG: pick a DEM tile -> align zones
  to the DEM CRS -> dem_zonal(measure=...) -> handle no-data zones -> report.
inclusion: fileMatch
fileMatchPattern:
  - "**/*terrain*"
  - "**/*dem*"
  - "**/*slope*"
  - "**/*aspect*"
---

# Steering Workflow: Terrain Statistics over a Polygon (DEM COG)

Summarize elevation, slope, or aspect within a set of vector zones by reading a
Cloud-Optimized DEM directly over byte ranges — the terrain analogue of
`zonal-statistics`.

- Provider tools: `geo-terrain.dem_zonal`, `geo-ops.transform_crs`,
  `geo-formats.to_geoparquet`
- Requirements: 8.7 (per-zone min/max/mean/sum/count, plus `std`), 8.8 (no-data
  zones still return; others unaffected), 8.1 (CRS transform), 12.6 (byte-range
  read aborts cleanly on failure)
- Related skills: `crs-handling`, `cloud-optimized-formats`

## Prerequisites

- Python 3.10 or higher.
- The `geo-terrain`, `geo-ops`, and `geo-formats` servers installed and
  registered in `mcp.json` (e.g. `uvx geo-terrain`).
- A DEM COG href (e.g. a Copernicus GLO-30 or USGS 3DEP tile) and vector zones.
- Familiarity with coordinate reference systems and CRS alignment.

## Activation

Active when the open/edited file matches `**/*terrain*`, `**/*dem*`,
`**/*slope*`, or `**/*aspect*`. If steps cannot be presented, deactivate and
report this workflow name plus the reason.

## Steps (in order)

1. **Resolve the DEM.** Either pass a specific `dem_href`, or a named
   `dem_source` (`"glo30"` — Copernicus GLO-30) and let `dem_zonal` resolve and
   mosaic the overlapping tiles. Copernicus GLO-30 is EPSG:4326 (degrees); most
   USGS 3DEP COGs are projected (metres).
2. **Align CRS.** The zones must be in the DEM's CRS. If they differ, reproject
   with `geo-ops.transform_crs(...)` before calling `dem_zonal`. Never assume
   the CRSs already match.
3. **Compute.** Call `geo-terrain.dem_zonal(dem_source="glo30", zones=...,
   measure="elevation"|"slope"|"aspect", stats=[...])` (or `dem_href=...` for a
   specific COG). For `slope`/`aspect`
   the cell size comes from the DEM geotransform; for a **geographic** DEM
   (degrees) pass `pixel_size_m=[x, y]` (ground metres) so slope is in the
   expected degrees. Only the tiles overlapping the zones are read.
4. **Handle no-data zones.** A zone with no overlapping DEM cells returns a
   no-data indication (`None`-valued statistics) while the others still receive
   theirs (Requirement 8.8). Do not fail the whole request because one zone is
   empty.
5. **Emit GeoParquet.** Persist the per-zone results with
   `geo-formats.to_geoparquet(...)`, preserving zone id, statistics, and
   no-data flags.

## Notes

- CRS alignment is the most common source of silent errors — reproject
  explicitly.
- `measure="elevation"` is unit-agnostic; `slope`/`aspect` depend on ground cell
  size, so supply `pixel_size_m` for a degrees DEM.
- For planetary-scale runs (very large DEMs or huge zone sets), route the job
  through `aws-geo-compute` (Amazon EMR with Apache Sedona) rather than the
  single-process path.
