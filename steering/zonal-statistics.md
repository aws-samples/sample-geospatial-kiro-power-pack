---
name: zonal-statistics
description: >-
  Compute zonal statistics for a raster over vector zones: load raster + zones
  -> align CRS -> zonal_statistics -> handle no-data zones -> emit GeoParquet.
inclusion: fileMatch
fileMatchPattern:
  - "**/*zonal*"
  - "**/*stats*"
  - "**/*zonal_stats*"
---

# Steering Workflow: Zonal Statistics over Raster and Vector

Summarize a raster within a set of vector zones, returning a defined statistic
per zone and a no-data indication for zones that do not overlap the raster.

- Provider tools: `geo-raster.zonal_statistics`, `geo-ops.transform_crs`,
  `geo-formats.to_geoparquet`
- Requirements: 8.7 (per-zone min/max/mean/sum/count, plus `std` population
  standard deviation), 8.8 (no-data zones still
  return; other zones unaffected), 8.1 (CRS transform), 12.3 (tabular vector
  output is GeoParquet), Property 14 (statistics equal a reference computation)
- Related skills: `crs-handling`, `cloud-optimized-formats`

## Prerequisites

- Python 3.10 or higher.
- The `geo-raster`, `geo-ops`, and `geo-formats` servers installed and
  registered in `mcp.json` (e.g. `uvx geo-raster`, `uvx geo-ops`,
  `uvx geo-formats`), providing `zonal_statistics`, `transform_crs`, and
  `to_geoparquet`.
- Familiarity with coordinate reference systems and CRS alignment.
- Input data: a raster (GeoTIFF/COG) and vector zones (GeoJSON/GeoParquet) for
  the same area.

## Activation

Active when the open/edited file matches `**/*zonal*`, `**/*stats*`, or
`**/*zonal_stats*` (raster × vector statistics code). If steps cannot be
presented, deactivate and report this workflow name plus the reason. Deactivate
when the triggering file no longer matches.

## Steps (in order)

1. **Load inputs.** Resolve the raster href and the vector zones
   (`FeatureCollection`). Read the raster via S3 byte ranges where possible
   rather than copying the full asset (Requirement 12.1). Confirm the requested
   statistics (default: `min`, `max`, `mean`, `sum`, `count`; `std` is also
   supported as an opt-in population standard deviation).
2. **Align CRS.** Compare the raster CRS and the zones CRS. If they differ,
   reproject the zones to the raster CRS with
   `geo-ops.transform_crs(geometry=..., src_crs=zones_crs, dst_crs=raster_crs)`
   before any overlap is computed. Never assume the CRSs already match.
3. **Compute zonal statistics.** Call
   `geo-raster.zonal_statistics(raster_href=..., zones=..., stats=[...])` to get
   one `ZoneStat` per zone (Requirement 8.7).
4. **Handle no-data zones.** For any zone with no overlapping raster cells,
   confirm a no-data indication (e.g. `None`-valued statistics) is returned for
   that zone while every other zone still receives its statistics
   (Requirement 8.8). Do not drop or fail the whole request because one zone is
   empty.
5. **Emit GeoParquet.** Write the per-zone results as GeoParquet via
   `geo-formats.to_geoparquet(...)`, preserving the zone identifier, the
   computed statistics, and the no-data flags (Requirement 12.3).

## Notes

- CRS alignment is the most common source of silent errors here — always
  reproject explicitly and keep axis order correct (`crs-handling` skill).
- Validate inputs first: a malformed raster/zone geometry must yield an
  `Error_Taxonomy` validation error with no partial output.
- Performance: the default engine is pure-Python (dependency-light). Installing
  the optional `geo-raster[fast]` extra enables a numpy-accelerated zonal engine
  (same results, verified by a differential test) that is markedly faster on
  larger windows and many-zone requests — worthwhile for interactive and
  mid-size work.
- Scale boundary: for planetary-scale zonal statistics (very large rasters or
  huge zone sets), do not push the single-process engine — route the job through
  `aws-geo-compute` (`large-zonal-statistics` → Amazon EMR with Apache Sedona).
