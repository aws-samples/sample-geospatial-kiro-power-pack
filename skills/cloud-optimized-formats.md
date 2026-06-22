---
name: cloud-optimized-formats
description: When to use COG/GeoParquet/Zarr/COPC and how to write cloud-optimized outputs.
activation:
  - "**/*.tif"
  - "**/*.tiff"
  - "**/*.parquet"
  - "**/*.zarr"
  - "raster/vector code"
---

# Cloud-Optimized Formats

Encoded best practice for the Geospatial Power Pack: **all persisted geospatial
outputs are written in cloud-optimized formats.** This lets consumers read only
the byte ranges they need directly from object storage (S3) instead of copying
whole assets locally. It is the format side of the "bring compute to the data"
principle.

## Activation context

This skill applies whenever you open or edit raster/vector code or files
matching `**/*.tif`, `**/*.tiff`, `**/*.parquet`, or `**/*.zarr` — that is, any
code that **reads, transforms, or writes** geospatial data assets.

## Pick the format by data shape

| Data shape | Format | Server | Notes |
|---|---|---|---|
| 2D raster / imagery (single or multi-band) | **COG** (Cloud-Optimized GeoTIFF) | `geo-formats`, `geo-raster` | Internal tiling + overviews; HTTP range reads. |
| Tabular vector (points/lines/polygons + attributes) | **GeoParquet** | `geo-formats`, `geo-query` | Columnar, predicate/column pushdown, bbox covering metadata. |
| N-dimensional / time-series / multivariate arrays | **Zarr** | `geo-formats` | Chunked, parallel, good for data cubes and climate stacks. |
| LiDAR / 3D point clouds | **COPC** (Cloud-Optimized Point Cloud) | `geo-formats`, `geo-pointcloud` | LAZ with a clustered octree for spatial-range streaming. |

Default decision: **raster → COG, vector → GeoParquet, cube → Zarr, points →
COPC.** Only deviate when a downstream consumer requires another format, and
document why.

## Best practices when writing outputs

- **Write to the curated tiers of the medallion layout** (`silver/` cleaned,
  `gold/` curated). Treat raw drops as `bronze/` and never overwrite them.
- **COG:** build internal overviews (power-of-two pyramid), use internal tiling
  (typically 512×512), apply appropriate compression (e.g. DEFLATE/ZSTD for
  continuous data, LZW for categorical), and preserve the **nodata** value.
  Validate the result with a COG validator before publishing.
- **GeoParquet:** store the geometry column per the GeoParquet spec, embed the
  CRS in the column metadata, and include bbox/covering metadata so readers can
  prune row groups by spatial extent.
- **Zarr:** choose chunk sizes aligned to expected read windows; record CRS and
  dimension coordinates; keep consolidated metadata for fast catalog reads.
- **COPC:** keep the clustered octree intact; do not rewrite to plain LAS/LAZ
  when streaming access is needed.

## Round-trip integrity is mandatory

Format conversion must be **lossless** and is verified by property tests
(`geo-formats`):

- **COG round-trip is pixel-exact** for every band, including nodata-flagged
  pixels (Property 2 / Requirements 12.2, 15.3). Never silently rescale,
  resample, or drop nodata on conversion.
- **GeoParquet round-trip preserves features exactly** — feature count,
  geometries coordinate-for-coordinate, and every attribute value (Property 3 /
  Requirements 12.3, 15.4).

If a transform cannot guarantee an exact round-trip, surface it explicitly
rather than persisting lossy output.

## Failure handling

- A **write failure stops the operation, removes any partial output, and
  returns an `Error_Taxonomy` error** — never leave a half-written asset
  (Requirement 12.7).
- A **read failure** (after the retry budget) stops and leaves local storage
  unchanged (Requirement 12.6).
- Malformed input is rejected with a `VALIDATION` error **before** any bytes are
  written, so there is no partial or persisted output (Requirement 15.5).
