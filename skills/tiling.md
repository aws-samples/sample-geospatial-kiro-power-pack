---
name: tiling
description: Dynamic tiling, overview levels, and web-mercator pyramids.
activation:
  - "raster/tiling code"
---

# Tiling

Encoded best practice for the Geospatial Power Pack: serve raster data through
**dynamic tiling** backed by **internal overviews**, so clients fetch only the
pixels they can see at the zoom they're viewing. This is the visualization side
of "bring compute to the data."

## Activation context

This skill applies to raster/tiling code — anything that builds overviews,
generates map tiles, serves XYZ/TMS endpoints, or wires up a dynamic tiler
(e.g. TiTiler, as an external service) over COGs the pack reads or produces.

## Concepts

- **Overviews (pyramids):** downsampled copies of a raster at successively
  coarser resolutions, stored **inside** the COG. They let a tiler answer a
  low-zoom request by reading a small overview instead of decimating full-res
  pixels on the fly.
- **Web Mercator (EPSG:3857) tile pyramid:** the standard web map scheme. The
  world is a quadtree of 256×256 tiles addressed by `z/x/y`; each zoom level `z`
  doubles the resolution of `z-1`.
- **Dynamic tiling:** generate tiles on demand from a COG (e.g. TiTiler) using
  HTTP range reads, rather than pre-rendering and storing every tile.

## Best practices

- **Tile from COGs with overviews.** Dynamic tiling is efficient only when the
  source is cloud-optimized — internal tiling plus an overview pyramid means a
  tile request reads a bounded byte range, not the whole file. (See the
  `cloud-optimized-formats` skill.)
- **Build a full power-of-two overview pyramid** down to where the whole image
  fits in one tile. Stop adding levels once the coarsest overview is ≤ one tile.
- **Match the resampling method to the data:**
  - Continuous (elevation, reflectance) → `average` / `bilinear` / `cubic`.
  - Categorical (land cover, classes) → `nearest` / `mode`. Never average class
    codes — it invents nonexistent categories.
- **Pick internal tile/block size deliberately** (commonly 256 or 512). 512
  blocks reduce overhead for large tiles; 256 aligns with the standard web map
  tile size.
- **Reproject for display, not for storage.** Keep the archival COG in its
  native/analysis CRS; let the tiler reproject to Web Mercator at request time.
- **Set nodata and alpha correctly** so tile edges and gaps render transparent
  rather than as black/!-value borders.
- **Apply rescaling/colormap at the tiler**, parameterized per request, instead
  of baking a single visualization into the stored raster.

## Validation & limits

- Reject tile requests with **empty, oversized, or unsupported** inputs using an
  `Error_Taxonomy` validation error (Property 4 / Requirement 15.5) — don't
  attempt to render them.
- Treat out-of-range `z/x/y` (outside the pyramid's valid extent) as a validation
  error, not a silent empty tile.
- Honor the in-process vs. delegate threshold: large mosaics/pyramids that exceed
  the memory/dataset limits go to `aws-geo-compute` (see `tool-selection`).
