---
name: cog-conversion
description: >-
  Convert a raster to a Cloud-Optimized GeoTIFF: read -> validate -> to_cog
  (overviews) -> verify pixel-exact round-trip -> write to silver/gold.
inclusion: fileMatch
fileMatchPattern:
  - "**/*.tif"
  - "**/*.tiff"
  - "**/*cog*"
  - "**/*convert*"
---

# Steering Workflow: COG Conversion

Produce a valid Cloud-Optimized GeoTIFF (COG) from a source raster, verify the
conversion is loss-free, and promote the result to a cloud-optimized location.

- Provider tools: `geo-formats.to_cog`, `geo-formats.validate_format`
- Requirements: 8.4 (produce COG), 12.2 (raster output is COG), 12.7 (stop and
  remove partial output on write failure), 15.3 / Property 2 (pixel-exact
  round-trip including nodata)
- Related skills: `cloud-optimized-formats`, `crs-handling`, `tiling`

## Prerequisites

- Python 3.10 or higher.
- The `geo-formats` server installed and registered in `mcp.json` (e.g.
  `uvx geo-formats`, or `uv pip install -e ./packages/geo-formats`), which
  provides `to_cog` and `validate_format`.
- Familiarity with raster data formats (GeoTIFF/COG) and nodata handling.
- Optional: AWS credentials in `mcp.json` only if reading from or writing the
  output to Amazon S3; local file paths work without them.

## Activation

Active when the open/edited file matches a raster (`**/*.tif`, `**/*.tiff`) or a
COG/conversion script (`**/*cog*`, `**/*convert*`). If steps cannot be
presented, deactivate and report this workflow name plus the reason. Deactivate
when the triggering file no longer matches.

## Steps (in order)

1. **Identify source and destination.** Resolve the source raster href and the
   destination href. Prefer reading metadata only (band count, dtype, CRS,
   nodata, block layout) from S3 byte ranges — do not copy the full asset to
   local storage for inspection (Requirement 12.1).
2. **Validate the source raster.** Call `geo-formats.validate_format(href=src,
   fmt="geotiff")`. If the source is malformed or unreadable, stop and surface
   the `Error_Taxonomy` validation/upstream error — do not attempt conversion.
3. **Convert to COG with overviews.** Call
   `geo-formats.to_cog(src_href=src, dst_href=tmp_or_dst, overviews=True)`.
   Use internal tiling, compression, and overview pyramids appropriate to the
   data (preserve the source nodata value). Write to a temporary or staging
   path first.
4. **Verify the output is a valid COG.** Call
   `geo-formats.validate_format(href=dst, fmt="cog")` and confirm the
   cloud-optimized structure (tiled, overviews present, valid IFD layout).
5. **Verify the pixel-exact round-trip.** Read the COG back and confirm that, for
   every band, pixel values equal the source exactly — including nodata-flagged
   pixels (Requirement 15.3, Property 2). If any pixel differs, treat the
   conversion as failed.
6. **Handle write/verify failure.** On a write or verification failure, stop,
   remove any partial output that was written, and return the `Error_Taxonomy`
   error (Requirement 12.7). Do not leave a partial COG behind.
7. **Promote to silver/gold.** On success, write/move the verified COG to its
   cloud-optimized silver or gold S3 location and record provenance (source
   href, conversion parameters, destination href).

## Notes

- Always write raster outputs as COG (Requirement 12.2); never emit a plain,
  non-tiled GeoTIFF from this workflow.
- Keep the nodata value explicit through every step so masked pixels round-trip
  exactly.
