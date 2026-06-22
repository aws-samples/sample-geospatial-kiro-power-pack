---
name: stac-metadata
description: "Proper STAC item/asset metadata: datetime, bbox, collections."
activation:
  - "**/*stac*"
  - "catalog code"
---

# STAC Metadata

Encoded best practice for the Geospatial Power Pack: when discovering or
producing **SpatioTemporal Asset Catalog (STAC)** records, get the metadata
right. Searches and downstream analysis are only as sound as the `datetime`,
`bbox`, and `collection` fields they rely on.

## Activation context

This skill applies to files matching `**/*stac*` and to catalog code — anything
that searches STAC APIs (`geo-stac`) or builds STAC Items/Collections.

## The STAC model in one paragraph

A **Collection** groups related Items and declares shared metadata (license,
providers, spatial/temporal `extent`, summaries). An **Item** is a GeoJSON
Feature describing one spatiotemporal observation; it carries `geometry`,
`bbox`, a `datetime` (or `start_datetime`/`end_datetime` range), `properties`,
and one or more **Assets**. An **Asset** is a pointer (href) to actual data
(e.g. a COG band) with `type`, `roles`, and optional `eo:bands` / `raster:bands`.

## Required / high-value fields

- **`datetime`** — RFC 3339, UTC (`Z`). For an acquisition window, set
  `datetime: null` and provide both `start_datetime` and `end_datetime`. Never
  emit a naive local timestamp.
- **`bbox`** — `[west, south, east, north]` in **WGS84 (EPSG:4326), lon/lat
  order**, and consistent with `geometry`. Handle antimeridian crossings per the
  STAC/GeoJSON rules rather than emitting a bbox that wraps the globe.
- **`geometry`** — valid GeoJSON in lon/lat; must agree with `bbox`.
- **`collection`** — every Item references the Collection id it belongs to, and
  that Collection exists.
- **`properties`** — include relevant extensions (`eo:cloud_cover`, `gsd`,
  `proj:epsg`, `view:*`) so queries can filter precisely.
- **Assets** — give each a stable `href`, correct media `type` (e.g.
  `image/tiff; application=geotiff; profile=cloud-optimized` for COGs), and
  meaningful `roles` (`data`, `overview`, `thumbnail`, `metadata`).

## Searching STAC well (`geo-stac`)

- Constrain by **`collections`, `bbox`/`intersects`, and `datetime`** to keep
  result sets small and relevant.
- Use the **`query`/CQL2 filter** for property predicates (cloud cover,
  platform, EPSG) instead of fetching everything and filtering client-side.
- Page through results; don't assume the first page is complete.
- The Power Pack searches across **Earth Search (Element84), MS Planetary
  Computer, NASA CMR-STAC, Copernicus Data Space, USGS** — treat per-source
  failures as graceful degradation (`partial=True` with provenance), not a hard
  error.
- **Carry provenance forward.** After discovery, summarize which source/collection
  each chosen asset came from so analysis stays auditable.

## Producing STAC records

- Build the **Item geometry/bbox from the actual data footprint**, not a rough
  guess.
- Point assets at **cloud-optimized** files (COG/GeoParquet) and set the media
  `type` accordingly.
- Validate against the STAC spec (and any declared extensions) before publishing
  to a catalog.

## Validation

Malformed catalog input — a start-after-end time range, an out-of-range or
malformed bbox, an unparseable datetime — is rejected with an `Error_Taxonomy`
**validation** error and produces no partial output (Property 4 / Requirements
2.7, 15.5).
