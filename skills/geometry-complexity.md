---
name: geometry-complexity
description: Reduce high-vertex geometry (simplify -> hull -> bbox) and prefer referencing geometry by path before inlining it into a tool call.
activation:
  - "code referencing zones"
  - "code referencing FeatureCollection"
  - "code referencing perimeter"
  - "code referencing GeoJSON geometry"
---

# Geometry Complexity

Encoded best practice for the Geospatial Power Pack: **vertex count and request
payload size are a scaling axis of their own**, independent of how many *bytes*
of raster or dataset a task touches. Many tools accept **inline GeoJSON**
geometry — zonal statistics, band-math zones, CRS transforms, spatial joins,
overlays. For those, a high-vertex geometry can make a call slow or fail even
when the underlying data is tiny: a 3-KB raster window with a 3,400-vertex zone
fails on the **inline geometry payload**, not the data.

## Activation context

This skill applies to any code that constructs, passes, or transforms vector
geometry — anything referencing `zones`, a `FeatureCollection`, a `perimeter`,
or a GeoJSON geometry — especially just before a tool call that consumes inline
geometry.

## The core rule

**Reduce the geometry to the coarsest form that still answers the question,
before the call.** In order of shape fidelity:

1. **`geo-ops.simplify(geometry=..., tolerance=..., preserve_topology=True)`** —
   Douglas-Peucker vertex reduction. Every point of the result stays within
   `tolerance` (in the geometry's coordinate units) of the original, so it
   **keeps concavity**. This is the default choice: it shrinks a detailed
   watershed / admin boundary / fire perimeter to a fraction of its vertices
   while remaining faithful to the footprint. `preserve_topology=True` avoids
   invalid/collapsed output.
2. **`geo-ops.convex_hull(geometry=...)`** — the smallest convex shape that
   contains the input. It is a **lossy superset**: it fills in every concavity,
   so per-zone results computed against a hull *over-count* the true area. Use
   only when a coarse, convex approximation is explicitly acceptable, and flag
   the accuracy caveat on any result.
3. **Bounding box** — the coarsest reduction; a rectangle around the geometry.
   Cheapest to pass, least faithful.

If the geometry already exists as a file or URL (`.geojson`, `.parquet`), prefer
**referencing it by path/href** over inlining thousands of coordinates —
`geo-formats.to_geoparquet` already accepts a path/href for its `src`, and that
is the pattern to reach for whenever a tool supports it.

## Choosing a tolerance for `simplify`

- `tolerance` is in the geometry's **coordinate units** (degrees for EPSG:4326,
  metres for a projected CRS) — so pick it *after* you know the CRS, and prefer
  simplifying in a projected CRS where the unit is metres and intuitive.
- Start around the raster's cell size (or the smallest feature you care about):
  simplifying below one pixel adds vertices with no analytical benefit.
- `tolerance=0` is a no-op; increase it until the vertex count drops to a
  manageable size without visibly distorting the footprint.

## Order with CRS handling

Reduce **first, then reproject** — run `simplify` (or hull/bbox) and reproject
the *reduced* geometry, not the full perimeter. Transforming thousands of
vertices you are about to discard is wasted work, and it keeps the payload small
through the whole chain (`crs-handling` skill).

## Checklist

- [ ] Estimated the geometry's vertex count before passing it inline.
- [ ] Reduced high-vertex geometry (`simplify` → `convex_hull` → bbox) to the
      coarsest form that still answers the question.
- [ ] Used `simplify` (not `convex_hull`) whenever concavity matters; flagged the
      superset caveat if a hull was used.
- [ ] Referenced geometry by path/href instead of inlining where the tool allows.
- [ ] Reduced the geometry *before* reprojecting it (`crs-handling`).
