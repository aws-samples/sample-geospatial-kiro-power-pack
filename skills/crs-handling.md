---
name: crs-handling
description: Always track CRS; reproject explicitly; avoid silent axis-order bugs.
activation:
  - "code referencing pyproj"
  - "code referencing crs"
  - "code referencing EPSG"
---

# CRS Handling

Encoded best practice for the Geospatial Power Pack: a coordinate is meaningless
without its **coordinate reference system (CRS)**. Most production geospatial
bugs are silent CRS bugs — mismatched datums, assumed EPSG codes, and axis-order
swaps. Track CRS everywhere and reproject **explicitly**.

## Activation context

This skill applies to any code referencing `pyproj`, `crs`, or `EPSG`, and more
broadly to any code that reads coordinates, transforms geometries, or joins
datasets that may live in different reference systems.

## Core rules

1. **Every geometry/dataset carries its CRS.** Never store or pass bare
   coordinates. Read the CRS from the source; if it is missing, treat it as an
   error condition, not an assumption.
2. **Never assume EPSG:4326.** A dataset that "looks like" lat/lon may be in a
   projected or differently-datumed CRS. Confirm before operating.
3. **Reproject explicitly, never implicitly.** Any cross-CRS operation must go
   through an explicit transform step (`geo-ops` CRS transform). Do not rely on
   a library to "figure it out."
4. **Align CRS before any spatial operation** — overlay, spatial join, zonal
   statistics, distance/area. Operands must share one CRS first.
5. **Choose the right CRS for the operation.** Geographic CRS (degrees) for
   storage/exchange; an appropriate projected CRS (meters) for distance, area,
   and buffering. Computing area/length in degrees is a bug.

## The axis-order trap

EPSG:4326 is formally defined as **(latitude, longitude)** — lat first — while
GeoJSON, most web tooling, and many file formats use **(longitude, latitude)**.
This mismatch silently swaps coordinates and places data in the wrong hemisphere.

- Be explicit about axis order when constructing a `pyproj` transformer; prefer
  `always_xy=True` so transforms consume and produce **(x, y) = (lon, lat)**
  consistently.
- When ingesting from a source, confirm whether it emits lon/lat or lat/lon
  before building geometries.
- When emitting GeoJSON, ensure coordinates are **(longitude, latitude)** per
  RFC 7946.

## Round-trip correctness (verified by property test)

`geo-ops` CRS transforms are covered by a property test: **transforming a
geometry to a target CRS and back to the source CRS reproduces it within a
documented tolerance** (max per-vertex deviation in source-CRS units — Property
1 / Requirements 8.1, 15.2).

- Document the tolerance; do not treat "close enough" as exact.
- Round-trip drift beyond tolerance signals a wrong CRS, a datum-shift gap
  (missing grid), or an axis-order swap — investigate, don't paper over it.

## Practical checklist

- [ ] Source CRS read and recorded (not assumed).
- [ ] Target CRS chosen deliberately for the operation (storage vs. measurement).
- [ ] Transform built explicitly with known axis order (`always_xy=True`).
- [ ] All operands reprojected to a common CRS before the spatial op.
- [ ] Output carries its CRS in the file/column metadata.
- [ ] Malformed coordinates rejected with a `VALIDATION` error (Requirement 15.5).
