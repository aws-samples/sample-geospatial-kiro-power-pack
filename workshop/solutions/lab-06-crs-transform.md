# Lab 06 — CRS transforms and validation (solution)

Server: `geo-ops` · Starter data: `exercises/03-crs-transform/sample_points.geojson`

## Exercise 1 — Transform coordinates

**Prompt:** *"Transform the coordinate (-104.99, 39.74) from EPSG:4326 to EPSG:32613 (UTM Zone 13N). Explain what UTM Zone 13N means."*

**Tool:** `geo-ops` → `transform_crs`.

**Expected output:** approximately `(500,000 , 4,399,000)` meters (easting near 500 km because -105° is the central meridian of zone 13; northing ~4,399 km from the equator). UTM Zone 13N covers longitudes -108° to -102°, centered on -105° — the right zone for Denver.

## Exercise 2 — Validate geometry

**Prompt:** *"Create a self-intersecting polygon (a bowtie shape) and validate it. Then fix it using validate_geometry."*

**Tool:** `geo-ops` → `validate_geometry`. (The starter file includes an intentional bowtie polygon, feature id 5.)

**Expected output:** validation reports the geometry as invalid (self-intersection); `make_valid` returns a corrected geometry (typically a MultiPolygon of the two triangles). Point: invalid geometries silently corrupt area and join results.

## Exercise 3 — Choose the right CRS

**Prompt:** *"I want to calculate the total area of building footprints in Denver. What CRS should I use, and why?"*

**Expected answer:** not EPSG:4326 (degrees aren't constant length). Use EPSG:32613 (UTM 13N) or EPSG:2232 (Colorado Central State Plane) — both projected, meter-based, accurate near Denver.

## Exercise 4 — Round-trip validation

**Prompt:** *"Transform (-104.99, 39.74) from EPSG:4326 to EPSG:32613, then back to EPSG:4326. Is the round-trip accurate?"*

**Expected output:** original recovered within ~1e-8 degrees (sub-millimeter). Larger drift means a pipeline bug.

## Exercise 5 — Transform a GeoJSON feature collection

**Prompt:** *"Take the Denver boundary GeoJSON from the previous lab and transform it from EPSG:4326 to EPSG:32613. Calculate its area in square kilometers."*

**Tool:** `geo-ops` → `transform_crs` then area. **Expected output:** Denver County is ~401 km² (~155 sq mi); a rough AOI rectangle will differ. The value must be computed in the projected CRS, not in degrees.

## Checkpoint answers

- Transformed coordinates and understood the output ✅
- Validated + fixed an invalid geometry ✅
- Knew which CRS to use for area ✅
- Verified a round-trip transform ✅
