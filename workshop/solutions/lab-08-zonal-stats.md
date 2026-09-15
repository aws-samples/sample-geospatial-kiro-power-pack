# Lab 08 — Zonal statistics and spatial joins (solution)

Servers: `geo-raster` (`zonal_stats`) / `geo-ops` (`buffer`, `spatial_join`, `overlay`) · Starter data: `exercises/05-zonal-stats/denver_neighborhoods.geojson`

Values vary with the underlying scene; the rankings below are the expected pattern for Denver.

## Exercise 1 — Average NDVI per neighborhood

**Prompt:** *"Calculate the mean NDVI for each Denver neighborhood using the NDVI raster from Lab 07 and the Denver neighborhood boundary polygons. Which neighborhoods have the greenest vegetation?"*

**Tool:** `geo-raster` → `zonal_stats` (NDVI raster × neighborhood polygons).

**Expected output:** each neighborhood polygon gains `mean_ndvi`, `max_ndvi`, `std_ndvi`, and `count`. Park-heavy neighborhoods — **City Park West** and **Washington Park** — rank greenest (high mean NDVI), while **Downtown** and **Five Points** rank lowest. Absolute values shift with the scene, but the ordering holds.

## Exercise 2 — Spatial join: buildings near parks

**Prompt:** *"Perform a spatial join between Denver building footprints and park polygons. Which buildings are within 100 meters of a park? What percentage of buildings have park access?"*

**Tools:** `geo-ops` → `buffer` (100 m around parks) then `spatial_join` (intersects predicate).

**Expected output:** each building tagged `near_park: true/false`, plus an overall percentage with park access. The exact share depends on the building/park extent pulled, so report it as a computed figure rather than a fixed number.

## Exercise 3 — Overlay operations

**Prompt:** *"Find the intersection of Denver's flood plain zones with residential land-use areas. What area (in hectares) is both residential AND in a flood plain?"*

**Tool:** `geo-ops` → `overlay` (intersection).

**Expected output:** a polygon layer covering the residential ∩ flood-plain area with a hectare figure. Area must be computed in a projected CRS (UTM 13N / EPSG:32613), not in degrees. Union and difference are available from the same tool for the complementary questions.

## Exercise 4 — Combined pipeline

**Prompt:** *"For Denver County: 1) Get the county boundary, 2) Search for Sentinel-2 imagery, 3) Calculate NDVI, 4) Get neighborhood boundaries, 5) Calculate mean NDVI per neighborhood, 6) Rank neighborhoods by vegetation health"*

**Tools (chained):** `geo-vector` (boundary) → `geo-stac` (scene) → `geo-raster` (NDVI) → `geo-vector` (neighborhoods) → `geo-raster` (`zonal_stats`) → ranking.

**Expected output:** a ranked table of neighborhoods by `mean_ndvi`, greenest first (City Park West / Washington Park at the top, Downtown / Five Points at the bottom) — the same six-step pipeline the capstone builds on, driven by one prompt.

## Exercise 5 — Buffer analysis

**Prompt:** *"Create a 500-meter buffer around all rivers in Denver. How many buildings fall within the river buffer? These are flood-risk buildings."*

**Tools:** `geo-ops` → `buffer` (500 m around river lines) then `spatial_join` / count.

**Expected output:** a river buffer polygon and a count of intersecting buildings flagged as flood-risk — concentrated along the South Platte and Cherry Creek corridors. The count varies with the building layer pulled.

## Checkpoint answers

- Computed zonal statistics (mean NDVI per neighborhood) ✅
- Performed a spatial join (buildings near parks) ✅
- Ran an overlay operation (residential ∩ flood plain) ✅
- Chained 3+ MCP tools in a single analysis workflow ✅
