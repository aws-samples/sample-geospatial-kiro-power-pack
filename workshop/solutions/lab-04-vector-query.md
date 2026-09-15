# Lab 04 — Vector data queries (solution)

Server: `geo-vector` · Starter data: `exercises/02-vector-query/denver_aoi.geojson`

Live OSM/Overture data changes; counts and attributes below are representative shapes, not fixed values.

## Exercise 1 — Query building footprints

**Prompt:** *"Find all building footprints within 200 meters of the coordinate (-104.99, 39.74) in Denver"*

**Tool:** `geo-vector` → spatial query against OpenStreetMap / Overture Maps.

**Expected output:** a GeoJSON FeatureCollection of building polygons — typically dozens within a 200 m radius of downtown Denver. Each feature carries attributes like `building` (type), `height`/`levels`, and sometimes `name`. Many buildings have partial attributes; missing height is normal in OSM.

## Exercise 2 — Query land use zones

**Prompt:** *"What are the land-use zones within 1 kilometer of Denver's Civic Center Park?"*

**Tool:** `geo-vector` → land-use polygon query.

**Expected output:** polygons tagged `residential`, `commercial`, `retail`, `industrial`, `park`/`grass`, etc. Around Civic Center you'll see a commercial/government core ringed by residential and parkland. Coverage varies by how well the area is mapped.

## Exercise 3 — Administrative boundaries

**Prompt:** *"Get the boundary polygon for Denver County, Colorado"*

**Tool:** `geo-vector` → administrative boundary lookup.

**Expected output:** a single (multi)polygon for Denver County (~401 km²), including the detached DIA panhandle to the northeast — a good reminder that the county outline is not a simple rectangle.

**Follow-up prompt:** *"Now get the boundary for all counties in the Denver metro area (Denver, Adams, Arapahoe, Jefferson, Douglas)"* → five county polygons returned as a FeatureCollection.

## Exercise 4 — Combine vector + STAC search

**Prompt:** *"Get the bounding box of Denver County, then search for Sentinel-2 imagery covering that exact area from last month"*

**Tools (chained):** `geo-vector` (boundary → bbox) then `geo-stac` → `stac_search` with that bbox.

**Expected output:** a Denver County bbox near `[-105.11, 39.61, -104.60, 39.91]`, then a handful of Sentinel-2 L2A scenes covering it from the last month. The point is the chaining — Kiro extracts the bbox from the vector result and feeds it to the STAC search without you copying coordinates.

## Exercise 5 — Export for later use

**Prompt:** *"Save the Denver County boundary as a GeoJSON file called denver_boundary.geojson in my exercises directory"*

**Tool:** `geo-vector` result written to `exercises/02-vector-query/denver_boundary.geojson`.

**Expected output:** a valid GeoJSON file with the Denver County boundary. Later labs (zonal stats, capstone ETL) reuse this file as their area of interest, so confirm it exists and opens as one county polygon.

## Checkpoint answers

- Queried building footprints and received GeoJSON polygons ✅
- Understood points vs. lines vs. polygons ✅
- Chained `geo-vector` + `geo-stac` in one workflow ✅
- Saved `denver_boundary.geojson` for later labs ✅
