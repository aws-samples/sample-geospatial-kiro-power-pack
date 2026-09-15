# Lab 05 — Federated multi-catalog search (solution)

Servers: `geo-stac` / `geo-vector` · Starter data: reuses the Denver AOI from Lab 04

Catalog contents change daily; the shapes and tradeoffs below are what a correct result looks like.

## Exercise 1 — Compare data sources

**Prompt:** *"Search for the best available satellite imagery over Denver from June 2024 across all available STAC catalogs. Compare what Sentinel-2 and Landsat offer for this area."*

**Tool:** `geo-stac` → `stac_search_multi` (Element84 Earth Search, USGS LandsatLook, Microsoft Planetary Computer).

**Expected output:** merged results grouped by source — several Sentinel-2 L2A scenes at 10 m and Landsat C2 L2 scenes at 30 m. Sentinel-2's shorter revisit usually gives the most recent cloud-free scene; Landsat fills temporal gaps. The comparison should surface the 10 m vs. 30 m resolution difference explicitly.

## Exercise 2 — Find complementary data

**Prompt:** *"I need both optical imagery (Sentinel-2) and radar imagery (Sentinel-1) over Denver from the same week in June 2024. Search across catalogs to find matching temporal coverage."*

**Tool:** `geo-stac` → `stac_search_multi` across `sentinel-2-l2a` and `sentinel-1-grd`.

**Expected output:** optical scenes (color/vegetation, cloud-affected) paired with radar scenes (structure/moisture, cloud-penetrating) that fall in the same week. If the week has heavy cloud, the Sentinel-1 acquisitions are the ones that still carry usable signal — the reason to pair sensors.

## Exercise 3 — Discover available collections

**Prompt:** *"What data collections are available in the Element84 Earth Search catalog? List them with their temporal and spatial coverage."*

**Tool:** `geo-stac` → collections listing.

**Expected output:** collection IDs such as `sentinel-2-l2a`, `sentinel-2-l1c`, `sentinel-1-grd`, `landsat-c2-l2`, plus each collection's temporal extent (e.g. 2015→present for Sentinel-2) and global/near-global spatial extent.

## Exercise 4 — Resolution-appropriate selection

**Prompt:** *"I need to analyze vegetation health across the entire state of Colorado. Which available satellite product gives me the best balance of spatial coverage and temporal frequency?"*

**Tool:** reasoning over catalog metadata (no single tool call required).

**Expected answer:** for a whole state, favor coverage/frequency over fine detail — MODIS/VIIRS (250 m+, daily) is best for state-level temporal monitoring; Landsat (30 m, 16-day) is a middle ground; Sentinel-2 (10 m, 5-day) gives detail but needs many tiles to cover Colorado. The right choice depends on the analysis scale, not just "highest resolution."

## Checkpoint answers

- Searched across multiple catalogs and compared results ✅
- Understood the optical vs. radar sensor tradeoff ✅
- Identified the right data product for a given analysis scale ✅
