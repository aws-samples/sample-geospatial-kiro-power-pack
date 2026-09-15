# Lab 07 — Raster band math (NDVI, NDWI, NDBI) (solution)

Server: `geo-raster` · Starter data: `exercises/04-band-math/scene_urls.json`

Pixel values vary by scene and date; the ranges below are what correct indices look like over Denver.

## Exercise 1 — Calculate NDVI

**Prompt:** *"Calculate NDVI for a Sentinel-2 scene over Denver using the formula (B08 - B04) / (B08 + B04). Use a scene from June 2024. Show me the value range and explain what different values mean."*

**Tool:** `geo-raster` → `band_math` reading B08 (NIR) and B04 (Red) from the S3 COGs.

**Expected output:** a single-band NDVI raster with values in `[-1, +1]`. Over June Denver you'll see parks/irrigated areas at 0.6–0.9, grassland at 0.3–0.6, and downtown/water below 0.1 (water goes negative). The explanation should tie high NIR + low Red to healthy vegetation.

## Exercise 2 — Calculate NDWI

**Prompt:** *"Calculate NDWI using the formula (B03 - B08) / (B03 + B08) for the same Denver scene. Where does NDWI show water?"*

**Tool:** `geo-raster` → `band_math` with B03 (Green) and B08 (NIR).

**Expected output:** an NDWI raster where open water (Cherry Creek Reservoir, Sloan's Lake, the South Platte) reads > 0.3, moist soil sits near 0, and dry land/vegetation goes negative — the inverse pattern of NDVI over the same pixels.

## Exercise 3 — Windowed read (spatial subset)

**Prompt:** *"Read only a 5km x 5km window of the NIR band centered on Denver's City Park (-104.955, 39.748). How many pixels does this represent at 10m resolution?"*

**Tool:** `geo-raster` → windowed COG read (HTTP byte-range).

**Expected output:** 5000 m ÷ 10 m = 500 pixels per side, so a 500 × 500 window = **250,000 pixels**. The read pulls ~1 MB instead of the full 100+ MB band — the payoff of the COG format.

## Exercise 4 — Multiple indices comparison

**Prompt:** *"For the same Denver scene, calculate both NDVI and NDBI (Normalized Difference Built-up Index: (B11 - B08) / (B11 + B08)). Where are the highest NDBI values? Do they inversely correlate with high NDVI?"*

**Tool:** `geo-raster` → `band_math` twice (NDVI, then NDBI with B11 SWIR + B08 NIR).

**Expected output:** highest NDBI over downtown/Five Points and industrial corridors; those same pixels have low NDVI, while City Park has the reverse — a clear inverse correlation. Note B11 is 20 m and must be resampled to align with the 10 m NIR grid.

## Exercise 5 — Time-series index

**Prompt:** *"Calculate NDVI for Denver in January 2024 and June 2024. What's the seasonal difference? Where is the change most dramatic?"*

**Tool:** `geo-raster` → `band_math` on a January scene and a June scene, then differenced.

**Expected output:** June NDVI well above January across vegetated zones; the biggest jump is in parks, golf courses, and irrigated/agricultural land (bare/dormant → full canopy), while downtown barely changes. Exact deltas depend on the chosen scenes and cloud cover.

## Checkpoint answers

- Calculated NDVI and can interpret the value range ✅
- Calculated NDWI and identified water bodies ✅
- Performed a windowed read (500×500 = 250,000 pixels) of a COG ✅
- Compared indices between two time periods ✅
