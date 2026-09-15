# Lab 11 — Change detection (temporal analysis) (solution)

Server: `geo-foundation-models` (`change_detect`) · Starter data: `exercises/08-change-detection/baseline_tiles.json`

The model backend is a deterministic stand-in, so change values are reproducible; the patterns below are the expected outcome.

## Exercise 1 — Detect seasonal change

**Prompt:** *"Run change detection between January 2024 and June 2024 for the Denver metro area (-105.1, 39.6, -104.8, 39.9). Use a threshold of 0.3. What changes does it find?"*

**Tool:** `geo-foundation-models` → `change_detect` (threshold 0.3).

**Expected output:** change polygons over vegetation green-up (deciduous bare → canopy), agricultural fields (bare soil → crops), and snow melt in the foothills. The 0.3 threshold catches moderate seasonal change; stable built areas stay below it.

## Exercise 2 — Urban development detection

**Prompt:** *"Run change detection on a known development area. Use the area around Denver International Airport where new construction has been ongoing. Compare 2023 to 2024. Set threshold to 0.4 to catch only significant structural changes."*

**Tool:** `geo-foundation-models` → `change_detect` (threshold 0.4).

**Expected output:** high-magnitude change where bare soil became buildings/pavement near DIA; agricultural→suburban transitions register moderate-to-high; already-built parcels fall below 0.4. Raising the threshold filters out seasonal noise so only structural change survives.

## Exercise 3 — Multi-area comparison

**Prompt:** *"Run change detection for 3 different areas: 1) Downtown Denver (stable urban), 2) Eastern suburbs (active development), 3) Mountain foothills (natural). Which area shows the most change? Does this match expectations?"*

**Tool:** `geo-foundation-models` → `change_detect` on three AOIs.

**Expected output:** ranking — **eastern suburbs** highest (active construction), mountain foothills moderate (seasonal vegetation), **downtown** lowest (stable built environment). Development happens at the margins, not the core.

## Exercise 4 — Quantify change magnitude

**Prompt:** *"For the highest-change area from Exercise 3, show me the change magnitude values and the dominant change type classification. What physical process does this represent?"*

**Tool:** `geo-foundation-models` → `change_detect` detail on the eastern-suburbs AOI.

**Expected output:** `change_polygons` (GeoJSON), `change_magnitude` on a 0–1 scale, and `dominant_change_type: urbanization` — representing bare soil / agriculture converting to built structures.

## Exercise 5 — Build a monitoring baseline

**Prompt:** *"Create a baseline embedding for a forest area west of Denver. Then compare it against a more recent observation. If the similarity drops below 0.85, flag it as potential deforestation."*

**Tool:** `geo-foundation-models` → embed baseline, embed monitor tile, compare.

**Expected output:** a stable forest yields similarity above 0.85 (no alert); a drop **below 0.85** flags potential deforestation for investigation. This baseline → monitor → alert loop is the template for any early-warning system.

## Checkpoint answers

- Ran change detection between two time periods ✅
- Can interpret change magnitude values ✅
- Compared change rates across land-use types (suburbs > foothills > downtown) ✅
- Understood the baseline-monitoring pattern for alerts (similarity < 0.85) ✅
