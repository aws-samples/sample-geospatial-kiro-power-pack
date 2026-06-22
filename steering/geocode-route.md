---
name: geocode-route
description: >-
  Geocode two addresses and compute a route between them:
  geocode origin+destination -> route -> emit GeoJSON.
inclusion: fileMatch
fileMatchPattern:
  - "**/*geocode*"
  - "**/*route*"
  - "**/*routing*"
---

# Steering Workflow: Combined Geocode and Route

Turn an origin and destination address into coordinates, compute a route between
them, and emit the result as GeoJSON.

- Provider tools: `geo-geocode-route.geocode`, `geo-geocode-route.route`
- Requirements: 7.4 (address → coordinate ≤10s), 7.10 (origin + destination →
  route ≤30s), 7.12 (reject unparseable addresses with a validation error),
  12.3 (tabular vector output is GeoParquet — apply when persisting tabular
  results)
- Related skills: `crs-handling`, `tool-selection`

## Prerequisites

- Python 3.10 or higher.
- The `geo-geocode-route` server installed and registered in `mcp.json` (e.g.
  `uvx geo-geocode-route`), which provides `geocode` and `route`.
- No credentials required for the open defaults (Nominatim, OSRM, Valhalla);
  configure a `source` credential only to select Amazon Location.
- Familiarity with the supported routing profiles (car, bike, foot) and with
  GeoJSON output.

## Activation

Active when the open/edited file matches `**/*geocode*`, `**/*route*`, or
`**/*routing*`. If steps cannot be presented, deactivate and report this
workflow name plus the reason. Deactivate when the triggering file no longer
matches.

## Steps (in order)

1. **Validate the addresses.** Confirm the origin and destination addresses are
   present and parseable, and choose a routing `profile` (e.g. `car`, `bike`,
   `foot`). Reject empty or unparseable addresses with an `Error_Taxonomy`
   validation error before any lookup (Requirement 7.12).
2. **Geocode the origin.** Call `geo-geocode-route.geocode(address=origin)` to
   obtain the origin coordinate within 10s (Requirement 7.4).
3. **Geocode the destination.** Call
   `geo-geocode-route.geocode(address=destination)` to obtain the destination
   coordinate within 10s.
4. **Compute the route.** Call
   `geo-geocode-route.route(origin=..., destination=..., profile=...)` to get the
   route geometry and summary (distance, duration) within 30s
   (Requirement 7.10).
5. **Emit GeoJSON.** Build a GeoJSON `FeatureCollection` containing the origin
   point, the destination point, and the route `LineString`, with properties for
   distance and duration. When persisting a tabular form, write GeoParquet
   (Requirement 12.3).

## Notes

- Geocoding results are in WGS84 (EPSG:4326); keep coordinates in that CRS for
  GeoJSON output and reproject explicitly if a downstream step needs another CRS
  (`crs-handling` skill).
- If a configured provider is unreachable within the time budget, surface the
  `Error_Taxonomy` availability error rather than returning a partial route.
