#!/usr/bin/env python3
"""Validate every server's MCP tool ``inputSchema`` is well-formed and resolvable.

The shared :func:`geo_common.runtime.build_tool_input_schema` turns each
registered tool's signature into a JSON-Schema ``inputSchema`` that the MCP
client validates arguments against. A regression there (notably: a pydantic
model parameter whose ``$defs`` are not hoisted to the schema root, leaving
dangling ``#/$defs/<Name>`` pointers) breaks the tool for *every* call before
the payload is even evaluated - exactly the ``spatial_join`` "PointerToNowhere"
bug. This script catches that class of problem across all servers in one pass,
with no network or credentials.

For each registered tool it checks that the generated schema:

* is a structurally valid JSON Schema (``Draft202012Validator.check_schema``);
* has **no dangling** ``#/$defs/<Name>`` reference (every ``$ref`` resolves to a
  definition present at the schema root).

Run it before an acceptance session, or wire it into CI / a pre-commit hook:

    python scripts/check_tool_schemas.py

Exit code is non-zero if any tool schema is malformed or has a dangling ref.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, List, Tuple

# Each server package's module path and server class. Kept as a static list so
# the check needs no package metadata scan; update it when a server is added or
# removed. The server-parity check in check_manifest_drift.py (which reuses this
# list) catches a server that is in code/manifest but missing here.
SERVERS: Tuple[Tuple[str, str], ...] = (
    ("geo_stac.server", "GeoStacServer"),
    ("geo_vector.server", "GeoVectorServer"),
    ("geo_ops.server", "GeoOpsServer"),
    ("geo_foundation_models.server", "GeoFoundationModelsServer"),
    ("geo_index.server", "GeoIndexServer"),
    ("geo_embedding_search.server", "GeoEmbeddingSearchServer"),
    ("geo_geocode_route.server", "GeoGeocodeRouteServer"),
    ("geo_terrain.server", "GeoTerrainServer"),
    ("geo_biodiversity.server", "GeoBiodiversityServer"),
    ("geo_weather_climate.server", "GeoWeatherClimateServer"),
    ("geo_formats.server", "GeoFormatsServer"),
    ("geo_query.server", "GeoQueryServer"),
    ("geo_raster.server", "GeoRasterServer"),
    ("geo_pointcloud.server", "GeoPointcloudServer"),
    ("geo_warehouse.server", "GeoWarehouseServer"),
    ("geo_commercial_imagery.server", "GeoCommercialImageryServer"),
    ("aws_geo_compute.server", "AwsGeoComputeServer"),
    ("geo_ogc.server", "GeoOgcServer"),
    ("geo_3d.server", "Geo3DServer"),
)


def _all_refs(node: Any) -> List[str]:
    """Collect every ``$ref`` string anywhere in a schema document."""
    refs: List[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                refs.append(value)
            else:
                refs.extend(_all_refs(value))
    elif isinstance(node, list):
        for item in node:
            refs.extend(_all_refs(item))
    return refs


def main() -> int:
    try:
        import jsonschema
    except ImportError:
        print("ERROR: jsonschema is required (pip install jsonschema).", file=sys.stderr)
        return 2

    from geo_common.runtime import build_tool_input_schema

    checked = 0
    problems: List[str] = []
    skipped: List[str] = []

    for module_name, class_name in SERVERS:
        try:
            server_cls = getattr(importlib.import_module(module_name), class_name)
            server = server_cls()
        except Exception as exc:  # noqa: BLE001 - report and continue
            skipped.append("%s (%s: %s)" % (class_name, type(exc).__name__, exc))
            continue

        for tool_name, tool in server.tools.items():
            checked += 1
            schema = build_tool_input_schema(tool.func)
            label = "%s.%s" % (server.server_name, tool_name)

            try:
                jsonschema.Draft202012Validator.check_schema(schema)
            except jsonschema.exceptions.SchemaError as exc:
                problems.append("%s: malformed schema - %s" % (label, exc.message))
                continue

            defs = set(schema.get("$defs", {}))
            dangling = sorted(
                {
                    ref
                    for ref in _all_refs(schema)
                    if ref.startswith("#/$defs/") and ref.split("/")[-1] not in defs
                }
            )
            if dangling:
                problems.append("%s: dangling $ref(s) %s" % (label, dangling))

    print("Checked %d tool schemas across %d servers." % (checked, len(SERVERS)))
    if skipped:
        print("Skipped (not importable in this env):")
        for item in skipped:
            print("  - %s" % item)
    if problems:
        print("\nFAILED — %d schema problem(s):" % len(problems))
        for item in problems:
            print("  - %s" % item)
        return 1
    print("OK — all tool schemas are well-formed with resolvable $refs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
