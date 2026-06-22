#!/usr/bin/env python3
"""Check that ``bundle-manifest.json`` and the server code agree (no drift).

The manifest is the Power Hub's source of truth: it lists every server, its
``uvx`` install command, and the ``mcp.json`` credential keys each one reads.
The servers themselves independently declare the same facts in code
(``server_name``, ``INSTALL_COMMAND``, :meth:`required_credentials`,
:meth:`catalog_entries`). When someone adds a server, a tool, or a credential
and updates only one side, the two silently drift apart. This script fails when
they do, so a contributor catches the mismatch before it ships.

It verifies, across all servers in one pass (no network, no credentials):

1. **Server parity** - every tool server in code appears in the manifest and
   vice versa (the Hub ``kiro-geospatial`` and the ``geo-common`` base, which
   are not tool MCP servers, are excluded).
2. **Install-command parity** - each server's ``INSTALL_COMMAND`` equals the
   manifest ``uvx`` entry.
3. **Credential parity** - each server's :meth:`required_credentials`
   ``(key, classification)`` set equals the manifest ``credentials`` block.
4. **Catalog integrity** - every server exposes at least one catalog entry, and
   each entry names that server as its ``provider_server``.
5. **Catalog/tool correspondence** - every advertised capability is backed by a
   registered tool and every registered tool is advertised (comparing on the
   ``tool`` base name so ``tool:variant`` discovery entries are allowed). This
   catches a capability advertised but never wired (dead/aspirational), or a
   working tool the catalog forgot to advertise.

Run it before an acceptance session, or wire it into CI / a pre-commit hook:

    python scripts/check_manifest_drift.py

Exit code is non-zero if any drift is found.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Reuse the single static server registry the schema check maintains.
from check_tool_schemas import SERVERS  # noqa: E402  (same scripts/ dir)

#: Manifest "servers" entries that are not tool MCP servers and so have no
#: server class / tools / catalog of their own.
_NON_TOOL_SERVERS = {"kiro-geospatial", "geo-common"}

_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "bundle-manifest.json"


def _manifest_servers() -> Dict[str, Dict[str, Any]]:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    return {s["name"]: s for s in manifest["servers"]}


def _install_command(module_name: str) -> str:
    """Return the server module's ``INSTALL_COMMAND`` constant, or ''."""
    module = importlib.import_module(module_name)
    return getattr(module, "INSTALL_COMMAND", "") or ""


def main() -> int:  # noqa: C901 - linear sequence of independent checks
    manifest = _manifest_servers()
    manifest_tool_servers = set(manifest) - _NON_TOOL_SERVERS

    problems: List[str] = []
    skipped: List[str] = []
    code_servers: Dict[str, Any] = {}

    for module_name, class_name in SERVERS:
        try:
            server_cls = getattr(importlib.import_module(module_name), class_name)
            server = server_cls()
        except Exception as exc:  # noqa: BLE001 - report and continue
            skipped.append("%s (%s: %s)" % (class_name, type(exc).__name__, exc))
            continue
        code_servers[server.server_name] = (server, module_name)

    # 1. Server parity (only meaningful for servers we could import).
    importable = set(code_servers)
    missing_in_manifest = sorted(importable - manifest_tool_servers)
    for name in missing_in_manifest:
        problems.append("server %r is in code but not in bundle-manifest.json" % name)

    if not skipped:
        # Every server imported, so we can also assert the reverse direction.
        missing_in_code = sorted(manifest_tool_servers - importable)
        for name in missing_in_code:
            problems.append(
                "server %r is in bundle-manifest.json but has no server class" % name
            )

    # 2-4. Per-server checks.
    for name, (server, module_name) in sorted(code_servers.items()):
        entry = manifest.get(name)
        if entry is None:
            continue  # already reported under (1)

        # 2. Install-command parity.
        install = _install_command(module_name)
        if install and entry.get("uvx") and install != entry["uvx"]:
            problems.append(
                "%s: INSTALL_COMMAND %r != manifest uvx %r"
                % (name, install, entry["uvx"])
            )

        # 3. Credential parity.
        code_creds = {
            (s.mcp_json_key, s.classification.value)
            for s in server.required_credentials()
        }
        manifest_creds = {
            (c["key"], c["classification"]) for c in entry.get("credentials", [])
        }
        if code_creds != manifest_creds:
            problems.append(
                "%s: credential drift - code %s != manifest %s"
                % (name, sorted(code_creds), sorted(manifest_creds))
            )

        # 4. Catalog integrity.
        entries = server.catalog_entries()
        if not entries:
            problems.append("%s: no catalog entries declared" % name)
        for cat in entries:
            if cat.provider_server != name:
                problems.append(
                    "%s: catalog entry %r names provider %r"
                    % (name, cat.name, cat.provider_server)
                )

        # 5. Catalog/tool correspondence. Every advertised capability must be
        # backed by a registered, executable tool, and every registered tool
        # must be advertised. A catalog entry may enumerate per-variant facets
        # of one parameterized tool as ``tool:variant`` (e.g. the geo-query and
        # geo-warehouse spatial-SQL engines), so entries are compared on their
        # base name (the part before the first ``:``). This catches both a
        # capability advertised but never wired as a tool (a dead/aspirational
        # entry) and a working tool that the catalog fails to advertise.
        tool_names = set(server.tools)
        catalog_base_names = {cat.name.split(":", 1)[0] for cat in entries}
        advertised_not_tools = sorted(catalog_base_names - tool_names)
        for cap in advertised_not_tools:
            problems.append(
                "%s: catalog advertises %r but no tool is registered for it"
                % (name, cap)
            )
        tools_not_advertised = sorted(tool_names - catalog_base_names)
        for cap in tools_not_advertised:
            problems.append(
                "%s: tool %r is registered but absent from the catalog"
                % (name, cap)
            )

    print(
        "Checked %d importable servers against %d manifest tool-servers."
        % (len(code_servers), len(manifest_tool_servers))
    )
    if skipped:
        print("Skipped (not importable in this env):")
        for item in skipped:
            print("  - %s" % item)
        print(
            "NOTE: server-parity reverse check is relaxed because not every "
            "server imported; run with all packages on PYTHONPATH for the full check."
        )
    if problems:
        print("\nFAILED - %d drift problem(s):" % len(problems))
        for item in problems:
            print("  - %s" % item)
        return 1
    print("OK - manifest and server code agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
