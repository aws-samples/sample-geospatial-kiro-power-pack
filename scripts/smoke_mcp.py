#!/usr/bin/env python3
"""Layer-2/3 smoke test: MCP stdio handshake + local tool calls.

For each installed server console command, this:

* launches it as a subprocess over MCP stdio (exactly how Kiro runs it from a
  ``mcp.json`` ``command`` entry),
* performs the MCP ``initialize`` + ``tools/list`` handshake and asserts the
  server advertises at least one tool (layer 2 - validates the
  ``geo_common.runtime`` serving wiring end to end), and
* optionally calls one **no-network, local** tool and asserts a sane result
  (layer 3 - validates the ``call_tool`` dispatch + result serialization).

It requires the ``mcp`` SDK and the target servers installed in the current
environment (e.g. an editable install in a Python >=3.10 venv). It needs no
network for the default targets.

Usage (from the repo root, with the servers installed in the active venv):

    python scripts/smoke_mcp.py                     # default local-only targets
    python scripts/smoke_mcp.py geo-index geo-ops   # specific servers

Each named server must be on PATH as a console script (as installed by
``uv pip install -e ./packages/<server>``). Exit code is non-zero on any
failure, so it is CI-friendly.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:  # pragma: no cover - guidance when the SDK is absent
    sys.stderr.write(
        "The 'mcp' SDK is required. In a Python >=3.10 venv run:\n"
        "  pip install mcp\n"
        "  pip install -e ./packages/geo-common -e ./packages/<server>\n"
    )
    raise SystemExit(2)

#: Optional layer-3 probe per server: a tool that needs no network and the
#: arguments to call it with, plus a predicate validating the returned text.
#: Servers not listed here are still handshake-tested (layer 2) but not called.
LocalCall = Tuple[str, Dict[str, Any], "Any"]

LOCAL_CALLS: "Dict[str, LocalCall]" = {
    "geo-index": (
        "index_cell",
        {"lon": -122.4194, "lat": 37.7749, "scheme": "h3", "resolution": 9},
        lambda text: isinstance(text, str) and len(text) >= 3,
    ),
    "geo-ops": (
        "transform_crs",
        {
            "geometry": {"type": "Point", "coordinates": [-122.4194, 37.7749]},
            "src_crs": "EPSG:4326",
            "dst_crs": "EPSG:3857",
        },
        lambda text: "Point" in text or "coordinates" in text,
    ),
    "geo-embedding-search": (
        "store_embedding",
        {"embedding": [0.1, 0.2, 0.3], "metadata": {"area": "smoke"}},
        lambda text: '"retrievable": true' in text.replace(" ", "")
        or "retrievable" in text,
    ),
}

#: Default targets: local, no-network servers safe to call in CI.
DEFAULT_TARGETS = ["geo-index"]


#: Credentialed servers: a tool to invoke with NO credentials configured, plus a
#: substring that must appear in the returned auth error (the mcp.json key, or
#: "authentication"). Verifies the credential guard fires over MCP without any
#: account. ``aws-geo-compute`` is intentionally absent (its credential is
#: Optional in code, so it serves and is handshake-only).
AUTH_GUARD_CALLS: "Dict[str, Tuple[str, Dict[str, Any], str]]" = {
    "geo-commercial-imagery": (
        "search_commercial",
        {
            "provider": "MAXAR",
            "bbox": [-122.6, 37.6, -122.3, 37.9],
            "datetime_range": ["2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z"],
        },
        "SENTINELHUB",
    ),
    "geo-warehouse": (
        "warehouse_spatial_sql",
        {"query": "SELECT 1", "engine": "bigquery", "connection": "proj.dataset"},
        "BIGQUERY_CREDENTIALS",
    ),
    "aws-geo-compute": (
        "submit",
        {"task_type": "bulk-cog-conversion"},
        "AWS_ACCESS_KEY_ID",
    ),
}


async def _check_server(command: str) -> Tuple[bool, str]:
    resolved = _resolve_command(command)
    if resolved is None:
        return False, "console command %r not found (install it in this env first)" % command

    params = StdioServerParameters(command=resolved, args=[])
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                listed = await session.list_tools()
                tool_names = [t.name for t in listed.tools]
                if not tool_names:
                    return False, "server advertised no tools"

                detail = "tools/list -> %s" % ", ".join(tool_names)

                probe = LOCAL_CALLS.get(command)
                if probe is not None:
                    tool_name, args, predicate = probe
                    if tool_name not in tool_names:
                        return False, "%s; expected tool %r not advertised" % (detail, tool_name)
                    result = await session.call_tool(tool_name, args)
                    text = _first_text(result)
                    if text is None:
                        return False, "%s; call_tool(%s) returned no text content" % (detail, tool_name)
                    if not predicate(text):
                        return False, "%s; call_tool(%s) unexpected result: %.80r" % (detail, tool_name, text)
                    detail += "; call_tool(%s) OK -> %.60r" % (tool_name, text)

                guard = AUTH_GUARD_CALLS.get(command)
                if guard is not None:
                    tool_name, args, expected = guard
                    if tool_name not in tool_names:
                        return False, "%s; expected tool %r not advertised" % (detail, tool_name)
                    is_error, text = await _call_expecting_error(session, tool_name, args)
                    if not is_error:
                        return False, "%s; call_tool(%s) without creds did NOT error" % (detail, tool_name)
                    low = text.lower()
                    if expected.lower() not in low and "authentication" not in low:
                        return False, (
                            "%s; auth error missing %r: %.80r" % (detail, expected, text)
                        )
                    detail += "; auth guard OK -> names %s" % expected

                return True, detail
    except Exception as exc:  # noqa: BLE001 - report, don't crash the runner
        return False, "handshake failed: %s: %s" % (type(exc).__name__, exc)


def _resolve_command(command: str) -> Optional[str]:
    """Resolve a server console command to an executable path.

    Checks ``PATH`` first, then falls back to the directory of the running
    interpreter (the venv's ``bin/``), so the smoke test works whether or not
    the venv is "activated" - e.g. when invoked as ``.venv/bin/python
    scripts/smoke_mcp.py``.
    """
    found = shutil.which(command)
    if found:
        return found
    # Look in the active environment's script dir. Use sys.prefix (the venv
    # root) rather than resolving sys.executable, which would follow the venv's
    # python symlink back out to the base interpreter.
    for bindir in ("bin", "Scripts"):
        candidate = Path(sys.prefix) / bindir / command
        if candidate.exists():
            return str(candidate)
    return None


def _first_text(call_result: Any) -> Optional[str]:
    """Return the first text-content payload from a call_tool result."""
    for item in getattr(call_result, "content", []) or []:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            return text
    return None


async def _call_expecting_error(session: Any, tool_name: str, args: Dict[str, Any]) -> Tuple[bool, str]:
    """Call a tool expecting failure; return (is_error, combined_text).

    The MCP runtime surfaces a tool's ``GeoError`` either as a result with
    ``isError`` true (text content carrying the message) or, depending on SDK
    version, as a raised exception. Both are treated as the error signal so the
    auth-guard assertion is robust.
    """
    try:
        result = await session.call_tool(tool_name, args)
    except Exception as exc:  # noqa: BLE001 - the error path we are verifying
        return True, "%s: %s" % (type(exc).__name__, exc)

    texts = [
        getattr(item, "text", "")
        for item in (getattr(result, "content", []) or [])
        if getattr(item, "text", None)
    ]
    return bool(getattr(result, "isError", False)), " ".join(texts)


async def _run(targets: List[str]) -> int:
    failures = 0
    for command in targets:
        ok, detail = await _check_server(command)
        print("[%s] %-20s %s" % ("PASS" if ok else "FAIL", command, detail))
        if not ok:
            failures += 1
    print(
        "\n%d/%d servers passed the MCP handshake smoke test."
        % (len(targets) - failures, len(targets))
    )
    return 1 if failures else 0


def main() -> int:
    targets = sys.argv[1:] or DEFAULT_TARGETS
    return asyncio.run(_run(targets))


if __name__ == "__main__":
    raise SystemExit(main())
