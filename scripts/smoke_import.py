#!/usr/bin/env python3
"""Layer-1 smoke test: import + construction across every package.

This is the cheapest deploy-confidence check for the Geospatial Power Pack. For
every package under ``packages/`` it:

* puts the package source root on ``sys.path`` (so it works against a source
  checkout without an editable install),
* imports the package and its ``<module>.server`` entry-point module,
* asserts a ``main`` callable exists (the ``[project.scripts]`` target), and
* for the domain/peer servers, constructs the server class and asserts its
  tools register and its catalog/credential declarations resolve.

It needs no network, no AWS, and not even the ``mcp`` SDK (the MCP runtime is
imported lazily only when a server actually serves). It catches the most common
breakage: a package that won't import, a renamed/missing server class, or a
broken console entry point.

Run from the repo root:

    python3 scripts/smoke_import.py

Exit code is non-zero if any package fails, so it is CI-friendly.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"

#: Packages that are libraries / the Power hub, not tool MCP servers: we only
#: assert they import and expose a ``main`` entry point (no server class).
NON_SERVER_PACKAGES = {"geo-common", "kiro-geospatial"}


def _package_dirs() -> List[Path]:
    return sorted(p for p in PACKAGES_DIR.iterdir() if p.is_dir() and (p / "pyproject.toml").exists())


def _module_name(package_dir: Path) -> str:
    """``geo-foundation-models`` -> ``geo_foundation_models``."""
    return package_dir.name.replace("-", "_")


def _find_server_class(server_module) -> Optional[type]:
    """Return the BaseGeoServer subclass defined in ``server_module``, if any."""
    from geo_common.server import BaseGeoServer

    bases = {BaseGeoServer}

    candidates = []
    for _, obj in inspect.getmembers(server_module, inspect.isclass):
        if not issubclass(obj, BaseGeoServer) or obj in bases:
            continue
        # Only classes defined in this package (not imported base classes).
        if obj.__module__.startswith(server_module.__name__.split(".")[0]):
            candidates.append(obj)
    # Prefer a class whose module is exactly the server module.
    candidates.sort(key=lambda c: c.__module__ != server_module.__name__)
    return candidates[0] if candidates else None


def _check_package(package_dir: Path) -> Tuple[bool, str]:
    name = package_dir.name
    module_name = _module_name(package_dir)

    # Make this package's source importable without an editable install.
    src = str(package_dir)
    if src not in sys.path:
        sys.path.insert(0, src)

    try:
        importlib.import_module(module_name)
        server_module = importlib.import_module("%s.server" % module_name)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the runner
        return False, "import failed: %s: %s" % (type(exc).__name__, exc)

    if not callable(getattr(server_module, "main", None)):
        return False, "missing callable main() in %s.server" % module_name

    if name in NON_SERVER_PACKAGES:
        return True, "import OK; main() present (library/hub)"

    server_cls = _find_server_class(server_module)
    if server_cls is None:
        return False, "no BaseGeoServer subclass found in %s.server" % module_name

    try:
        server = server_cls()
        tools = server.tool_names()
        catalog = server.catalog_entries()
        creds = server.required_credentials()
    except Exception as exc:  # noqa: BLE001
        return False, "construct/declare failed: %s: %s" % (type(exc).__name__, exc)

    if not tools:
        return False, "%s registered no tools" % server_cls.__name__

    return True, "%s: tools=%d, catalog=%d, creds=%d" % (
        server_cls.__name__,
        len(tools),
        len(catalog),
        len(creds),
    )


def main() -> int:
    # geo-common must be importable first (every server depends on it).
    sys.path.insert(0, str(PACKAGES_DIR / "geo-common"))

    failures = 0
    for package_dir in _package_dirs():
        ok, detail = _check_package(package_dir)
        status = "PASS" if ok else "FAIL"
        print("[%s] %-24s %s" % (status, package_dir.name, detail))
        if not ok:
            failures += 1

    total = len(_package_dirs())
    print("\n%d/%d packages passed the import+construction smoke test." % (total - failures, total))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
