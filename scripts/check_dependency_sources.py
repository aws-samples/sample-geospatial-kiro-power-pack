#!/usr/bin/env python3
"""Check that nothing in the pack can resolve one of its own packages from PyPI.

None of this pack's packages (``geo-common``, the servers, the hub) are
published on PyPI. A bare name - ``geo-common>=0.1`` in a ``pyproject.toml``,
or ``uvx geo-stac`` in the manifest / docs - therefore resolves against the
public index by default, so anyone who registers that name on PyPI could get
their code installed on a developer, CI, or agent host (dependency confusion;
reported via the AWS VDP). One name (``geo-raster``) is already taken there by
an unrelated project.

Two rules keep every install path local to the repository checkout:

1. **Dependency pin.** Every package that depends on ``geo-common`` pins it to
   the in-repo path via ``[tool.uv.sources]`` so ``uv pip install``,
   ``uv tool install`` and ``uvx --from <path>`` never consult PyPI for it.
2. **Install command.** Every server's ``bundle-manifest.json`` ``uvx`` entry
   is the path-based ``uvx --from ./packages/<name> <name>`` (the drift check
   separately keeps each server's ``INSTALL_COMMAND`` equal to it), never a
   bare ``uvx <name>``. The ``geo-common`` entry is a placeholder, not a
   command, and is exempt. Companion servers (third-party, really on PyPI)
   use the ``install`` key and are out of scope.

Wire it into ``make check`` / CI:

    python scripts/check_dependency_sources.py

Exit code is non-zero if any rule is violated.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGES = _ROOT / "packages"
_MANIFEST = _ROOT / "bundle-manifest.json"

#: Dependencies that live only in this repo and must never come from PyPI.
_IN_REPO_ONLY = {"geo-common": "../geo-common"}

#: Manifest servers whose ``uvx`` field is descriptive text, not a command.
_NO_INSTALL_COMMAND = {"geo-common"}

_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _dep_name(requirement: str) -> str:
    """Return the normalized project name from a PEP 508 requirement string."""
    match = _NAME_RE.match(requirement)
    return match.group(1).lower().replace("_", "-") if match else ""


def _declared_deps(project: Dict[str, Any]) -> List[str]:
    deps = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        deps.extend(extra)
    return deps


def _expected_install_command(name: str) -> str:
    """Mirror of ``kiro_geospatial.installation.install_command`` (no import needed)."""
    return "uvx --from ./packages/%s %s" % (name, name)


def check_dependency_pins(problems: List[str]) -> int:
    """Rule 1: in-repo-only dependencies are pinned to their local path."""
    checked = 0
    for pyproject in sorted(_PACKAGES.glob("*/pyproject.toml")):
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = data.get("project", {})
        names = {_dep_name(d) for d in _declared_deps(project)}
        sources = data.get("tool", {}).get("uv", {}).get("sources", {})
        rel = pyproject.relative_to(_ROOT)
        for dep, expected_path in _IN_REPO_ONLY.items():
            if dep not in names:
                continue
            checked += 1
            source = sources.get(dep)
            if not isinstance(source, dict) or "path" not in source:
                problems.append(
                    "%s: depends on %r without a [tool.uv.sources] path entry "
                    "(would resolve from public PyPI)" % (rel, dep)
                )
            elif Path(source["path"]).as_posix() != expected_path:
                problems.append(
                    "%s: [tool.uv.sources] %s path is %r, expected %r"
                    % (rel, dep, source["path"], expected_path)
                )
            elif not (pyproject.parent / source["path"] / "pyproject.toml").exists():
                problems.append(
                    "%s: [tool.uv.sources] %s path %r does not exist"
                    % (rel, dep, source["path"])
                )
    return checked


def check_install_commands(problems: List[str]) -> int:
    """Rule 2: manifest ``uvx`` commands install from ``./packages/<name>``."""
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    checked = 0
    for entry in manifest.get("servers", []):
        name = entry["name"]
        if name in _NO_INSTALL_COMMAND:
            continue
        checked += 1
        command = str(entry.get("uvx", "")).strip()
        expected = _expected_install_command(name)
        if command != expected:
            problems.append(
                "bundle-manifest.json: server %r uvx is %r, expected %r "
                "(bare names resolve from public PyPI)" % (name, command, expected)
            )
        if not (_PACKAGES / name / "pyproject.toml").exists():
            problems.append(
                "bundle-manifest.json: server %r has no packages/%s/pyproject.toml"
                % (name, name)
            )
    return checked


def main() -> int:
    problems: List[str] = []
    pinned = check_dependency_pins(problems)
    commands = check_install_commands(problems)

    if problems:
        print("Dependency-source check FAILED:")
        for problem in problems:
            print("  - " + problem)
        return 1
    print(
        "Dependency-source check OK: %d package(s) pin geo-common to the in-repo "
        "path; %d manifest install command(s) are path-based." % (pinned, commands)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
