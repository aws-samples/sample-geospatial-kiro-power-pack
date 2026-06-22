#!/usr/bin/env python3
"""Scaffold a new Geospatial Power Pack server package.

Generates a ready-to-run server skeleton under ``packages/<name>/`` that follows
the pack's conventions: a :class:`~geo_common.server.BaseGeoServer` subclass with
one registered tool, a pydantic result model, a Resource Catalog entry, an empty
credential declaration, an ``INSTALL_COMMAND``, a ``pyproject.toml`` with the
``uvx`` console script, and a test module covering the **required test trio**
(registration/catalog, a happy-path call, and a validation-error guard).

Usage::

    python scripts/new_server.py --name geo-foo --pillar A --tool do_thing
    make new-server NAME=geo-foo PILLAR=A TOOL=do_thing

After scaffolding, follow the printed next steps: add the server to the
``SERVERS`` list in ``scripts/check_tool_schemas.py``, add a ``servers`` entry to
``bundle-manifest.json``, editable-install it, and run the checks. See
``CONTRIBUTING.md`` for the full contract.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGES = _REPO_ROOT / "packages"


def _camel(module: str) -> str:
    """``geo_foo_bar`` -> ``GeoFooBar``."""
    return "".join(part.capitalize() for part in module.split("_"))


def _render(template: str, ctx: Dict[str, str]) -> str:
    out = template
    for key, value in ctx.items():
        out = out.replace("{{%s}}" % key, value)
    return out


_PYPROJECT = '''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{{NAME}}"
version = "0.1.0"
description = "Pillar {{PILLAR}} MCP server: {{TOOL}} (scaffolded skeleton - edit me)."
requires-python = ">=3.10"
license = "MIT-0"
authors = [{ name = "Geospatial Power Pack Contributors" }]
keywords = ["kiro", "geospatial", "mcp"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: GIS",
]
dependencies = [
    "geo-common>=0.1",
    "httpx>=0.27",
    "pydantic>=2",
    "mcp>=1.0,<2",
]

[project.optional-dependencies]
dev = ["pytest>=8.0,<9", "pytest-asyncio>=0.23,<1", "hypothesis>=6.100"]

[project.scripts]
{{NAME}} = "{{MODULE}}.server:main"

[tool.hatch.build.targets.wheel]
packages = ["{{MODULE}}"]
'''


_MODELS = '''"""Data models for the ``{{NAME}}`` server (scaffolded skeleton).

Define the pydantic result type(s) your tool returns here. The example
:class:`{{TOOL_TITLE}}Result` is a minimal placeholder - replace its fields
with the real shape of your capability's output.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["{{TOOL_TITLE}}Result"]


class {{TOOL_TITLE}}Result(BaseModel):
    """The result returned by the ``{{TOOL}}`` tool (scaffolded placeholder)."""

    echo: str = Field(description="Echoes the validated input (replace me).")
    source: str = Field(description="The server that produced this result.")
'''


_SERVER = '''"""The ``{{NAME}}`` MCP server (Pillar {{PILLAR}}; scaffolded skeleton).

Wires a single ``{{TOOL}}`` tool into the shared
:class:`~geo_common.server.BaseGeoServer` contract. This is a generated
skeleton: replace the placeholder logic, model, and catalog description with the
real capability, and keep the error-taxonomy and validation conventions
(see ``CONTRIBUTING.md``).
"""

from __future__ import annotations

from typing import List

from geo_common.errors import ValidationError
from geo_common.models import CatalogEntry, CredentialSpec, OpennessTier
from geo_common.server import BaseGeoServer

from {{MODULE}}.models import {{TOOL_TITLE}}Result

__all__ = ["{{CLASS}}", "INSTALL_COMMAND", "main"]

#: The ``uvx`` command that installs this server (must equal the manifest ``uvx``).
INSTALL_COMMAND = "uvx {{NAME}}"


class {{CLASS}}(BaseGeoServer):
    """Pillar {{PILLAR}} server exposing the ``{{TOOL}}`` capability (scaffolded)."""

    pillar = "{{PILLAR}}"
    server_name = "{{NAME}}"
    version = "0.1.0"

    def __init__(self, http=None) -> None:
        super().__init__(http=http)
        self.register_tool("{{TOOL}}", self.{{TOOL}})

    async def {{TOOL}}(self, *, value: str) -> {{TOOL_TITLE}}Result:
        """Validate ``value`` and return a result (scaffolded placeholder).

        Replace this body with the real capability. The convention: validate
        inputs first and raise an ``Error_Taxonomy`` ``ValidationError`` naming
        the bad parameter before doing any work; map any downstream library /
        transport failure through :meth:`map_error`.
        """
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(
                "value must be a non-empty string",
                source=self.server_name,
                detail={"parameter": "value"},
            )
        return {{TOOL_TITLE}}Result(echo=value.strip(), source=self.server_name)

    def catalog_entries(self) -> "List[CatalogEntry]":
        """Register this server's capability in the Resource Catalog (Req 2.1)."""
        return [
            CatalogEntry(
                name="{{TOOL}}",
                pillar=self.pillar,
                capability_description=(
                    "{{TOOL}} (scaffolded skeleton - edit this description)."
                ),
                openness_tier=OpennessTier.OPEN,
                provider_server=self.server_name,
                installed=True,
            )
        ]

    def required_credentials(self) -> "List[CredentialSpec]":
        """Declare ``mcp.json`` credential keys this server reads (Req 16.1).

        A scaffolded skeleton is open and credential-free, so this is empty. If
        your server reads a credential, add a ``CredentialSpec`` here AND a
        matching entry to the server's ``credentials`` block in
        ``bundle-manifest.json`` (the drift check enforces they agree).
        """
        return []


def main() -> None:
    """Console entry point: serve {{NAME}} over MCP stdio."""
    {{CLASS}}().run()


if __name__ == "__main__":  # pragma: no cover - module CLI shim
    main()
'''


_INIT = '''"""{{NAME}}: Pillar {{PILLAR}} MCP server (scaffolded skeleton)."""

from __future__ import annotations

from {{MODULE}}.models import {{TOOL_TITLE}}Result
from {{MODULE}}.server import {{CLASS}}, INSTALL_COMMAND, main

__all__ = ["{{TOOL_TITLE}}Result", "{{CLASS}}", "INSTALL_COMMAND", "main"]
'''


_TEST = '''"""Tests for ``{{NAME}}`` (scaffolded skeleton - the required test trio).

Every server ships at least: (1) a registration/catalog check, (2) a happy-path
tool call, and (3) a validation-error guard. The repo's ``asyncio_mode = "auto"``
lets the ``async def`` tests run directly.
"""

from __future__ import annotations

import pytest

from geo_common.errors import ErrorCategory, ValidationError
from geo_common.models import OpennessTier

from {{MODULE}}.server import INSTALL_COMMAND, {{CLASS}}


def test_tool_is_registered() -> None:
    server = {{CLASS}}()
    assert "{{TOOL}}" in server.tool_names()


def test_catalog_entry_names_this_server_as_provider() -> None:
    server = {{CLASS}}()
    entries = server.catalog_entries()
    assert entries, "expected at least one catalog entry"
    for entry in entries:
        assert entry.provider_server == "{{NAME}}"
        assert entry.pillar == "{{PILLAR}}"
        assert entry.openness_tier is OpennessTier.OPEN


def test_starts_without_credentials() -> None:
    server = {{CLASS}}()
    server.start(configured_keys=[])
    assert server.started is True


def test_install_command_constant() -> None:
    assert INSTALL_COMMAND == "uvx {{NAME}}"


async def test_happy_path_returns_result() -> None:
    server = {{CLASS}}()
    result = await server.{{TOOL}}(value="hello")
    assert result.echo == "hello"
    assert result.source == "{{NAME}}"


@pytest.mark.parametrize("bad", ["", "   "])
async def test_blank_value_is_validation_error(bad: str) -> None:
    server = {{CLASS}}()
    with pytest.raises(ValidationError) as exc_info:
        await server.{{TOOL}}(value=bad)
    assert exc_info.value.category is ErrorCategory.VALIDATION
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Scaffold a new server package.")
    parser.add_argument("--name", required=True, help="server name, e.g. geo-foo")
    parser.add_argument(
        "--pillar", required=True, choices=["A", "B", "C"], help="pillar (A/B/C)"
    )
    parser.add_argument(
        "--tool", required=True, help="the single tool's snake_case name, e.g. do_thing"
    )
    args = parser.parse_args()

    name = args.name.strip()
    if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
        print("ERROR: --name must be lowercase, dash-separated (e.g. geo-foo).", file=sys.stderr)
        return 2
    tool = args.tool.strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", tool):
        print("ERROR: --tool must be a snake_case identifier (e.g. do_thing).", file=sys.stderr)
        return 2

    module = name.replace("-", "_")
    ctx = {
        "NAME": name,
        "MODULE": module,
        "CLASS": _camel(module) + "Server",
        "PILLAR": args.pillar,
        "TOOL": tool,
        "TOOL_TITLE": _camel(tool),
    }

    pkg_dir = _PACKAGES / name
    if pkg_dir.exists():
        print("ERROR: %s already exists; refusing to overwrite." % pkg_dir, file=sys.stderr)
        return 1

    src_dir = pkg_dir / module
    tests_dir = pkg_dir / "tests"
    files = {
        pkg_dir / "pyproject.toml": _PYPROJECT,
        src_dir / "__init__.py": _INIT,
        src_dir / "models.py": _MODELS,
        src_dir / "server.py": _SERVER,
        tests_dir / ("test_%s.py" % module): _TEST,
    }
    for path, template in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_render(template, ctx), encoding="utf-8")
        print("created %s" % path.relative_to(_REPO_ROOT))

    print(
        "\nScaffolded %s (class %s, tool %s).\n\nNext steps:\n"
        "  1. Add (\"%s.server\", \"%s\") to SERVERS in scripts/check_tool_schemas.py\n"
        "  2. Add a 'servers' entry for '%s' to bundle-manifest.json (uvx '%s')\n"
        "  3. Install it:   .venv/bin/python -m pip install -e ./packages/%s\n"
        "  4. Run checks:   make check    (schemas + manifest drift)\n"
        "  5. Run tests:    PYTHONPATH=packages/geo-common:packages/%s "
        ".venv/bin/python -m pytest packages/%s -q\n"
        "  6. Replace the placeholder tool/model/catalog with the real capability "
        "(see CONTRIBUTING.md)."
        % (
            name, ctx["CLASS"], tool,
            module, ctx["CLASS"],
            name, ctx["NAME"],
            name,
            name, name,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
