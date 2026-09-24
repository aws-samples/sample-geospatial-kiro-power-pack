"""Dependency-confusion hardening for the Hub's ``uv tool`` installer.

None of this pack's packages are on PyPI, so the production
:class:`SubprocessUvxRunner` must install from the repository checkout
(``uv tool install <packages_dir>/<name>``) and must never fall back to a bare
``uv tool install <name>``, which would resolve the name from the public index.
These tests stub ``subprocess.run`` to assert the exact argv, and check the
documented install-command contract shared by the manifest and every server.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List

import pytest

from kiro_geospatial.installation import (
    PACKAGES_DIR_ENV,
    SubprocessUvxRunner,
    UvxCommandError,
    install_command,
    resolve_packages_dir,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST = _REPO_ROOT / "bundle-manifest.json"


def _fake_packages_dir(tmp_path: Path, *names: str) -> Path:
    packages = tmp_path / "packages"
    for name in names:
        (packages / name).mkdir(parents=True)
        (packages / name / "pyproject.toml").write_text('[project]\nname = "%s"\n' % name)
    return packages


class _Recorder:
    """Stand-in for ``subprocess.run`` that records argv and scripts output."""

    def __init__(self, list_stdout: str = "") -> None:
        self.calls: List[List[str]] = []
        self._list_stdout = list_stdout

    def __call__(self, argv, **kwargs):  # noqa: ANN001 - subprocess.run signature
        self.calls.append(list(argv))
        stdout = self._list_stdout if argv[2] == "list" else ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")


# --------------------------------------------------------------------------- #
# install_command / resolve_packages_dir
# --------------------------------------------------------------------------- #
def test_install_command_is_path_based():
    assert install_command("geo-stac") == "uvx --from ./packages/geo-stac geo-stac"


def test_manifest_uvx_entries_follow_the_install_command_contract():
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest["servers"]:
        if entry["name"] == "geo-common":
            continue  # placeholder text, installed as a dependency
        assert entry["uvx"] == install_command(entry["name"]), entry["name"]
        assert not entry["uvx"].startswith("uvx %s" % entry["name"])


def test_resolve_packages_dir_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv(PACKAGES_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_packages_dir() == tmp_path / "packages"

    monkeypatch.setenv(PACKAGES_DIR_ENV, str(tmp_path / "from-env"))
    assert resolve_packages_dir() == tmp_path / "from-env"

    assert resolve_packages_dir(tmp_path / "explicit") == tmp_path / "explicit"


# --------------------------------------------------------------------------- #
# SubprocessUvxRunner
# --------------------------------------------------------------------------- #
def test_install_passes_the_local_source_path_not_a_bare_name(tmp_path, monkeypatch):
    packages = _fake_packages_dir(tmp_path, "geo-stac")
    recorder = _Recorder(list_stdout="geo-stac v0.3.0\n- geo-stac\n")
    monkeypatch.setattr(subprocess, "run", recorder)

    runner = SubprocessUvxRunner(uv_bin="uv", packages_dir=packages)
    version = runner.install("geo-stac")

    assert version == "0.3.0"
    install_calls = [c for c in recorder.calls if c[2] == "install"]
    assert install_calls == [["uv", "tool", "install", str(packages / "geo-stac")]]
    # The bare name must never be handed to `uv tool install`.
    assert ["uv", "tool", "install", "geo-stac"] not in recorder.calls


def test_install_refuses_when_local_source_is_missing(tmp_path, monkeypatch):
    packages = _fake_packages_dir(tmp_path, "geo-stac")
    recorder = _Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    runner = SubprocessUvxRunner(packages_dir=packages)
    with pytest.raises(UvxCommandError) as exc_info:
        runner.install("geo-vector")

    assert exc_info.value.package == "geo-vector"
    assert "not on PyPI" in exc_info.value.message
    assert recorder.calls == []  # nothing was executed; no public-index fallback


@pytest.mark.parametrize("bad", ["", ".", "..", "../geo-stac", "geo-stac/../../etc"])
def test_install_rejects_names_that_are_not_a_single_path_component(
    bad, tmp_path, monkeypatch
):
    packages = _fake_packages_dir(tmp_path, "geo-stac")
    recorder = _Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    runner = SubprocessUvxRunner(packages_dir=packages)
    with pytest.raises(UvxCommandError):
        runner.install(bad)
    assert recorder.calls == []


def test_uninstall_and_list_address_tools_by_name(tmp_path, monkeypatch):
    packages = _fake_packages_dir(tmp_path, "geo-stac")
    recorder = _Recorder(list_stdout="geo-common v0.2.0\n- geo-common\ngeo-stac v0.3.0\n- geo-stac\n")
    monkeypatch.setattr(subprocess, "run", recorder)

    runner = SubprocessUvxRunner(packages_dir=packages)
    assert runner.list_installed() == {"geo-common": "0.2.0", "geo-stac": "0.3.0"}
    runner.uninstall("geo-stac")
    assert ["uv", "tool", "uninstall", "geo-stac"] in recorder.calls


def test_real_packages_dir_resolves_every_manifest_server():
    """Every manifest server has a local source the runner would install from."""
    runner = SubprocessUvxRunner(packages_dir=_REPO_ROOT / "packages")
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest["servers"]:
        source = runner._source_path(entry["name"])
        assert (source / "pyproject.toml").is_file(), entry["name"]
