"""Example-based unit tests for the Installation Manager (Req 6.8, 6.9, 16.7).

The Installation Manager talks to ``uvx`` only through the :class:`UvxRunner`
abstraction, so these tests drive it with a :class:`FakeUvxRunner` that records
every install/uninstall call and can be told to fail on a chosen package. That
lets us assert the contract without shelling out to ``uv``:

* **Atomic rollback on partial failure** (Requirements 6.8, 16.7): when one
  package in a multi-package operation fails to install, every package this
  operation already installed is uninstalled, leaving nothing partially
  installed, and the error names the failed module.
* **Listing installed servers with versions** (Requirement 6.5).
* **Installing the MVP set together** (Requirements 16.2, 16.3).
* **Errors for not-installed removal** (Requirements 6.9, 16.7) and unknown
  modules.

These complement the property tests; here we pin concrete, representative
examples and edge cases.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from geo_common.errors import NotFoundError, UpstreamError, ValidationError
from kiro_geospatial.installation import (
    GEO_COMMON,
    HUB_DEFAULT_NAME,
    InstallationManager,
    ServerSpec,
    UvxCommandError,
    UvxRunner,
)


# --------------------------------------------------------------------------- #
# Fake uvx runner: in-memory install state with scriptable failures.
# --------------------------------------------------------------------------- #
class FakeUvxRunner(UvxRunner):
    """An in-memory :class:`UvxRunner` for tests.

    * ``state`` maps installed package -> version.
    * ``fail_on`` is a set of package names whose ``install`` raises
      :class:`UvxCommandError` (simulating a uvx failure).
    * ``install_calls`` / ``uninstall_calls`` record the order of operations so
      tests can assert exactly what the manager did (e.g. rollback order).
    * ``fail_uninstall_on`` lets a test simulate a best-effort rollback where
      one uninstall itself fails.
    """

    def __init__(
        self,
        *,
        preinstalled: Optional[Dict[str, str]] = None,
        fail_on: Optional[set] = None,
        version: str = "1.0.0",
        fail_uninstall_on: Optional[set] = None,
    ) -> None:
        self.state: Dict[str, str] = dict(preinstalled or {})
        self.fail_on: set = set(fail_on or set())
        self.fail_uninstall_on: set = set(fail_uninstall_on or set())
        self.version = version
        self.install_calls: List[str] = []
        self.uninstall_calls: List[str] = []

    def install(self, package: str) -> str:
        self.install_calls.append(package)
        if package in self.fail_on:
            raise UvxCommandError(
                package,
                "uv tool install %s failed" % package,
                detail="simulated failure",
            )
        self.state[package] = self.version
        return self.version

    def uninstall(self, package: str) -> None:
        self.uninstall_calls.append(package)
        if package in self.fail_uninstall_on:
            raise UvxCommandError(package, "uv tool uninstall %s failed" % package)
        self.state.pop(package, None)

    def list_installed(self) -> Dict[str, str]:
        return dict(self.state)


# --------------------------------------------------------------------------- #
# A small but representative server vocabulary mirroring the MVP set.
# --------------------------------------------------------------------------- #
MVP_SET = [
    GEO_COMMON,
    "geo-stac",
    "geo-vector",
    "geo-ops",
    "geo-foundation-models",
]


def _specs() -> Dict[str, ServerSpec]:
    pillars = {
        GEO_COMMON: "common",
        "geo-stac": "A",
        "geo-vector": "A",
        "geo-ops": "B",
        "geo-foundation-models": "C",
        "geo-terrain": "A",
    }
    return {
        name: ServerSpec(name=name, pillar=pillar)
        for name, pillar in pillars.items()
    }


def _manager(
    runner: FakeUvxRunner,
    *,
    configured_keys=None,
) -> InstallationManager:
    return InstallationManager(
        servers=_specs(),
        mvp_set=MVP_SET,
        first_expansions=["geo-warehouse"],
        hub_name=HUB_DEFAULT_NAME,
        runner=runner,
        configured_keys=configured_keys if configured_keys is not None else [],
    )


# --------------------------------------------------------------------------- #
# install_server: single server + geo-common, nothing else (Req 6.2, 6.3)
# --------------------------------------------------------------------------- #
def test_install_server_installs_server_plus_geo_common_only():
    runner = FakeUvxRunner()
    report = _manager(runner).install_server("geo-stac")

    # geo-common is ensured first, then the requested server.
    assert runner.install_calls == [GEO_COMMON, "geo-stac"]
    assert set(runner.state) == {GEO_COMMON, "geo-stac"}
    assert report.all_installed is True
    assert report.requested == ["geo-stac"]
    assert {v.name for v in report.installed} == {"geo-stac"}


def test_install_server_does_not_reinstall_existing_geo_common():
    runner = FakeUvxRunner(preinstalled={GEO_COMMON: "1.0.0"})
    _manager(runner).install_server("geo-vector")

    # geo-common already present -> only the new server is installed.
    assert runner.install_calls == ["geo-vector"]
    assert set(runner.state) == {GEO_COMMON, "geo-vector"}


def test_install_unknown_server_is_rejected_and_installs_nothing():
    runner = FakeUvxRunner()
    with pytest.raises(ValidationError) as exc_info:
        _manager(runner).install_server("geo-nonexistent")

    assert "geo-nonexistent" in str(exc_info.value)
    assert runner.install_calls == []
    assert runner.state == {}


# --------------------------------------------------------------------------- #
# Atomic rollback on partial failure (Requirements 6.8, 16.7)
# --------------------------------------------------------------------------- #
def test_install_set_rolls_back_atomically_on_partial_failure():
    # geo-vector fails AFTER geo-common and geo-stac install successfully.
    runner = FakeUvxRunner(fail_on={"geo-vector"})
    with pytest.raises(UpstreamError) as exc_info:
        _manager(runner).install_set(["geo-stac", "geo-vector", "geo-ops"])

    err = exc_info.value
    # The error names the failed module (Requirement 6.8).
    assert err.detail["failed_module"] == "geo-vector"
    # Nothing is left partially installed: everything this op added is gone.
    assert runner.state == {}
    # Rollback uninstalled exactly what was installed, in reverse order.
    assert runner.uninstall_calls == ["geo-stac", GEO_COMMON]
    assert set(err.detail["rolled_back"]) == {"geo-stac", GEO_COMMON}


def test_rollback_leaves_preexisting_installs_untouched():
    # geo-common + geo-stac were already installed before this operation.
    runner = FakeUvxRunner(
        preinstalled={GEO_COMMON: "1.0.0", "geo-stac": "1.0.0"},
        fail_on={"geo-ops"},
    )
    with pytest.raises(UpstreamError) as exc_info:
        _manager(runner).install_set(["geo-vector", "geo-ops"])

    # Only geo-vector was newly installed by this op, so only it is rolled back;
    # the pre-existing geo-common and geo-stac are left intact (Req 6.3, 6.6).
    assert runner.uninstall_calls == ["geo-vector"]
    assert runner.state == {GEO_COMMON: "1.0.0", "geo-stac": "1.0.0"}
    assert exc_info.value.detail["failed_module"] == "geo-ops"


def test_rollback_when_geo_common_install_itself_fails():
    runner = FakeUvxRunner(fail_on={GEO_COMMON})
    with pytest.raises(UpstreamError) as exc_info:
        _manager(runner).install_server("geo-stac")

    assert exc_info.value.detail["failed_module"] == GEO_COMMON
    # geo-common failed first; nothing installed, nothing to roll back.
    assert runner.state == {}
    assert runner.uninstall_calls == []


def test_rollback_is_best_effort_when_an_uninstall_fails():
    # geo-ops fails to install; rolling back geo-stac fails too, but geo-common
    # is still rolled back and the operation still reports the failure.
    runner = FakeUvxRunner(
        fail_on={"geo-ops"},
        fail_uninstall_on={"geo-stac"},
    )
    with pytest.raises(UpstreamError) as exc_info:
        _manager(runner).install_set(["geo-stac", "geo-ops"])

    # Both uninstalls were attempted (reverse order); geo-stac raised but
    # geo-common was still removed.
    assert runner.uninstall_calls == ["geo-stac", GEO_COMMON]
    assert GEO_COMMON not in runner.state
    # geo-stac remained because its uninstall failed (best-effort rollback).
    assert runner.state == {"geo-stac": "1.0.0"}
    assert exc_info.value.detail["rolled_back"] == [GEO_COMMON]


# --------------------------------------------------------------------------- #
# install_mvp_set: the five servers together (Requirements 16.2, 16.3)
# --------------------------------------------------------------------------- #
def test_install_mvp_set_installs_all_five_and_reports_startable():
    runner = FakeUvxRunner()
    report = _manager(runner).install_mvp_set()

    assert set(runner.state) == set(MVP_SET)
    assert report.all_installed is True
    # No Required credentials configured for these servers -> all can start.
    assert report.all_can_start is True
    assert {v.name for v in report.installed} == set(MVP_SET)
    assert all(v.can_start for v in report.installed)


def test_install_mvp_set_excludes_the_hub_package():
    # Even if the hub package is listed in the MVP set, it is not an
    # a-la-carte installable server (Requirement 16.2).
    runner = FakeUvxRunner()
    mgr = InstallationManager(
        servers=_specs(),
        mvp_set=[HUB_DEFAULT_NAME, *MVP_SET],
        hub_name=HUB_DEFAULT_NAME,
        runner=runner,
        configured_keys=[],
    )
    assert HUB_DEFAULT_NAME not in mgr.mvp_install_set()
    mgr.install_mvp_set()
    assert HUB_DEFAULT_NAME not in runner.state


def test_install_mvp_set_reports_not_startable_when_required_credential_missing():
    # geo-foundation-models requires a credential that is NOT configured.
    specs = _specs()
    specs["geo-foundation-models"] = ServerSpec(
        name="geo-foundation-models",
        pillar="C",
        required_credential_keys=["FM_API_KEY"],
    )
    runner = FakeUvxRunner()
    mgr = InstallationManager(
        servers=specs,
        mvp_set=MVP_SET,
        hub_name=HUB_DEFAULT_NAME,
        runner=runner,
        configured_keys=[],  # FM_API_KEY absent
    )
    report = mgr.install_mvp_set()

    assert report.all_installed is True
    assert report.all_can_start is False
    blocked = {v.name: v.blocking_credentials for v in report.installed if not v.can_start}
    assert blocked == {"geo-foundation-models": ["FM_API_KEY"]}


# --------------------------------------------------------------------------- #
# list_installed: report installed servers with versions (Requirement 6.5)
# --------------------------------------------------------------------------- #
def test_list_installed_reports_known_servers_with_versions_sorted():
    runner = FakeUvxRunner(
        preinstalled={
            "geo-stac": "2.1.0",
            GEO_COMMON: "1.0.0",
            "unrelated-tool": "9.9.9",  # not a known server -> excluded
        }
    )
    views = _manager(runner).list_installed()

    names = [v.name for v in views]
    assert names == sorted([GEO_COMMON, "geo-stac"])  # sorted, unknown excluded
    versions = {v.name: v.version for v in views}
    assert versions == {GEO_COMMON: "1.0.0", "geo-stac": "2.1.0"}


def test_list_installed_empty_when_nothing_installed():
    runner = FakeUvxRunner()
    assert _manager(runner).list_installed() == []


# --------------------------------------------------------------------------- #
# remove_server: not-installed and protected-package errors (Req 6.6, 6.9)
# --------------------------------------------------------------------------- #
def test_remove_not_installed_server_raises_and_changes_nothing():
    runner = FakeUvxRunner(preinstalled={GEO_COMMON: "1.0.0", "geo-stac": "1.0.0"})
    with pytest.raises(NotFoundError) as exc_info:
        _manager(runner).remove_server("geo-vector")

    assert "geo-vector" in str(exc_info.value)
    # Nothing was uninstalled; installed components are unchanged (Req 6.9).
    assert runner.uninstall_calls == []
    assert runner.state == {GEO_COMMON: "1.0.0", "geo-stac": "1.0.0"}


def test_remove_server_removes_only_that_server():
    runner = FakeUvxRunner(
        preinstalled={GEO_COMMON: "1.0.0", "geo-stac": "1.0.0", "geo-vector": "1.0.0"}
    )
    view = _manager(runner).remove_server("geo-stac")

    assert view.name == "geo-stac"
    assert view.version == "1.0.0"
    assert runner.uninstall_calls == ["geo-stac"]
    # geo-common and the other server remain (Requirement 6.6).
    assert runner.state == {GEO_COMMON: "1.0.0", "geo-vector": "1.0.0"}


def test_remove_geo_common_is_refused():
    runner = FakeUvxRunner(preinstalled={GEO_COMMON: "1.0.0"})
    with pytest.raises(ValidationError):
        _manager(runner).remove_server(GEO_COMMON)

    assert runner.uninstall_calls == []
    assert runner.state == {GEO_COMMON: "1.0.0"}


def test_remove_unknown_server_is_rejected():
    runner = FakeUvxRunner(preinstalled={GEO_COMMON: "1.0.0"})
    with pytest.raises(ValidationError):
        _manager(runner).remove_server("geo-nonexistent")

    assert runner.uninstall_calls == []
