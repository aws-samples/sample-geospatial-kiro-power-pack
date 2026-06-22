"""Unit tests for the steering activation engine (Requirement 14).

Example-based coverage of the three :class:`SteeringEngine` state transitions
called out by task 9.4:

* **match -> activate** with steps presented in their defined order
  (Requirements 14.2, 14.3).
* **fail -> deactivate with a named reason** when an active workflow's steps
  cannot be presented (Requirement 14.4).
* **unmatch -> deactivate** when the triggering file no longer matches, both by
  rename and by close (Requirement 14.6).

These complement the engine's docstring contract and exercise the public
event surface (``open_or_edit``/``rename``/``close``) only.
"""

from __future__ import annotations

import pytest

from kiro_geospatial.steering import (
    DeactivationReport,
    SteeringEngine,
    SteeringWorkflow,
    StepPresentation,
    StepPresentationError,
)


# --------------------------------------------------------------------------- #
# Fixtures: a small set of representative, ordered-step workflows.
# --------------------------------------------------------------------------- #
def _cog_workflow() -> SteeringWorkflow:
    return SteeringWorkflow(
        name="cog-conversion",
        description="Convert a raster to a Cloud-Optimized GeoTIFF.",
        file_match_patterns=["**/*.tif", "**/*.tiff"],
        steps=[
            "Identify source",
            "Validate CRS",
            "Convert to COG",
            "Verify round-trip",
        ],
    )


def _zonal_workflow() -> SteeringWorkflow:
    return SteeringWorkflow(
        name="zonal-statistics",
        description="Compute zonal statistics over raster and vector.",
        file_match_patterns=["**/*.geojson"],
        steps=["Load zones", "Aggregate", "Report"],
    )


def _engine(*, step_presenter=None) -> SteeringEngine:
    return SteeringEngine(
        [_cog_workflow(), _zonal_workflow()],
        step_presenter=step_presenter,
    )


# --------------------------------------------------------------------------- #
# match -> activate, steps presented in defined order (Req 14.2, 14.3)
# --------------------------------------------------------------------------- #
def test_open_matching_file_activates_workflow_with_ordered_steps():
    engine = _engine()

    update = engine.open_or_edit("data/raw/scene.tif")

    assert [p.workflow_name for p in update.activated] == ["cog-conversion"]
    presentation = update.activated[0]
    assert presentation.triggering_file == "data/raw/scene.tif"
    assert presentation.newly_activated is True
    # Req 14.3: steps presented in their exact defined order.
    assert presentation.steps == [
        "Identify source",
        "Validate CRS",
        "Convert to COG",
        "Verify round-trip",
    ]
    assert update.deactivated == []
    assert update.active_workflows == ["cog-conversion"]
    assert engine.is_active("cog-conversion")
    assert not engine.is_active("zonal-statistics")


def test_open_non_matching_file_activates_nothing():
    engine = _engine()

    update = engine.open_or_edit("notes/readme.md")

    assert update.activated == []
    assert update.deactivated == []
    assert update.active_workflows == []


def test_one_file_can_activate_multiple_matching_workflows():
    overlap = SteeringWorkflow(
        name="tif-audit",
        file_match_patterns=["**/*.tif"],
        steps=["Audit"],
    )
    engine = SteeringEngine([_cog_workflow(), overlap])

    update = engine.open_or_edit("scene.tif")

    # Both matching workflows activate; names are reported sorted/deterministic.
    activated = sorted(p.workflow_name for p in update.activated)
    assert activated == ["cog-conversion", "tif-audit"]
    assert update.active_workflows == ["cog-conversion", "tif-audit"]


def test_reedit_while_active_re_presents_without_newly_activated_flag():
    engine = _engine()
    engine.open_or_edit("scene.tif")

    again = engine.open_or_edit("scene.tif")

    assert [p.workflow_name for p in again.activated] == ["cog-conversion"]
    # Already active -> re-presentation, not a fresh activation.
    assert again.activated[0].newly_activated is False
    assert again.activated[0].steps == _cog_workflow().steps
    assert engine.is_active("cog-conversion")


# --------------------------------------------------------------------------- #
# fail -> deactivate with a named reason (Req 14.4)
# --------------------------------------------------------------------------- #
def test_presentation_failure_deactivates_and_reports_name_and_reason():
    def failing_presenter(workflow: SteeringWorkflow, file_path: str) -> None:
        raise StepPresentationError("editor channel unavailable")

    engine = _engine(step_presenter=failing_presenter)

    update = engine.open_or_edit("scene.tif")

    # Not activated.
    assert update.activated == []
    assert not engine.is_active("cog-conversion")
    # Req 14.4: a single deactivation report naming the workflow and the reason.
    assert len(update.deactivated) == 1
    report = update.deactivated[0]
    assert isinstance(report, DeactivationReport)
    assert report.workflow_name == "cog-conversion"
    assert "editor channel unavailable" in report.reason
    assert report.triggering_file == "scene.tif"
    assert update.active_workflows == []


def test_default_presenter_deactivates_workflow_with_no_steps():
    # The default presenter treats an empty step list as not presentable.
    stepless = SteeringWorkflow(
        name="empty-workflow",
        file_match_patterns=["**/*.tif"],
        steps=[],
    )
    engine = SteeringEngine([stepless])

    update = engine.open_or_edit("scene.tif")

    assert update.activated == []
    assert len(update.deactivated) == 1
    report = update.deactivated[0]
    assert report.workflow_name == "empty-workflow"
    assert "no ordered steps" in report.reason
    assert not engine.is_active("empty-workflow")


def test_presentation_failure_for_one_workflow_does_not_block_another():
    def selective_presenter(workflow: SteeringWorkflow, file_path: str) -> None:
        if workflow.name == "cog-conversion":
            raise StepPresentationError("boom")

    overlap = SteeringWorkflow(
        name="tif-audit",
        file_match_patterns=["**/*.tif"],
        steps=["Audit"],
    )
    engine = SteeringEngine(
        [_cog_workflow(), overlap], step_presenter=selective_presenter
    )

    update = engine.open_or_edit("scene.tif")

    assert [p.workflow_name for p in update.activated] == ["tif-audit"]
    assert [d.workflow_name for d in update.deactivated] == ["cog-conversion"]
    assert engine.is_active("tif-audit")
    assert not engine.is_active("cog-conversion")


def test_non_steering_exception_still_deactivates_with_reason():
    def exploding_presenter(workflow: SteeringWorkflow, file_path: str) -> None:
        raise RuntimeError("unexpected failure")

    engine = _engine(step_presenter=exploding_presenter)

    update = engine.open_or_edit("scene.tif")

    assert update.activated == []
    assert len(update.deactivated) == 1
    assert update.deactivated[0].workflow_name == "cog-conversion"
    assert "unexpected failure" in update.deactivated[0].reason


# --------------------------------------------------------------------------- #
# unmatch -> deactivate via rename (Req 14.6)
# --------------------------------------------------------------------------- #
def test_rename_to_non_matching_extension_deactivates_workflow():
    engine = _engine()
    engine.open_or_edit("scene.tif")
    assert engine.is_active("cog-conversion")

    update = engine.rename("scene.tif", "scene.txt")

    assert update.activated == []
    assert len(update.deactivated) == 1
    report = update.deactivated[0]
    assert report.workflow_name == "cog-conversion"
    assert "renamed" in report.reason
    assert not engine.is_active("cog-conversion")
    assert update.active_workflows == []


def test_rename_between_matching_extensions_keeps_workflow_active():
    engine = _engine()
    engine.open_or_edit("scene.tif")

    update = engine.rename("scene.tif", "scene.tiff")

    # Rename = withdraw the old path's triggers, then open the new path. The
    # old path's trigger is withdrawn (one deactivation report citing the
    # rename) and the still-matching new path re-activates the workflow, so the
    # net end-state is active.
    assert [p.workflow_name for p in update.activated] == ["cog-conversion"]
    assert [d.workflow_name for d in update.deactivated] == ["cog-conversion"]
    assert update.deactivated[0].triggering_file == "scene.tif"
    assert "renamed" in update.deactivated[0].reason
    assert engine.is_active("cog-conversion")
    assert update.active_workflows == ["cog-conversion"]


def test_rename_can_swap_one_workflow_for_another():
    engine = _engine()
    engine.open_or_edit("zones.geojson")
    assert engine.is_active("zonal-statistics")

    update = engine.rename("zones.geojson", "zones.tif")

    assert [p.workflow_name for p in update.activated] == ["cog-conversion"]
    assert [d.workflow_name for d in update.deactivated] == ["zonal-statistics"]
    assert engine.is_active("cog-conversion")
    assert not engine.is_active("zonal-statistics")


# --------------------------------------------------------------------------- #
# unmatch -> deactivate via close (Req 14.6)
# --------------------------------------------------------------------------- #
def test_close_triggering_file_deactivates_workflow():
    engine = _engine()
    engine.open_or_edit("scene.tif")

    update = engine.close("scene.tif")

    assert len(update.deactivated) == 1
    report = update.deactivated[0]
    assert report.workflow_name == "cog-conversion"
    assert "closed" in report.reason
    assert report.triggering_file == "scene.tif"
    assert not engine.is_active("cog-conversion")
    assert update.active_workflows == []


def test_close_one_of_two_triggers_keeps_workflow_active():
    engine = _engine()
    engine.open_or_edit("a.tif")
    engine.open_or_edit("b.tif")
    assert engine.is_active("cog-conversion")

    update = engine.close("a.tif")

    # The other open file still matches, so the workflow stays active and no
    # deactivation is reported.
    assert update.deactivated == []
    assert engine.is_active("cog-conversion")
    assert update.active_workflows == ["cog-conversion"]

    # Closing the last triggering file finally deactivates it.
    final = engine.close("b.tif")
    assert [d.workflow_name for d in final.deactivated] == ["cog-conversion"]
    assert not engine.is_active("cog-conversion")


def test_close_untracked_file_is_a_noop():
    engine = _engine()
    engine.open_or_edit("scene.tif")

    update = engine.close("never-opened.tif")

    assert update.deactivated == []
    assert engine.is_active("cog-conversion")
    assert update.active_workflows == ["cog-conversion"]


# --------------------------------------------------------------------------- #
# Engine construction guard
# --------------------------------------------------------------------------- #
def test_duplicate_workflow_names_are_rejected():
    with pytest.raises(ValueError):
        SteeringEngine([_cog_workflow(), _cog_workflow()])
