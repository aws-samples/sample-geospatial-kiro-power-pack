"""Steering activation engine for the Geospatial Power Pack (Requirement 14).

A :class:`SteeringWorkflow` is an executable, file-pattern-triggered, multi-step
workflow guide (the five authored under ``steering/``: ``cog-conversion``,
``zonal-statistics``, ``stac-discover-analyze``, ``embedding-change-detection``,
``geocode-route``). The :class:`SteeringEngine` governs their activation and
deactivation against editor file events:

* **Activate on match (Req 14.2).** When a file is opened or edited, every
  workflow whose ``fileMatchPattern`` matches that file is activated.
* **Present steps in order (Req 14.3).** While a workflow is active, its steps
  are presented to the model in their defined order.
* **Deactivate + report on presentation failure (Req 14.4).** If an active
  workflow's steps cannot be presented, the workflow is deactivated and a
  report is emitted that *names the workflow and the reason*.
* **Deactivate on unmatch (Req 14.6).** When the file that triggered a
  workflow's activation no longer matches that workflow's pattern (e.g. it is
  renamed or closed), the workflow is deactivated.

The engine is deliberately pure/in-memory and free of editor or MCP coupling so
it can be unit-tested directly: editor events are delivered through
:meth:`SteeringEngine.open_or_edit`, :meth:`SteeringEngine.rename`, and
:meth:`SteeringEngine.close`, and each returns a :class:`SteeringUpdate`
describing the activations and deactivations that resulted.

Python 3.9 compatibility: ``from __future__ import annotations`` plus
``typing`` generics keeps the pydantic models resolvable on 3.9+ (the package
targets 3.10+, but the shared monorepo test interpreter may be older).
"""

from __future__ import annotations

import os
import re
from typing import Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field

__all__ = [
    "SteeringWorkflow",
    "StepPresentation",
    "DeactivationReport",
    "SteeringUpdate",
    "StepPresentationError",
    "SteeringEngine",
    "load_workflows_from_directory",
]


# ---------------------------------------------------------------------------
# File-pattern matching (`**`-style globs)
# ---------------------------------------------------------------------------
def _normalize_path(file_path: str) -> str:
    """Normalize a path for matching: forward slashes, no leading ``./``.

    Matching is performed on the POSIX-style relative form so a workflow's
    ``**/*.tif`` pattern matches regardless of the host path separator.
    """
    norm = file_path.replace(os.sep, "/").replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def _glob_to_regex(pattern: str) -> "re.Pattern[str]":
    """Translate a ``**``-aware glob into an anchored, case-insensitive regex.

    Semantics (POSIX glob with globstar):

    * ``**/`` matches any number of leading path segments, including none, so
      ``**/*.tif`` matches both ``a.tif`` and ``data/raw/a.tif``.
    * ``**`` (not followed by ``/``) matches anything, across separators.
    * ``*`` matches any run of characters except ``/``.
    * ``?`` matches a single character except ``/``.

    Matching is case-insensitive so geospatial extensions such as ``.TIF`` and
    ``.tif`` both trigger their workflow.
    """
    norm = pattern.replace(os.sep, "/").replace("\\", "/")
    out: List[str] = []
    i = 0
    n = len(norm)
    while i < n:
        if norm[i] == "*":
            if norm[i : i + 3] == "**/":
                out.append("(?:.*/)?")
                i += 3
                continue
            if norm[i : i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
            i += 1
            continue
        if norm[i] == "?":
            out.append("[^/]")
            i += 1
            continue
        out.append(re.escape(norm[i]))
        i += 1
    return re.compile("^" + "".join(out) + "$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class SteeringWorkflow(BaseModel):
    """An executable, file-pattern-triggered steering workflow (Req 14.1).

    Carries the workflow ``name`` (the report identity used in deactivation
    indications, Req 14.4), its ``file_match_patterns`` (the
    ``fileMatchPattern`` triggers, Req 14.2/14.6), and its ordered ``steps``
    (presented in defined order, Req 14.3). ``source_path`` records the
    authoring ``steering/*.md`` file when loaded from disk.
    """

    name: str = Field(min_length=1)
    description: str = ""
    file_match_patterns: List[str] = Field(min_length=1)
    steps: List[str] = Field(default_factory=list)
    source_path: Optional[str] = None

    def matches(self, file_path: str) -> bool:
        """True iff ``file_path`` matches any of this workflow's patterns."""
        candidate = _normalize_path(file_path)
        for pattern in self.file_match_patterns:
            if _glob_to_regex(pattern).match(candidate):
                return True
        return False


class StepPresentation(BaseModel):
    """An active workflow's steps presented in defined order (Req 14.2/14.3).

    ``newly_activated`` distinguishes a fresh activation from a re-presentation
    triggered by another matching file while the workflow was already active.
    """

    workflow_name: str
    triggering_file: str
    steps: List[str]
    newly_activated: bool = True


class DeactivationReport(BaseModel):
    """A deactivation indication naming the workflow and the reason.

    Emitted both when steps cannot be presented (Req 14.4) and when the
    triggering file no longer matches (Req 14.6). ``reason`` is a human-readable
    explanation; ``triggering_file`` is the file whose event caused it when
    known.
    """

    workflow_name: str
    reason: str
    triggering_file: Optional[str] = None


class SteeringUpdate(BaseModel):
    """The result of one editor file event.

    Reports every workflow activated/re-presented (``activated``) with its
    ordered steps, every workflow deactivated (``deactivated``) with its named
    reason, and the names of all workflows still active afterwards.
    """

    file_path: str
    activated: List[StepPresentation] = Field(default_factory=list)
    deactivated: List[DeactivationReport] = Field(default_factory=list)
    active_workflows: List[str] = Field(default_factory=list)


class StepPresentationError(Exception):
    """Raised by a step presenter when a workflow's steps cannot be presented.

    The engine catches this, deactivates the affected workflow, and emits a
    :class:`DeactivationReport` naming the workflow and this error's message as
    the reason (Requirement 14.4).
    """


# A step presenter is invoked when a workflow activates; it presents the
# workflow's ordered steps to the model. Raising ``StepPresentationError``
# (or any exception) signals that the steps could not be presented (Req 14.4).
StepPresenter = Callable[[SteeringWorkflow, str], None]


def _default_step_presenter(workflow: SteeringWorkflow, file_path: str) -> None:
    """Default presenter: a workflow with no steps cannot be presented.

    The real hub wires a presenter that surfaces the ordered steps to the
    model. The default encodes the one universal precondition — there must be
    ordered steps to present (Req 14.3) — so that an empty/malformed workflow
    deactivates with a clear reason (Req 14.4) rather than activating silently.
    """
    if not workflow.steps:
        raise StepPresentationError(
            "workflow has no ordered steps to present"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class SteeringEngine:
    """Governs steering workflow activation/deactivation (Requirement 14).

    State is two complementary maps:

    * ``_active``: workflow name -> set of file paths currently triggering it.
      A workflow is active iff its trigger set is non-empty.
    * ``_file_triggers``: file path -> set of workflow names it currently
      triggers. Lets a file event withdraw exactly the triggers that file was
      responsible for (Req 14.6) without disturbing workflows held active by
      other files.
    """

    def __init__(
        self,
        workflows: List[SteeringWorkflow],
        *,
        step_presenter: Optional[StepPresenter] = None,
    ) -> None:
        # Reject duplicate workflow names: the name is the report identity.
        seen: Set[str] = set()
        for wf in workflows:
            if wf.name in seen:
                raise ValueError(f"duplicate steering workflow name: {wf.name!r}")
            seen.add(wf.name)
        self._workflows: Dict[str, SteeringWorkflow] = {w.name: w for w in workflows}
        self._present: StepPresenter = step_presenter or _default_step_presenter
        self._active: Dict[str, Set[str]] = {}
        self._file_triggers: Dict[str, Set[str]] = {}

    # -- introspection ------------------------------------------------------
    @property
    def workflows(self) -> List[SteeringWorkflow]:
        """All registered workflows."""
        return list(self._workflows.values())

    @property
    def active_workflows(self) -> List[str]:
        """Names of currently active workflows (sorted for determinism)."""
        return sorted(name for name, files in self._active.items() if files)

    def is_active(self, workflow_name: str) -> bool:
        """True iff the named workflow is currently active."""
        return bool(self._active.get(workflow_name))

    # -- events -------------------------------------------------------------
    def open_or_edit(self, file_path: str) -> SteeringUpdate:
        """Handle a file being opened or edited (Req 14.2, 14.3, 14.4, 14.6).

        Activates (or re-presents) every workflow whose pattern matches
        ``file_path`` and, for any workflow this same file previously triggered
        but no longer matches, withdraws that trigger — deactivating the
        workflow if no other file keeps it active.
        """
        path = _normalize_path(file_path)
        update = SteeringUpdate(file_path=path)

        matching = {
            name for name, wf in self._workflows.items() if wf.matches(path)
        }
        previously = self._file_triggers.get(path, set())

        # Req 14.6: this file used to trigger these workflows but no longer
        # matches them -> withdraw its trigger.
        for name in sorted(previously - matching):
            self._withdraw_trigger(
                name,
                path,
                update,
                reason=(
                    f"triggering file {path!r} no longer matches workflow "
                    f"{name!r} file pattern"
                ),
            )

        # Req 14.2/14.3: activate or re-present each matching workflow.
        confirmed: Set[str] = set()
        for name in sorted(matching):
            if self._activate(name, path, update):
                confirmed.add(name)

        # Record exactly the triggers this file is now responsible for.
        if confirmed:
            self._file_triggers[path] = confirmed
        else:
            self._file_triggers.pop(path, None)

        update.active_workflows = self.active_workflows
        return update

    def rename(self, old_path: str, new_path: str) -> SteeringUpdate:
        """Handle a file rename (Req 14.6 then 14.2).

        The old path can no longer trigger its workflows; the new path is then
        evaluated as a normal open/edit. Modeled as: withdraw every trigger the
        old path held, then :meth:`open_or_edit` the new path.
        """
        old = _normalize_path(old_path)
        new = _normalize_path(new_path)
        update = SteeringUpdate(file_path=new)

        for name in sorted(self._file_triggers.get(old, set())):
            self._withdraw_trigger(
                name,
                old,
                update,
                reason=(
                    f"triggering file {old!r} was renamed to {new!r} and no "
                    f"longer matches workflow {name!r} file pattern"
                ),
            )
        self._file_triggers.pop(old, None)

        opened = self.open_or_edit(new)
        update.activated.extend(opened.activated)
        # Deactivations from re-evaluating the new path (if any) are appended
        # after the rename-driven ones.
        for report in opened.deactivated:
            if report not in update.deactivated:
                update.deactivated.append(report)
        update.active_workflows = self.active_workflows
        return update

    def close(self, file_path: str) -> SteeringUpdate:
        """Handle a file being closed: withdraw all triggers it held.

        A closed file can no longer keep its workflows active; each workflow it
        triggered is deactivated unless another open file still matches it.
        """
        path = _normalize_path(file_path)
        update = SteeringUpdate(file_path=path)
        for name in sorted(self._file_triggers.get(path, set())):
            self._withdraw_trigger(
                name,
                path,
                update,
                reason=f"triggering file {path!r} was closed",
            )
        self._file_triggers.pop(path, None)
        update.active_workflows = self.active_workflows
        return update

    # -- internals ----------------------------------------------------------
    def _activate(
        self, name: str, file_path: str, update: SteeringUpdate
    ) -> bool:
        """Activate or re-present ``name`` for ``file_path``.

        Returns True if the workflow is active for this file afterwards. On a
        step-presentation failure the workflow is deactivated and a report is
        emitted (Req 14.4), and the file does not retain it as a trigger.
        """
        workflow = self._workflows[name]
        was_active = self.is_active(name)
        try:
            # Req 14.3: present steps in their defined order.
            self._present(workflow, file_path)
        except Exception as exc:  # noqa: BLE001 - any failure => deactivate
            reason = (
                str(exc)
                or f"{type(exc).__name__} while presenting steps"
            )
            # Req 14.4: deactivate (drop every trigger) and report by name.
            self._drop_workflow(name)
            update.deactivated.append(
                DeactivationReport(
                    workflow_name=name,
                    reason=f"steps could not be presented: {reason}",
                    triggering_file=file_path,
                )
            )
            return False

        self._active.setdefault(name, set()).add(file_path)
        update.activated.append(
            StepPresentation(
                workflow_name=name,
                triggering_file=file_path,
                steps=list(workflow.steps),
                newly_activated=not was_active,
            )
        )
        return True

    def _withdraw_trigger(
        self,
        name: str,
        file_path: str,
        update: SteeringUpdate,
        *,
        reason: str,
    ) -> None:
        """Remove ``file_path`` as a trigger of ``name``; deactivate if last."""
        triggers = self._active.get(name)
        if not triggers or file_path not in triggers:
            return
        triggers.discard(file_path)
        if not triggers:
            self._active.pop(name, None)
            update.deactivated.append(
                DeactivationReport(
                    workflow_name=name,
                    reason=reason,
                    triggering_file=file_path,
                )
            )

    def _drop_workflow(self, name: str) -> None:
        """Fully deactivate ``name`` and detach it from every triggering file."""
        self._active.pop(name, None)
        for triggers in self._file_triggers.values():
            triggers.discard(name)


# ---------------------------------------------------------------------------
# Loading workflows from the authored steering/*.md files
# ---------------------------------------------------------------------------
_FRONT_MATTER_KEY = re.compile(r"^([A-Za-z_][\w-]*):\s*(.*)$")
_STEPS_HEADING = re.compile(r"^##\s+Steps\b", re.IGNORECASE)
_HEADING = re.compile(r"^##\s+")
_NUMBERED = re.compile(r"^\s*\d+\.\s+(.*)$")
_BOLD_TITLE = re.compile(r"^\*\*(.+?)\*\*")


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _parse_front_matter(text: str):
    """Parse the leading ``---`` YAML-ish front matter; return (data, body).

    Supports exactly the shapes used by the authored steering files: inline
    scalars (``name: cog-conversion``), folded/literal block scalars
    (``description: >-``), and block sequences (``fileMatchPattern:`` followed
    by ``  - "**/*.tif"`` items). No third-party YAML dependency is required.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end = idx
            break
    if end is None:
        return {}, text

    fm = lines[1:end]
    body = "\n".join(lines[end + 1 :])
    data: Dict[str, object] = {}
    i = 0
    while i < len(fm):
        line = fm[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        match = _FRONT_MATTER_KEY.match(line)
        if not match:
            i += 1
            continue
        key, raw = match.group(1), match.group(2).strip()

        if raw in {">", ">-", ">+", "|", "|-", "|+"}:
            block: List[str] = []
            i += 1
            while i < len(fm) and (fm[i].startswith((" ", "\t")) or not fm[i].strip()):
                block.append(fm[i].strip())
                i += 1
            folded = " " if raw.startswith(">") else "\n"
            data[key] = folded.join(part for part in block if part).strip()
            continue

        if raw == "":
            items: List[str] = []
            i += 1
            while i < len(fm) and fm[i].lstrip().startswith("- "):
                items.append(_strip_quotes(fm[i].lstrip()[2:]))
                i += 1
            data[key] = items
            continue

        data[key] = _strip_quotes(raw)
        i += 1
    return data, body


def _parse_steps(body: str) -> List[str]:
    """Extract ordered step labels from the ``## Steps (in order)`` section.

    Each numbered item's bold title (``**Identify source.**``) becomes the
    step label; items without a bold title fall back to their first sentence.
    Continuation lines are folded into the current step.
    """
    lines = body.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if _STEPS_HEADING.match(line.strip()):
            start = idx + 1
            break
    if start is None:
        return []

    raw_steps: List[str] = []
    current: Optional[str] = None
    for line in lines[start:]:
        if _HEADING.match(line.strip()):
            break
        numbered = _NUMBERED.match(line)
        if numbered:
            if current is not None:
                raw_steps.append(current)
            current = numbered.group(1).strip()
        elif current is not None and line.strip():
            current += " " + line.strip()
    if current is not None:
        raw_steps.append(current)

    labels: List[str] = []
    for step in raw_steps:
        bold = _BOLD_TITLE.match(step)
        if bold:
            labels.append(bold.group(1).strip().rstrip("."))
        else:
            labels.append(step.split(". ", 1)[0].strip())
    return labels


def _parse_workflow_file(path: str) -> SteeringWorkflow:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    data, body = _parse_front_matter(text)

    name = data.get("name") or os.path.splitext(os.path.basename(path))[0]
    patterns = data.get("fileMatchPattern", [])
    if isinstance(patterns, str):
        patterns = [patterns] if patterns else []
    description = data.get("description", "")
    if not isinstance(description, str):
        description = ""

    return SteeringWorkflow(
        name=str(name),
        description=description,
        file_match_patterns=[str(p) for p in patterns],
        steps=_parse_steps(body),
        source_path=path,
    )


def load_workflows_from_directory(directory: str) -> List[SteeringWorkflow]:
    """Load every ``*.md`` steering workflow from ``directory`` (sorted).

    Parses the front matter (``name``, ``description``, ``fileMatchPattern``)
    and the ordered steps from each authored steering file so a
    :class:`SteeringEngine` can be constructed directly from the ``steering/``
    directory.
    """
    workflows: List[SteeringWorkflow] = []
    for entry in sorted(os.listdir(directory)):
        if not entry.endswith(".md"):
            continue
        full = os.path.join(directory, entry)
        if not os.path.isfile(full):
            continue
        workflow = _parse_workflow_file(full)
        if workflow.file_match_patterns:
            workflows.append(workflow)
    return workflows
