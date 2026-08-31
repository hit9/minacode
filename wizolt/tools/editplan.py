"""Batch edit planning: resolve a batch of Edit calls against an in-memory file model."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from wizolt.base import ToolCall, ToolError, split_lines
from wizolt.source import (
    SOURCE_TARGET_CHANGED,
    SOURCE_TARGET_CONSUMED,
    SourceBlock,
    SourceView,
    ToolOutput,
    fresh_context_block,
    insertion_witness,
    range_lines,
    render_tool_output,
    source_error,
)
from wizolt.tools.files import Edit, EditTool

if TYPE_CHECKING:
    from wizolt.session import Session


class EditBatchPlan:
    """Resolve a batch of Edit calls against an in-memory file model before anything is written.

    Every line in the planned model carries the source view and line it came from, so a later
    call in the same batch can use a pre-edit view after earlier insertions shifted its untouched
    lines. A later call targeting a line consumed by an earlier edit is refused; calls using
    different views of one path validate against the planned current state in model order.
    This is also what lets confirmation show the final result rather than the first step of it.

    Planning touches no file. Each planned edit records the content it expects and re-checks it
    at write time, so an edit computed against a file that changed underneath is rejected instead
    of clobbering it. A call that cannot be planned records its error against the call id rather
    than raising, keeping the one-result-per-call contract.
    """

    @dataclass
    class Line:
        text: str
        origin: tuple[str, int] | None  # (view key, 1-based source line); None = written by a batch edit

    @dataclass
    class FileState:
        path: str
        lines: list[EditBatchPlan.Line]
        original: list[str]
        exists: bool
        base_key: str | None = None  # view key whose 1-based lines index the original file

        def text(self) -> str:
            return "".join(line.text for line in self.lines)

        def current_index(self, origin: tuple[str, int]) -> int | None:
            for index, line in enumerate(self.lines):
                if line.origin == origin:
                    return index
            return None

    @dataclass
    class ApplyResult:
        lines: list[EditBatchPlan.Line]
        changes: list[tuple[int, int, int, int]]
        replacements: list[tuple[int, int, list[str]]]
        replace_all: bool = False
        relocations: list[str] = field(default_factory=list)

    @dataclass
    class PlannedEdit:
        path: str
        before: str
        after: str
        created: bool
        changes: list[tuple[int, int, int, int]]
        warnings: str
        relocations: list[str] = field(default_factory=list)

        def preview(self, tool: EditTool) -> str:
            return tool.diff(self.path, self.before, self.after) or f"Edit({self.path})"

        def call(self, tool: EditTool) -> ToolOutput:
            """Check-before-write, then write and render the Edit envelope with a fresh view."""
            if os.path.isdir(self.path):
                raise ToolError("planned edit is stale; path is a directory")
            if os.path.exists(self.path):
                with open(self.path, encoding="utf-8") as file:
                    current = file.read()
            elif self.created and not self.before:
                current = ""
            else:
                raise ToolError("planned edit is stale; file changed")
            if current != self.before:
                raise ToolError("planned edit is stale; file changed")
            if self.created:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as file:
                file.write(self.after)
            tool.last_path = tool.session.relpath(self.path)
            tool.last_diff = tool.diff(self.path, self.before, self.after)
            tool.last_before = self.before
            tool.last_after = self.after
            lines = split_lines(self.after)
            parts: list[str | SourceBlock] = [f"<Edit path={json.dumps(tool.last_path)}>", tool.last_diff.rstrip()]
            if self.warnings:
                parts.append(self.warnings)
            if self.relocations:
                parts.append("\n".join(self.relocations))
            parts.append(tool.fresh_block(self.path, lines, self.changes))
            parts.append("</Edit>")
            retained = render_tool_output(ToolOutput("", tuple(parts)), [None] * len(parts))
            return ToolOutput(retained, tuple(parts))

    def __init__(self, session: Session):
        self.session = session
        self.files: dict[str, EditBatchPlan.FileState] = {}
        self.planned: dict[str, EditBatchPlan.PlannedEdit] = {}
        self.errors: dict[str, tuple[str, object | None]] = {}

    def build(self, calls: list[ToolCall]) -> EditBatchPlan:
        for call in calls:
            if call.name != "Edit":
                continue
            try:
                self.plan_call(call, EditTool(self.session, call.args))
            except ToolError as error:
                self.errors[call.id] = (str(error), getattr(error, "recovery", None))
        return self

    def plan_call(self, call: ToolCall, tool: EditTool) -> None:
        path, source_name, edits = tool.parse()
        creating = edits[0].op == "create"
        view = tool.resolve_view(path, source_name, creating)
        state = self.file_state(tool, path, creating, view)
        before, created = state.text(), not state.exists
        before_lines = [line.text for line in state.lines]
        result = self.apply(tool, state, edits, view)
        after = "".join(line.text for line in result.lines)
        if after == before and not created:
            raise ToolError(EditTool.no_changes_error_from_lines(before_lines, result.replacements, result.replace_all))
        self.planned[call.id] = self.PlannedEdit(path, before, after, created, result.changes, tool.warnings_block(before, after, edits), result.relocations)
        state.lines, state.exists = result.lines, True

    def file_state(self, tool: EditTool, path: str, creating: bool, view: SourceView | None) -> FileState:
        if path in self.files:
            state = self.files[path]
            if not state.exists and not creating:
                raise ToolError("file does not exist; use op=create to create it")
            if state.exists and creating:
                raise ToolError("file already exists")
            return state
        if tool._validate_target(path, creating):
            with open(path, encoding="utf-8") as file:
                original = file.readlines()
            key = view.key if view is not None else ""
            state = self.FileState(
                path,
                [self.Line(line, (key, i + 1)) for i, line in enumerate(original)],
                original,
                True,
                base_key=key or None,
            )
        else:
            state = self.FileState(path, [], [], False)
        self.files[path] = state
        return state

    def apply(self, tool: EditTool, state: FileState, edits: list[Edit], view: SourceView | None) -> ApplyResult:
        if edits[0].op == "create":
            lines = tool.content_lines(edits[0].content, False)
            return self.ApplyResult(self.new_lines(lines), [(0, 0, 0, len(lines))], [], relocations=[])
        assert view is not None
        replacements, relocations = self.resolve_batch(tool, state, edits, view)
        result = tool.splice_lines([line.text for line in state.lines], replacements, relocations)
        lines = list(state.lines)
        for start, end, replacement in sorted(result.replacements, reverse=True):
            lines[start:end] = self.new_lines(replacement)
        return self.ApplyResult(lines, result.changes, result.replacements, relocations=result.relocations)

    def resolve_batch(self, tool: EditTool, state: FileState, edits: list[Edit], view: SourceView) -> tuple[list[tuple[int, int, list[str]]], list[str]]:
        """Resolve every operation against the planned state, by source-view line origin."""
        replacements: list[tuple[int, int, list[str]]] = []
        relocations: list[str] = []
        for edit in edits:
            try:
                if edit.op in {"replace", "delete"}:
                    target = range_lines(view, edit.start, edit.end)
                    indices = self.planned_indices(state, view, edit.start, edit.end)
                    if not indices or any(index is None for index in indices):
                        raise source_error(
                            SOURCE_TARGET_CONSUMED,
                            f"{view.key} lines {edit.start}:{edit.end} were replaced or deleted by an earlier edit in this batch; Read again",
                        )
                    resolved = [index for index in indices if index is not None]
                    start_idx = resolved[0]
                    if resolved != list(range(start_idx, start_idx + len(resolved))):
                        raise source_error(
                            SOURCE_TARGET_CONSUMED,
                            f"{view.key} lines {edit.start}:{edit.end} were split by an earlier edit in this batch; Read again",
                        )
                    planned = [line.text for line in state.lines[start_idx : start_idx + len(target)]]
                    if planned != list(target):
                        raise source_error(
                            SOURCE_TARGET_CHANGED,
                            f"{view.key} lines {edit.start}:{edit.end} were changed by an earlier edit in this batch; Read again",
                        )
                    replacement = [] if edit.op == "delete" else tool.content_lines(edit.content, start_idx + len(target) < len(state.lines))
                    replacements.append((start_idx, start_idx + len(target), replacement))
                else:
                    after = edit.op == "insert_after"
                    witness, boundary, index = insertion_witness(view, edit.line, after)
                    if not witness and view.total_lines == 0:
                        if state.lines:
                            raise source_error(SOURCE_TARGET_CHANGED, "the empty file now has content; Read again for a current view")
                        at = 0
                    else:
                        witness_lines = [index + 1 - boundary + offset for offset in range(len(witness))]
                        indices = self.planned_indices(state, view, witness_lines[0], witness_lines[-1])
                        if not indices or any(item is None for item in indices):
                            raise source_error(
                                SOURCE_TARGET_CONSUMED,
                                f"{view.key} line {edit.line} boundary was replaced or deleted by an earlier edit in this batch; Read again",
                            )
                        resolved = [index for index in indices if index is not None]
                        if resolved != list(range(resolved[0], resolved[0] + len(resolved))):
                            raise source_error(
                                SOURCE_TARGET_CONSUMED,
                                f"{view.key} line {edit.line} boundary was split by an earlier edit in this batch; Read again",
                            )
                        planned = [line.text for line in state.lines[resolved[0] : resolved[-1] + 1]]
                        if planned != list(witness):
                            raise source_error(
                                SOURCE_TARGET_CHANGED,
                                f"{view.key} line {edit.line} boundary was changed by an earlier edit in this batch; Read again",
                            )
                        at = resolved[0] + boundary
                    replacement = tool.content_lines(edit.content, at < len(state.lines))
                    replacements.append((at, at, replacement))
            except ToolError as error:
                raise self._with_recovery(error, state, view, edit) from error
        return replacements, relocations

    def planned_indices(self, state: FileState, view: SourceView, start: int, end: int) -> list[int | None]:
        """Planned 0-based indices for the view's lines `start..end` (1-based, inclusive).

        Lines are found by their (view key, source line) origin. A different view of the same
        path is aligned to the state's base view when its text still matches the original file,
        so calls in one batch may mix views of one path.
        """
        out: list[int | None] = []
        for line_no in range(start, end + 1):
            index = state.current_index((view.key, line_no))
            if (
                index is None
                and state.base_key
                and state.base_key != view.key
                and line_no - 1 < len(state.original)
                and self.view_line_text(view, line_no) == state.original[line_no - 1]
            ):
                index = state.current_index((state.base_key, line_no))
            out.append(index)
        return out

    @staticmethod
    def view_line_text(view: SourceView, line_no: int) -> str | None:
        for span in view.spans:
            if span.start <= line_no <= span.end:
                return span.lines[line_no - span.start]
        return None

    def _with_recovery(self, error: ToolError, state: FileState, view: SourceView, edit: Edit) -> ToolError:
        """Attach a fresh bounded view of the pre-batch file around the requested coordinates."""
        center = (edit.start - 1) if edit.op in {"replace", "delete"} else max(0, edit.line - 1)
        recovery = ToolOutput("", (fresh_context_block(view.path, view.display_path, state.original, center),))
        return ToolError(str(error), recovery=recovery)

    @staticmethod
    def new_lines(lines: list[str]) -> list[Line]:
        return [EditBatchPlan.Line(line, None) for line in lines]
