"""Batch edit planning: resolve a batch of Edit calls against an in-memory file model."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from minacode.base import ToolCall, ToolError, split_lines
from minacode.tools.files import Edit, EditTool, ReadTool

if TYPE_CHECKING:
    from minacode.session import Session


class EditBatchPlan:
    """Resolve a batch of Edit calls against an in-memory file model before anything is written.

    Every anchor names a line as the model read it, but the second edit in a batch lands on a file the
    first already shifted. Each line therefore carries the index it came from, so `12:hash` still
    resolves after an insertion moved that line down:

        read as        after edit 1
        11 ...         11 ...
        12 target      12 <inserted>
                        13 target      <- origin 12, still the anchor's line

    Planning the batch first is also what lets confirmation show the final result rather than the
    first step of it.

    Planning touches no file. Each planned edit records the content it expects and re-checks it at
    write time, so an edit computed against a file that changed underneath is rejected instead of
    clobbering it. A call that cannot be planned records its error against the call id rather than
    raising, keeping the one-result-per-call contract.
    """

    @dataclass
    class Line:
        text: str
        origin: int | None

    @dataclass
    class FileState:
        path: str
        lines: list[EditBatchPlan.Line]
        original: list[str]
        exists: bool

        def text(self) -> str:
            return "".join(line.text for line in self.lines)

        def current_origin(self, origin: int) -> int | None:
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

    @dataclass
    class PlannedEdit:
        path: str
        before: str
        after: str
        created: bool
        changes: list[tuple[int, int, int, int]]
        warnings: str

        def preview(self, tool: EditTool) -> str:
            return tool.diff(self.path, self.before, self.after) or f"Edit({self.path})"

        def call(self, tool: EditTool) -> str:
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
            parts = [f"<Edit path={json.dumps(tool.last_path)}>", tool.file_stat(self.path), tool.last_diff.rstrip()]
            if self.warnings:
                parts.append(self.warnings)
            parts.extend((tool.edit_context(self.after, self.changes), "</Edit>"))
            return "\n".join(parts)

    def __init__(self, session: Session):
        self.session = session
        self.files: dict[str, EditBatchPlan.FileState] = {}
        self.planned: dict[str, EditBatchPlan.PlannedEdit] = {}
        self.errors: dict[str, str] = {}

    def build(self, calls: list[ToolCall]) -> EditBatchPlan:
        for call in calls:
            if call.name != "Edit":
                continue
            try:
                self.plan_call(call, EditTool(self.session, call.args))
            except ToolError as error:
                self.errors[call.id] = str(error)
        return self

    def plan_call(self, call: ToolCall, tool: EditTool) -> None:
        path, edits = tool.parse()
        state = self.file_state(tool, path, edits[0].op == "create")
        before, created = state.text(), not state.exists
        before_lines = [line.text for line in state.lines]
        result = self.apply(tool, state, edits)
        after = "".join(line.text for line in result.lines)
        if after == before and not created:
            raise ToolError(EditTool.no_changes_error_from_lines(before_lines, result.replacements, result.replace_all))
        self.planned[call.id] = self.PlannedEdit(path, before, after, created, result.changes, tool.warnings_block(before, after, edits))
        state.lines, state.exists = result.lines, True

    def file_state(self, tool: EditTool, path: str, creating: bool) -> FileState:
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
            state = self.FileState(path, [self.Line(line, index) for index, line in enumerate(original)], original, True)
        else:
            state = self.FileState(path, [], [], False)
        self.files[path] = state
        return state

    def apply(self, tool: EditTool, state: FileState, edits: list[Edit]) -> ApplyResult:
        result = tool.apply(state.text(), edits, lambda anchor: self.resolve_anchor(state, anchor))
        if edits[0].op == "create" or result.replace_all:
            return self.ApplyResult(self.new_lines(split_lines(result.content)), result.changes, result.replacements, result.replace_all)
        lines = list(state.lines)
        for start, end, replacement in sorted(result.replacements, reverse=True):
            lines[start:end] = self.new_lines(replacement)
        return self.ApplyResult(lines, result.changes, result.replacements)

    @staticmethod
    def new_lines(lines: list[str]) -> list[Line]:
        return [EditBatchPlan.Line(line, None) for line in lines]

    def resolve_anchor(self, state: FileState, anchor: str) -> int:
        index, expected = ReadTool.require_anchor(anchor)
        if 0 <= index < len(state.lines) and ReadTool.anchor_matches(state.lines[index].text, expected):
            return index
        if 0 <= index < len(state.original) and ReadTool.anchor_matches(state.original[index], expected):
            current = state.current_origin(index)
            if current is not None:
                return current
            raise ToolError(
                f"stale anchor {anchor}; original line was changed in this batch; Read again unless the returned context verifies the intended line; "
                "for a small exact edit whose old text is unique, prefer replace_unique\n"
                + EditTool.current_file_context([line.text for line in state.lines], index)
            )
        relocated = ReadTool.relocated_anchor([line.text for line in state.lines], index, expected)
        if relocated is not None:
            return relocated
        if 0 <= index < len(state.lines):
            current_line = ReadTool.anchor_line(index, state.lines[index].text)
            raise ToolError(
                f"stale anchor {anchor}; current is {current_line}; retry with a returned anchor only if its content is the line you meant; "
                "otherwise Read again; for a small exact edit whose old text is unique, prefer replace_unique\n"
                + EditTool.current_file_context([line.text for line in state.lines], index)
            )
        raise ToolError(
            f"anchor line {index + 1} out of range; file has {len(state.lines)} lines; "
            "Read again unless the returned context verifies the intended line\n" + EditTool.current_file_context([line.text for line in state.lines], index)
        )
