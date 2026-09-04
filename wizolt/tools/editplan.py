"""Batch edit planning: resolve a batch of Edit calls against an in-memory file model."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from wizolt.base import ToolCall, ToolError, run_blocking, split_lines
from wizolt.source import SOURCE_TARGET_CONSUMED, SourceView, ToolOutput, source_error
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
        consumed: set[tuple[str, int]] = field(default_factory=set)  # origins an earlier edit in this batch replaced or deleted
        edited: bool = False  # an earlier call in this batch already applied an edit to this file

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
        relocations: list[str] = field(default_factory=list)
        consumed: set[tuple[str, int]] = field(default_factory=set)
        seam_duplicates: list[str] = field(default_factory=list)

    @dataclass(frozen=True)
    class FileSnapshot:
        """Immutable disk state used to build the virtual edit model on the loop."""

        exists: bool
        is_directory: bool
        content: str
        parent_exists: bool
        parent_is_directory: bool

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
            return tool.diff(self.path, self.before, self.after, created=self.created) or f"Edit({self.path})"

        def transact(self) -> EditBatchPlan.PlannedEdit:
            """Check-before-write and write on a blocking worker, returning its receipt."""
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
            return self

        async def apply(self, tool: EditTool) -> ToolOutput:
            """Apply the transaction off-loop, then publish its result on the owning loop."""
            output: list[ToolOutput] = []

            def commit(receipt: EditBatchPlan.PlannedEdit) -> None:
                output.append(
                    tool.result(
                        receipt.path,
                        receipt.before,
                        receipt.after,
                        receipt.changes,
                        receipt.warnings,
                        receipt.relocations,
                        created=receipt.created,
                    )
                )

            await run_blocking(self.transact, commit=commit)
            return output[0]

    def __init__(self, session: Session):
        self.session = session
        self.files: dict[str, EditBatchPlan.FileState] = {}
        self.planned: dict[str, EditBatchPlan.PlannedEdit] = {}
        self.errors: dict[str, tuple[str, object | None]] = {}

    async def build(self, calls: list[ToolCall]) -> EditBatchPlan:
        prepared: list[tuple[ToolCall, EditTool, str, bool, SourceView | None, list[Edit]]] = []
        missing: list[tuple[ToolCall, EditTool, str, str, list[Edit]]] = []
        for call in calls:
            if call.name != "Edit":
                continue
            try:
                tool = EditTool(self.session, call.args)
                path, source_name, edits = tool.parse()
                creating = edits[0].op == "create"
                try:
                    view = tool.resolve_view(path, source_name, creating, edits, recover_missing=False)
                except ToolError as error:
                    if str(error).startswith("source missing"):
                        missing.append((call, tool, path, source_name, edits))
                        continue
                    raise
                prepared.append((call, tool, path, creating, view, edits))
            except ToolError as error:
                self.errors[call.id] = (str(error), error.recovery)
        recoverable_missing_paths = (path for _, _, path, _, _ in missing if self.session.in_cwd(path) or self.session.owns_asset(path))
        paths = tuple(dict.fromkeys([*(path for _, _, path, _, _, _ in prepared), *recoverable_missing_paths]))
        snapshots = await run_blocking(lambda: {path: self.snapshot(path) for path in paths}) if paths else {}
        for call, tool, path, source_name, edits in missing:
            snapshot = snapshots.get(path)
            recovery = (
                tool.current_view_recovery_from_lines(path, edits, split_lines(snapshot.content))
                if snapshot is not None and snapshot.exists and not snapshot.is_directory
                else None
            )
            hint = "use the fresh view below" if recovery else "Read or Search again to obtain a current view"
            self.errors[call.id] = (f"source missing {source_name} is unknown or expired; {hint}", recovery)
        for call, tool, path, creating, view, edits in prepared:
            try:
                self.plan_call(call, tool, path, creating, view, edits, snapshots[path])
            except ToolError as error:
                self.errors[call.id] = (str(error), error.recovery)
        return self

    def plan_call(
        self,
        call: ToolCall,
        tool: EditTool,
        path: str,
        creating: bool,
        view: SourceView | None,
        edits: list[Edit],
        snapshot: FileSnapshot,
    ) -> None:
        state = self.file_state(path, creating, view, snapshot)
        before, created = state.text(), not state.exists
        before_lines = [line.text for line in state.lines]
        result = self.apply(tool, state, edits, view)
        after = "".join(line.text for line in result.lines)
        if after == before and not created:
            raise ToolError(
                EditTool.no_changes_error_from_lines(before_lines, result.replacements),
                recovery=tool.no_op_recovery(path, view, before, result.replacements),
            )
        self.planned[call.id] = self.PlannedEdit(
            path, before, after, created, result.changes, tool.warnings_block(edits, result.seam_duplicates), result.relocations
        )
        state.lines, state.exists, state.edited = result.lines, True, True
        state.consumed |= result.consumed

    def file_state(self, path: str, creating: bool, view: SourceView | None, snapshot: FileSnapshot) -> FileState:
        if path in self.files:
            state = self.files[path]
            if not state.exists and not creating:
                raise ToolError("file does not exist; use op=create to create it")
            if state.exists and creating and state.lines:
                raise ToolError("file already exists")
            return state
        if snapshot.exists:
            if creating:
                if not snapshot.is_directory and not snapshot.content:
                    state = self.FileState(path, [], [], False)
                    self.files[path] = state
                    return state
                raise ToolError("file already exists")
            if snapshot.is_directory:
                raise ToolError("path is a directory")
            original = split_lines(snapshot.content)
            key = view.key if view is not None else ""
            state = self.FileState(
                path,
                [self.Line(line, (key, i + 1)) for i, line in enumerate(original)],
                original,
                True,
                base_key=key or None,
            )
        elif not creating:
            raise ToolError("file does not exist; use op=create to create it")
        else:
            parent = os.path.dirname(path) or "."
            if snapshot.parent_exists and not snapshot.parent_is_directory:
                raise ToolError("parent path is not a directory")
            if not snapshot.parent_exists and not self.session.in_cwd(parent):
                raise ToolError("parent directory outside workspace does not exist; create it with an approved Bash mkdir, then retry Edit")
            state = self.FileState(path, [], [], False)
        self.files[path] = state
        return state

    @staticmethod
    def snapshot(path: str) -> FileSnapshot:
        """Read one edit target and the metadata needed to validate creation."""
        if os.path.exists(path):
            if os.path.isdir(path):
                return EditBatchPlan.FileSnapshot(True, True, "", True, True)
            with open(path, encoding="utf-8") as file:
                return EditBatchPlan.FileSnapshot(True, False, file.read(), True, True)
        parent = os.path.dirname(path) or "."
        return EditBatchPlan.FileSnapshot(False, False, "", os.path.exists(parent), os.path.isdir(parent))

    def apply(self, tool: EditTool, state: FileState, edits: list[Edit], view: SourceView | None) -> ApplyResult:
        """Apply one call to the planned file, then rebuild the planned lines with their origins.

        The resolution itself is EditTool's, run against the planned text instead of the file on
        disk: this plan only supplies where earlier edits in the batch moved each line to, and the
        fresh view a failure hands back, which must describe the file as it still is on disk.
        """
        if edits[0].op == "create":
            lines = tool.content_lines(edits[0].content, False)
            return self.ApplyResult(self.new_lines(lines), [(0, 0, 0, len(lines))], [], relocations=[])
        assert view is not None
        result = tool.apply(
            state.text(),
            edits,
            view,
            locate=lambda edit, first, last: self.planned_start(state, view, edit, first, last),
            recover=lambda _edit: tool.current_view_recovery_from_lines(view.path, edits, state.original),
        )
        lines = list(state.lines)
        consumed: set[tuple[str, int]] = set()
        for start, end, replacement in sorted(result.replacements, reverse=True):
            consumed.update(line.origin for line in lines[start:end] if line.origin is not None)
            lines[start:end] = self.new_lines(replacement)
        return self.ApplyResult(
            lines, result.changes, result.replacements, relocations=result.relocations, consumed=consumed, seam_duplicates=result.seam_duplicates
        )

    def planned_start(self, state: FileState, view: SourceView, edit: Edit, start: int, end: int) -> int | None:
        """Where the view's lines `start..end` (1-based, inclusive) sit in the planned state, or
        None when this plan has no position of its own to offer and the view's coordinate stands.

        Lines are found by their (view key, source line) origin; a different view of the same path
        is aligned to the state's base view when its text still matches the original file, so calls
        in one batch may mix views of one path. A line an earlier edit in this batch replaced or
        deleted is refused: its content is gone, and hunting for a copy of it elsewhere is exactly
        the guess relocation must never make.

        Until this batch has actually edited the file, an origin is only the view's own coordinate
        wearing a label -- the lines were enumerated from the file as read, so a file that drifted
        before the batch began leaves no trace in them. None says so, and EditTool validates that
        coordinate the same way it does outside a batch. Once an edit has been applied, the index
        reflects where this plan moved the line, over a state the plan owns and re-verifies against
        disk before writing, so it is a position rather than an assumption.
        """
        label = f"{view.key} lines {edit.start}:{edit.end}"
        origins = [self.origin_key(state, view, line_no) for line_no in range(start, end + 1)]
        if any(origin in state.consumed for origin in origins):
            raise source_error(SOURCE_TARGET_CONSUMED, f"{label} were replaced or deleted by an earlier edit in this batch; Read again")
        if not state.edited:
            return None
        indices = [state.current_index(origin) for origin in origins]
        if any(index is None for index in indices):
            return None  # not tracked in the planned model: validate or relocate by content
        resolved = [index for index in indices if index is not None]
        if resolved != list(range(resolved[0], resolved[0] + len(resolved))):
            raise source_error(SOURCE_TARGET_CONSUMED, f"{label} were split by an earlier edit in this batch; Read again")
        return resolved[0]

    def origin_key(self, state: FileState, view: SourceView, line_no: int) -> tuple[str, int]:
        """The origin that identifies view line `line_no` in this file's planned state."""
        if state.base_key and state.base_key != view.key and line_no - 1 < len(state.original) and view.line(line_no) == state.original[line_no - 1]:
            return (state.base_key, line_no)
        return (view.key, line_no)

    @staticmethod
    def new_lines(lines: list[str]) -> list[Line]:
        return [EditBatchPlan.Line(line, None) for line in lines]
