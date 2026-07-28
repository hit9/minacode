"""minacode tool runner: batched edit planning, confirmation, and tool execution."""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import ClassVar

from minacode.base import (
    ActiveResource,
    Json,
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
    ToolCall,
    ToolError,
)
from minacode.context import ContextManager
from minacode.session import Session, TurnDiff
from minacode.tools import (
    TOOL_REGISTRY,
    AskSpec,
    AskTool,
    BashTool,
    CodeIndex,
    Edit,
    EditTool,
    JobTool,
    ReadTool,
    Tool,
)


class EditBatchPlan:
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
            return "\n".join(
                [
                    f"<Edit path={json.dumps(tool.last_path)}>",
                    tool.file_stat(self.path),
                    tool.last_diff.rstrip(),
                    tool.edit_context(self.after, self.changes),
                    "</Edit>",
                ]
            )

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
        self.planned[call.id] = self.PlannedEdit(path, before, after, created, result.changes)
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
            return self.ApplyResult(self.new_lines(ReadTool.split_lines(result.content)), result.changes, result.replacements, result.replace_all)
        lines = list(state.lines)
        for start, end, replacement in sorted(result.replacements, reverse=True):
            lines[start:end] = self.new_lines(replacement)
        return self.ApplyResult(lines, result.changes, result.replacements)

    @staticmethod
    def new_lines(lines: list[str]) -> list[Line]:
        return [EditBatchPlan.Line(line, None) for line in lines]

    def resolve_anchor(self, state: FileState, anchor: str) -> int:
        index, expected = ReadTool.require_anchor(anchor)
        if index < len(state.lines) and ReadTool.anchor_matches(state.lines[index].text, expected):
            return index
        if index < len(state.original) and ReadTool.anchor_matches(state.original[index], expected):
            current = state.current_origin(index)
            if current is not None:
                return current
            raise ToolError(f"stale anchor {anchor}; original line was changed in this batch")
        relocated = ReadTool.relocated_anchor([line.text for line in state.lines], index, expected)
        if relocated is not None:
            return relocated
        current_line = ReadTool.anchor_line(index, state.lines[index].text) if index < len(state.lines) else "out of range"
        raise ToolError(f"stale anchor {anchor}; current is {current_line}")


@dataclass
class ToolDisplay:
    """How one tool call renders: the batch-counter suffix, the short call line, whether it prints
    as a nested tree, and whether it was auto/user approved. Threaded from run_one into finish/reject."""

    batch_suffix: str = ""
    display: str | None = None
    nested_display: bool = False
    approved: bool = False
    auto: bool = False


class ToolRunner:
    BASH_TRANSCRIPT_PREVIEW_LINES: ClassVar[int] = 3
    BASH_PREVIEW_LINES: ClassVar[int] = 24
    BASH_PREVIEW_LINE_LIMIT: ClassVar[int] = 220
    EDIT_PATH_RE: ClassVar[re.Pattern] = re.compile(r'<Edit\s+path=(".*?")')
    MCP_CALL_RE: ClassVar[re.Pattern] = re.compile(r"(?s)<MCPCall\b[^>]*>\n?(.*?)\n?</MCPCall>\s*$")

    def __init__(self, session: Session, context: ContextManager, input_fn=input, output_fn=print):
        self.session = session
        self.context = context
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.live_output: Callable[[str, str], None] | None = None
        self.live_start: Callable[[], None] | None = None
        self.question_fn: Callable[[AskSpec, str], str] | None = None
        self._active_bash: ActiveResource[BashTool] = ActiveResource()

    def cancel(self) -> None:
        self._active_bash.apply(lambda tool: tool.cancel())

    def call_tool(self, tool: Tool, planned_edit: EditBatchPlan.PlannedEdit | None = None) -> str:
        if not isinstance(tool, BashTool):
            return planned_edit.call(tool) if planned_edit and isinstance(tool, EditTool) else tool.call()
        with self._active_bash.track(tool):
            return tool.call()

    def run(self, calls: list[ToolCall], batch_suffix: str = "") -> list[Json]:
        messages: list[Json] = []
        observations: list[Json] = []
        # Shared, mutated across segments: `first` controls which display carries batch_suffix;
        # `refused` short-circuits the rest of the batch once a confirmation is declined.
        state = {"first": True, "refused": False}
        index = 0
        while index < len(calls):
            if state["refused"]:
                messages.append(self.skip_message(calls[index]))
                index += 1
                continue
            end = self.parallel_segment_end(calls, index)
            if end - index >= 2 and self.session.settings.max_parallel_tools > 1:
                messages.extend(self.run_parallel(calls[index:end], batch_suffix, state))
                index = end
                continue
            end = index + 1 if self.edit_barrier(calls[index]) else self.edit_segment_end(calls, index)
            messages.extend(self.run_serial(calls[index:end], batch_suffix, state, observations))
            index = end
        return [*messages, *observations]

    def skip_message(self, call: ToolCall) -> Json:
        content = self.tool_message(call, "", "Skipped: previous tool call was refused", failed=True)
        return {"role": "tool", "tool_call_id": call.id, "content": content}

    def run_serial(self, segment: list[ToolCall], batch_suffix: str, state: dict[str, bool], observations: list[Json]) -> list[Json]:
        messages: list[Json] = []
        plan = EditBatchPlan(self.session).build(segment) if any(call.name == "Edit" for call in segment) else EditBatchPlan(self.session)
        for call in segment:
            suffix = batch_suffix if state["first"] else ""
            state["first"] = False
            status, content, observation = self.run_one(
                call, batch_suffix=suffix, planned_edit=plan.planned.get(call.id), plan_error=plan.errors.get(call.id, "")
            )
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
            if observation is not None:
                observations.append(observation)
            if status == "refused":
                state["refused"] = True
        return messages

    def run_parallel(self, segment: list[ToolCall], batch_suffix: str, state: dict[str, bool]) -> list[Json]:
        # Run the pure tool.call() work concurrently, but apply all side effects (display, session
        # bookkeeping, tool messages) on this thread in request order, so output and the results
        # handed back to the model match the order the model issued the calls.
        cap = max(1, self.session.settings.max_parallel_tools)
        outcomes: list[tuple[str, str, str | None, float] | None] = [None] * len(segment)
        with ThreadPoolExecutor(max_workers=min(len(segment), cap), thread_name_prefix="tool") as executor:
            futures = {executor.submit(self.execute_readonly, call): position for position, call in enumerate(segment)}
            for future in as_completed(futures):
                outcomes[futures[future]] = future.result()
        messages: list[Json] = []
        for call, outcome in zip(segment, outcomes):
            suffix = batch_suffix if state["first"] else ""
            state["first"] = False
            assert outcome is not None
            content = self.finalize_outcome(call, outcome, batch_suffix=suffix)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        return messages

    def parallel_segment_end(self, calls: list[ToolCall], start: int) -> int:
        end = start
        while end < len(calls) and self.parallel_safe(calls[end]):
            end += 1
        return end

    def parallel_safe(self, call: ToolCall) -> bool:
        # A call may run concurrently only if it neither mutates state nor blocks on interactive
        # input: read-only, auto-approved, non-interactive tools (Read/Search/Recall/InspectCode,
        # read-only MCP). Edit is coordinated serially by EditBatchPlan;
        # Bash streams live output and mutates; Ask blocks on the user.
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is None or call.name == "Edit" or tool_class in (BashTool, JobTool, AskTool) or tool_class.PRODUCES_MODEL_OBSERVATION:
            return False
        try:
            return not tool_class(self.session, call.args).needs_confirmation()
        except Exception:  # noqa: BLE001 - malformed third-party tool implementations are never parallel-safe.
            return False

    def execute_readonly(self, call: ToolCall) -> tuple[str, str, str | None, float]:
        # Pure execution for a parallel worker: returns (kind, output, display, elapsed) and performs
        # no display or session writes (those happen in finalize_outcome on the main thread). Mirrors
        # run_one's branches, minus confirmation (parallel_safe guarantees none is needed).
        started = time.monotonic()
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is None:
            return "reject", f"ToolError: unknown tool {call.name}", None, 0.0
        tool = tool_class(self.session, call.args)
        display = None
        try:
            display = self.short_call(call, tool.short_args())
            if call.error:
                raise ToolError(call.error)
            output = tool.call()
        except ToolError as error:
            return "reject", f"ToolError: {error}", display, time.monotonic() - started
        except Exception as error:  # noqa: BLE001 - tool failures are serialized back to the model.
            return "error", f"ToolError: {error}", display, time.monotonic() - started
        return "ok", output, display, time.monotonic() - started

    def finalize_outcome(self, call: ToolCall, outcome: tuple[str, str, str | None, float], batch_suffix: str = "") -> str:
        kind, output, display, elapsed = outcome
        d = ToolDisplay(batch_suffix=batch_suffix, display=display)
        if kind == "ok":
            return self.finish(call, output, elapsed=elapsed, d=d)
        if kind == "reject":
            return self.reject(call, output, d=d)
        return self.finish(call, output, failed=True, elapsed=elapsed, d=d)

    def edit_segment_end(self, calls: list[ToolCall], start: int) -> int:
        end = start
        while end < len(calls) and not self.edit_barrier(calls[end]):
            end += 1
        return end

    def edit_barrier(self, call: ToolCall) -> bool:
        tool_class = TOOL_REGISTRY.get(call.name)
        return call.name != "Edit" and (tool_class is None or tool_class.MUTATES or tool_class.PRODUCES_MODEL_OBSERVATION)

    def run_one(
        self,
        call: ToolCall,
        batch_suffix: str = "",
        planned_edit: EditBatchPlan.PlannedEdit | None = None,
        plan_error: str = "",
    ) -> tuple[str, str, Json | None]:
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is None:
            return "failed", self.reject(call, f"ToolError: unknown tool {call.name}", d=ToolDisplay(batch_suffix=batch_suffix)), None
        if call.error:
            return "failed", self.reject(call, f"ToolError: {call.error}", d=ToolDisplay(batch_suffix=batch_suffix)), None
        tool = tool_class(self.session, call.args)
        if isinstance(tool, BashTool):
            tool.live_output = self.live_output
        started = time.monotonic()
        d = ToolDisplay(batch_suffix=batch_suffix)
        if isinstance(tool, AskTool):
            tool.question_fn = self.question_fn
        try:
            d.display = self.short_call(call, tool.short_args())
            if plan_error:
                raise ToolError(plan_error)
            needs_confirmation = tool.needs_confirmation()
            if needs_confirmation and self.session.settings.yolo:
                d.auto = True
                pre = self.approval_display(call, tool, "auto", batch_suffix=batch_suffix, planned_edit=planned_edit)
                # The "auto …" header duplicates the result line; only surface it when it carries a
                # preview the result line won't repeat (e.g. an Edit diff). The auto-approval itself
                # is recorded by the [auto] tag on the result line below.
                if pre.has_children:
                    self.output_fn(pre)
                    d.nested_display = True
            elif needs_confirmation:
                d.nested_display = True
                confirmed, reason = self.confirm(call, tool, batch_suffix=batch_suffix, planned_edit=planned_edit)
                if not confirmed:
                    output = "Cancelled: user refused tool call" + ((": " + reason) if reason else "")
                    return "refused", self.finish(call, output, failed=True, elapsed=time.monotonic() - started, d=d), None
                d.approved = True
            if isinstance(tool, BashTool) and self.live_start is not None:
                if not d.nested_display:
                    self.output_fn(LogBlock.hierarchy(self.log_root(d.display or self.short_call(call), batch_suffix=batch_suffix, call=call), []))
                    d.nested_display = True
                self.live_start()
            output = self.call_tool(tool, planned_edit)
            observation = tool.model_observation()
        except ToolError as error:
            return "failed", self.reject(call, f"ToolError: {error}", d=d), None
        except Exception as error:  # noqa: BLE001 - tool failures are serialized back to the model.
            return "failed", self.finish(call, f"ToolError: {error}", failed=True, elapsed=time.monotonic() - started, d=d), None
        return "ok", self.finish(call, output, elapsed=time.monotonic() - started, turn_diff=tool.turn_diff(), d=d), observation

    def reject(
        self,
        call: ToolCall,
        output: str,
        *,
        d: ToolDisplay | None = None,
    ) -> str:
        d = d or ToolDisplay()
        self.session.record_tool_error("-", call.name, call.args, output)
        self.output_fn(
            LogBlock.hierarchy(None, [LogLine("error", self.oneline(output.removeprefix("ToolError:").strip(), 220), LogRole.ERROR, LogEdge.END)])
            if d.nested_display
            else self.reject_display(call, output, d=d)
        )
        return self.tool_message(call, "", output, failed=True, display=d.display)

    def reject_display(self, call: ToolCall, output: str, *, d: ToolDisplay) -> LogBlock:
        # Argument/usage rejections are usually self-corrected on retry, so show a quiet one-liner
        # (rendered dim by UiPrinter) instead of the full red failed block. The model still receives
        # the complete error so it can correct the call.
        reason = self.oneline(output.removeprefix("ToolError:").strip(), 60)
        return LogBlock.hierarchy(self.log_root((d.display or self.short_call(call)) + " · rejected: " + reason, LogRole.MUTED, d.batch_suffix, call), [])

    def finish(
        self,
        call: ToolCall,
        output: str,
        *,
        failed: bool = False,
        elapsed: float | None = None,
        store: bool = True,
        turn_diff: TurnDiff | None = None,
        d: ToolDisplay | None = None,
    ) -> str:
        d = d or ToolDisplay()
        tool_class = TOOL_REGISTRY.get(call.name)
        key = self.session.store_tool_result(call.name, call.args, output) if not failed and store and (tool_class is None or tool_class.STORES_RESULT) else ""
        if failed:
            self.session.record_tool_error(key or "-", call.name, call.args, output)
        elif key:
            self.update_code_index(call, output)
            if turn_diff and turn_diff.path and turn_diff.diff:
                self.session.store_turn_diff(
                    key,
                    self.session.state.turn_step,
                    turn_diff.path,
                    turn_diff.diff,
                    before=turn_diff.before,
                    after=turn_diff.after,
                    round=self.session.state.round_count,
                )
        self.output_fn(self.finish_display(call, key, output, failed=failed, elapsed=elapsed, d=d))
        return self.tool_message(call, key, output, failed=failed, display=d.display)

    def tool_message(self, call: ToolCall, key: str, output: str, *, failed: bool = False, display: str | None = None) -> str:
        head = "tool " + ((key + " ") if key else ("- " if failed else "")) + (display or self.short_call(call))
        rows = [head]
        if failed:
            rows.append("status: failed")
        rows.extend(["output:", self.context.bound_output(output, key).rstrip()])
        return "\n".join(rows).strip()

    def update_code_index(self, call: ToolCall, output: str) -> None:
        if call.name != "Edit":
            return
        paths = [str(call.args[0])] if call.args and isinstance(call.args[0], str) else []
        for match in self.EDIT_PATH_RE.finditer(output):
            with contextlib.suppress(json.JSONDecodeError):
                paths.append(str(json.loads(match.group(1))))
        CodeIndex(self.session).update(list(dict.fromkeys(paths)))

    def confirm(self, call: ToolCall, tool: Tool, batch_suffix: str = "", planned_edit: EditBatchPlan.PlannedEdit | None = None) -> tuple[bool, str]:
        self.output_fn(self.approval_display(call, tool, "confirm", batch_suffix=batch_suffix, planned_edit=planned_edit))
        answer = self.input_fn(LogBlock.prefix(2, LogEdge.CONTINUE) + "[Y/n or reason] ").strip()
        lower = answer.lower()
        if lower in {"", "y", "yes"}:
            return True, ""
        return False, "" if lower in {"n", "no"} else answer

    def approval_display(
        self,
        call: ToolCall,
        tool: Tool,
        status: str,
        batch_suffix: str = "",
        planned_edit: EditBatchPlan.PlannedEdit | None = None,
    ) -> LogBlock:
        role = LogRole.TOOL if status == "confirm" else LogRole.AUTO
        root = self.log_root(self.short_call(call), role, batch_suffix, call)
        children = []
        if tool.NAME != "Edit":
            return LogBlock.hierarchy(root, children)
        preview = planned_edit.preview(tool) if planned_edit and isinstance(tool, EditTool) else tool.preview()
        preview_lines = preview.rstrip().splitlines()
        if preview_lines:
            children.append(LogLine("preview", role=LogRole.META, edge=LogEdge.BRANCH))
            children.extend(LogLine("", line, LogRole.DIFF, LogEdge.CONTINUE) for line in preview_lines)
        return LogBlock.hierarchy(root, children)

    def finish_display(
        self,
        call: ToolCall,
        key: str,
        output: str,
        *,
        failed: bool,
        elapsed: float | None = None,
        d: ToolDisplay | None = None,
    ) -> str | LogBlock:
        d = d or ToolDisplay()
        if call.name == "Note" and not failed and d.display:
            return self.with_batch_suffix(d.display.removeprefix("Note ").strip(), d.batch_suffix)
        tag = " [refused]" if failed and "user refused" in output else " [failed]" if failed else " [approved]" if d.approved else " [auto]" if d.auto else ""
        tree = d.nested_display or call.name == "Bash"
        root = self.log_root(d.display or self.short_call(call), LogRole.ERROR if failed else LogRole.TOOL, d.batch_suffix, call)
        children = []
        if failed:
            label = "refused" if "user refused" in output else "error"
            children.append(LogLine(label, self.oneline(output, 220), LogRole.ERROR, LogEdge.END))
        elif call.name == "MCP":
            summary = self.mcp_result_summary(call, output, elapsed)
            if summary:
                children.append(LogLine("", summary, LogRole.META, LogEdge.END))
        elif call.name == "Bash":
            preview = self.bash_result_preview(output, self.BASH_TRANSCRIPT_PREVIEW_LINES)
            if preview:
                duration = f" · {elapsed:.1f}s" if elapsed is not None else ""
                children.append(LogLine("output" + duration, "Ctrl-O for more", LogRole.META, LogEdge.BRANCH))
                children.extend(LogLine("", line, LogRole.OUTPUT, LogEdge.CONTINUE) for line in preview.splitlines())
        elif call.name == "Ask":
            children.append(LogLine("answer", self.oneline(output, 220), LogRole.META, LogEdge.END))
        if tree and not failed:
            children.append(LogLine("stored" if key else "done", key + tag if key else tag.strip(), LogRole.META, LogEdge.END))
        elif not tree:
            tail = ((" → " + key) if key else "") + tag
            root = LogLine(root.label, root.text, root.role, meta=root.meta + tail, syntax=root.syntax)
        return LogBlock.hierarchy(None if d.nested_display else root, children)

    def log_root(self, display: str, role: LogRole = LogRole.TOOL, batch_suffix: str = "", call: ToolCall | None = None) -> LogLine:
        name, _, args = display.partition(" ")
        tool_class = TOOL_REGISTRY.get(name)
        syntax = ""
        if tool_class is not None:
            syntax = tool_class.log_lexer(call.args) if call is not None else tool_class.LOG_LEXER
        if role is LogRole.MUTED:
            syntax = ""
        # The batch counter goes into `meta` (rendered gray) instead of `args` (syntax-highlighted),
        # so it reads as a subdued tag on the same line rather than another highlighted token.
        meta = ("  " + batch_suffix) if batch_suffix else ""
        return LogLine(name, args, role, meta=meta, syntax=syntax)

    def bash_result_preview(self, output: str, line_limit: int | None = None) -> str:
        sections = []
        for name in ("stdout", "stderr"):
            text = self.tagged_output(output, name).strip()
            if text:
                sections.extend([name + ":", *("  " + line for line in self.preview_lines(text, line_limit))])
        return "\n".join(sections)

    @staticmethod
    def tagged_output(output: str, name: str) -> str:
        start_tag = f"<{name}>"
        end_tag = f"</{name}>"
        start = output.find(start_tag)
        if start < 0:
            return ""
        start += len(start_tag)
        if output.startswith("\n", start):
            start += 1
        next_section = output.find("\n<stderr>\n", start) if name == "stdout" else output.find("\n</BashToolResult>", start)
        end = output.rfind(end_tag, start, next_section if next_section >= 0 else len(output))
        if end < 0:
            return ""
        text = output[start:end]
        return text.removesuffix("\n")

    def preview_lines(self, text: str, line_limit: int | None = None) -> list[str]:
        line_limit = self.BASH_PREVIEW_LINES if line_limit is None else line_limit
        lines = [self.clip_preview_line(line) for line in text.splitlines()]
        if len(lines) <= line_limit:
            return lines
        head = line_limit // 2
        tail = line_limit - head
        omitted = len(lines) - line_limit
        noun = "line" if omitted == 1 else "lines"
        return [*lines[:head], f"... {omitted} {noun} omitted ...", *lines[-tail:]]

    def clip_preview_line(self, line: str) -> str:
        line = line.rstrip()
        return line if len(line) <= self.BASH_PREVIEW_LINE_LIMIT else line[: self.BASH_PREVIEW_LINE_LIMIT - 3].rstrip() + "..."

    def mcp_result_summary(self, call: ToolCall, output: str, elapsed: float | None) -> str:
        if str((call.args[0] if call.args and isinstance(call.args[0], dict) else {}).get("action")) != "call":
            return ""
        inner = output
        match = self.MCP_CALL_RE.match(output)
        if match:
            inner = match.group(1).strip()
        if not inner:
            shape = "empty"
        else:
            try:
                data = json.loads(inner)
            except (json.JSONDecodeError, ValueError):
                data = None
            if isinstance(data, list):
                shape = f"{len(data)} items"
            elif isinstance(data, dict):
                shape = f"{len(data)} fields"
            else:
                shape = f"{inner.count(chr(10)) + 1} lines"
        parts = [f"{shape}, {self.human_size(len(inner))}"]
        if elapsed is not None:
            parts.append(f"{elapsed:.1f}s")
        return "→ " + " · ".join(parts)

    @staticmethod
    def human_size(num_bytes: int) -> str:
        if num_bytes < 1024:
            return f"{num_bytes}B"
        if num_bytes < 1024 * 1024:
            return f"{num_bytes / 1024:.1f}KB"
        return f"{num_bytes / (1024 * 1024):.1f}MB"

    @staticmethod
    def with_batch_suffix(text: str, suffix: str) -> str:
        return text + (("  " + suffix) if suffix else "")

    def short_call(self, call: ToolCall, args: list[str] | None = None) -> str:
        tool_class = TOOL_REGISTRY.get(call.name)
        if args is None:
            try:
                args = tool_class(self.session, call.args).short_args() if tool_class is not None else [Tool.compact(arg) for arg in call.args]
            except Exception:  # noqa: BLE001 - display formatting must fall back for malformed tool arguments.
                args = [Tool.compact(arg) for arg in call.args]
        text = " ".join([call.name, *args]).strip()
        return text if "\n" in text else self.oneline(text, 200)

    @staticmethod
    def oneline(text: str, limit: int) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."
