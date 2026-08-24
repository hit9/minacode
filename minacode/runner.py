"""minacode tool runner: batched edit planning, confirmation, and tool execution."""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, NamedTuple

from prompt_toolkit.utils import get_cwidth

from minacode.base import (
    ActiveResource,
    ApprovalView,
    Json,
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
    Text,
    ToolCall,
    ToolError,
    builtin_tool_label,
)
from minacode.context import ContextManager
from minacode.session import Session, TurnDiff
from minacode.tools import (
    TOOL_REGISTRY,
    AskSpec,
    AskTool,
    BashTool,
    CodeIndex,
    DelegateTool,
    Edit,
    EditTool,
    JobTool,
    ReadTool,
    Tool,
    ToolScript,
    ViewImageTool,
)

if TYPE_CHECKING:
    from minacode.engine import Agent
    from minacode.model import ModelClient


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


@dataclass
class ToolDisplay:
    """How one tool call renders: the batch-counter suffix, the short call line, whether it prints
    as a nested tree, and whether it was auto/user approved. Threaded from run_one into finish/reject."""

    batch_suffix: str = ""
    display: str | None = None
    nested_display: bool = False
    approved: bool = False
    auto: bool = False
    # Non-empty when a ViewImage call bridged to the [vision] entry; finish_display draws it as a
    # child of the call line so the bridge trace can never precede the call's own root.
    vision_entry: str = ""


class ToolRunner:
    """Execute one batch of tool calls, returning exactly one result per call the model emitted.

    That count is what replay depends on: refused, failed, skipped, malformed, and interrupted calls
    each still produce a matching tool message, because a history with an unanswered call is invalid
    on every provider.

    A batch is segmented rather than flat. Independent read-only calls run concurrently; mutating and
    interactive ones stay ordered, and edits in one segment are planned together so their anchors
    resolve against the file the earlier edits will have left behind.

    Concurrency covers only `call()`. Every side effect — display, session bookkeeping, the returned
    messages — is applied on this thread in the model's original order. A declined confirmation
    short-circuits the rest of the batch, and observations follow all results so it stays replayable.
    """

    BASH_TRANSCRIPT_PREVIEW_LINES: ClassVar[int] = 3
    BASH_PREVIEW_LINE_LIMIT: ClassVar[int] = 220
    EDIT_PATH_RE: ClassVar[re.Pattern] = re.compile(r'<Edit\s+path=(".*?")')
    MCP_CALL_RE: ClassVar[re.Pattern] = re.compile(r"(?s)<MCPCall\b[^>]*>\n?(.*?)\n?</MCPCall>\s*$")
    # The envelope DelegateTool._send returns for a finished delegation: attributes in fixed order,
    # the worker's answer wrapped in <worker> tags. Parsed with a couple of string scans — the
    # format is ours, so no XML parser is needed.
    DELEGATE_META_RE: ClassVar[re.Pattern] = re.compile(
        r'<Delegate action="send" steps="(\d+)" elapsed="([^"]+)" files="([^"]*)" stopped_at_max_steps="(true|false)"(?: tokens="([^"]*)")?(?: rounds="(\d+)")?(?: context_percent="(\d+)")?>'
    )

    def __init__(self, session: Session, context: ContextManager, input_fn=input, output_fn=print):
        self.session = session
        self.context = context
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.live_output: Callable[[str, str], None] | None = None
        self.live_start: Callable[..., None] | None = None
        self.worker_rule: Callable | None = None
        # Renders the worker's interim and final model text like an agent answer (markdown), wired by the loop;
        # None lets the worker publish it through its ordinary output channel (headless).
        self.worker_answer: Callable[[str], None] | None = None
        self.question_fn: Callable[[list[AskSpec]], list[str]] | None = None
        # Injected by CommandLoop: drives the Delegate confirm-time `c` config loop through the
        # shared choice selector (see CommandLoop.run_worker_config). None degrades the `c` key to
        # printing the current worker config only (headless / non-CommandLoop runners).
        self.worker_config_picker: Callable[[], None] | None = None
        # Injected by CommandLoop: opens a read-only viewer for the text behind a confirmation --
        # a Delegate order, a ToolScript body -- for the confirm-time `v`/`view` key (see
        # cli.modals.approval_text_viewer). None degrades the `v` key to printing the whole text
        # (headless / non-CommandLoop runners). The return value is the viewer's close signal and is
        # discarded here; the Ctrl-O browser's reopen loop reads it on its own.
        self.text_viewer: Callable[[ApprovalView], object] | None = None
        # How many enclosing tool calls are running the calls being logged right now. Nested calls
        # (a ToolScript's call()) are printed one level deeper per enclosing call, so the log shows
        # who made them; see nested().
        self.nesting = 0
        # Injected by CommandLoop: offers the next approval prompt's actions as a selectable row,
        # and reports whether it took (see TuiApp.set_approval_form). None, or a False return, means
        # the answer has to be typed out -- headless runs, piped stdin.
        self.approval_form: Callable[[list[tuple[str, str]]], bool] | None = None
        # Injected by CommandLoop for the Delegate worker: ModelClient only streams when on_stream
        # is set, so an unwired worker would run unstreamed and its thinking would stay invisible.
        self.model_stream: Callable[[str, str], None] | None = None
        # Injected by CommandLoop: lifecycle callbacks the worker agent must see, so retry backoff,
        # provider-side builtin calls, and automatic compaction of the worker show up in the parent
        # TUI exactly as they do for the main agent. None degrades to the default (unstreamed,
        # unlogged) behavior for headless runners.
        self.retry_wait: Callable[[bool], None] | None = None
        self.builtin_call: Callable[[str, str], None] | None = None
        self.compaction: Callable[[bool, str], None] | None = None
        # Injected by CommandLoop: a ToolScript body is the one stretch of a turn where nothing is
        # streaming and no single tool line is pending, so the divider would otherwise sit on
        # "working" for the whole batch. The source rides along so Ctrl-O can offer the script
        # while it runs. None degrades to no phase label (headless runners).
        self.script_status: Callable[[bool, str], None] | None = None
        # Bash and Job are the tools that block on something outside the agent, so Ctrl-C has to
        # reach them: Bash kills its process group, Job abandons a wait and leaves the job running.
        self._active_bash: ActiveResource[BashTool] = ActiveResource()
        self._active_job: ActiveResource[JobTool] = ActiveResource()
        # The in-flight worker agent, so Ctrl-C fans out to it (see DelegateTool).
        self._active_worker: ActiveResource[Agent] = ActiveResource()
        # The client behind explicit ViewImage vision requests, owned
        # here so cancel() reaches the in-flight request. Created lazily -- most sessions never
        # bridge an image tool call -- and shared across calls, since tool calls never overlap a
        # main-model request. See vision_client().
        self._vision_client: ModelClient | None = None

    @contextlib.contextmanager
    def nested(self):
        """Run a block with everything it logs indented one level deeper.

        Held by a tool that runs other tool calls (ToolScript), so its nested calls are printed as
        children of the call that made them. Restored on the way out even if the script raised, so
        one failure cannot leave the rest of the session permanently indented."""
        self.nesting += 1
        try:
            yield
        finally:
            self.nesting -= 1

    def emit(self, block: str | LogBlock) -> None:
        """Print a log block at the current nesting depth. Wrapping a block in another LogBlock is
        exactly one indent level to LogBlock.walk, so depth costs nothing but the wrapper. Plain
        strings (a tool display that renders itself, e.g. Note) carry no tree to indent."""
        if isinstance(block, LogBlock):
            for _ in range(self.nesting):
                block = LogBlock([self.rooted(block)], gutter=True)
        self.output_fn(block)

    @classmethod
    def rooted(cls, block: LogBlock) -> LogBlock:
        """Give a nested block's root lines the tree's continuation edge.

        Indent alone leaves a nested call looking like an ordinary one that happens to sit further
        right. The edge here and the rail the gutter draws under it are the same column, so the
        script, each call it made -- including everything that call logged below itself -- and the
        result it returned read as one unbroken bracket. Lines that already carry an edge are a
        block's own children and keep it."""
        return LogBlock(
            [
                cls.rooted(item) if isinstance(item, LogBlock) else replace(item, edge=LogEdge.CONTINUE) if item.edge is LogEdge.NONE else item
                for item in block.items
            ]
        )

    def vision_client(self) -> ModelClient:
        """The client behind tool-side vision requests, owned here so cancel() can abort one.

        The ViewImage observation would otherwise run on a throwaway client nobody can
        reach, leaving Ctrl-C dead until the provider timeout. Created lazily because most
        sessions never bridge an image tool call."""
        if self._vision_client is None:
            from minacode.model import ModelClient  # local import: model.py imports the tool registry

            self._vision_client = ModelClient(self.session)
        return self._vision_client

    def cancel(self) -> None:
        self._active_bash.apply(lambda tool: tool.cancel())
        self._active_job.apply(lambda tool: tool.cancel())
        self._active_worker.apply(lambda agent: agent.cancel())
        if self._vision_client is not None:
            self._vision_client.cancel()

    def call_tool(self, tool: Tool, planned_edit: EditBatchPlan.PlannedEdit | None = None) -> str:
        if isinstance(tool, DelegateTool):
            tool.runner = self
            return tool.call()
        if isinstance(tool, ToolScript):
            tool.runner = self
            return tool.call()
        if isinstance(tool, ViewImageTool):
            # The runner owns the vision client, so Agent.cancel() reaches an in-flight
            # observation instead of leaving it to wait out the provider timeout.
            tool.vision_observe = lambda images, question: self.vision_client().vision_observe(images, question)
            return tool.call()
        if isinstance(tool, BashTool):
            with self._active_bash.track(tool):
                return tool.call()
        if isinstance(tool, JobTool):
            with self._active_job.track(tool):
                return tool.call()
        return planned_edit.call(tool) if planned_edit and isinstance(tool, EditTool) else tool.call()

    def run(self, calls: list[ToolCall], batch_suffix: str = "") -> list[Json]:
        messages: list[Json] = []
        observations: list[Json] = []
        # Shared, mutated across segments: `first` controls which display carries batch_suffix;
        # `refused` short-circuits the rest of the batch once a confirmation is declined.
        state = {"first": True, "refused": False}
        echoed = self.session.config.provider.builtin_function_names()
        index = 0
        while index < len(calls):
            if state["refused"]:
                messages.append(self.skip_message(calls[index]))
                index += 1
                continue
            if calls[index].name in echoed:
                messages.append(self.builtin_echo_message(calls[index]))
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

    def builtin_echo_message(self, call: ToolCall) -> Json:
        """Answer a provider's own builtin function by returning its arguments unchanged.

        The provider runs the tool; the call it emits is a handshake, and the documented client
        side of it is to send the arguments straight back. The result therefore skips confirmation,
        the registry, and the usual `tool ... output:` framing — anything added here would reach the
        provider as part of its own protocol. It is logged like a tool call so the transcript still
        shows that the work happened. Evidence:
        https://platform.kimi.ai/docs/guide/use-web-search
        """
        # An unrecognized name is parsed as a single raw payload, which is exactly what to echo.
        payload = call.args[0] if len(call.args) == 1 else call.args
        content = json.dumps(payload, ensure_ascii=False)
        label = builtin_tool_label(call.name)
        self.emit(LogBlock([LogLine(label, self.oneline(content, 120), LogRole.TOOL, LogEdge.BRANCH)]))
        return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}

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
        if (
            (self.session.tool_names and call.name not in self.session.tool_names)
            or tool_class is None
            or call.name in ("Delegate", "Edit", "NextHints")
            or tool_class in (BashTool, JobTool, AskTool, ToolScript)
            or tool_class.PRODUCES_MODEL_OBSERVATION
        ):
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
        """Run one tool call, returning (status, tool message, optional observation).

        Every exit produces a message — unknown tool, malformed arguments, refusal, exception — because
        the batch owes the model one result per emitted call. The status is what the caller acts on:
        "refused" short-circuits the remaining calls, "failed" does not.

        Ordering carries meaning. The display line is built before confirmation so a declined call still
        shows what was asked, and Bash's live preview starts only after approval so nothing streams out
        of a call the user has not agreed to run.
        """
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is None:
            return "failed", self.reject(call, f"ToolError: unknown tool {call.name}", d=ToolDisplay(batch_suffix=batch_suffix)), None
        if self.session.tool_names and call.name not in self.session.tool_names:
            return "failed", self.reject(call, f"ToolError: {call.name} is not available in this session", d=ToolDisplay(batch_suffix=batch_suffix)), None
        if call.error:
            return "failed", self.reject(call, f"ToolError: {call.error}", d=ToolDisplay(batch_suffix=batch_suffix)), None
        tool = tool_class(self.session, call.args)
        if isinstance(tool, (BashTool, JobTool)):
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
            if needs_confirmation and self.session.settings.yolo and not tool.always_confirms():
                d.auto = True
                pre = self.approval_display(call, tool, "auto", batch_suffix=batch_suffix, planned_edit=planned_edit)
                # The "auto …" header duplicates the result line; only surface it when it carries a
                # preview the result line won't repeat (e.g. an Edit diff). The auto-approval itself
                # is recorded by the [auto] tag on the result line below.
                if pre.has_children:
                    self.emit(pre)
                    d.nested_display = True
            elif needs_confirmation:
                if not isinstance(tool, DelegateTool):
                    # Nested displays drop the root line to avoid duplicating the confirmation
                    # block's own root. A Delegate send keeps its root: the finish block is the
                    # closing marker of the delegation bracket and must carry the same yellow
                    # [worker] identity as the start marker.
                    d.nested_display = True
                confirmed, reason = self.confirm(call, tool, batch_suffix=batch_suffix, planned_edit=planned_edit)
                if not confirmed:
                    output = "Cancelled: user refused tool call" + ((": " + reason) if reason else "")
                    return "refused", self.finish(call, output, failed=True, elapsed=time.monotonic() - started, d=d), None
                d.approved = True
            if isinstance(tool, BashTool) and self.live_start is not None:
                if not d.nested_display:
                    self.emit(LogBlock.hierarchy(self.log_root(d.display or self.short_call(call), batch_suffix=batch_suffix, call=call), []))
                    d.nested_display = True
                self.live_start()
            elif isinstance(tool, JobTool) and tool.blocks_agent() and self.live_start is not None:
                # A blocking Job wait streams the job's log into the same live preview as Bash, so
                # it draws the root line up front and hands the preview the wait budget for the
                # countdown, exactly like Bash's pre-block.
                if not d.nested_display:
                    self.emit(LogBlock.hierarchy(self.log_root(d.display or self.short_call(call), batch_suffix=batch_suffix, call=call), []))
                    d.nested_display = True
                self.live_start(tool.wait_budget(tool.payload()))
            elif tool.blocks_agent() and not d.nested_display:
                # A blocking call with no live preview wired up (e.g. headless, or a Job wait
                # outside the runner) still prints its call line now -- as a leaf the finish block
                # will hang children under -- so the user sees the agent is waiting instead of a
                # blank screen until the result lands. Skipped when something already drew a root
                # (an approval block, an auto preview); a second copy of the same line is noise,
                # not reassurance.
                self.emit(LogBlock.hierarchy(self.log_root(d.display or self.short_call(call), batch_suffix=batch_suffix, call=call), []))
                d.nested_display = True
            output = self.call_tool(tool, planned_edit)
            if isinstance(tool, ViewImageTool) and tool.vision_entry_label:
                d.vision_entry = tool.vision_entry_label
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
        self.emit(
            LogBlock.hierarchy(None, [LogLine("error", self.oneline(output.removeprefix("ToolError:").strip(), 220), LogRole.ERROR, LogEdge.END)])
            if d.nested_display
            else self.reject_display(call, output, d=d)
        )
        return self.tool_message(call, "", output, failed=True, display=d.display)

    def reject_display(self, call: ToolCall, output: str, *, d: ToolDisplay) -> LogBlock:
        # Argument/usage rejections are usually self-corrected on retry, so show a quiet one-liner
        # (rendered dim by UiPrinter) instead of the full red failed block. The model still receives
        # the complete error so it can correct the call.
        #
        # One line has to be enforced here, not assumed: a display is whatever the tool's short_args
        # produced, and Note's is the whole rendered note so that a successful call can print it.
        # Left alone, a rejected Note dims its entire body and hides the reason at the end of the
        # last line -- the reason for the rejection is the only part of a rejection worth reading.
        reason = self.oneline(output.removeprefix("ToolError:").strip(), 60)
        display = self.oneline(d.display or self.short_call(call), 120)
        return LogBlock.hierarchy(self.log_root(display + " · rejected: " + reason, LogRole.MUTED, d.batch_suffix, call), [])

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
        if not (tool_class is not None and tool_class.SILENT) or failed:
            self.emit(self.finish_display(call, key, output, failed=failed, elapsed=elapsed, d=d))
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
        always_option = isinstance(tool, DelegateTool) and tool.always_confirms()
        # Decided before the brief is drawn: the brief needs the actions either way -- live in the
        # form, or spelled out in the typed legend when there is no form to show them.
        actions = self.approval_actions(tool, always_option)
        form = actions if self.declare_approval_form(actions) else []
        # Printed once, outside the loop. The `c` and `v` actions come back here to ask again, and
        # a second copy of the brief in the transcript is noise: the first one is still on screen,
        # and what those actions changed (or showed) they report themselves.
        self.emit(self.approval_display(call, tool, "confirm", batch_suffix=batch_suffix, planned_edit=planned_edit, form=form, actions=actions))
        while True:
            self.declare_approval_form(actions)  # the TUI drops the form when a prompt resolves
            reply = self.input_fn(LogBlock.prefix(2, LogEdge.CONTINUE) + self.approval_prompt(always_option, form))
            if reply is None:
                # The TUI cancelled the prompt (Ctrl-C, Ctrl-D on an empty line, app shutdown).
                # A cancel is a plain refusal: never the default approve, and never a reason —
                # the model would read placeholder text as something the user actually typed.
                return False, ""
            answer = reply.strip()
            lower = answer.lower()
            if always_option and lower in {"c", "config"}:
                # The whole-line `c`/`config` opens the worker configuration loop; anything else
                # (e.g. "cost too high") is an ordinary refusal reason, so only exact matches enter.
                self.delegate_config_cycle()
                continue  # re-ask; the config cycle printed what it changed
            if lower in {"v", "view"} and (view := tool.approval_view()) is not None:
                # Same whole-line exact-match rule as `c`: `v`/`view` opens the read-only viewer on
                # whatever text this call commits to -- an order, a script -- and anything else
                # (e.g. "cost too high") stays an ordinary refusal reason.
                #
                # Built here rather than before the loop: `c` can have edited the worker config
                # since, and a viewer that reports the configuration a send will run under has to
                # read it now, not as it stood when the prompt was first drawn.
                self.view_text(view)
                continue  # re-ask; viewing changed nothing, so there is nothing to redraw
            if lower in {"", "y", "yes"}:
                return True, ""
            return False, "" if lower in {"n", "no"} else answer

    def approval_actions(self, tool: Tool, always_option: bool) -> list[tuple[str, str]]:
        """What this prompt can do, as (label, answer) pairs with the default first.

        The one answer to the question, so the action row and the typed legend cannot disagree about
        it. Ordered by how often they are wanted, because reaching the nth costs n-1 Tabs. Refusing
        sits last despite being common: Escape already refuses in one key. Every action's answer is
        the whole line the user could have typed, so the typed protocol underneath is untouched and
        a headless run loses the row, not the actions."""
        actions = [("Approve", "")]
        view = tool.approval_view()
        if view is not None:
            actions.append((f"View {view.label}", "v"))  # a tool with nothing to view returns None, so it is never offered
        if always_option:
            actions.append(("Worker config", "c"))
        actions.append(("Refuse", "n"))
        return actions

    def declare_approval_form(self, actions: list[tuple[str, str]]) -> bool:
        """Offer the actions to the TUI as a selectable row; report whether it took them. False
        (headless, piped stdin) sends the brief back to printing the typed legend."""
        return self.approval_form is not None and self.approval_form(actions)

    @staticmethod
    def approval_prompt(always_option: bool, form: list[tuple[str, str]]) -> str:
        """The one-line prompt after the brief. The form renders its own labels above the input row,
        so it only needs the field name; without one the prompt carries the typed protocol, which is
        all a headless run has."""
        if form:
            return "reason › "
        return "Approve delegation? [Y/n/c] " if always_option else "Approve? [Y/n or reason] "

    def delegate_config_cycle(self) -> None:
        """The `c` action of a Delegate send prompt: hand the interactive editing to the injected
        picker loop (CommandLoop.run_worker_config), which reuses the shared choice selector and
        writes back through the /worker pickers, then print the worker's provider/model/effort/api.

        Printed after the picker, not before: the picker already shows each current value as the
        preselected option, so what is worth logging is the config the send would now run under.
        The approval brief above keeps its original rows, so the two together read as a change.
        Without an injected picker (headless, or a runner outside CommandLoop) this just prints the
        current values; the confirmation prompt re-asks either way."""
        if self.worker_config_picker is not None:
            self.worker_config_picker()
        self.emit(self.worker_config_block())

    def view_text(self, view: ApprovalView) -> None:
        """The `v` action of a confirmation prompt: open a read-only viewer with the full,
        untruncated text behind the call. Without an injected viewer (headless, or a runner outside
        CommandLoop) this prints the whole thing; the confirmation prompt re-asks either way."""
        if self.text_viewer is not None:
            self.text_viewer(view)
        else:
            self.emit(self.full_text_block(view))

    def full_text_block(self, view: ApprovalView) -> LogBlock:
        """Headless fallback for the `v` action: header rows then the whole text, one line each, so
        nothing behind the confirmation is dropped. Code keeps its lexer here too -- the fallback is
        what a piped-stdin run reads instead of the viewer, not a lesser copy of it."""
        role = LogRole.CODE if view.lexer else LogRole.OUTPUT
        return LogBlock(
            [
                LogLine(view.label, "", LogRole.FIELD, LogEdge.BRANCH),
                *(LogLine(label, value, LogRole.FIELD, LogEdge.CONTINUE) for label, value in self._field_pairs(view.rows)),
                *(LogLine("", line, role, LogEdge.CONTINUE, syntax=view.lexer) for line in view.text.splitlines()),
            ]
        )

    def worker_config_block(self) -> LogBlock:
        """The current effective worker config as a log block: one row per knob, inherited values
        marked `(inherit)`, matching the approval brief's four worker rows (same cyan aligned
        labels)."""
        return LogBlock(
            [
                LogLine("worker config", "", LogRole.FIELD, LogEdge.BRANCH),
                *(LogLine(label, value, LogRole.FIELD, LogEdge.CONTINUE) for label, value in self._field_pairs(self.worker_config_rows())),
            ]
        )

    def worker_config_rows(self) -> list[tuple[str, str]]:
        """The effective worker provider/model/effort/api as (label, value) rows. Owned by
        DelegateTool, which also builds the send brief's header rows from it, so the `c` cycle and
        the brief can never disagree about what the worker is configured as."""
        return DelegateTool.worker_config_rows(self.session.config)

    @staticmethod
    def _field_pairs(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Pad each label to the widest label in the block, CJK-safe (visible width via
        get_cwidth), so the values start on one column."""
        width = max(get_cwidth(label) for label, _ in rows) if rows else 0
        return [(label + " " * max(0, width - get_cwidth(label)), value) for label, value in rows]

    def approval_display(
        self,
        call: ToolCall,
        tool: Tool,
        status: str,
        batch_suffix: str = "",
        planned_edit: EditBatchPlan.PlannedEdit | None = None,
        form: list[tuple[str, str]] | None = None,
        actions: list[tuple[str, str]] | None = None,
    ) -> LogBlock:
        role = LogRole.TOOL if status == "confirm" else LogRole.AUTO
        root = self.log_root(self.short_call(call), role, batch_suffix, call)
        children = []
        if isinstance(tool, DelegateTool) and tool.always_confirms():
            children.extend(self.delegate_approval_children(tool, form or [], actions))
        elif tool.NAME == "Edit":
            preview = planned_edit.preview(tool) if planned_edit and isinstance(tool, EditTool) else tool.preview()
            preview_lines = preview.rstrip().splitlines()
            if preview_lines:
                children.append(LogLine("preview", role=LogRole.META, edge=LogEdge.BRANCH))
                children.extend(LogLine("", line, LogRole.DIFF, LogEdge.CONTINUE) for line in preview_lines)
        elif (view := tool.approval_view()) is not None:
            children.extend(self.view_excerpt_children(view, status, form or [], actions or self.approval_actions(tool, False)))
        return LogBlock.hierarchy(root, children)

    # How much of an approval view's text the block itself shows. The rest is one keypress away in
    # the viewer (`v`, or Ctrl-O afterwards), and a script long enough to overflow this is exactly
    # the one whose full body in the transcript would bury everything it then goes on to do.
    VIEW_EXCERPT_LINES: ClassVar[int] = 10
    # Slack before clipping is worth it: the `… +N more lines` line costs a row of its own, so
    # hiding one or two lines buys nothing and merely sends the reader to the viewer for them.
    VIEW_EXCERPT_SLACK: ClassVar[int] = 2

    def view_excerpt_children(self, view: ApprovalView, status: str, form: list[tuple[str, str]], actions: list[tuple[str, str]]) -> list[LogLine]:
        """The opening lines of an approval view, syntax-highlighted, under a header naming what is
        clipped. CODE-role lines are lexed as one block by the renderer, so a construct spanning
        lines (a triple-quoted string) still highlights correctly inside the excerpt."""
        lines = view.text.rstrip().splitlines()
        if not lines:
            return []
        kept = len(lines) if len(lines) <= self.VIEW_EXCERPT_LINES + self.VIEW_EXCERPT_SLACK else self.VIEW_EXCERPT_LINES
        shown, hidden = lines[:kept], len(lines) - kept
        children = [
            LogLine(view.label, "", LogRole.META, LogEdge.BRANCH),
            *(LogLine("", line, LogRole.CODE, LogEdge.CONTINUE, syntax=view.lexer) for line in shown),
        ]
        # The tail line is where the rest of the text is advertised, so it is also the legend's home
        # when there is no action row to carry the keys (headless, piped stdin). Under yolo nothing
        # stops to ask, so it points at the one door that is still open afterwards: the Ctrl-O
        # browser, which is the only way to read a script that was never confirmed.
        tail = f"… +{hidden} more line{'' if hidden == 1 else 's'}" if hidden else ""
        if status != "confirm":
            tail = (tail + " · " if tail else "") + "Ctrl-O for more"
        elif not form:
            tail = (tail + " · " if tail else "") + self.approval_legend(actions, view.label)
        if tail:
            # CONTINUE, not END: the call is about to run, and what it does next -- the prompt, the
            # calls the script makes, the result line -- hangs off this same gutter. Closing the
            # tree here and reopening it below would break the bracket in the middle.
            children.append(LogLine("", tail, LogRole.META, LogEdge.CONTINUE))
        return children

    # The typed protocol, spelled out for runs with no action row to show it: headless, piped stdin.
    # Keyed by the action's answer so the legend can be built from the offered actions and cannot
    # advertise one the call has no use for; ordered here rather than by the row, which leads with
    # what is reached most often while a legend reads best with the two answers first.
    APPROVAL_LEGEND_SEGMENTS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("", "Y/Enter approve"),
        ("n", "n refuse"),
        ("c", "c worker config"),
        ("v", "v view {label}"),
    )

    @classmethod
    def approval_legend(cls, actions: list[tuple[str, str]], view_label: str = "") -> str:
        offered = {answer for _, answer in actions}
        segments = [text.format(label=view_label) for answer, text in cls.APPROVAL_LEGEND_SEGMENTS if answer in offered]
        return " · ".join(segments) + " · else reason"

    def delegate_approval_children(
        self, tool: DelegateTool, form: list[tuple[str, str]] | None = None, actions: list[tuple[str, str]] | None = None
    ) -> list[LogLine]:
        """Approval brief for a Delegate send: title, a one-line order excerpt, explicit send
        parameters, and the worker configuration the send will run under. FIELD-role rows render
        cyan left-aligned labels (padded to one column, CJK-safe) with default-foreground values;
        everything is derived from the call and the session config, never from mutable worker state.

        The key legend closes the brief only when there is no action row: with one, the same
        choices sit live above the input line, and a copy frozen into the transcript would go stale
        the moment the worker config is edited."""
        order = tool.payload_dict().get("order")
        order_row = None
        if isinstance(order, str) and order.strip():
            lines = order.strip().splitlines()
            text = self.oneline(lines[0].strip(), 100)
            if len(lines) > 1:
                text += f"  (… {len(lines) - 1} more lines)"
            order_row = ("order", text)
        rows: list[tuple[str, str, LogRole]] = [(label, value, LogRole.FIELD) for label, value in self._field_pairs(tool.header_rows(order_row))]
        if not form:
            rows.append(("", self.approval_legend(actions if actions is not None else self.approval_actions(tool, True), "order"), LogRole.META))
        last = len(rows) - 1
        return [
            LogLine(label, value, role, LogEdge.END if index == last else LogEdge.BRANCH if index == 0 else LogEdge.CONTINUE)
            for index, (label, value, role) in enumerate(rows)
        ]

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
        tree = d.nested_display or call.name in ("Bash", "Delegate") or bool(d.vision_entry)
        # A failed call explains itself in the error child below, so its root only has to identify
        # the call -- collapsed to one line, or a multi-line display (Note keeps the whole rendered
        # note there) paints its entire body red under the tag.
        label = d.display or self.short_call(call)
        root = self.log_root(self.oneline(label, 120) if failed else label, LogRole.ERROR if failed else LogRole.TOOL, d.batch_suffix, call)
        is_reset = call.name == "Delegate" and not failed and 'action="reset"' in output
        if call.name == "Delegate" and not failed and not is_reset:
            # The delegation bracket: the start marker opens with the yellow full-width rule; the
            # finish closes it with the sibling rule carrying the done summary. Without a wired
            # worker_rule the finish block falls back to the "[worker] ◀" root line with the detail
            # in the child lines below. Reset is a one-shot tool call, not a bracket: it keeps its
            # ordinary tool root and does not print a full-width rule.
            if self.worker_rule is not None:
                root = None
            else:
                root = self.log_root("[worker] ◀", LogRole.WORKER, d.batch_suffix, call)
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
        elif call.name == "ToolScript":
            # Closes the bracket the nested calls were indented under: how many of them there were,
            # how long the script took, and the first lines of what it printed -- the printed output
            # being the whole point of a script, since only that comes back to the model. The script
            # body itself stays one keypress away rather than repeated here.
            fields = self.toolscript_result_fields(output)
            if fields is not None:  # a describe returns tool shapes, not a script envelope
                counted, stdout, error = fields
                duration = f" · {elapsed:.1f}s" if elapsed is not None else ""
                # A script that raised returns its envelope normally, so the call itself did not
                # fail and nothing above this line says otherwise. Said here, or a script that died
                # on call 2 of 40 reads exactly like one that finished -- and the error, unlike the
                # printed output, is the part the reader needs.
                head = ("failed · " if error else "") + f"calls {counted}" + duration
                children.append(LogLine(head, "Ctrl-O for more", LogRole.ERROR if error else LogRole.META, LogEdge.BRANCH))
                body = error or stdout
                children.extend(
                    LogLine("", line, LogRole.ERROR if error else LogRole.OUTPUT, LogEdge.CONTINUE)
                    for line in self.preview_lines(body, self.BASH_TRANSCRIPT_PREVIEW_LINES)
                )
        elif call.name == "Ask":
            children.append(LogLine("answer", self.oneline(output, 220), LogRole.META, LogEdge.END))
        elif call.name == "Delegate":
            if 'action="reset"' in output:
                # Reset is a one-shot tool call, not a delegation bracket: it keeps its ordinary
                # tool root (above) and only adds a plain done child stating what it cleared and
                # what survives. No full-width `worker reset` rule runs.
                children.append(LogLine("done", "worker context cleared; file changes and merged diffs kept", LogRole.META, LogEdge.BRANCH))
            else:
                summary = self.delegate_result_summary(output)
                if summary:
                    if self.worker_rule is not None:
                        fields = self.delegate_result_fields(output)
                        # `summary` only renders when the envelope parsed, so fields is never None
                        # here; the guard exists for the type checker.
                        if fields is not None:
                            title = ""
                            if call.args and isinstance(call.args[0], dict):
                                raw_title = call.args[0].get("title")
                                title = raw_title.strip() if isinstance(raw_title, str) else ""
                            parts = ([title] if title else []) + [f"steps {fields.steps}", fields.elapsed]
                            if fields.in_tokens:
                                parts.append(f"{fields.in_tokens} in / {fields.out_tokens} out")
                            if fields.files != "(none)":
                                parts.append(fields.files if len(fields.files) <= 48 else fields.files[:47].rstrip() + "…")
                            self.worker_rule("worker done · " + " · ".join(parts))
                    else:
                        children.append(LogLine("done", summary, LogRole.META, LogEdge.BRANCH))
                    preview = self.delegate_answer_preview(output)
                    if preview:
                        children.extend(LogLine("", line, LogRole.OUTPUT, LogEdge.CONTINUE) for line in preview.splitlines())
        elif call.name == "ViewImage" and d.vision_entry:
            # The vision-provider observation is a child of the call line, drawn before the stored row, so
            # the trace can never appear above its own call (the attachment path's standalone
            # line is the engine's, not a tool's). TOOL, not sibling children's META: the stored
            # row is bookkeeping, this is a real request on another paid entry.
            children.append(LogLine("described by", d.vision_entry, LogRole.TOOL, LogEdge.BRANCH))
        if tree and not failed:
            children.append(LogLine("stored" if key else "done", key + tag if key else tag.strip(), LogRole.META, LogEdge.END))
        elif not tree and root is not None:
            # root can be None only on the Delegate worker_rule path, where tree is always True.
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

    def bash_result_preview(self, output: str, line_limit: int, char_limit: int | None = None) -> str:
        sections = []
        for name in ("stdout", "stderr"):
            text = self.tagged_output(output, name).strip()
            if text:
                sections.extend([name + ":", *("  " + line for line in self.preview_lines(text, line_limit, char_limit))])
        return "\n".join(sections)

    # What a scrolling viewer renders: generous next to the three-line transcript preview, but not
    # unbounded. Stored output has no cap of its own, and the text wrapper costs time quadratic in
    # the length of a single line, so one minified-JSON line would freeze the modal until it gave
    # up. Whatever these drop is still whole under the result's own tr.N key.
    VIEWER_LINES: ClassVar[int] = 2000
    VIEWER_LINE_CHARS: ClassVar[int] = 1000

    # The marker preview_lines writes where it elided; read back to describe the bound, so the note
    # in the viewer's header cannot drift from the text under it.
    OMITTED_RE: ClassVar[re.Pattern] = re.compile(r"\.\.\. (\d+) lines? omitted \.\.\.")

    def viewer_text(self, text: str) -> tuple[str, str]:
        """Arbitrary result text bounded for a scrolling viewer, with a note saying what the bound
        dropped. The same head/tail elision the transcript preview uses, at a size meant to be read
        rather than glanced at."""
        bounded = "\n".join(self.preview_lines(text, self.VIEWER_LINES, self.VIEWER_LINE_CHARS))
        return bounded, self.viewer_note(text, bounded)

    def bash_viewer_output(self, output: str) -> tuple[str, str]:
        """A Bash result's streams, labeled and bounded for a scrolling viewer, with the same note.

        Measured against the streams rather than the stored envelope: the envelope's tag lines are
        not output, and a note that counts them reads as nonsense on a one-line result."""
        bounded = self.bash_result_preview(output, self.VIEWER_LINES, self.VIEWER_LINE_CHARS)
        streams = [text for name in ("stdout", "stderr") if (text := self.tagged_output(output, name).strip())]
        return bounded, self.viewer_note("\n".join(streams), bounded)

    def viewer_note(self, source: str, bounded: str) -> str:
        """What the viewer's bound dropped, as a header phrase -- empty when it dropped nothing.

        Both facts come from the source, not from sniffing the rendered text: a clipped line whose
        tail was whitespace comes back shorter than the limit, so a length test on the output would
        stay silent about exactly the clip it was meant to report. Silence is the dangerous half --
        a reader who cannot tell an elided result from a complete one has to distrust every one."""
        lines = source.splitlines()
        omitted = sum(int(match.group(1)) for match in self.OMITTED_RE.finditer(bounded))
        parts = []
        if omitted:
            parts.append(f"{len(lines) - omitted} shown of {len(lines)}")
        if any(len(line.rstrip()) > self.VIEWER_LINE_CHARS for line in lines):
            parts.append(f"long lines clipped at {self.VIEWER_LINE_CHARS}")
        return " · ".join(parts)

    @staticmethod
    def bash_exit_code(output: str) -> str:
        """The exit code the envelope recorded, or "" when the output is not one."""
        for line in output.splitlines():
            if line.startswith("* exit_code: "):
                return line.removeprefix("* exit_code: ").strip()
        return ""

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

    # `calls: 5 [tr.95-99]`, `calls: 0`, or the bounded `calls: ... +120 keys` form, all of which
    # lead with the count -- the keys themselves are already in the log, one per nested call line.
    TOOLSCRIPT_CALLS_RE: ClassVar[re.Pattern] = re.compile(r"^calls: (?:\.\.\. \+)?(\d+)", re.MULTILINE)

    def toolscript_result_fields(self, output: str) -> tuple[str, str, str] | None:
        """(nested call count, printed stdout, error) from a ToolScript envelope, or None when the
        output is not one -- a `describe` returns tool shapes, and has no script to summarize."""
        if not output.startswith(("ToolScript ok", "ToolScript failed")):
            return None
        match = self.TOOLSCRIPT_CALLS_RE.search(output)
        sections: dict[str, list[str]] = {"stdout:": [], "error:": []}
        section = ""
        for line in output.splitlines():
            if line in ("stdout:", "stderr:", "error:"):
                section = line
            elif section in sections:
                sections[section].append(line)
        # The traceback's last line is the one that names what went wrong; the frames above it are
        # in the viewer, against the numbered source.
        error = "\n".join(sections["error:"]).strip()
        return match.group(1) if match else "0", "\n".join(sections["stdout:"]), error.splitlines()[-1] if error else ""

    def preview_lines(self, text: str, line_limit: int, char_limit: int | None = None) -> list[str]:
        lines = [self.clip_preview_line(line, char_limit) for line in text.splitlines()]
        if len(lines) <= line_limit:
            return lines
        head = line_limit // 2
        tail = line_limit - head
        omitted = len(lines) - line_limit
        noun = "line" if omitted == 1 else "lines"
        return [*lines[:head], f"... {omitted} {noun} omitted ...", *lines[-tail:]]

    def clip_preview_line(self, line: str, char_limit: int | None = None) -> str:
        limit = self.BASH_PREVIEW_LINE_LIMIT if char_limit is None else char_limit
        line = line.rstrip()
        return line if len(line) <= limit else line[: limit - 3].rstrip() + "..."

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

    class DelegateFields(NamedTuple):
        """The display fields of a finished Delegate send envelope. Named rather than a plain tuple
        because both readers want a different subset, and the envelope keeps gaining attributes."""

        steps: str
        elapsed: str
        files: str
        in_tokens: str
        out_tokens: str
        stopped: bool
        rounds: str
        context_percent: str

    def delegate_result_fields(self, output: str) -> DelegateFields | None:
        """Parse a finished Delegate send envelope into its display fields, or None when the
        envelope is missing. rounds/context_percent are "" when the envelope was written before
        they existed. Shared by delegate_result_summary (the fallback child line) and the finish
        rule label, so both show the same numbers.
        """
        match = self.DELEGATE_META_RE.search(output)
        if not match:
            return None
        steps, elapsed, files, stopped, tokens, rounds, context_percent = match.groups()
        if tokens is not None:
            in_tokens, out_tokens = tokens.split("/", 1)
            in_tokens = Text.abbreviate_count(int(in_tokens))
            out_tokens = Text.abbreviate_count(int(out_tokens))
        else:
            in_tokens = out_tokens = ""
        return self.DelegateFields(steps, elapsed, files, in_tokens, out_tokens, stopped == "true", rounds or "", context_percent or "")

    def delegate_result_summary(self, output: str) -> str:
        """The one-line summary of a finished Delegate send, from its envelope attributes."""
        fields = self.delegate_result_fields(output)
        if fields is None:
            return ""
        parts = [f"steps {fields.steps}", fields.elapsed, fields.files]
        if fields.in_tokens:
            parts.append(f"{fields.in_tokens} in / {fields.out_tokens} out")
        if fields.rounds:
            parts.append(f"round {fields.rounds}")
        if fields.context_percent:
            parts.append(f"ctx {fields.context_percent}%")
        if fields.stopped:
            parts.append("stopped at max steps")
        return " · ".join(parts)

    def delegate_answer_preview(self, output: str) -> str:
        """The worker's answer (the text between <worker> and </worker>), bounded like the Bash
        transcript preview: clipped per line and capped at BASH_TRANSCRIPT_PREVIEW_LINES."""
        start = output.find("<worker>")
        end = output.find("</worker>")
        if start < 0 or end <= start:
            return ""
        answer = output[start + len("<worker>") : end].strip()
        if not answer:
            return ""
        return "\n".join(self.preview_lines(answer, self.BASH_TRANSCRIPT_PREVIEW_LINES))

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
