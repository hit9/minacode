"""wizolt tool runner: batched edit planning, confirmation, and tool execution."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import TYPE_CHECKING

from wizolt.base import (
    MAX_TOOL_OUTPUT_TOKENS,
    ActiveResource,
    ApprovalView,
    Json,
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
    ToolCall,
    ToolError,
    builtin_tool_label,
    fail_if_running_loop,
    oneline,
)
from wizolt.context import ContextManager
from wizolt.model import ModelClient
from wizolt.session import Session, TurnDiff
from wizolt.source import SourceBlock, TextBlock, ToolOutput
from wizolt.tools import (
    TOOL_REGISTRY,
    AskSpec,
    AskTool,
    BashTool,
    CodeIndex,
    DelegateTool,
    EditTool,
    JobTool,
    Tool,
    ToolScript,
    ViewImageTool,
    toolblocks,
    tooloutput,
)
from wizolt.tools.editplan import EditBatchPlan
from wizolt.tools.toolblocks import ToolDisplay
from wizolt.vision import VisionObserver

if TYPE_CHECKING:
    from wizolt.engine import Agent


class ToolRunner:
    """Execute one batch of tool calls, returning exactly one result per call the model emitted.

    That count is what replay depends on: refused, failed, skipped, malformed, and interrupted calls
    each still produce a matching tool message, because a history with an unanswered call is invalid
    on every provider.

    A batch is segmented rather than flat. Independent read-only calls run concurrently; mutating and
    interactive ones stay ordered, and edits in one segment are planned together so their line
    origins resolve against the file the earlier edits will have left behind.

    Concurrency covers only `call()`. Every side effect — display, session bookkeeping, the returned
    messages — is applied on this thread in the model's original order. A declined confirmation
    short-circuits the rest of the batch, and observations follow all results so it stays replayable.
    """

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
            self._vision_client = ModelClient(self.session)
        return self._vision_client

    def cancel(self) -> None:
        self._active_bash.apply(lambda tool: tool.cancel())
        self._active_job.apply(lambda tool: tool.cancel())
        self._active_worker.apply(lambda agent: agent.cancel())
        if self._vision_client is not None:
            # TODO(async-phase-4): tool cancellation is still a fan-out from the caller's thread;
            # ask the in-flight observation to end rather than waiting out the provider timeout.
            self._vision_client.cancel_active_request()

    def call_tool(self, tool: Tool, planned_edit: EditBatchPlan.PlannedEdit | None = None) -> str | ToolOutput:
        if isinstance(tool, DelegateTool):
            tool.runner = self
            return tool.call()
        if isinstance(tool, ToolScript):
            tool.runner = self
            return tool.call()
        if isinstance(tool, ViewImageTool):
            # The runner owns the vision client, so Agent.cancel() reaches an in-flight
            # observation instead of leaving it to wait out the provider timeout.
            # TODO(async-phase-3): ViewImage is still a synchronous tool, so the observation gets an
            # outer boundary here. Phase 3 gives the tool a native `call_async` and this disappears.
            observer = VisionObserver(self.vision_client())

            def observe(images, question: str = "", observer=observer) -> str:
                fail_if_running_loop("use await ViewImageTool.call_async(...)")
                return asyncio.run(observer.observe_async(images, question))

            tool.vision_observe = observe
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
        shows that the work happened.
        """
        # An unrecognized name is parsed as a single raw payload, which is exactly what to echo.
        payload = call.args[0] if len(call.args) == 1 else call.args
        content = json.dumps(payload, ensure_ascii=False)
        label = builtin_tool_label(call.name)
        self.emit(LogBlock([LogLine(label, oneline(content, 120), LogRole.TOOL, LogEdge.BRANCH)]))
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
        outcomes: list[tuple[str, str | ToolOutput, str | None, float, object | None] | None] = [None] * len(segment)
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

    def execute_readonly(self, call: ToolCall) -> tuple[str, str | ToolOutput, str | None, float, object | None]:
        # Pure execution for a parallel worker: returns (kind, output, display, elapsed, recovery)
        # and performs no display or session writes (those happen in finalize_outcome on the main
        # thread). Mirrors run_one's branches, minus confirmation (parallel_safe guarantees none is
        # needed). Recovery is the structured source output a ToolError carries, if any.
        started = time.monotonic()
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is None:
            return "reject", f"ToolError: unknown tool {call.name}", None, 0.0, None
        tool = tool_class(self.session, call.args)
        display = None
        try:
            display = tooloutput.short_call(self.session, call, tool.short_args())
            if call.error:
                raise ToolError(call.error)
            output = tool.call()
        except ToolError as error:
            return "reject", f"ToolError: {error}", display, time.monotonic() - started, error.recovery
        except Exception as error:  # noqa: BLE001 - tool failures are serialized back to the model.
            return "error", f"ToolError: {error}", display, time.monotonic() - started, None
        return "ok", output, display, time.monotonic() - started, None

    def finalize_outcome(self, call: ToolCall, outcome: tuple[str, str | ToolOutput, str | None, float, object | None], batch_suffix: str = "") -> str:
        kind, output, display, elapsed, recovery = outcome
        d = ToolDisplay(batch_suffix=batch_suffix, display=display)
        if kind == "ok":
            return self.finish(call, output, elapsed=elapsed, d=d)
        if kind == "reject":
            return self.reject(call, str(output), d=d, recovery=recovery)
        # An unexpected exception carries no structured recovery; only ToolError does, and that is
        # the "reject" branch above.
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
        plan_error: str | tuple[str, object | None] = "",
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
            d.display = tooloutput.short_call(self.session, call, tool.short_args())
            if plan_error:
                # A batch plan failure carries (message, recovery); attach the recovery so the
                # rendered failure still hands the model a fresh view.
                message, recovery = plan_error if isinstance(plan_error, tuple) else (plan_error, None)
                raise ToolError(message, recovery=recovery)
            needs_confirmation = tool.needs_confirmation()
            if needs_confirmation and self.session.settings.yolo and not tool.always_confirms():
                d.auto = True
                pre = toolblocks.approval_display(self.session, call, tool, "auto", batch_suffix=batch_suffix, planned_edit=planned_edit)
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
                    self.emit(
                        LogBlock.hierarchy(
                            toolblocks.log_root(d.display or tooloutput.short_call(self.session, call), batch_suffix=batch_suffix, call=call), []
                        )
                    )
                    d.nested_display = True
                self.live_start()
            elif isinstance(tool, JobTool) and tool.blocks_agent() and self.live_start is not None:
                # A blocking Job wait streams the job's log into the same live preview as Bash, so
                # it draws the root line up front and hands the preview the wait budget for the
                # countdown, exactly like Bash's pre-block.
                if not d.nested_display:
                    self.emit(
                        LogBlock.hierarchy(
                            toolblocks.log_root(d.display or tooloutput.short_call(self.session, call), batch_suffix=batch_suffix, call=call), []
                        )
                    )
                    d.nested_display = True
                self.live_start(tool.wait_budget(tool.payload()))
            elif tool.blocks_agent() and not d.nested_display:
                # A blocking call with no live preview wired up (e.g. headless, or a Job wait
                # outside the runner) still prints its call line now -- as a leaf the finish block
                # will hang children under -- so the user sees the agent is waiting instead of a
                # blank screen until the result lands. Skipped when something already drew a root
                # (an approval block, an auto preview); a second copy of the same line is noise,
                # not reassurance.
                self.emit(
                    LogBlock.hierarchy(toolblocks.log_root(d.display or tooloutput.short_call(self.session, call), batch_suffix=batch_suffix, call=call), [])
                )
                d.nested_display = True
            output = self.call_tool(tool, planned_edit)
            if isinstance(tool, ViewImageTool) and tool.vision_entry_label:
                d.vision_entry = tool.vision_entry_label
            observation = tool.model_observation()
        except ToolError as error:
            return "failed", self.reject(call, f"ToolError: {error}", d=d, recovery=error.recovery), None
        except Exception as error:  # noqa: BLE001 - tool failures are serialized back to the model.
            return "failed", self.finish(call, f"ToolError: {error}", failed=True, elapsed=time.monotonic() - started, d=d), None
        return "ok", self.finish(call, output, elapsed=time.monotonic() - started, turn_diff=tool.turn_diff(), d=d), observation

    def reject(
        self,
        call: ToolCall,
        output: str,
        *,
        d: ToolDisplay | None = None,
        recovery: object | None = None,
    ) -> str:
        """A refused or failed call. `recovery` is structured source output carried by a ToolError
        (e.g. a fresh view from a stale Edit): its views are registered and rendered here, on the
        main thread, without parsing error strings."""
        d = d or ToolDisplay()
        recovery_output = isinstance(recovery, ToolOutput) and recovery.has_source
        text = output + "\n" + self._render_source_output(recovery) if recovery_output else output
        self.session.record_tool_error("-", call.name, call.args, text)
        self.emit(
            LogBlock.hierarchy(None, [LogLine("error", oneline(output.removeprefix("ToolError:").strip(), 220), LogRole.ERROR, LogEdge.END)])
            if d.nested_display
            else toolblocks.reject_display(self.session, call, output, d=d)
        )
        return self.tool_message(call, "", text, failed=True, display=d.display, bound=not recovery_output)

    def finish(
        self,
        call: ToolCall,
        output: str | ToolOutput,
        *,
        failed: bool = False,
        elapsed: float | None = None,
        store: bool = True,
        turn_diff: TurnDiff | None = None,
        d: ToolDisplay | None = None,
    ) -> str:
        d = d or ToolDisplay()
        tool_class = TOOL_REGISTRY.get(call.name)
        tool_output = ToolOutput.of(output)
        retain = not failed and store and (tool_class is None or tool_class.STORES_RESULT)
        key = ""
        bound = True
        if tool_output.has_source:
            # Source-bearing output is projected, keyed, and rendered here on the main thread:
            # view ids follow model call order, and the retained plain text is stored under tr.N.
            model_text, key = self._source_output(call, tool_output, retain=retain)
            bound = False
        else:
            model_text = tool_output.retained_text
            key = self.session.store_tool_result(call.name, call.args, model_text) if retain else ""
        if failed:
            self.session.record_tool_error(key or "-", call.name, call.args, model_text)
        elif key:
            self.update_code_index(call, model_text)
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
            self.emit(toolblocks.finish_display(self.session, call, key, model_text, failed=failed, elapsed=elapsed, d=d, worker_rule=self.worker_rule))
        return self.tool_message(call, key, model_text, failed=failed, display=d.display, bound=bound)

    def _source_output(self, call: ToolCall, tool_output: ToolOutput, *, retain: bool) -> tuple[str, str]:
        """Project source blocks, store the retained plain text, register views, and render.

        Returns (model_text, tr.N key or ""). The note inside a bounded block names the
        retained key and its materialized file, so the model can Read the omitted middle back
        into a fresh view; without a retained copy to point at there is nothing to name.
        """
        projected = tool_output.project(max_tokens=MAX_TOOL_OUTPUT_TOKENS, estimate=self.context.estimated_text_tokens)
        retained = tool_output.retained_text
        key = ""
        if retain:
            key = self.session.store_tool_result(call.name, call.args, retained)
            bounded = [index for index, part in enumerate(projected.parts) if isinstance(part, TextBlock) or (isinstance(part, SourceBlock) and part.bounded)]
            if bounded:
                path = self.context.materialize_output(key, retained)
                hint = self.context.OMITTED_OUTPUT_HINT if path else self.context.OMITTED_OUTPUT_RECALL_HINT
                parts = list(projected.parts)
                for index in bounded:
                    part = parts[index]
                    if isinstance(part, (SourceBlock, TextBlock)):
                        parts[index] = replace(part, note_recall=key, note_file=path, note_hint=hint)
                projected = ToolOutput(projected.retained_text, tuple(parts))
        keys = self.session.register_source_drafts(list(projected.drafts))
        return projected.render(keys), key

    def _render_source_output(self, recovery: object | None) -> str:
        """Register a ToolError's recovery drafts and render them with their fresh view ids.

        Projected like any other source output: a failure message skips the generic bounding, so
        this is the only thing keeping a recovery view inside the budget. Only the lines that
        survive projection are registered, so a clipped one still cannot authorize what it hid.
        """
        if not isinstance(recovery, ToolOutput) or not recovery.has_source:
            return ""
        projected = recovery.project(max_tokens=MAX_TOOL_OUTPUT_TOKENS, estimate=self.context.estimated_text_tokens)
        keys = self.session.register_source_drafts(list(projected.drafts))
        return projected.render(keys)

    def tool_message(self, call: ToolCall, key: str, output: str, *, failed: bool = False, display: str | None = None, bound: bool = True) -> str:
        head = "tool " + ((key + " ") if key else ("- " if failed else "")) + (display or tooloutput.short_call(self.session, call))
        rows = [head]
        if failed:
            rows.append("status: failed")
        body = self.context.bound_output(output, key).rstrip() if bound else output.rstrip()
        rows.extend(["output:", body])
        return "\n".join(rows).strip()

    def update_code_index(self, call: ToolCall, output: str) -> None:
        if call.name != "Edit":
            return
        paths = [str(call.args[0])] if call.args and isinstance(call.args[0], str) else []
        for match in tooloutput.EDIT_PATH_RE.finditer(output):
            with contextlib.suppress(json.JSONDecodeError):
                paths.append(str(json.loads(match.group(1))))
        CodeIndex(self.session).update(list(dict.fromkeys(paths)))

    def confirm(self, call: ToolCall, tool: Tool, batch_suffix: str = "", planned_edit: EditBatchPlan.PlannedEdit | None = None) -> tuple[bool, str]:
        always_option = isinstance(tool, DelegateTool) and tool.always_confirms()
        # Decided before the brief is drawn: the brief needs the actions either way -- live in the
        # form, or spelled out in the typed legend when there is no form to show them.
        actions = toolblocks.approval_actions(tool, always_option)
        form = actions if self.declare_approval_form(actions) else []
        # Printed once, outside the loop. The `c` and `v` actions come back here to ask again, and
        # a second copy of the brief in the transcript is noise: the first one is still on screen,
        # and what those actions changed (or showed) they report themselves.
        self.emit(
            toolblocks.approval_display(self.session, call, tool, "confirm", batch_suffix=batch_suffix, planned_edit=planned_edit, form=form, actions=actions)
        )
        while True:
            self.declare_approval_form(actions)  # the TUI drops the form when a prompt resolves
            reply = self.input_fn(LogBlock.prefix(2, LogEdge.CONTINUE) + toolblocks.approval_prompt(always_option, form))
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

    def declare_approval_form(self, actions: list[tuple[str, str]]) -> bool:
        """Offer the actions to the TUI as a selectable row; report whether it took them. False
        (headless, piped stdin) sends the brief back to printing the typed legend."""
        return self.approval_form is not None and self.approval_form(actions)

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
        self.emit(toolblocks.worker_config_block(self.session))

    def view_text(self, view: ApprovalView) -> None:
        """The `v` action of a confirmation prompt: open a read-only viewer with the full,
        untruncated text behind the call. Without an injected viewer (headless, or a runner outside
        CommandLoop) this prints the whole thing; the confirmation prompt re-asks either way."""
        if self.text_viewer is not None:
            self.text_viewer(view)
        else:
            self.emit(toolblocks.full_text_block(view))
