"""wizolt tool runner: batched edit planning, confirmation, and tool execution."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import inspect
import json
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypeVar

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
    MCPTool,
    NextHintsTool,
    SearchTool,
    Tool,
    ToolScript,
    ViewImageTool,
    toolblocks,
    tooloutput,
)
from wizolt.tools.editplan import EditBatchPlan
from wizolt.tools.toolblocks import ToolDisplay
from wizolt.tools.toolscript import ScriptCancelled
from wizolt.vision import VisionObserver

if TYPE_CHECKING:
    from wizolt.engine import Agent

_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True)
class NestedRequest:
    """One ToolScript nested invocation, as data.

    The gateway deliberately takes structured calls rather than a coroutine or a callback: nothing
    a tool hands over gets to decide what runs on the runtime loop, and the runner keeps ownership
    of validation, approval, segmentation, and cancellation. `batch` is the shape the script asked
    for -- one call, or a `call_many` whose segmentation and ordering the runner applies."""

    calls: tuple[ToolCall, ...]
    batch: bool


class _NestedGateway:
    """The loop-side owner of a running ToolScript's nested calls.

    The script is synchronous Python on one dedicated worker, so its `call()` has to reach the
    runner across a thread boundary and wait for an answer. It does that by handing over call data
    and blocking its own thread on a plain future -- never by submitting a coroutine, which would
    put a tool in charge of what the loop runs.

    Admission and shutdown share one lock, so a request cannot pass the gate and then find the
    runner gone: whoever loses the race completes the worker's future with cancellation instead of
    leaving it blocked forever."""

    def __init__(self, runner: ToolRunner, loop: asyncio.AbstractEventLoop):
        self._runner = runner
        self._loop = loop
        self._lock = threading.Lock()
        self._open = True
        self._futures: set[concurrent.futures.Future] = set()
        self._active: set[asyncio.Task] = set()

    def submit(self, request: NestedRequest) -> list[tuple[str, str]]:
        """Called on the script worker: hand the calls over and block this thread on the answer."""

        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            self._futures.add(future)
        try:
            self._loop.call_soon_threadsafe(self._accept, request, future)
        except RuntimeError:
            # The loop is already gone; nothing will ever answer, so answer here.
            self._settle(future, ScriptCancelled())
        return future.result()

    def _accept(self, request: NestedRequest, future: concurrent.futures.Future) -> None:
        """On the loop: admit the request, or refuse it because shutdown got here first."""

        with self._lock:
            if not self._open:
                self._settle_locked(future, ScriptCancelled())
                return
            task = self._loop.create_task(self._serve(request, future))
            self._active.add(task)
        task.add_done_callback(self._active.discard)

    async def _serve(self, request: NestedRequest, future: concurrent.futures.Future) -> None:
        try:
            outcomes = await self._runner.run_nested(request)
        except asyncio.CancelledError:
            self._settle(future, ScriptCancelled())
            raise
        except BaseException as error:  # noqa: BLE001 - the worker must learn every ending, not hang.
            self._settle(future, error)
        else:
            self._settle(future, None, outcomes)

    def _settle(self, future: concurrent.futures.Future, error: BaseException | None, result=None) -> None:
        with self._lock:
            self._settle_locked(future, error, result)

    def _settle_locked(self, future: concurrent.futures.Future, error: BaseException | None, result=None) -> None:
        self._futures.discard(future)
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    async def close(self) -> None:
        """Close admission, quiesce active nested work, and unblock every waiting worker.

        Awaited before `run` returns, so no gateway task and no blocked worker future can
        outlive the ToolScript call that owns them."""

        with self._lock:
            self._open = False
            active = list(self._active)
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        with self._lock:
            waiting = list(self._futures)
        for future in waiting:
            self._settle(future, ScriptCancelled())


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
        self.question_fn: Callable[[list[AskSpec]], Awaitable[list[str]]] | None = None
        # Injected by CommandLoop: drives the Delegate confirm-time `c` config loop through the
        # shared choice selector (see CommandLoop.run_worker_config). None degrades the `c` key to
        # printing the current worker config only (headless / non-CommandLoop runners).
        self.worker_config_picker: Callable[[], Awaitable[None] | None] | None = None
        # Injected by CommandLoop: opens a read-only viewer for the text behind a confirmation --
        # a Delegate order, a ToolScript body -- for the confirm-time `v`/`view` key (see
        # cli.modals.approval_text_viewer). None degrades the `v` key to printing the whole text
        # (headless / non-CommandLoop runners). The return value is the viewer's close signal and is
        # discarded here; the Ctrl-O browser's reopen loop reads it on its own.
        self.text_viewer: Callable[[ApprovalView], Awaitable[object] | object] | None = None
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
        # Injected by CommandLoop: resolves a pending TUI approval/Ask prompt with "cancelled", so
        # a worker parked on the user can be unblocked when the turn is cancelled. None (headless,
        # piped stdin) means the injected input function owns its own unblocking.
        self.cancel_input: Callable[[], None] | None = None
        # Loop-bound state for one run() invocation. Never reused across invocations: a
        # semaphore belongs to the loop that created it, and this runner outlives any single loop.
        self._capacity: asyncio.Semaphore | None = None
        self._gateway: _NestedGateway | None = None
        # The cancellation token of the ToolScript currently running, if any.
        self._script_budget: object | None = None

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

    def request_tool_stop(self, tool: Tool) -> None:
        """Ask one tool to stop, and note it. Never a claim that anything stopped -- the runner
        still waits for the invocation to finish before it reports cancellation."""

        with contextlib.suppress(Exception):
            tool.request_stop()
        if isinstance(tool, AskTool) and self.cancel_input is not None:
            # A worker parked on the user answers nothing on its own: resolve the prompt as
            # cancelled so it returns, which `Ask` reads as a dismissal.
            with contextlib.suppress(Exception):
                self.cancel_input()

    @contextlib.asynccontextmanager
    async def _bounded(self):
        """Hold one unit of the invocation's tool-execution capacity.

        Acquired before a worker is submitted and released only after that worker has finished,
        cancellation included: capacity a cancelled call is still quiescing is not capacity."""

        capacity = self._capacity
        if capacity is None:
            yield
            return
        async with capacity:
            yield

    async def _run_in_executor(self, invoke: Callable[[], _ResultT], tool: Tool | None = None, *, executor=None, bounded: bool = True) -> _ResultT:
        """Run one synchronous tool body on a worker, and never abandon it.

        Cancelling the task that is waiting does not cancel the work: the worker keeps running,
        with whatever files, subprocesses, and session state it is holding. So cancellation here
        means ask the tool to stop, keep waiting, and only then report cancellation upward. A
        failure raised by a worker that was already being cancelled is a cleanup detail and must
        not replace the cancellation the turn is unwinding on."""

        loop = asyncio.get_running_loop()
        async with contextlib.AsyncExitStack() as stack:
            if bounded:
                await stack.enter_async_context(self._bounded())
            future = loop.run_in_executor(executor, invoke)
            cancel_error: asyncio.CancelledError | None = None
            while not future.done():
                try:
                    # `wait` rather than awaiting the future: it never raises the worker's own
                    # outcome, so that outcome is still there to be read off the future below. A
                    # shield would leave it to be collected by the loop's exception handler, which
                    # is exactly the kind of unobserved ending this runner is not allowed to have.
                    await asyncio.wait({future})
                except asyncio.CancelledError as error:
                    cancel_error = cancel_error or error
                    if tool is not None:
                        self.request_tool_stop(tool)
            try:
                result = future.result()
            except BaseException as worker_error:
                if cancel_error is None:
                    raise
                self.report_cleanup_error(worker_error)
                raise cancel_error from None
            if cancel_error is not None:
                raise cancel_error
            return result

    def report_cleanup_error(self, error: BaseException) -> None:
        """A worker that failed while it was being cancelled. Recorded, never raised over the
        cancellation: the turn is ending either way, and the tool's own result boundary is gone."""

        if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, ScriptCancelled)):
            return
        self.session.record_tool_error("-", "cancel", [], f"ToolError while cancelling: {error}")

    async def call_tool(self, tool: Tool, planned_edit: EditBatchPlan.PlannedEdit | None = None) -> str | ToolOutput:
        """Run one tool's own work under the execution policy for its kind.

        The policy lives here rather than on each Tool: what may overlap, what must not be
        abandoned, and what must stay on the loop are properties of the batch this runner is
        executing, not of the tool's business logic."""

        if isinstance(tool, ViewImageTool):
            # The runner owns the vision client, so cancelling the turn reaches an in-flight
            # observation instead of leaving it to wait out the provider timeout.
            tool.vision_observe = VisionObserver(self.vision_client()).observe
            return await tool.call()
        if isinstance(tool, ToolScript):
            tool.runner = self
            return await self._run_script(tool)
        if isinstance(tool, DelegateTool):
            # Awaited directly: the worker's turn is a child of this one, so the parent's
            # cancellation reaches it by propagation and its diffs still merge on every ending.
            tool.runner = self
            return await tool.call()
        if isinstance(tool, BashTool):
            with self._active_bash.track(tool):
                return await tool.call()
        if isinstance(tool, JobTool):
            with self._active_job.track(tool):
                # Waiting is native asyncio. The process handle itself remains Popen-backed so a
                # background job can outlive the turn and the loop that started it.
                return await tool.call()
        if isinstance(tool, AskTool):
            # The user is asked on the loop and may take as long as they like; the prompt stays
            # live and cancellable rather than parking a worker on them.
            return await tool.call()
        if isinstance(tool, MCPTool):
            # Native: the manager's operations are coroutines on this loop, so a cancelled turn
            # reaches the FastMCP client itself and its teardown is awaited before this returns.
            return await tool.call()
        if isinstance(tool, SearchTool):
            # ripgrep is an asyncio subprocess, so cancellation kills and reaps it directly.
            return await tool.call()
        if tool.MUTATES or tool.PRODUCES_MODEL_OBSERVATION or isinstance(tool, NextHintsTool):
            # Short local mutation stays on the loop: single-writer, and an Edit or a Note can
            # never outlive the turn that ordered it. Cancellation is observed on both sides.
            self._raise_if_cancelled()
            try:
                return planned_edit.call(tool) if planned_edit and isinstance(tool, EditTool) else tool.call()
            finally:
                self._raise_if_cancelled()
        # Everything else -- reads, searches, MCP calls, Ask -- is bounded synchronous work.
        return await self._run_in_executor(tool.call, tool)

    @staticmethod
    def _raise_if_cancelled() -> None:
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise asyncio.CancelledError

    def run_sync(self, calls: list[ToolCall], batch_suffix: str = "") -> list[Json]:
        """Synchronous entry point for direct tests and embedding; the agent awaits run."""

        fail_if_running_loop("use await ToolRunner.run(...)")
        return asyncio.run(self.run(calls, batch_suffix))

    async def run(self, calls: list[ToolCall], batch_suffix: str = "") -> list[Json]:
        loop = asyncio.get_running_loop()
        # Per invocation, never per runner: these are loop-bound, and this runner outlives loops.
        self._capacity = asyncio.Semaphore(max(1, self.session.settings.max_parallel_tools))
        self._gateway = _NestedGateway(self, loop)
        try:
            return await self._run_batch(calls, batch_suffix)
        finally:
            gateway, self._gateway = self._gateway, None
            self._capacity = None
            if gateway is not None:
                await gateway.close()

    async def _run_batch(self, calls: list[ToolCall], batch_suffix: str = "") -> list[Json]:
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
                messages.extend(await self.run_parallel(calls[index:end], batch_suffix, state))
                index = end
                continue
            end = index + 1 if self.edit_barrier(calls[index]) else self.edit_segment_end(calls, index)
            messages.extend(await self.run_serial(calls[index:end], batch_suffix, state, observations))
            index = end
        return [*messages, *observations]

    async def run_nested(self, request: NestedRequest) -> list[tuple[str, str]]:
        """The loop-side half of ToolScript's gateway: run one nested request the runner's own way.

        A `call_many` gets the baseline segmentation, concurrency cap, and original-order
        publication; a single `call` takes the ordinary path, confirmation included. Every call was
        validated by the script boundary before any of them reached here."""

        calls = list(request.calls)
        if not request.batch:
            status, message, _ = await self._run_nested_one(calls[0])
            return [(status, message)]
        outcomes: list[tuple[str, str]] = []
        index = 0
        while index < len(calls):
            end = index
            while end < len(calls) and self.parallel_safe(calls[end]):
                end += 1
            segment = calls[index:end]
            if len(segment) > 1:
                raw = await self._execute_readonly_segment(segment)
                for call, outcome in zip(segment, raw):
                    outcomes.append(("ok" if outcome[0] == "ok" else "failed", self.finalize_outcome(call, outcome)))
                index = end
                continue
            # A lone parallel-safe call has nothing to overlap with, and anything else must run on
            # its own anyway: both take the ordinary single-call path, confirmation included.
            status, message, _ = await self._run_nested_one(calls[index])
            outcomes.append((status, message))
            index += 1
        return outcomes

    async def _run_nested_one(self, call: ToolCall) -> tuple[str, str, Json | None]:
        """One nested call. An Edit goes through a single-element plan so it behaves exactly like a
        top-level single Edit -- source-view planning, stale checks, write-time verification."""

        if call.name == "Edit":
            plan = EditBatchPlan(self.session).build([call])
            return await self.run_one(call, planned_edit=plan.planned.get(call.id), plan_error=plan.errors.get(call.id, ""))
        return await self.run_one(call)

    async def _run_script(self, tool: ToolScript) -> str | ToolOutput:
        """Run a ToolScript on a worker of its own, with its nested calls served from this loop.

        The dedicated worker is the point: the script blocks synchronously while its nested calls
        run, so it must not sit in the same bounded capacity those calls need to execute. On
        cancellation the token is set, the gateway stops admitting and unblocks whatever the script
        is waiting on, and only then is the worker awaited -- a script is never abandoned."""

        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="toolscript")
        try:
            budget_holder: list[Any] = []
            tool.on_budget = budget_holder.append
            future = loop.run_in_executor(executor, tool.call)
            cancel_error: asyncio.CancelledError | None = None
            while not future.done():
                try:
                    # `wait`, not the future itself: the script's own outcome stays on the future
                    # for the read below instead of escaping into the loop's exception handler.
                    await asyncio.wait({future})
                except asyncio.CancelledError as error:
                    cancel_error = cancel_error or error
                    for budget in budget_holder:
                        budget.cancel()
                    gateway = self._gateway
                    if gateway is not None:
                        await gateway.close()
            try:
                result = future.result()
            except BaseException as worker_error:
                if cancel_error is None:
                    raise
                self.report_cleanup_error(worker_error)
                result = ""
            if cancel_error is not None:
                raise cancel_error
            return result
        finally:
            tool.on_budget = None
            executor.shutdown(wait=True)

    def nested_calls(self, calls: list[ToolCall], *, batch: bool) -> list[tuple[str, str]]:
        """Called on the ToolScript worker: hand nested calls to the loop and wait for their results."""

        gateway = self._gateway
        if gateway is None:
            raise ToolError("ToolScript nested calls require a running tool runner")
        return gateway.submit(NestedRequest(tuple(calls), batch))

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

    async def run_serial(self, segment: list[ToolCall], batch_suffix: str, state: dict[str, bool], observations: list[Json]) -> list[Json]:
        messages: list[Json] = []
        plan = EditBatchPlan(self.session).build(segment) if any(call.name == "Edit" for call in segment) else EditBatchPlan(self.session)
        for call in segment:
            suffix = batch_suffix if state["first"] else ""
            state["first"] = False
            status, content, observation = await self.run_one(
                call, batch_suffix=suffix, planned_edit=plan.planned.get(call.id), plan_error=plan.errors.get(call.id, "")
            )
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
            if observation is not None:
                observations.append(observation)
            if status == "refused":
                state["refused"] = True
        return messages

    async def run_parallel(self, segment: list[ToolCall], batch_suffix: str, state: dict[str, bool]) -> list[Json]:
        # Run the pure tool.call() work concurrently, but apply all side effects (display, session
        # bookkeeping, tool messages) on the loop in request order, so output and the results handed
        # back to the model match the order the model issued the calls.
        outcomes = await self._execute_readonly_segment(segment)
        messages: list[Json] = []
        for call, outcome in zip(segment, outcomes):
            suffix = batch_suffix if state["first"] else ""
            state["first"] = False
            content = self.finalize_outcome(call, outcome, batch_suffix=suffix)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        return messages

    async def _execute_readonly_segment(self, segment: list[ToolCall]) -> list[tuple[str, str | ToolOutput, str | None, float, object | None]]:
        """Run one run of independent read-only calls concurrently, bounded by the invocation's
        capacity, and hand their outcomes back in the model's original order.

        Each call is its own child task so cancellation reaches each of them; the whole set is then
        awaited before cancellation propagates, because a read that is still running is still
        holding a file handle and a worker. An ordinary tool failure is converted at that call's own
        result boundary and never cancels a healthy sibling."""

        children = [asyncio.ensure_future(self.execute_readonly(call)) for call in segment]
        try:
            results = await asyncio.gather(*children)
        except asyncio.CancelledError:
            for child in children:
                child.cancel()
            await asyncio.gather(*children, return_exceptions=True)
            raise
        except BaseException:
            for child in children:
                child.cancel()
            await asyncio.gather(*children, return_exceptions=True)
            raise
        return list(results)

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

    async def execute_readonly(self, call: ToolCall) -> tuple[str, str | ToolOutput, str | None, float, object | None]:
        # Pure execution for one parallel call: returns (kind, output, display, elapsed, recovery)
        # and performs no display or session writes (those happen in finalize_outcome on the loop).
        # Mirrors run_one's branches, minus confirmation (parallel_safe guarantees none is needed).
        # Recovery is the structured source output a ToolError carries, if any.
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
            # Native async calls are cancelled at their resource; other read-only tools remain
            # bounded blocking work whose executor future is awaited through cancellation.
            output = await (tool.call() if isinstance(tool, (MCPTool, SearchTool)) else self._run_in_executor(tool.call))
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

    async def run_one(
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
                confirmed, reason = await self.confirm(call, tool, batch_suffix=batch_suffix, planned_edit=planned_edit)
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
            output = await self.call_tool(tool, planned_edit)
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

    async def confirm(self, call: ToolCall, tool: Tool, batch_suffix: str = "", planned_edit: EditBatchPlan.PlannedEdit | None = None) -> tuple[bool, str]:
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
            reply = await self.request_input(LogBlock.prefix(2, LogEdge.CONTINUE) + toolblocks.approval_prompt(always_option, form))
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
                await self.delegate_config_cycle()
                continue  # re-ask; the config cycle printed what it changed
            if lower in {"v", "view"} and (view := tool.approval_view()) is not None:
                # Same whole-line exact-match rule as `c`: `v`/`view` opens the read-only viewer on
                # whatever text this call commits to -- an order, a script -- and anything else
                # (e.g. "cost too high") stays an ordinary refusal reason.
                #
                # Built here rather than before the loop: `c` can have edited the worker config
                # since, and a viewer that reports the configuration a send will run under has to
                # read it now, not as it stood when the prompt was first drawn.
                await self.view_text(view)
                continue  # re-ask; viewing changed nothing, so there is nothing to redraw
            if lower in {"", "y", "yes"}:
                return True, ""
            return False, "" if lower in {"n", "no"} else answer

    async def request_input(self, prompt: str) -> str | None:
        """Await one line of user input for an approval.

        The injected `input_fn` is awaitable under the CLI, which puts the prompt on the runtime
        loop. A plain callable (headless, a test, an embedding) runs on a worker instead, and its
        injector owns unblocking it -- nothing here can interrupt a blocking read."""

        reply = self.input_fn(prompt)
        if inspect.isawaitable(reply):
            return await reply
        return reply

    def declare_approval_form(self, actions: list[tuple[str, str]]) -> bool:
        """Offer the actions to the TUI as a selectable row; report whether it took them. False
        (headless, piped stdin) sends the brief back to printing the typed legend."""
        return self.approval_form is not None and self.approval_form(actions)

    async def delegate_config_cycle(self) -> None:
        """The `c` action of a Delegate send prompt: hand the interactive editing to the injected
        picker loop (CommandLoop.run_worker_config), which reuses the shared choice selector and
        writes back through the /worker pickers, then print the worker's provider/model/effort/api.

        Printed after the picker, not before: the picker already shows each current value as the
        preselected option, so what is worth logging is the config the send would now run under.
        The approval brief above keeps its original rows, so the two together read as a change.
        Without an injected picker (headless, or a runner outside CommandLoop) this just prints the
        current values; the confirmation prompt re-asks either way."""
        if self.worker_config_picker is not None:
            result = self.worker_config_picker()
            if inspect.isawaitable(result):
                await result
        self.emit(toolblocks.worker_config_block(self.session))

    async def view_text(self, view: ApprovalView) -> None:
        """The `v` action of a confirmation prompt: open a read-only viewer with the full,
        untruncated text behind the call. Without an injected viewer (headless, or a runner outside
        CommandLoop) this prints the whole thing; the confirmation prompt re-asks either way."""
        if self.text_viewer is not None:
            result = self.text_viewer(view)
            if inspect.isawaitable(result):
                await result
        else:
            self.emit(toolblocks.full_text_block(view))
