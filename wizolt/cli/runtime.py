"""TuiRuntime: drive the interactive session timeline while CommandLoop owns session behavior."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wizolt.base import MalformedToolCallError, TurnBox, WizoltError
from wizolt.cli.modals import tool_output_viewer
from wizolt.image import UserInput
from wizolt.render import search_sources_footer
from wizolt.tui import TuiApp

# The TUI status label shown while a resumed session's transcript is being restored: a quiet
# lead-in before the single-write replay, so the wait reads as a restore in progress rather than a
# stuck prompt.
RESUME_STATUS_LABEL = "resuming session…"

if TYPE_CHECKING:
    from wizolt.cli import CommandLoop


class _TurnCancelled(Exception):
    """A child turn that settled after cancellation, carried back to the runtime task.

    CancelledError cannot cross that task boundary as itself -- the runtime would read it as its
    own cancellation -- so this distinguishes a cancelled turn from a failed runtime."""


@dataclass(frozen=True)
class _Submission:
    """One thing a prompt-toolkit callback accepted and could not persist itself.

    `value` is an input to queue before saving; a submission with none is a queue mutation the
    callback already applied and only needs written down. `resume_notice` prints the paste-ready
    resume line once the save that carries it has returned a uid. `turn_boundary` is queued after
    a turn's last output: the single consumer persists and flushes every earlier input before it
    opens admission for the next turn."""

    value: UserInput | None = None
    resume_notice: bool = False
    turn_boundary: asyncio.Future[None] | None = None


class ScrollbackWriter:
    """One ordered queue for completed scrollback writes while the TUI is live.

    Printing above a live application means suspending it, so a write is not something a producer
    can just do: it has to be handed to the loop that owns the terminal. Everything goes through one
    FIFO queue, so what the reader sees is the order things happened in -- a promoted answer above
    the tool output of the batch that followed it, not interleaved with it.

    Nothing blocks the loop. Producers submit and move on; a caller that genuinely needs a write to
    be on screen before it continues -- the turn, between a promoted response and its tool batch --
    awaits a `barrier` instead of a threading Event.

    Admission and scheduling share one lock, so a producer cannot pass the gate and then find the
    loop gone: whoever loses that race is refused, and a refused write takes the direct-output
    fallback rather than disappearing."""

    def __init__(self, loop: asyncio.AbstractEventLoop, write: Callable[[Callable[[], None]], Awaitable[None]], fallback: Callable[[Callable[[], None]], None]):
        self._loop = loop
        self._write = write
        self._fallback = fallback
        self._queue: asyncio.Queue = asyncio.Queue()
        self._lock = threading.Lock()
        self._open = True
        self._error: BaseException | None = None
        self._task = loop.create_task(self._pump(), name="scrollback-writer")

    @property
    def lock(self) -> threading.Lock:
        """The admission lock. Shared with the background-output gate: closing that gate and
        scheduling a write are the same race, so they are decided by the same lock."""
        return self._lock

    def submit(self, callback: Callable[[], None]) -> None:
        """Queue one completed write. FIFO, from any thread, and never blocking.

        Scheduling happens under the lock so the queue order is the submit order: `call_soon` is
        itself FIFO, so ordering the schedule calls orders the writes."""

        with self._lock:
            if not self._open:
                self._fallback(callback)
                return
            self._loop.call_soon_threadsafe(self._queue.put_nowait, callback)

    async def barrier(self) -> None:
        """Return once every write submitted before this call has reached the terminal.

        Re-raises the first write failure: a terminal that refused a write is the runtime's problem,
        not something to bury in a background task nobody observes."""

        done: asyncio.Future[None] = self._loop.create_future()
        with self._lock:
            if not self._open:
                self._raise_pending()
                return
            self._loop.call_soon_threadsafe(self._queue.put_nowait, done)
        await done
        self._raise_pending()

    async def close(self) -> None:
        """Stop accepting writes, drain the ones already accepted, and end the writer task."""

        with self._lock:
            if not self._open:
                return
            self._open = False
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    def _raise_pending(self) -> None:
        error, self._error = self._error, None
        if error is not None:
            raise error

    async def _pump(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            if isinstance(item, asyncio.Future):
                if not item.done():
                    item.set_result(None)
                continue
            try:
                await self._write(item)
            except Exception as error:  # noqa: BLE001 - kept for the next barrier or close to raise.
                if self._error is None:
                    self._error = error


class TuiRuntime:
    """Own the interactive session timeline while CommandLoop owns session behavior."""

    def __init__(self, command_loop: CommandLoop):
        self.loop = command_loop
        # Submitted user input, on the runtime loop. TUI callbacks run there too, so one FIFO can
        # order idle input with follow-ups accepted while a turn is finishing.
        self.pending: asyncio.Queue[UserInput] = asyncio.Queue()
        # Work a prompt-toolkit callback accepted but must not do inline: queueing an input and
        # persisting it. One FIFO and one consumer, because submit order is the persistence order
        # and task scheduling order is not a contract.
        self.submissions: asyncio.Queue[_Submission] = asyncio.Queue()
        self.submissions_task: asyncio.Task | None = None
        self.accepting = False
        self.turn_active = False
        self.cancel_pending = False
        self.force_exit_timer: threading.Timer | None = None
        self.error: BaseException | None = None
        # The loop this runtime owns, and everything it started on it. Every task created here is
        # either awaited in its lexical scope or registered below with a done callback that
        # observes its result, so nothing is left to an event-loop exception handler.
        self.runtime_loop: asyncio.AbstractEventLoop | None = None
        self.tasks: set[asyncio.Task] = set()
        # The tool-output browser, while one is open. Runtime-owned like every other task here;
        # kept as a handle only so a second Ctrl-O can tell that one is already showing.
        self.browser: asyncio.Task | None = None
        self.scrollback: ScrollbackWriter | None = None
        # Set to end the input loop: an asyncio event, so waiting for it is an await rather than a
        # poll, and it is owned by the loop that sets it.
        self.shutdown: asyncio.Event | None = None
        self.application_ready: asyncio.Event | None = None

    @property
    def tui(self) -> TuiApp:
        assert self.loop.tui is not None
        return self.loop.tui

    def interrupt(self) -> None:
        """Ctrl-C from the TUI: ask the active turn to cancel, and say so on the status line.

        No process signal: the turn is a task, and `Agent.cancel()` schedules its cancellation on
        the loop that owns it. The status stays on `cancelling` until the turn has quiesced and
        settled, which is the honest state -- an uncooperative tool may still be unwinding."""

        if self.cancel_pending:
            return
        self.cancel_pending = True
        self.tui.set_running("cancelling")
        self.loop.agent.cancel()

    def _request_model_retry(self) -> None:
        """`/resend`: ask the model client to drop the exact attempt in flight and send it again.

        Not a turn cancellation and not a signal: the client's own thread-safe claim is the entire
        wake-up mechanism, and it is also the debounce -- an attempt already claimed, or none in
        flight, answers False and nothing about the session changes. The retry counters move only
        once the request has actually been accepted."""

        state = self.loop.session.state
        if state.model_retry_until > 0 or state.manual_model_retry_requested:
            return
        if not self.loop.agent.model.retry_active_request():
            return
        state.manual_model_retry_requested = True
        state.model_retry_count += 1
        self.tui.invalidate()

    def submit_running(self, value: str | UserInput) -> None:
        value = value if isinstance(value, UserInput) else UserInput(value)
        text = str(value).strip()
        if not text:
            return
        if not value.images and "\n" not in text and text.startswith("/"):
            self.spawn(self.loop.run_queued_command(text), name="queued-command")
        else:
            self.submit_accepted(_Submission(value))
        self.tui.invalidate()

    def recall(self) -> str | UserInput:
        recalled = self.loop.recall_pending_input(self._request_model_retry)
        if recalled:
            # The queue already changed; only writing it down is left, and this key handler owes
            # the editor an answer now rather than after a file write.
            self.submit_accepted(_Submission())
        return recalled

    def submit_accepted(self, submission: _Submission) -> None:
        """Hand one accepted callback result to the consumer, in submit order.

        Refused once shutdown has closed admission: a save scheduled then would land after the
        session said it was finished."""

        if self.accepting:
            self.submissions.put_nowait(submission)

    async def _consume_submissions(self) -> None:
        """Apply accepted submissions one at a time, each fully persisted before the next.

        One consumer rather than a task per keystroke: two saves racing would make task scheduling
        order the order of the log, and the second could capture a queue the first had not written
        yet."""

        while True:
            submission = await self.submissions.get()
            try:
                if submission.turn_boundary is not None:
                    await self.loop.session.save_snapshot()
                    self.turn_active = False
                    self.submit_next(self.loop.take_pending_inputs())
                    if not submission.turn_boundary.done():
                        submission.turn_boundary.set_result(None)
                    continue
                if submission.value is not None:
                    admitted = await self._admit_input(submission.value)
                    if admitted is None:
                        continue  # refused: the draft went back to the editor, nothing was queued
                    if not self.turn_active:
                        self.pending.put_nowait(admitted)
                        continue
                    self.loop.session.enqueue_user_input(admitted)
                uid = await self.loop.session.save_snapshot()
                if submission.resume_notice:
                    self.loop.emit_resume_line(uid)
            except BaseException as error:
                if submission.turn_boundary is not None and not submission.turn_boundary.done():
                    submission.turn_boundary.set_exception(error)
                raise
            finally:
                self.submissions.task_done()

    async def _admit_input(self, value: str | UserInput) -> UserInput | None:
        """Store one submitted input's images off the loop before anything owns it.

        Admission failure -- a recognized image vanished or changed before its copy -- refuses the
        input: the draft goes back to the editor with the error, and nothing was queued. Text
        without images passes straight through."""

        value = value if isinstance(value, UserInput) else UserInput(str(value))
        if not value.images:
            return value
        try:
            return await self.loop.session.images.admit(value)
        except WizoltError as error:
            self.tui.restore_submission(value.display_text(), str(error))
            return None

    async def _close_submissions(self) -> None:
        """Stop accepting, let the consumer finish what it already accepted, then end it."""

        self.accepting = False
        task, self.submissions_task = self.submissions_task, None
        if task is None:
            return
        drained = asyncio.ensure_future(self.submissions.join())
        try:
            # Raced against the consumer itself: if it died, its work is never going to drain and
            # waiting for the join would hold shutdown open forever.
            await asyncio.wait({drained, task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not drained.done():
                drained.cancel()
                await asyncio.wait({drained})
            task.cancel()
            await asyncio.wait({task})

    def expand_output(self) -> None:
        """Ctrl-O: open the tool-output browser as one runtime task on the TUI's own loop.

        One browser at a time. The browser is a modal, and a second Ctrl-O used to queue a second
        one behind it on the modal idle event -- so closing the first opened another the reader
        never asked for. While one is open the keystroke does nothing; Esc/Ctrl-O still closes it."""

        if self.browser is not None and not self.browser.done():
            return
        self.browser = self.spawn(tool_output_viewer(self.loop), name="tool-output")

    def complete_mentions(self, query: str, ready: Callable[[], None]) -> None:
        """prompt-toolkit's file-completion callback: admission only, no work.

        The callback fires from the input handler on every keystroke, so it may not do the ranking
        itself. It hands one coroutine to the background owner and returns; `complete` is what
        decides that a newer query has superseded this one."""

        mentions = self.loop.session.mentions
        if mentions is not None:
            self.loop.spawn_background(mentions.complete(query, ready), name="mention-completion")

    def spawn(self, coroutine, *, name: str = "") -> asyncio.Task | None:
        """Start one runtime-owned task and keep it until it is done.

        Registered rather than fired and forgotten: the done callback consumes the result, so an
        expected cancellation is absorbed and an unexpected failure brings the runtime down through
        its own shutdown path instead of surfacing as an unretrieved-exception warning."""

        loop = self.runtime_loop
        if loop is None:
            return None
        task = loop.create_task(coroutine, name=name or None)
        self.tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        # An owned task failed on its own. Record it for the caller of run() and unwind: leaving
        # the runtime up around a dead command or viewer would be a quieter, worse failure.
        if self.error is None:
            self.error = error
        self.request_shutdown()

    def request_shutdown(self) -> None:
        """Stop accepting new turns and let run begin its shutdown sequence."""
        event, loop = self.shutdown, self.runtime_loop
        if event is None or loop is None:
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(event.set)

    def request_exit(self) -> None:
        """Ctrl-D or /exit: ask the runtime to shut down.

        The application is not stopped here. It stays up until the shutdown sequence has settled
        the active turn and drained accepted scrollback; exiting it first would take the terminal
        away from output that was already accepted."""

        self.request_shutdown()
        # Through the consumer, not inline: this is a key handler, and the drain that shutdown runs
        # before it cancels anything is what keeps the resume line on screen.
        self.submit_accepted(_Submission(resume_notice=True))

    def force_exit(self) -> None:
        """The emergency escape hatch: ask for the graceful path, then arm a deadline.

        A forced signal cannot promise that anything was closed, which is why it is last: the
        ordinary path is requested first and given its chance to unwind."""

        self.request_shutdown()
        self.loop.agent.cancel()
        self.force_exit_timer = threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM))
        self.force_exit_timer.daemon = True
        self.force_exit_timer.start()

    def submit_chat(self, value: UserInput) -> None:
        """Accept one submitted line. Runs on the runtime loop, where the TUI's callbacks run."""
        self.submit_accepted(_Submission(value))

    def build_tui(self) -> TuiApp:
        return TuiApp(
            on_chat_submit=self.submit_chat,
            on_running_submit=self.submit_running,
            on_exit_request=self.request_exit,
            on_force_exit=self.force_exit,
            on_interrupt=self.interrupt,
            on_retry=self._request_model_retry,
            on_recall=self.recall,
            on_expand_output=self.expand_output,
            status_fragments_fn=self.loop.status_bar.fragments,
            activity_fragments_fn=self.loop.view.tui_activity_fragments,
            input_hint_fn=self.loop.view.tui_input_hint,
            quick_hints_fn=lambda: self.loop.session.quick_hints,
            file_picker_available_fn=self.loop.session.mentions.picker.available if self.loop.session.mentions else None,
            file_picker_fn=self.loop.pick_file if self.loop.session.mentions else None,
            file_complete_fn=self.complete_mentions if self.loop.session.mentions else None,
            editor_context_fn=self.loop.editor_context,
            images=self.loop.session.images,
            history=self.loop.input_history,
            completer=self.loop.input_completer,
            on_app_stop=lambda: self.loop.ui.drain_scrollback(),
        )

    def submit_next(self, entered: Sequence[str | UserInput]) -> None:
        if not entered:
            return
        first = entered[0] if isinstance(entered[0], UserInput) else UserInput(entered[0])
        self.pending.put_nowait(first)
        for text in entered[1:]:
            self.loop.session.enqueue_user_input(text)

    def reset_turn(self) -> None:
        self.loop.model_stream_output("", "")
        # A request can fail after permanent promotion but before Agent re-publishes the text and
        # consumes its marker. Never let that stale marker suppress an identical later response.
        self.loop.model_stream_promoted_text = ""
        self.tui.set_idle()
        self.cancel_pending = False

    async def dispatch(self, user_input: str | UserInput) -> bool:
        """Dispatch one input. Return true when it was fully handled as a command."""
        user_input = user_input if isinstance(user_input, UserInput) else UserInput(user_input)
        self.loop.ui.emit_answer(user_input.display_text(), role="user", rule=False)
        try:
            handled, exit_now = await self.loop.command(user_input.strip())
        except (KeyboardInterrupt, WizoltError) as error:
            self.loop.emit_turn("Cancelled" if isinstance(error, KeyboardInterrupt) else f"Error: {error}")
            self.submit_next(self.loop.take_pending_inputs())
            self.reset_turn()
            return True
        if exit_now:
            self.request_shutdown()
            return True
        if handled:
            # A command must not strand queued follow-ups: flush them as run_agent_turn does, so
            # they keep chaining once the command completes (e.g. /compact then queued input).
            # Submit before restoring the idle prompt, where newer input can enter `pending`.
            self.submit_next(self.loop.take_pending_inputs())
            self.reset_turn()
            return True
        return False

    async def _drive_turn(self, user_input: str | UserInput) -> None:
        """Run one turn as its own task, and name its settled cancellation.

        A child task rather than a bare await: the turn is the thing Ctrl-C cancels, and cancelling
        it must not read as the runtime itself being torn down. It is awaited to completion first,
        so by the time this raises, Agent has settled or retracted the turn and saved its snapshot."""

        turn = asyncio.ensure_future(self.loop.agent.run(user_input))
        try:
            await turn
        except asyncio.CancelledError:
            if turn.cancelled():
                raise _TurnCancelled from None
            raise

    async def run_agent_turn(self, user_input: str | UserInput) -> None:
        user_input = user_input if isinstance(user_input, UserInput) else UserInput(user_input)
        self.loop.emit("")
        self.loop.user_turn_rule()
        self.loop.status_bar.begin()
        self.tui.set_running("working")
        self.turn_active = True
        started = time.monotonic()
        cancelled = False
        malformed_tool_call = False
        answered = False
        try:
            await self._drive_turn(user_input)
            answered = True
        except _TurnCancelled:
            cancelled = True
        except MalformedToolCallError as error:
            answer = str(error)
            malformed_tool_call = True
        except WizoltError as error:
            answer = f"Error: {error}"
        finally:
            self.loop.session.state.manual_model_retry_requested = False
            self.loop.schedule_index_freshness()
        try:
            if cancelled:
                self.loop.model_stream_output("", "")
                self.loop.emit_turn("Cancelled")
            else:
                # The engine publishes its own final answer through output_fn now; only errors it
                # raised before publishing land here.
                if not answered:
                    self.loop.ui.separate()
                    self.loop.ui.emit_answer(answer, rule=False, indent=TurnBox.CONTENT_LEVEL)
                # Emitted outside the promotion check: a promoted answer is already in scrollback
                # without its sources, so skipping the footer there would drop them exactly when a
                # search ran. It shares the answer's content indent.
                if footer := search_sources_footer(self.loop.agent.turn_sources):
                    self.loop.ui.emit_answer(footer, rule=False, indent=TurnBox.CONTENT_LEVEL)
                if not malformed_tool_call:
                    self.loop.ui.emit_turn_end(started)
            await self._finish_turn_submissions()
        finally:
            self.turn_active = False
            self.reset_turn()

    async def _finish_turn_submissions(self) -> None:
        """Place a FIFO boundary after inputs accepted during this turn and await its commit."""

        if self.submissions_task is None:
            await self.loop.session.save_snapshot()
            self.turn_active = False
            self.submit_next(self.loop.take_pending_inputs())
            return
        boundary = asyncio.get_running_loop().create_future()
        self.submissions.put_nowait(_Submission(turn_boundary=boundary))
        await boundary

    async def run_agent_loop(self) -> None:
        """Take submitted input one at a time, until shutdown is requested.

        The wait is an await on two futures, not a poll: whichever of the next input and the
        shutdown request arrives first ends it."""

        assert self.shutdown is not None
        while not self.shutdown.is_set():
            waiting = asyncio.ensure_future(self.pending.get())
            stopping = asyncio.ensure_future(self.shutdown.wait())
            try:
                await asyncio.wait({waiting, stopping}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in (waiting, stopping):
                    if not task.done():
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
            if not waiting.done() or waiting.cancelled():
                return
            user_input = waiting.result()
            self.loop.session.clear_quick_hints()  # the user acted; drop last turn's offerings (also covers slash commands, which skip Agent.run)
            if self.cancel_pending:
                self.loop.emit_turn("Cancelled")
                self.reset_turn()
                continue
            if (admitted := await self._admit_input(user_input)) is None:
                continue  # an image submission was refused; its draft is back in the editor
            user_input = admitted
            if not await self.dispatch(user_input):
                await self._turn_until_shutdown(user_input)

    async def _turn_until_shutdown(self, user_input: str | UserInput) -> None:
        """Run one turn, and let a shutdown request end the *wait* for it.

        The turn is not abandoned by that: it is a runtime-owned task, so `_shutdown` cancels and
        awaits it in its own fixed order, after asking the agent to stop. Without this race an
        application that ended under a live turn -- a force exit, a terminal that went away -- would
        leave the runtime waiting for a turn nobody can see any more, and shutdown could never
        reach the step that cancels it."""

        assert self.shutdown is not None
        turn = self.spawn(self.run_agent_turn(user_input), name="agent-turn")
        assert turn is not None
        stopping = asyncio.ensure_future(self.shutdown.wait())
        try:
            await asyncio.wait({turn, stopping}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not stopping.done():
                stopping.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stopping

    async def _run_application(self) -> None:
        """The prompt-toolkit application, as one task of this runtime.

        Its ending is the runtime's ending: a normal return means the app stopped on its own, and a
        failure is recorded for run() before shutdown proceeds. Either way the runtime unwinds
        rather than leaving the turn and the writer running with no terminal under them."""

        try:
            await self.tui.run(style=self.loop.view.style())
        except BaseException as error:
            if self.error is None:
                self.error = error
            raise
        finally:
            self.request_shutdown()

    async def run(self, *, show_banner: bool = True) -> int:
        """Own the interactive session: the application, the active turn, and the output queue.

        One loop for all of it, so a model request, a tool batch, an MCP call and a keystroke are
        just tasks that take turns -- and a turn's cancellation reaches everything it awaits."""

        self.runtime_loop = asyncio.get_running_loop()
        self.loop.open_background()
        self.submissions = asyncio.Queue()
        self.accepting = True
        self.shutdown = asyncio.Event()
        self.application_ready = asyncio.Event()
        self.loop.tui = self.build_tui()
        self.tui.on_ready = self.application_ready.set
        application = self.spawn(self._run_application(), name="tui-application")
        assert application is not None
        self.submissions_task = self.spawn(self._consume_submissions(), name="submissions")
        try:
            await self._await_ready(application)
            self.scrollback = ScrollbackWriter(self.runtime_loop, self.tui.write_to_scrollback, self.loop.ui.write_direct)
            self.loop.scrollback = self.scrollback
            self.loop.background_output_lock = self.scrollback.lock
            self.loop.agent.output_barrier = self.scrollback.barrier
            # Restored transcript lines wait until patch_stdout owns the terminal. The normal
            # CommandLoop entry already put its static banner in scrollback before terminal
            # probing; direct runtime callers retain the default banner here.
            resuming = self.loop.session.resumed
            if resuming:
                self.tui.set_running(RESUME_STATUS_LABEL)
            if show_banner:
                self.loop.start_session()
            else:
                self.loop.start_session(show_banner=False)
            if resuming:
                self.tui.set_idle()
            self.spawn(self.loop.discover_mcp(), name="mcp-discovery")
            # Git discovery can cost hundreds of milliseconds in a large worktree. Warm the
            # runtime-only snapshot after the prompt is live so the first picker need not wait.
            self.loop.refresh_mentions()
            self.submit_next(self.loop.take_pending_inputs())
            await self.run_agent_loop()
        finally:
            await self._shutdown(application)
        if self.error is not None:
            raise self.error
        return 0

    async def _await_ready(self, application: asyncio.Task) -> None:
        """Wait for the application to be live, or for it to have failed trying."""
        assert self.application_ready is not None
        ready = asyncio.ensure_future(self.application_ready.wait())
        try:
            await asyncio.wait({ready, application}, return_when=asyncio.FIRST_COMPLETED)
            if application.done():
                await application  # re-raises whatever stopped it
            else:
                await ready
        finally:
            if not ready.done():
                ready.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ready

    async def _shutdown(self, application: asyncio.Task) -> None:
        """Unwind in a fixed order, so nothing is closed while something still needs it.

        Stop taking new work; close submission admission and let the consumer finish the saves it
        already accepted; cancel and await the active turn, including tool quiescence and turn
        settlement; drain the tasks this runtime started; then close output -- admission first,
        then the accepted writes -- and only then exit the application and await it. The loop is
        closed by `asyncio.run` after this returns, which is why every close happens here."""

        if self.shutdown is not None:
            self.shutdown.set()
        if self.force_exit_timer is not None:
            self.force_exit_timer.cancel()
        # Before anything is cancelled: an accepted submission is a keystroke the reader already
        # sent, and its save is bounded. Cancelling the consumer first would drop it.
        await self._close_submissions()
        self.loop.agent.cancel()
        owned = [task for task in self.tasks if task is not application]
        for task in owned:
            task.cancel()
        if owned:
            await asyncio.gather(*owned, return_exceptions=True)
        await self.loop.close_background()
        await self.loop.close_resources()
        writer, self.scrollback = self.scrollback, None
        if writer is not None:
            # Admission and the drain are one step: the gate they share is this writer's lock, so a
            # background worker cannot pass it and then find the queue closed behind it.
            self.loop.close_background_output()
            await writer.close()
        self.loop.scrollback = None
        self.loop.agent.output_barrier = None
        self.tui.exit()
        with contextlib.suppress(asyncio.CancelledError):
            await application
        self.loop.tui = None
