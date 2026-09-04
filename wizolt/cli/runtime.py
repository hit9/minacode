"""TuiRuntime: drive the interactive session timeline while CommandLoop owns session behavior."""

from __future__ import annotations

import os
import queue
import signal
import threading
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from wizolt.base import MalformedToolCallError, TurnBox, WizoltError
from wizolt.cli.modals import tool_output_viewer
from wizolt.image import UserInput
from wizolt.render import search_sources_footer
from wizolt.tools import CodeIndex
from wizolt.tui import TuiApp

# The TUI status label shown while a resumed session's transcript is being restored: a quiet
# lead-in before the single-write replay, so the wait reads as a restore in progress rather than a
# stuck prompt.
RESUME_STATUS_LABEL = "resuming session…"

if TYPE_CHECKING:
    from wizolt.cli import CommandLoop


class TuiRuntime:
    """Own the interactive session timeline while CommandLoop owns session behavior."""

    def __init__(self, command_loop: CommandLoop):
        self.loop = command_loop
        self.pending: queue.Queue[UserInput] = queue.Queue()
        self.stop = threading.Event()
        self.cancel_pending = threading.Event()
        self.main_busy = threading.Event()
        self.force_exit_timer: threading.Timer | None = None
        self.error: BaseException | None = None

    @property
    def tui(self) -> TuiApp:
        assert self.loop.tui is not None
        return self.loop.tui

    def _interrupt_active(self, cancel: Callable[[], None]) -> None:
        threading.Thread(target=cancel, daemon=True).start()
        if self.main_busy.is_set():
            os.kill(os.getpid(), signal.SIGINT)

    def interrupt(self) -> None:
        if self.cancel_pending.is_set():
            return
        self.cancel_pending.set()
        self.tui.set_running("cancelling")
        self._interrupt_active(self.loop.agent.cancel)

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
            threading.Thread(target=self.loop.run_queued_command, args=(text,), daemon=True).start()
        else:
            self.loop.session.enqueue_user_input(value)
            self.loop.session.save_snapshot()
        self.tui.invalidate()

    def recall(self) -> str | UserInput:
        return self.loop.recall_pending_input(self._request_model_retry)

    def expand_output(self) -> None:
        threading.Thread(target=lambda: tool_output_viewer(self.loop), name="tool-output", daemon=True).start()

    def request_exit(self) -> None:
        self.stop.set()
        self.loop.save_and_emit_resume()

    def force_exit(self) -> None:
        self.stop.set()
        threading.Thread(target=self.loop.agent.cancel, daemon=True).start()
        self.force_exit_timer = threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM))
        self.force_exit_timer.daemon = True
        self.force_exit_timer.start()
        os.kill(os.getpid(), signal.SIGINT)

    def build_tui(self) -> TuiApp:
        return TuiApp(
            on_chat_submit=self.pending.put,
            on_running_submit=self.submit_running,
            on_exit_request=self.request_exit,
            on_force_exit=self.force_exit,
            on_interrupt=self.interrupt,
            on_retry=self._request_model_retry,
            on_recall=self.recall,
            on_expand_output=self.expand_output,
            status_fragments_fn=lambda: self.loop.status_bar.display_fragments(active=self.tui.input_mode == "running"),
            activity_fragments_fn=self.loop.view.tui_activity_fragments,
            input_hint_fn=self.loop.view.tui_input_hint,
            quick_hints_fn=lambda: self.loop.session.quick_hints,
            file_picker_available_fn=self.loop.session.mentions.picker.available if self.loop.session.mentions else None,
            file_picker_fn=self.loop.session.mentions.picker.pick if self.loop.session.mentions else None,
            file_complete_fn=self.loop.session.mentions.complete_async if self.loop.session.mentions else None,
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
        self.pending.put(first)
        for text in entered[1:]:
            self.loop.session.enqueue_user_input(text)

    def reset_turn(self) -> None:
        self.loop.model_stream_output("", "")
        # A request can fail after permanent promotion but before Agent re-publishes the text and
        # consumes its marker. Never let that stale marker suppress an identical later response.
        with self.loop.model_stream_lock:
            self.loop.model_stream_promoted_text = ""
        self.tui.set_idle()
        self.cancel_pending.clear()
        self.main_busy.clear()

    def dispatch(self, user_input: str | UserInput) -> bool:
        """Dispatch one input. Return true when it was fully handled as a command."""
        user_input = user_input if isinstance(user_input, UserInput) else UserInput(user_input)
        self.loop.ui.emit_answer(user_input.display_text(), role="user", rule=False)
        try:
            handled, exit_now = self.loop.command(user_input.strip())
        except (KeyboardInterrupt, WizoltError) as error:
            self.loop.emit_turn("Cancelled" if isinstance(error, KeyboardInterrupt) else f"Error: {error}")
            self.submit_next(self.loop.take_pending_inputs())
            self.reset_turn()
            return True
        if exit_now:
            self.stop.set()
            self.main_busy.clear()
            self.tui.exit()
            return True
        if handled:
            # A command must not strand queued follow-ups: flush them as run_agent_turn does, so
            # they keep chaining once the command completes (e.g. /compact then queued input).
            # Submit before restoring the idle prompt, where newer input can enter `pending`.
            self.submit_next(self.loop.take_pending_inputs())
            self.reset_turn()
            return True
        return False

    def run_agent_turn(self, user_input: str | UserInput) -> None:
        user_input = user_input if isinstance(user_input, UserInput) else UserInput(user_input)
        self.loop.emit("")
        self.loop.user_turn_rule()
        self.loop.status_bar.begin()
        self.tui.set_running("working")
        started = time.monotonic()
        cancelled = False
        malformed_tool_call = False
        answered = False
        try:
            self.loop.agent.run(user_input)
            answered = True
        except KeyboardInterrupt:
            self.submit_next(self.loop.take_pending_inputs())
            cancelled = True
        except MalformedToolCallError as error:
            answer = str(error)
            malformed_tool_call = True
        except WizoltError as error:
            answer = f"Error: {error}"
        finally:
            self.reset_turn()
            self.loop.session.state.manual_model_retry_requested = False
            CodeIndex(self.loop.session).update_pending_async()
        if cancelled:
            self.loop.emit_turn("Cancelled")
            return
        # The engine publishes its own final answer through output_fn now; only errors it raised
        # before publishing land here.
        if not answered:
            if self.loop.ui.color:
                self.loop.emit()
            self.loop.ui.emit_answer(answer, rule=False, indent=TurnBox.CONTENT_LEVEL)
        # Emitted outside the promotion check: a promoted answer is already in scrollback without
        # its sources, so skipping the footer there would drop them exactly when a search ran.
        # Indented like the answer it belongs to, which the engine publishes through
        # emit_agent_output at CONTENT_LEVEL; at column 0 it would hang off the left of its answer.
        if footer := search_sources_footer(self.loop.agent.turn_sources):
            self.loop.ui.emit_answer(footer, rule=False, indent=TurnBox.CONTENT_LEVEL)
        if not malformed_tool_call:
            self.loop.ui.emit_turn_end(started)
        self.loop.session.save_snapshot()
        self.submit_next(self.loop.take_pending_inputs())

    def run_agent_loop(self) -> None:
        while not self.stop.is_set():
            try:
                user_input = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            self.main_busy.set()
            self.loop.session.clear_quick_hints()  # the user acted; drop last turn's offerings (also covers slash commands, which skip Agent.run)
            if self.cancel_pending.is_set():
                self.loop.emit_turn("Cancelled")
                self.reset_turn()
                continue
            if not self.dispatch(user_input):
                self.run_agent_turn(user_input)

    def run_tui_app(self) -> None:
        try:
            self.tui.run(style=self.loop.view.style())
        except BaseException as error:  # noqa: BLE001 - propagate every TUI-thread failure on the main thread.
            self.error = error
            self.stop.set()

    def run(self) -> int:
        """Run the agent on the main thread and prompt-toolkit on one joined UI thread."""
        self.loop.tui = self.build_tui()
        tui_thread = threading.Thread(target=self.run_tui_app, name="tui")
        tui_thread.start()
        try:
            self.tui.ready.wait()
            if self.error is not None:
                raise self.error
            # Emit startup and restored transcript lines only after patch_stdout owns the terminal,
            # so the primary-screen application places them in native terminal/tmux scrollback.
            resuming = self.loop.session.resumed
            if resuming:
                self.tui.set_running(RESUME_STATUS_LABEL)
            self.loop.start_session()
            if resuming:
                self.tui.set_idle()
            if self.loop.session.mentions is not None:
                # Git discovery can cost hundreds of milliseconds in a large worktree. Warm its
                # runtime-only snapshot after the prompt is live so the first picker need not wait.
                self.loop.session.mentions.refresh_async()
            self.submit_next(self.loop.take_pending_inputs())
            self.run_agent_loop()
        finally:
            self.stop.set()
            if self.force_exit_timer is not None:
                self.force_exit_timer.cancel()
            self.tui.exit()
            # Do not let interpreter finalization race a TUI thread flushing stdout. The emergency
            # force-exit timer remains responsible for terminating a genuinely wedged application.
            tui_thread.join()
            try:
                self.loop.close_background_output()
            finally:
                self.loop.tui = None
        if self.error is not None:
            raise self.error
        return 0
