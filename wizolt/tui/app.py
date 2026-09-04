"""wizolt prompt-toolkit application, its input-side processors, and modal plumbing."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from prompt_toolkit import search as pt_search
from prompt_toolkit.application import Application, create_app_session, run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions, is_done, is_searching
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import ConditionalContainer, Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import BeforeInput, HighlightIncrementalSearchProcessor, Processor, Transformation
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import SearchToolbar

from wizolt.base import (
    LogBlock,
    LogEdge,
    WizoltError,
    fail_if_running_loop,
)
from wizolt.image import IMAGE_MARKER, ImageInputs, ImageRef, UserInput
from wizolt.mentions import FilePick, MentionSpan, active_mention, encode_file_mention, scan_mentions
from wizolt.render import UiPrinter
from wizolt.tui.views import TUI_MODAL_PENDING


@dataclass
class TuiModal:
    fragments_fn: Callable[[], StyleAndTextTuples]
    key_fn: Callable[[str, str], Any]
    exclusive: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    # Set when the modal was opened from the loop, so its result is awaited there instead of
    # waited on from a worker thread.
    future: Any = None


@dataclass(frozen=True)
class _EditDelta:
    """The minimal single-edit difference between two strings (pure data)."""

    prefix: int  # length of the unchanged common prefix
    removed: str  # slice of the old text that was deleted
    inserted: str  # slice of the new text that was added


def _edit_delta(old: str, new: str) -> _EditDelta:
    """Compute the minimal edit turning ``old`` into ``new`` via common prefix/suffix framing."""
    prefix = 0
    limit = min(len(old), len(new))
    while prefix < limit and old[prefix] == new[prefix]:
        prefix += 1
    suffix = 0
    while suffix < len(old) - prefix and suffix < len(new) - prefix and old[-suffix - 1] == new[-suffix - 1]:
        suffix += 1
    return _EditDelta(
        prefix=prefix,
        removed=old[prefix : len(old) - suffix],
        inserted=new[prefix : len(new) - suffix],
    )


class CallbackPlaceholder(Processor):
    def __init__(self, text_fn: Callable[[], str]):
        self.text_fn = text_fn

    def apply_transformation(self, transformation_input) -> Transformation:
        ti = transformation_input
        text = self.text_fn()
        buffer = ti.buffer_control.buffer
        if not text or buffer is None or buffer.text or ti.lineno != ti.document.line_count - 1:
            return Transformation(ti.fragments)
        return Transformation([*ti.fragments, ("class:queue.hint", text)])


class ImageLabelProcessor(Processor):
    """Render each one-cell image marker as an atomic, readable inline label."""

    def __init__(self, images_fn: Callable[[], tuple[ImageRef, ...]]):
        self.images_fn = images_fn

    def apply_transformation(self, transformation_input) -> Transformation:
        ti = transformation_input
        images = self.images_fn()
        before = sum(line.count(IMAGE_MARKER) for line in ti.document.lines[: ti.lineno])
        source = "".join(fragment[1] for fragment in ti.fragments)
        labels: dict[int, str] = {}
        ordinal = before
        for index, char in enumerate(source):
            if char == IMAGE_MARKER and ordinal < len(images):
                ordinal += 1
                labels[index] = f"[Image #{ordinal} \u00b7 {images[ordinal - 1].name}]"
        if not labels:
            return Transformation(ti.fragments)
        fragments: StyleAndTextTuples = []
        source_index = 0
        for fragment in ti.fragments:
            style, text = fragment[0], fragment[1]
            for char in text:
                label = labels.get(source_index)
                fragments.append(("class:image.attachment" if label else style, label or char))
                source_index += 1

        def source_to_display(index: int) -> int:
            return index + sum(len(label) - 1 for position, label in labels.items() if position < index)

        def display_to_source(index: int) -> int:
            display = 0
            for position in range(len(source) + 1):
                if display >= index:
                    return position
                value = labels[position] if position in labels else (source[position] if position < len(source) else "")
                display += len(value)
            return len(source)

        return Transformation(fragments, source_to_display=source_to_display, display_to_source=display_to_source)


class TuiApp:
    """One primary-screen application for live activity, input, selectors, and status.

    The agent owns the main thread; prompt-toolkit owns the TUI thread. `request_input` bridges
    blocking approvals, while completed output is printed above the app into terminal scrollback.
    """

    MODAL_KEYS: ClassVar[tuple[str, ...]] = tuple(
        "j k h l g G up down left right tab s-tab enter escape q r pagedown pageup c-d c-u c-o backspace c-h /".split()  # noqa: SIM905 - compact key table.
    )
    # Frame budget for the running divider. Motion is smooth only while a moving highlight advances
    # about one cell per frame, so this rate is what `View.QUEUE_SWEEP_CELLS_PER_SEC` follows.
    ANIMATION_INTERVAL: ClassVar[float] = 1 / 30
    # Idle refresh: no animation runs on the idle screen, only the 0.2s index and MCP spinners.
    IDLE_REFRESH_INTERVAL: ClassVar[float] = 0.2
    # Debounce inline second-stage candidate refreshes while the user is still typing. The external
    # file picker overrides this with a zero-delay handoff at the namespace boundary.
    MENTION_TRANSITION_DELAY: ClassVar[float] = 0.12
    # Quick-hint chips wrap to a new line after this many per row, matching the tool's 2-4 range
    # with the upper end always reachable: four short hints never crowd one line.
    MAX_QUICK_HINTS_PER_ROW: ClassVar[int] = 3

    def __init__(
        self,
        *,
        on_chat_submit: Callable[[UserInput], None] | None = None,
        on_running_submit: Callable[[UserInput], None] | None = None,
        on_exit_request: Callable[[], None] | None = None,
        on_force_exit: Callable[[], None] | None = None,
        on_interrupt: Callable[[], None] | None = None,
        on_retry: Callable[[], None] | None = None,
        on_recall: Callable[[], str | UserInput] | None = None,
        on_expand_output: Callable[[], None] | None = None,
        status_fragments_fn: Callable[[], StyleAndTextTuples] | None = None,
        activity_fragments_fn: Callable[[], StyleAndTextTuples] | None = None,
        input_hint_fn: Callable[[], str] | None = None,
        quick_hints_fn: Callable[[], tuple[str, ...]] | None = None,
        file_picker_available_fn: Callable[[], bool] | None = None,
        file_picker_fn: Callable[[str], FilePick] | None = None,
        file_complete_fn: Callable[[str, Callable[[], None]], None] | None = None,
        editor_context_fn: Callable[[], str] | None = None,
        images: ImageInputs | None = None,
        image_cwd: str = "",
        history: FileHistory | None = None,
        completer: Completer | None = None,
        on_app_stop: Callable[[], None] | None = None,
    ) -> None:
        self.on_chat_submit = on_chat_submit or (lambda _: None)
        self.on_running_submit = on_running_submit or (lambda _: None)
        self.on_exit_request = on_exit_request or (lambda: None)
        self.on_force_exit = on_force_exit or (lambda: None)
        self.on_interrupt = on_interrupt or (lambda: None)
        self.on_retry = on_retry or (lambda: None)
        self.on_recall = on_recall or (lambda: "")
        self.on_expand_output = on_expand_output or (lambda: None)
        self.status_fragments_fn: Callable[[], StyleAndTextTuples] = status_fragments_fn or list
        # Called once after the application stops, before the terminal is handed back, so the
        # owner can flush anything still queued (see UiPrinter.drain_scrollback).
        self.on_app_stop = on_app_stop or (lambda: None)
        self.activity_fragments_fn: Callable[[], StyleAndTextTuples] = activity_fragments_fn or list
        self.input_hint_fn = input_hint_fn or (lambda: "")
        self.quick_hints_fn: Callable[[], tuple[str, ...]] = quick_hints_fn or (lambda: ())
        self.file_picker_available_fn = file_picker_available_fn or (lambda: False)
        self.file_picker_fn = file_picker_fn or (lambda _query: FilePick(unavailable=True))
        self.file_complete_fn = file_complete_fn or (lambda _query, ready: ready())
        self.editor_context_fn = editor_context_fn or (lambda: "")
        self.images = images if images is not None else ImageInputs(cwd=image_cwd)
        self.input_images: tuple[ImageRef, ...] = ()
        self._last_input_text = ""
        self._changing_input = False
        self._search_start_text = ""
        self.input_error = ""
        self.history = history
        self.input_buffer = Buffer(
            history=history,
            completer=completer,
            complete_while_typing=False,
            enable_history_search=True,
            multiline=True,
            accept_handler=self._accept,
        )
        self.input_buffer.on_text_changed += self._on_input_text_changed
        self.search_toolbar = SearchToolbar()
        self.app: Application | None = None
        self.ready = threading.Event()
        self.input_mode = "chat"  # chat | dispatch | running | approval
        self.quick_hint_focus = -1  # -1 = input focused; 0..n-1 = that quick-input chip
        self._quick_hint_resume_focus = -1  # last picked chip; Tab resumes after it once
        self.quick_hint_picked: list[str] = []  # chips picked into the input, in pick order
        self._last_quick_hints: tuple[str, ...] | None = None  # hints seen by the last quick_hints() call
        self._file_picker_active = False
        self._mention_transition_timer: asyncio.TimerHandle | None = None
        self.input_prompt = UiPrinter.PROMPT_PREFIX
        # Every line of the prompt except the last. The input row's prefix is a single-line
        # processor, so these are rendered as their own rows above it. See _set_mode.
        self._input_prompt_above: list[str] = []
        self._input_pending: asyncio.Future[str | None] | None = None
        self._input_loop: asyncio.AbstractEventLoop | None = None
        # Set while no modal is visible, so an approval prompt can wait for a selector to close
        # without polling. Created on the loop that first awaits it.
        self._modal_idle: asyncio.Event | None = None
        # None is the cancellation signal, distinct from every string the user can submit (including
        # ""). See request_input: callers must not read a cancel as an answer.
        # (label, answer) actions of the current approval prompt, and which one is focused. See
        # set_approval_form.
        self._approval_actions: list[tuple[str, str]] = []
        self._approval_focus = 0
        self.status_label: str = ""
        self.modal: TuiModal | None = None
        self.modal_lock = threading.Lock()
        self.input_window: Window | None = None
        self.activity_window: Window | None = None
        self.modal_window: Window | None = None
        self.exclusive_modal_window: Window | None = None
        self.status_window: Window | None = None

    async def request_input_async(self, prompt: str) -> str | None:
        """Ask for a line of user input inline (an approval prompt, an Ask free-text page) and await it.

        Returns None when the input was cancelled instead of answered: Ctrl-C, Ctrl-D on an empty
        line, or the app exiting while something is still waiting here. Cancellation must be its own
        value, not a string: "" is a legitimate submission that `confirm` reads as the default
        approve, and any placeholder text ("cancelled") would reach the model as a real answer.
        Callers decide what a cancel means -- `confirm` refuses, Ask dismisses.

        At most one request may be pending; a second is an internal error, not a queue. The input
        mode is restored in `finally`, so a cancelled or failed request never strands the prompt."""

        if self._input_pending is not None:
            raise WizoltError("internal error: a TUI input request is already pending")
        # A tool approval must not replace an already-visible selector. Wait for that selector to
        # close, then reuse the shared input row.
        await self._modal_idle_event().wait()
        loop = asyncio.get_running_loop()
        pending: asyncio.Future[str | None] = loop.create_future()
        self._input_loop = loop
        self._input_pending = pending
        previous_mode, previous_prompt = self.input_mode, self.full_input_prompt()
        previous_document: Document | None = None
        previous_images = self.input_images

        def switch(document: Document, mode: str, prompt_text: str) -> None:
            nonlocal previous_document
            if previous_document is None:
                previous_document = self.input_buffer.document
            images = previous_images if mode == previous_mode else ()
            self._reset_input(UserInput(document.text, images), cursor_position=document.cursor_position)
            self._set_mode(mode, prompt_text)

        switch(Document(""), "approval", prompt)
        try:
            return await pending
        finally:
            self._input_pending = None
            self._input_loop = None
            self._approval_actions = []  # the form belongs to one prompt; the next one declares its own
            switch(previous_document or Document(""), previous_mode, previous_prompt)

    def resolve_input(self, value: str | None) -> None:
        """Answer the pending input request. Called on the loop; a later answer is ignored."""
        pending = self._input_pending
        if pending is not None and not pending.done():
            pending.set_result(value)

    def cancel_input(self) -> None:
        """Resolve a pending input request as cancelled, from any thread.

        Idempotent and safe when nothing is pending. Scheduled onto the loop that owns the future
        when it is asked for from elsewhere: a future belongs to its loop, whoever asks."""

        pending, loop = self._input_pending, self._input_loop
        if pending is None or loop is None or pending.done():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self.resolve_input(None)
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self.resolve_input, None)

    def _modal_idle_event(self) -> asyncio.Event:
        """Set while no modal is visible. Created on the loop that first awaits it."""
        if self._modal_idle is None:
            self._modal_idle = asyncio.Event()
            if self.modal is None:
                self._modal_idle.set()
        return self._modal_idle

    def set_approval_form(self, actions: list[tuple[str, str]]) -> bool:
        """Give the *next* approval prompt a row of selectable actions, as (label, answer) pairs
        with the default first. Returns whether the form was installed, so the caller can fall back
        to a typed legend when there is no TUI.

        The row is navigated with Tab/arrows and fired with Enter, which submits that action's
        `answer` -- the same whole line the user could have typed, so the typed protocol underneath
        is untouched. Typing anything moves to the reason field instead: no letter is ever a
        shortcut, so no reason is unwritable and there is nothing to memorize.

        Cleared when the prompt resolves. Set it before every prompt that wants it."""
        self._approval_actions = list(actions)
        self._approval_focus = 0
        return True

    def approval_form_fragments(self) -> StyleAndTextTuples:
        """The live action row. Dimmed whole once a reason is being typed, because Enter then sends
        the reason rather than firing the focused action -- the row has to stop looking armed."""
        if not self._approval_actions or self.input_mode != "approval":
            return []
        typing = bool(self.input_buffer.text)
        parts: StyleAndTextTuples = [("ansibrightblack", LogBlock.prefix(2, LogEdge.CONTINUE))]
        for index, (label, _) in enumerate(self._approval_actions):
            focused = index == self._approval_focus and not typing
            style = "class:approval.action.focused" if focused else "class:approval.action.dim" if typing else "class:approval.action"
            parts.append((style, f" {label} "))
            parts.append(("", "  "))
        parts.append(("class:approval.action.dim", "  Enter send · Esc back" if typing else "  Tab to move"))
        return parts

    def move_approval_focus(self, delta: int) -> None:
        if self._approval_actions:
            self._approval_focus = (self._approval_focus + delta) % len(self._approval_actions)
            self.invalidate()

    def set_running(self, label: str) -> None:
        self.status_label = label
        self._set_mode("running", "+> ")

    def set_dispatching(self, prompt: str = "") -> None:
        self._set_mode("dispatch", prompt)

    def set_idle(self) -> None:
        self.status_label = ""
        self._set_mode("chat", UiPrinter.PROMPT_PREFIX)

    def _set_mode(self, mode: str, prompt: str) -> None:
        self.input_mode = mode
        # The prompt reaches the input row through BeforeInput, a single-line processor, and
        # BufferControl does not split processor output on "\n" the way FormattedTextControl does --
        # a literal newline would reach the screen as the "^J" control character. Only the last line
        # can be a prefix; the rest are rows of their own above the input. A prompt that opens with
        # "\n" (Ask free-text asking for a blank line above the question) is just the empty-first-line
        # case of this, and multi-line questions are the common one.
        above, separator, last = prompt.rpartition("\n")
        self._input_prompt_above = above.split("\n") if separator else []
        self.input_prompt = last
        self._clear_quick_hint_selection()
        if mode not in {"chat", "running"}:
            self.input_error = ""
        self.invalidate()

    def invalidate(self) -> None:
        app = self.app
        if app is None:
            return
        try:
            app.invalidate()
        except RuntimeError:
            # prompt-toolkit checks `is_running` before scheduling, but shutdown can clear its
            # event loop between that check and the thread-safe call. A redraw that lost that race
            # is obsolete; an error from a still-running application is not.
            if app.is_running:
                raise

    def invalidate_frame(self) -> None:
        """Ask for a redraw from a source that fires far faster than the eye needs.

        Model output arrives token by token. While the running region is on screen the animation
        ticker already redraws at the frame rate, so redrawing per token only makes the cadence
        swing with the model's pace; anywhere else there is no ticker, so redraw normally.
        """
        if self.input_mode != "running":
            self.invalidate()

    async def write_to_scrollback_async(self, callback: Callable[[], None]) -> None:
        """Print above the live application, on its own loop, and return once the terminal took it.

        `run_in_terminal` owns the erase/write/redraw sequence, while `create_app_session` routes
        nested prompt-toolkit printers to this application's output. A write failure is raised to
        the caller -- the writer task -- rather than buried in a background task nobody observes.

        Once the application has stopped there is nothing to print above, so the callback runs
        directly: that is the same fallback the direct-output path uses while the runtime unwinds."""

        app = self.app
        if app is None or not app.is_running:
            callback()
            return

        def render() -> None:
            with create_app_session(output=app.output):
                callback()

        await run_in_terminal(render)

    def write_to_scrollback(self, callback: Callable[[], None]) -> None:
        """Print above the live application and wait until the terminal has accepted it.

        The blocking form, for callers that are not on the application's loop -- a modal viewer on
        its own thread, a standalone application. The runtime uses the awaitable form through its
        ordered writer instead.
        """
        app = self.app
        if app is None or not app.is_running:
            callback()
            return
        done = threading.Event()
        errors: list[Exception] = []

        async def write() -> None:
            try:

                def render() -> None:
                    with create_app_session(output=app.output):
                        callback()

                await run_in_terminal(render)
            except Exception as error:  # noqa: BLE001 - return terminal failures to the agent thread.
                errors.append(error)
            finally:
                done.set()

        def schedule_write() -> None:
            app.create_background_task(write())

        self._schedule(schedule_write)
        done.wait()
        if errors:
            raise errors[0]

    def _schedule(self, callback: Callable[..., None], *args: Any) -> None:
        app = self.app
        if app is not None and app.is_running:
            loop = app.loop
            assert loop is not None
            loop.call_soon_threadsafe(callback, *args)
        else:
            callback(*args)

    def exit(self) -> None:
        app = self.app
        if app is None:
            return

        def close() -> None:
            with contextlib.suppress(Exception):
                app.exit(result=None)

        self._schedule(close)

    def _accept(self, buffer: Buffer) -> bool:
        text = buffer.text
        if self.input_mode == "approval" and self._input_pending is not None:
            # Enter fires the focused action while the line is empty, and sends the reason once
            # there is one. Both submit a plain string, so the approval loop reads one protocol.
            self.resolve_input(self._approval_actions[self._approval_focus][1] if not text and self._approval_actions else text)
            return False
        if self.input_mode == "running":
            if text.strip():
                value = self._submitted_input()
                if value is None:
                    return True
                self._append_history(value)
                self._reset_input("")
                self.on_running_submit(value)
                return True
            return False
        if self.input_mode == "chat":
            if not text.strip():
                return False
            value = self._submitted_input()
            if value is None:
                return True
            self._append_history(value)
            self._reset_input("")
            self.set_dispatching()
            self.on_chat_submit(value)
            return True
        return False

    def _submitted_input(self) -> UserInput | None:
        value = self._recognize_input()
        try:
            value = self.images.prepare(value)
        except WizoltError as error:
            self.input_error = str(error)
            self.invalidate()
            return None
        self.input_error = ""
        return value

    def _append_history(self, value: UserInput) -> None:
        if self.history is not None:
            self.history.append_string(value.original_text())

    def _load_buffer_history_now(self) -> None:
        """Copy history entries into the input buffer synchronously when its async loader has not.

        `Buffer.reset` - every submit - cancels the background task that copies history into the
        buffer's working lines, and the copy only restarts at the next repaint. A recall key can
        arrive first, right after Enter, and `auto_up` then walks a list with no entries: nothing
        is recalled (or, with older history, a stale entry). Left alone when nothing has been
        loaded from disk yet - the first repaint's loader owns that, and there is nothing to race.
        """
        buffer = self.input_buffer
        entries = self.history.get_strings() if self.history is not None else []
        if not entries or buffer._load_history_task is not None:
            return
        for entry in reversed(entries):  # Oldest first: each appendleft lands before the last one.
            buffer._working_lines.appendleft(entry)
        # The index setter parks the cursor at zero; the native loader moves the index without
        # touching it, and the text is unchanged here (the edited line just moved), so keep the
        # cursor where it was - a draft being recalled over should not jump to the line start.
        cursor = buffer.cursor_position
        buffer.working_index = len(entries)  # The freshly reset line stays the one being edited.
        buffer.cursor_position = cursor
        done = asyncio.get_running_loop().create_future()
        done.set_result(None)
        buffer._load_history_task = done  # The next repaint must not copy the entries in again.

    def _recognize_input(self) -> UserInput:
        value = self.images.recognize(self.input_buffer.text, self.input_images)
        if str(value) != self.input_buffer.text or value.images != self.input_images:
            self._reset_input(value, cursor_position=len(value))
        return value

    def _reset_input(self, value: str | UserInput, *, cursor_position: int | None = None, preserve_quick_hints: bool = False) -> None:
        user_input = value if isinstance(value, UserInput) else UserInput(value)
        if not preserve_quick_hints:
            self._clear_quick_hint_selection()
        self._changing_input = True
        try:
            self.input_images = user_input.images
            self._last_input_text = str(user_input)
            position = len(user_input) if cursor_position is None else cursor_position
            self.input_buffer.reset(Document(str(user_input), cursor_position=position))
        finally:
            self._changing_input = False

    def _abort_history_search(self) -> None:
        """Abort an in-flight Ctrl-R search and restore the input as it was before the search started."""
        pt_search.stop_search()
        self._reset_input(self._search_start_text)

    def quick_hints(self) -> tuple[str, ...]:
        hints = self.quick_hints_fn()
        if hints != self._last_quick_hints:
            changed = self._last_quick_hints is not None
            self._last_quick_hints = hints
            if changed:
                self._clear_quick_hint_selection()
        return hints

    def _clear_quick_hint_selection(self) -> None:
        self.quick_hint_focus = -1
        self._quick_hint_resume_focus = -1
        self.quick_hint_picked = []

    def quick_hint_fragments(self) -> StyleAndTextTuples:
        hints = self.quick_hints()
        if not hints:
            return []
        return self._flow_quick_hints(hints, self._quick_hint_columns(), self.quick_hint_focus, tuple(self.quick_hint_picked))

    def _quick_hint_columns(self) -> int:
        """The terminal width in cells the hint row is rendered into; 0 when unknown."""
        app = self.app
        if app is None:
            return 0
        try:
            return app.output.get_size().columns
        except Exception:  # noqa: BLE001 - a terminal that refuses to report its size falls back to one unwrapped row.
            return 0

    @staticmethod
    def _flow_quick_hints(hints: tuple[str, ...], columns: int, focus: int, picked: tuple[str, ...]) -> StyleAndTextTuples:
        """Lay chips out left to right, at most three per row, wrapping only between chips.

        A row ends when it holds `MAX_QUICK_HINTS_PER_ROW` chips or the next chip would not fit
        in the remaining width. `columns` is the width in cells (0 = unknown: ignore width and
        let the window wrap as a fallback). Chips never wrap mid-text except for one extreme:
        a single chip wider than the whole terminal overflows its own row and the window's own
        `wrap_lines` splits it; every ordinary chip stays whole and distinguishable.
        """
        parts: StyleAndTextTuples = []
        line_width = 0
        chips_on_row = 0
        for index, hint in enumerate(hints):
            chip = f" \u2713 {hint} " if hint in picked else f" {hint} "
            chip_width = get_cwidth(chip)
            if index:
                if chips_on_row >= TuiApp.MAX_QUICK_HINTS_PER_ROW or (columns and line_width + 3 + chip_width > columns):
                    parts.append(("class:quickhint", "\n"))
                    line_width = 0
                    chips_on_row = 0
                else:
                    parts.append(("class:quickhint.sep", " \u2502 "))
                    line_width += 3
            style = "class:quickhint.focused" if index == focus else "class:quickhint"
            parts.append((style, chip))
            line_width += chip_width
            chips_on_row += 1
        return parts

    def cycle_quick_hint_focus(self, reverse: bool = False) -> None:
        count = len(self.quick_hints())
        if not count:
            return
        origin = self.quick_hint_focus
        if origin == -1 and self._quick_hint_resume_focus >= 0:
            origin = self._quick_hint_resume_focus
            self._quick_hint_resume_focus = -1
        focus = origin + (-1 if reverse else 1)
        if focus >= count or focus < -1:
            focus = count - 1 if reverse else -1
        self.quick_hint_focus = focus
        self.invalidate()

    def _live_quick_hints(self, buffer: Buffer) -> tuple[str, ...]:
        """The chips Tab and Enter act on, or () when the chip row is not in play.

        It is in play on a chat prompt that still holds exactly the picked text -- any manual edit
        leaves that agreement -- and with no completion menu open, which owns both keys while it
        is up. Refreshing the hints here is what makes the two keys decide from one snapshot.
        """
        if self.input_mode != "chat":
            return ()
        hints = self.quick_hints()
        if not hints or buffer.complete_state is not None or buffer.text != "\n".join(self.quick_hint_picked):
            return ()
        return hints

    def tab_or_complete(self, buffer: Buffer, *, reverse: bool) -> None:
        # On an idle prompt whose text is only picked suggestions Tab/Shift-Tab cycle the quick
        # inputs; anywhere else they complete.
        if self._live_quick_hints(buffer):
            self.cycle_quick_hint_focus(reverse=reverse)
            return
        target = self._file_mention_at_cursor(buffer)
        if not reverse and target is not None:
            span, end = target
            if self.file_picker_available_fn() and self.app is not None:
                # Commit a visible @file: preview before the picker snapshots the buffer. This also
                # makes Tab an immediate accelerator during the namespace transition delay.
                buffer.complete_state = None
                self._start_file_picker(buffer, span, end)
                return
            if buffer.complete_state is None:
                self._refresh_file_completions(buffer)
        self.complete_input(buffer, reverse=reverse)

    def _pick_quick_hint(self, buffer: Buffer) -> bool:
        """Toggle the chip Tab has focused into the input; True when the chip row was in play.

        Enter picks and returns focus to the input line, so a second Enter sends. The pick is
        a toggle: a chip already picked is unpicked."""
        hints = self._live_quick_hints(buffer)
        if not hints or not 0 <= self.quick_hint_focus < len(hints):
            return False
        picked_focus = self.quick_hint_focus
        hint = hints[picked_focus]
        if hint in self.quick_hint_picked:
            self.quick_hint_picked.remove(hint)
        else:
            self.quick_hint_picked.append(hint)
        self._reset_input("\n".join(self.quick_hint_picked), preserve_quick_hints=True)
        self.quick_hint_focus = -1
        self._quick_hint_resume_focus = picked_focus
        return True

    def placeholder_text(self) -> str:
        if self.input_mode == "chat" and self.quick_hints():
            return "" if self.quick_hint_focus >= 0 else "Tab cycles suggestions \u00b7 Enter picks \u00b7 Enter sends"
        return self.input_hint_fn()

    def _on_input_text_changed(self, buffer: Buffer) -> None:
        """Reconcile everything that tracks the input text after an edit (image markers, recognition)."""
        text = buffer.text
        if self._changing_input:
            self._last_input_text = text
            return
        old = self._last_input_text
        if old == text:
            return
        if self.quick_hint_picked and text != "\n".join(self.quick_hint_picked):
            # A manual edit leaves the picked-chip agreement: drop the picks so Tab stops cycling
            # chips and the edited text is what sends.
            self._clear_quick_hint_selection()
        self.input_error = ""
        delta = _edit_delta(old, text)
        self._sync_input_images(old, delta)
        self._last_input_text = text
        if delta.inserted and delta.inserted[-1].isspace() and self.input_mode in {"chat", "running"}:
            self._recognize_input()
        self._offer_mention_completions(buffer, delta)

    def _offer_mention_completions(self, buffer: Buffer, delta: _EditDelta) -> None:
        self._cancel_mention_transition()
        if self.input_mode not in {"chat", "running"} or not delta.inserted:
            return
        span = active_mention(buffer.document.text_before_cursor)
        if span is None:
            return
        old_text = buffer.text[: delta.prefix] + delta.removed + buffer.text[delta.prefix + len(delta.inserted) :]
        old_cursor = delta.prefix + len(delta.removed)
        old_span = active_mention(old_text[:old_cursor])
        selected_namespace = (
            not span.payload
            and old_span is not None
            and old_span.kind == "bare"
            and old_span.start == span.start
            and span.kind.startswith(old_span.payload.lower())
        )
        interactive_transition = len(delta.inserted) == 1 or selected_namespace
        if span.kind == "file":
            picker_available = self.file_picker_available_fn()
            if picker_available and self.app is not None and interactive_transition:
                self._schedule_file_picker(buffer)
            elif not picker_available:
                self._refresh_file_completions(buffer)
            return
        if span.kind in {"mcp", "skill"}:
            if not interactive_transition:
                return
            # Completion selection updates the document before prompt-toolkit publishes the newly
            # selected row in complete_state. Always defer this namespace transition so it observes
            # the settled state; continued typing cancels and reschedules it with the latest query.
            self._schedule_name_completions(buffer)
            return
        state = buffer.complete_state
        if state is not None:
            # No row is selected, so preserving the current document while dropping the stale
            # completion state is safe. cancel_completion() would restore its older document.
            buffer.complete_state = None
        buffer.start_completion(select_first=False)

    def _cancel_mention_transition(self) -> None:
        timer = self._mention_transition_timer
        self._mention_transition_timer = None
        if timer is not None:
            timer.cancel()

    def _schedule_mention_transition(self, buffer: Buffer, callback: Callable[[Buffer], None], *, delay: float | None = None) -> None:
        app = self.app
        if app is None or not app.is_running:
            return
        text, cursor = buffer.text, buffer.cursor_position

        def transition() -> None:
            self._mention_transition_timer = None
            if self.input_mode not in {"chat", "running"} or buffer.text != text or buffer.cursor_position != cursor:
                return
            callback(buffer)

        loop = app.loop
        assert loop is not None
        self._mention_transition_timer = loop.call_later(self.MENTION_TRANSITION_DELAY if delay is None else delay, transition)

    def _schedule_name_completions(self, buffer: Buffer) -> None:
        def show(current: Buffer) -> None:
            span = active_mention(current.document.text_before_cursor)
            if span is None or span.kind not in {"mcp", "skill"}:
                return
            # Keep the selected namespace as real input instead of restoring the first-stage
            # document, then replace that menu with the namespace's own candidates.
            current.complete_state = None
            current.start_completion(select_first=False)

        self._schedule_mention_transition(buffer, show)

    def _schedule_file_picker(self, buffer: Buffer) -> None:
        def open_picker(current: Buffer) -> None:
            target = self._file_mention_at_cursor(current)
            if target is None or self._file_picker_active:
                return
            # A namespace chosen from the first-stage menu is still a preview until its completion
            # state is discarded. Commit it before fzf takes over so Escape returns to @file:.
            current.complete_state = None
            self._start_file_picker(current, *target)

        self._schedule_mention_transition(buffer, open_picker, delay=0)

    @staticmethod
    def _file_mention_at_cursor(buffer: Buffer) -> tuple[MentionSpan, int] | None:
        cursor = buffer.cursor_position
        partial = active_mention(buffer.text[:cursor])
        if partial is None or partial.kind != "file":
            return None
        full = next((span for span in scan_mentions(buffer.text) if span.kind == "file" and span.start == partial.start and span.end >= cursor), None)
        return partial, full.end if full is not None else cursor

    def _refresh_file_completions(self, buffer: Buffer) -> None:
        text, cursor = buffer.text, buffer.cursor_position
        target = self._file_mention_at_cursor(buffer)
        if target is None:
            return
        query = target[0].payload

        def ready() -> None:
            if self.app is None:
                return

            def show() -> None:
                if buffer.text == text and buffer.cursor_position == cursor and self._file_mention_at_cursor(buffer) is not None:
                    if buffer.complete_state is not None:
                        # Preserve a namespace or path currently previewed in the input. Restoring
                        # the completion state's original document would jump back to the parent
                        # menu just as the file candidates become ready.
                        buffer.complete_state = None
                    buffer.start_completion(select_first=False)
                    self.invalidate()

            self._schedule(show)

        self.file_complete_fn(query, ready)

    def _start_file_picker(self, buffer: Buffer, span: MentionSpan, end: int) -> None:
        self._cancel_mention_transition()
        if self._file_picker_active or self.app is None:
            return
        self._file_picker_active = True
        document = buffer.document

        async def pick() -> None:
            try:
                # The app is suspended while fzf owns the terminal. Launch it inline so there is no
                # executor gap between erasing the prompt and the child drawing its first frame.
                result = await run_in_terminal(lambda: self.file_picker_fn(span.payload))
                if result.unavailable:
                    self._refresh_file_completions(buffer)
                    return
                if result.selection is None or buffer.text != document.text or buffer.cursor_position != document.cursor_position:
                    return
                mention = encode_file_mention(result.selection)
                text = document.text[: span.start] + mention + document.text[end:]
                position = span.start + len(mention)
                self._reset_input(UserInput(text, self.input_images), cursor_position=position)
                self.invalidate()
            finally:
                self._file_picker_active = False

        self.app.create_background_task(pick())

    def _sync_input_images(self, old: str, delta: _EditDelta) -> None:
        """Drop image markers that an edit removed from the input text."""
        removed = delta.removed.count(IMAGE_MARKER)
        if not removed:
            return
        first = old[: delta.prefix].count(IMAGE_MARKER)
        self.input_images = self.input_images[:first] + self.input_images[first + removed :]

    async def show_modal_async(
        self,
        fragments_fn: Callable[[], StyleAndTextTuples],
        key_fn: Callable[[str, str], Any],
        *,
        exclusive: bool = False,
    ) -> Any:
        """Show a modal inside this Application and await its result on the loop that runs it.

        The modal's key handling already runs on this loop, so awaiting here is what keeps a
        runtime-owned command or viewer task from parking the loop on a threading primitive."""

        app = self.app
        if app is None or not app.is_running or self.modal_window is None:
            return None
        await self._modal_idle_event().wait()
        modal = TuiModal(fragments_fn, key_fn, exclusive=exclusive)
        modal.future = asyncio.get_running_loop().create_future()
        self._activate_modal(app, modal, exclusive=exclusive)
        return await modal.future

    def _activate_modal(self, app: Application, modal: TuiModal, *, exclusive: bool) -> None:
        """Make `modal` the visible one. Runs on the loop, whichever entry point opened it."""
        self.modal = modal
        self._modal_idle_event().clear()
        target = self.exclusive_modal_window if exclusive else self.modal_window
        assert target is not None
        app.layout.focus(target)
        if exclusive:
            self._use_alternate_screen(True)
        app.invalidate()

    def show_modal(
        self,
        fragments_fn: Callable[[], StyleAndTextTuples],
        key_fn: Callable[[str, str], Any],
        *,
        exclusive: bool = False,
    ) -> Any:
        """Show a modal inside this Application and block the calling worker until it closes."""
        with self.modal_lock:
            app = self.app
            if app is None or not app.is_running or self.modal_window is None:
                return None
            modal = TuiModal(fragments_fn, key_fn, exclusive=exclusive)

            self._schedule(lambda: self._activate_modal(app, modal, exclusive=exclusive))
            modal.done.wait()
            return modal.result

    def close_modal(self, result: Any = None) -> None:
        modal = self.modal
        if modal is None:
            return
        modal.result = result
        self.modal = None
        if self._modal_idle is not None:
            self._modal_idle.set()
        if self.app is not None and self.input_window is not None:
            self.app.layout.focus(self.input_window)
        if modal.exclusive:
            self._use_alternate_screen(False)
        self.invalidate()
        if modal.future is not None and not modal.future.done():
            modal.future.set_result(result)
        modal.done.set()

    def _use_alternate_screen(self, enabled: bool) -> None:
        """Move the persistent app between the primary and alternate screen.

        Exclusive modals (the /diff viewer) fill the whole pane. Painted on the primary screen they
        push the transcript above them off the top into scrollback, and closing the modal only
        shrinks the app region back — the transcript never comes back down. Give them the alternate
        screen instead, so the terminal restores the transcript on exit the way `less` does.
        """
        app = self.app
        if app is None or app.renderer.full_screen == enabled:
            return
        # Erase the region we own on the screen we are leaving, so no stale footer is left behind
        # (on the way back this also drops us out of the alternate screen).
        app.renderer.erase()
        app.renderer.full_screen = enabled
        app._request_absolute_cursor_position()

    @staticmethod
    def alternate_screen_available() -> bool:
        """Whether an exclusive modal can preserve the primary screen in this terminal."""
        if not os.environ.get("TMUX"):
            return True
        # alternate-screen is a window option, so show-options reports it only when a window
        # overrides it and stays silent for the usual global `set -wg` form. Formatting the
        # resolved value instead answers for both, as 1 (enabled) or 0 (disabled).
        command = ["tmux", "display-message", "-p"]
        if pane := os.environ.get("TMUX_PANE"):
            command.extend(["-t", pane])
        command.append("#{alternate-screen}")
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=1, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return True
        return result.returncode != 0 or result.stdout.strip() != "0"

    def modal_fragments(self) -> StyleAndTextTuples:
        return self.modal.fragments_fn() if self.modal is not None else []

    def dispatch_modal_key(self, key: str, data: str = "") -> None:
        if self.modal is None:
            return
        result = self.modal.key_fn(key, data)
        if result is not TUI_MODAL_PENDING:
            self.close_modal(result)
        else:
            self.invalidate()

    def status_fragments(self) -> StyleAndTextTuples:
        if self.input_mode == "dispatch" and self.input_prompt:
            return [("ansibrightblack", self.input_prompt)]
        if self.input_mode == "approval" and self.input_prompt:
            frame = "|/-\\"[int(time.monotonic() / 0.2) % 4]
            connector = LogBlock.prefix(2, LogEdge.CONTINUE)
            prompt = (
                [("ansibrightblack", connector), ("class:approval", self.input_prompt[len(connector) :])]
                if self.input_prompt.startswith(connector)
                else [("class:approval", self.input_prompt)]
            )
            return [*prompt, ("class:approval.wait", frame + " ")]
        return [("class:prompt", self.input_prompt)]

    def full_input_prompt(self) -> str:
        """The prompt as the caller passed it, rejoined from the rows it was split across."""
        return "\n".join([*self._input_prompt_above, self.input_prompt])

    def input_prompt_above_fragments(self) -> StyleAndTextTuples:
        """The prompt's leading lines, rendered as their own rows above the input. Unlike the input
        row's prefix this is a FormattedTextControl, which does split its text on "\n"."""
        if not self._input_prompt_above:
            return []
        style = "class:approval" if self.input_mode == "approval" else "class:prompt"
        return [(style, "\n".join(self._input_prompt_above))]

    def input_error_fragments(self) -> StyleAndTextTuples:
        error = self.input_error
        return [("class:input.error", f"Error: {error}")] if error else []

    @staticmethod
    def complete_input(buffer: Buffer, *, reverse: bool = False) -> None:
        if buffer.complete_state is not None:
            buffer.complete_previous() if reverse else buffer.complete_next()
            return
        if buffer.completer is None:
            return
        event = CompleteEvent(completion_requested=True)
        completions = list(buffer.completer.get_completions(buffer.document, event))
        if len(completions) == 1:
            buffer.apply_completion(completions[0])
        elif completions:
            if reverse:
                buffer.start_completion(select_last=True)
            else:
                buffer.start_completion(select_first=False)

    def _status_bar_window(self, *, dont_extend_height: bool) -> Window:
        return Window(
            FormattedTextControl(self.status_fragments_fn, style="class:bottom-toolbar.text"),
            style="class:bottom-toolbar",
            height=1,
            dont_extend_height=dont_extend_height,
        )

    def build_layout(self) -> Layout:
        input_processors: list[Processor] = [
            HighlightIncrementalSearchProcessor(),
            ImageLabelProcessor(lambda: self.input_images),
            BeforeInput(self.status_fragments),
            CallbackPlaceholder(self.placeholder_text),
        ]
        self.input_window = Window(
            BufferControl(
                buffer=self.input_buffer,
                input_processors=input_processors,
                search_buffer_control=self.search_toolbar.control,
                preview_search=True,
            ),
            height=Dimension(min=1),
            dont_extend_height=True,
            wrap_lines=True,
            style=UiPrinter.user_log_style(),
        )
        completion_space = ConditionalContainer(Window(height=12, dont_extend_height=True), filter=has_completions & ~is_done)
        input_error = ConditionalContainer(
            Window(FormattedTextControl(self.input_error_fragments), dont_extend_height=True, wrap_lines=True),
            filter=Condition(lambda: bool(self.input_error_fragments())),
        )
        approval_form = ConditionalContainer(
            Window(FormattedTextControl(self.approval_form_fragments), dont_extend_height=True, wrap_lines=True),
            filter=Condition(lambda: bool(self._approval_actions) and self.input_mode == "approval"),
        )
        self.activity_window = Window(FormattedTextControl(self.activity_fragments_fn), dont_extend_height=True, wrap_lines=True)
        running = Condition(lambda: self.input_mode == "running")
        activity = ConditionalContainer(
            self.activity_window,
            filter=running,
        )
        running_gap_above = ConditionalContainer(
            Window(height=1, dont_extend_height=True),
            filter=running,
        )
        running_gap_below = ConditionalContainer(
            Window(height=1, dont_extend_height=True),
            filter=running,
        )
        prompt_above = ConditionalContainer(
            Window(FormattedTextControl(self.input_prompt_above_fragments), wrap_lines=True, dont_extend_height=True),
            filter=Condition(lambda: bool(self._input_prompt_above)),
        )
        self.modal_window = Window(FormattedTextControl(self.modal_fragments, focusable=True), wrap_lines=False, dont_extend_height=True)
        modal_active = Condition(lambda: self.modal is not None)
        exclusive_active = Condition(lambda: self.modal is not None and self.modal.exclusive)
        idle = Condition(lambda: self.input_mode == "chat")
        has_quick_hints = idle & Condition(lambda: bool(self.quick_hints()))
        quick_hints_gap = ConditionalContainer(Window(height=1, dont_extend_height=True), filter=has_quick_hints)
        quick_hints_row = ConditionalContainer(
            Window(FormattedTextControl(self.quick_hint_fragments), wrap_lines=True, dont_extend_height=True),
            filter=has_quick_hints,
        )
        normal_region = ConditionalContainer(
            HSplit(
                [
                    running_gap_above,
                    activity,
                    running_gap_below,
                    prompt_above,
                    input_error,
                    approval_form,
                    self.input_window,
                    quick_hints_gap,
                    quick_hints_row,
                    completion_space,
                    self.search_toolbar,
                    Window(height=1, dont_extend_height=True),
                ]
            ),
            filter=~modal_active,
        )
        modal_region = ConditionalContainer(
            HSplit([self.modal_window, Window(height=1, dont_extend_height=True)]),
            filter=modal_active & ~exclusive_active,
        )
        self.status_window = self._status_bar_window(dont_extend_height=True)
        # Keep the idle prompt padded from prior output, but start transient running/approval
        # regions at row zero. Otherwise patch_stdout can commit that leading empty row between a
        # tool's approval header and its eventual result when it suspends and redraws the app.
        content = HSplit(
            [
                ConditionalContainer(Window(height=1, dont_extend_height=True), filter=idle),
                modal_region,
                normal_region,
                self.status_window,
            ]
        )
        self.exclusive_modal_window = Window(FormattedTextControl(self.modal_fragments, focusable=True), wrap_lines=False)
        exclusive_status = self._status_bar_window(dont_extend_height=False)
        root = FloatContainer(
            HSplit(
                [
                    ConditionalContainer(content, filter=~exclusive_active),
                    ConditionalContainer(HSplit([self.exclusive_modal_window, exclusive_status]), filter=exclusive_active),
                ]
            ),
            [Float(CompletionsMenu(max_height=12, scroll_offset=1), xcursor=True, ycursor=True, attach_to_window=self.input_window, transparent=True)],
        )
        return Layout(root, focused_element=self.input_window)

    def make_bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        modal = Condition(lambda: self.modal is not None)
        running = Condition(lambda: self.input_mode == "running" and self.modal is None)

        for key in self.MODAL_KEYS:
            bindings.add(key, filter=modal, eager=True)(lambda event, key=key: self.dispatch_modal_key(key, event.data))
        for number in range(1, 10):
            bindings.add(str(number), filter=modal, eager=True)(lambda event, number=number: self.dispatch_modal_key(str(number), event.data))
        bindings.add(Keys.Any, filter=modal)(lambda event: self.dispatch_modal_key("any", event.data))

        def enter(event):
            # Enter ends a Ctrl-R history search by placing the match into the input box without
            # submitting it, so the text can be reviewed or edited first; a second Enter sends.
            # Without this, this eager binding wins over prompt_toolkit's search-mode Enter and
            # runs on the search field's own buffer, which has no accept handler, so the key
            # press would do nothing.
            if is_searching():
                pt_search.accept_search()
                return
            # Enter on a focused quick-hint chip picks it into the input and returns focus to the
            # input line, so a second Enter sends.
            buffer = event.current_buffer
            if self._pick_quick_hint(buffer):
                return
            # Enter with a completion row highlighted (Tab previews it) commits that row into the
            # input instead of sending the message: the menu closes, the prompt stays open, and a
            # second Enter sends. A menu opened by typing alone has no highlighted row, so Enter
            # still sends there, exactly as it always did.
            state = buffer.complete_state
            if state is not None and state.current_completion is not None:
                buffer.apply_completion(state.current_completion)
                return
            buffer.validate_and_handle()

        bindings.add("enter", filter=~modal, eager=True)(enter)
        bindings.add("escape", "enter", filter=~modal, eager=True)(lambda event: event.current_buffer.insert_text("\n"))
        for key, reverse in (("tab", False), ("s-tab", True)):
            bindings.add(key, filter=~modal)(lambda event, reverse=reverse: self.tab_or_complete(event.current_buffer, reverse=reverse))

        # The approval action row is live only while the line is empty: the moment a reason is being
        # typed, Tab completes and the arrows move the cursor, exactly as everywhere else. That is
        # the whole reason actions are selected rather than bound to letters -- no key a reason
        # might start with is ever spent on a shortcut.
        picking = Condition(lambda: self.input_mode == "approval" and bool(self._approval_actions) and not self.input_buffer.text)
        for key, delta in (("tab", 1), ("right", 1), ("s-tab", -1), ("left", -1)):
            bindings.add(key, filter=~modal & picking, eager=True)(lambda _, delta=delta: self.move_approval_focus(delta))

        # Not eager: an arrow key is an escape sequence, so Escape has to stay ambiguous long enough
        # for the parser to see whether more bytes follow. With a reason typed it clears back to the
        # action row; on an empty line there is nothing to take back, so it cancels the approval,
        # which `confirm` reads as a refusal with no reason.
        def escape(event):  # pragma: no cover — interactive path
            if self.input_buffer.text:
                self.input_buffer.reset(Document(""))
            elif self._input_pending is not None:
                self.resolve_input(None)

        bindings.add("escape", filter=~modal & Condition(lambda: self.input_mode == "approval" and bool(self._approval_actions)))(escape)

        def paste(event):
            event.current_buffer.insert_text(event.data.replace("\r\n", "\n").replace("\r", "\n"))
            if self.input_mode in {"chat", "running"}:
                self._recognize_input()

        bindings.add(Keys.BracketedPaste, filter=~modal)(paste)

        def history_search(event):
            direction = pt_search.SearchDirection.BACKWARD
            if event.app.layout.current_control is self.search_toolbar.control:
                pt_search.do_incremental_search(direction, count=event.arg)
            else:
                # Snapshot the input before starting a new search so aborting it (Ctrl-C / Ctrl-U
                # while searching) can restore exactly what was there, including any draft.
                self._search_start_text = self.input_buffer.text
                pt_search.start_search(direction=direction)

        bindings.add("c-r", filter=~modal, eager=True)(history_search)
        bindings.add("c-o", filter=~modal, eager=True)(lambda _: self.on_expand_output())

        # Ctrl-P mirrors Up here: readline treats them as synonyms, and both recall the latest
        # queued follow-up, or walk history when none is queued, while a turn is working.
        def recall(event):
            text = self.on_recall()
            if text:
                self._reset_input(text, cursor_position=len(text))
            else:
                self._load_buffer_history_now()
                event.current_buffer.auto_up(count=event.arg)

        bindings.add("c-p", filter=running, eager=True)(recall)
        bindings.add("up", filter=running, eager=True)(recall)

        # Ctrl-X Ctrl-E (readline `edit-and-execute-command`) and Ctrl-G hand the current input to
        # $VISUAL/$EDITOR (fallback vim) for editing, matching Claude Code's editor bindings. The
        # `c-x c-e` chord means a lone Ctrl-X waits for the second key instead of firing eagerly.
        # In-flight resend has no key; it is the `/resend` command typed in the running input.
        edits_input = Condition(lambda: self.input_mode in {"chat", "running", "approval"})

        def edit_in_editor(_):  # pragma: no cover — interactive path
            self.edit_input_in_editor()

        bindings.add("c-g", filter=~modal & edits_input)(edit_in_editor)
        bindings.add("c-x", "c-e", filter=~modal & edits_input)(edit_in_editor)

        def ctrl_c(event):  # pragma: no cover — interactive path
            # Never quit on Ctrl-C. Instead:
            #   * approval mode → cancel this specific prompt (None, never "" which confirm() reads
            #     as the default approve, and never placeholder text the model would see).
            #   * idle chat → clear the current input silently.
            #   * agent running → discard a draft, or interrupt the turn when the input is empty.
            # Exit remains reserved for Ctrl-D on an empty chat input or the /exit slash command.
            if self.modal is not None:
                result = self.modal.key_fn("c-c", event.data)
                self.close_modal(None if result is TUI_MODAL_PENDING else result)
                return
            if is_searching():
                # While a Ctrl-R search is in flight, Ctrl-C aborts the search and restores the
                # pre-search input (readline behavior); it must not clear the matched entry.
                self._abort_history_search()
                return
            if self.input_mode == "approval" and self._input_pending is not None:
                self.resolve_input(None)
                return
            if self.input_mode == "chat":
                if self.input_buffer.text:
                    self.input_buffer.reset(Document(""))
                return
            if self.input_mode in {"dispatch", "running"}:
                # A draft absorbs the first press, the way it already does at the idle prompt. The
                # queue hint only renders on an empty buffer, so "Ctrl-C interrupts" is shown
                # exactly when the next press interrupts.
                if self.input_buffer.text:
                    self.input_buffer.reset(Document(""))
                    return
                self.on_interrupt()

        bindings.add("<sigint>", eager=True)(ctrl_c)
        bindings.add("c-c", eager=True)(ctrl_c)

        def clear_input(_):  # pragma: no cover — interactive path
            # The readline convention for discarding the line, and the one key that means the same
            # thing in every editor here. Ctrl-C also clears, but while the agent runs it spends a
            # press that would otherwise interrupt; this one never competes with stopping the turn.
            if is_searching():
                # Like Ctrl-C, abort an in-flight Ctrl-R search and restore the pre-search input
                # instead of clearing the matched entry.
                self._abort_history_search()
                return
            self.input_buffer.reset(Document(""))

        bindings.add("c-u", filter=~modal & edits_input, eager=True)(clear_input)

        def ctrl_d(event):  # pragma: no cover — interactive path
            if self.input_mode == "approval" and self._input_pending is not None:
                # EOF on an empty approval line cancels rather than submitting "", which confirm()
                # would read as the default approve -- the same trap Ctrl-C used to fall into.
                self.resolve_input(self.input_buffer.text or None)
            elif self.input_buffer.text and self.input_mode in {"chat", "running"}:
                self.input_buffer.delete()
            elif self.input_mode == "chat":
                self.on_exit_request()
                event.app.exit()

        bindings.add("c-d", filter=~modal, eager=True)(ctrl_d)

        def force_exit(event):  # pragma: no cover — interactive emergency path
            self.on_force_exit()
            event.app.exit()

        bindings.add(Keys.ControlBackslash, eager=True)(force_exit)

        return bindings

    @staticmethod
    def editor_command() -> list[str]:
        """The editor to launch for Ctrl-X Ctrl-E / Ctrl-G: $VISUAL, then $EDITOR, then vim."""
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vim"
        return shlex.split(editor)

    # Scissors marker (the git "scissors" convention) separating the editable draft from the
    # read-only reference context that Ctrl-X Ctrl-E appends below it. Everything from this
    # line down is stripped before the message is sent.
    EDITOR_CONTEXT_MARKER = "# ------------------------ >8 ------------------------"

    @classmethod
    def _compose_editor_text(cls, draft: str, context: str) -> tuple[str, str]:
        """Text handed to the external editor: the draft, then (when context is available) a
        scissors line and the agent's recent reply for reference, since the full-screen editor
        hides the scrollback the reply is printed into. Returns the composed text together with
        the unique marker that separates the draft from the reference context (empty when no
        context was appended), so stripping later removes only the context this call added and
        never a scissors line the user typed themselves."""
        context = context.strip()
        if not context:
            return draft, ""
        marker = f"{cls.EDITOR_CONTEXT_MARKER} ({uuid.uuid4().hex[:12]})"
        composed = (
            draft
            + "\n\n"
            + marker
            + "\n"
            + "# Reference only: everything below the scissors line is stripped before your\n"
            + "# message is sent. The agent's most recent reply follows for reference.\n"
            + "\n"
            + context
        )
        return composed, marker

    @classmethod
    def _strip_editor_context(cls, text: str, marker: str) -> str:
        """Drop the reference context this composition added (its unique scissors line and
        everything below it). When no marker was appended there is nothing to strip, so a scissors
        line the user typed themselves is left untouched."""
        if marker:
            text = text.split(marker, 1)[0]
        return text.rstrip("\n")

    def _edit_text_in_editor(self, text: str) -> str | None:
        """Run the editor on `text` via a temp file and return the edited content, or None if the
        editor could not launch or exited non-zero. Runs off the event loop, inside run_in_terminal."""
        fd, path = tempfile.mkstemp(prefix="wizolt-input-", suffix=".md")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            try:
                completed = subprocess.run([*self.editor_command(), path], check=False)
            except OSError:
                return None
            if completed.returncode != 0:
                return None
            with open(path, encoding="utf-8") as handle:
                return handle.read()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    async def _run_input_editor(self) -> None:
        # `run_in_terminal` suspends the app and restores it afterward (the same primitive
        # prompt_toolkit uses for its own editor support), so a full-screen editor gets a clean
        # terminal. A non-zero exit or a launch failure leaves the input untouched. The editor
        # also receives the agent's recent reply below a scissors line for reference (the
        # full-screen editor hides the scrollback); that context is stripped back out on return.
        original = UserInput(self.input_buffer.text, self.input_images).original_text()
        composed, marker = self._compose_editor_text(original, self.editor_context_fn())
        edited = await run_in_terminal(lambda: self._edit_text_in_editor(composed), in_executor=True)
        if edited is None:
            return
        edited = self._strip_editor_context(edited, marker)
        if edited != original:
            self._reset_input(edited, cursor_position=len(edited))
            if self.input_mode in {"chat", "running"}:
                self._recognize_input()
            self.invalidate()

    def edit_input_in_editor(self) -> None:
        """Ctrl-X Ctrl-E / Ctrl-G: edit the current input in an external editor, then load the result back."""
        if self.app is not None:
            self.app.create_background_task(self._run_input_editor())

    async def animate(self) -> None:
        """Invalidate at the animation frame rate while the running region is on screen.

        prompt-toolkit captures `refresh_interval` when it starts its own refresh task, so the rate
        cannot be raised for the running turn alone. This second ticker owns the animated mode and
        stops asking for frames as soon as the divider is gone, leaving the idle screen slow.
        """
        while True:
            await asyncio.sleep(self.ANIMATION_INTERVAL)
            if self.input_mode == "running":
                self.invalidate()

    def _install_resize_reanchor(self, app: Application) -> None:
        """Handle terminal resizes by re-anchoring the app at the pane bottom.

        prompt-toolkit's resize path erases from where it last drew and then trusts the cursor
        position report (CPR). A multiplexer reflow (tmux zoom/unzoom) moves the already drawn app
        before the resize is even detected, so that erase misses the moved copy and the CPR answer
        carries the drifted row: the app creeps toward the top of the pane and every cycle leaves
        a stale copy behind. Erase from the cursor the terminal actually reports, then park the
        cursor where an app of the last rendered height belongs and run the stock CPR-and-redraw
        sequence from there, so the reported position describes the app instead of the drift.
        """
        vanilla_resize = app._on_resize

        def on_resize() -> None:
            renderer = app.renderer
            last_screen = renderer.last_rendered_screen
            # 0-based row of the app's top when it sits flush with the pane bottom.
            anchored_row = renderer.output.get_size().rows - last_screen.height if last_screen is not None and not renderer.full_screen else -1
            if anchored_row < 0:
                # Nothing rendered yet, or a full-screen app: the stock path is already right.
                vanilla_resize()
                return
            # Erase starting at the terminal's actual cursor — after a reflow only the terminal
            # knows where the moved app is, and erase_down() from there clears its every row.
            renderer.erase(leave_alternate_screen=False)
            renderer.output.cursor_goto(anchored_row + 1, 1)
            renderer.reset(leave_alternate_screen=False)
            app._request_absolute_cursor_position()
            app._redraw()

        app._on_resize = on_resize

    def run(self, style: Style | None = None) -> None:  # pragma: no cover — interactive
        """Standalone entry point: own a loop for this application alone.

        The interactive runtime does not use this. It owns one loop for the application, the active
        turn, and its own background work, and awaits `run_async` on it."""

        fail_if_running_loop("use await TuiApp.run_async(...)")
        return asyncio.run(self.run_async(style))

    async def run_async(self, style: Style | None = None) -> None:  # pragma: no cover — interactive
        app = self._build_application(style)
        self.app = app
        self.ready.clear()

        def start() -> None:
            # pre_run already runs inside the application's loop, and the task it starts is
            # cancelled with the rest of the application's background tasks on exit.
            app.create_background_task(self.animate())
            self.ready.set()

        try:
            with patch_stdout():
                await app.run_async(pre_run=start)
        finally:
            self._after_run()

    def _after_run(self) -> None:
        self.ready.set()
        # Flush anything still queued in the scrollback batching window before the terminal is
        # handed back; a timer fired inside the app loop would never get to run again.
        self.on_app_stop()
        self.app = None
        # Anything still parked on an input request unblocks as a cancel: a pending approval must
        # not be granted by the app shutting down.
        self.resolve_input(None)
        if self.modal is not None:
            self.close_modal(None)

    def _build_application(self, style: Style | None = None) -> Application:  # pragma: no cover — interactive
        app = Application(
            layout=self.build_layout(),
            key_bindings=self.make_bindings(),
            full_screen=False,
            mouse_support=False,
            refresh_interval=self.IDLE_REFRESH_INTERVAL,
            style=style,
            erase_when_done=True,
        )
        # A persistent primary-screen renderer needs CPR after a terminal resize; otherwise its
        # stale cursor coordinates can leave the transient footer in tmux scrollback. Keep the
        # legacy behavior of silently degrading on terminals that do not answer the probe.
        app.renderer.cpr_not_supported_callback = lambda: None
        self._install_resize_reanchor(app)
        return app
