"""minacode prompt-toolkit application and interactive view state."""

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
from enum import Enum, auto
from typing import Any, ClassVar, TypeVar

from prompt_toolkit import search as pt_search
from prompt_toolkit.application import Application, create_app_session, run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions, is_done
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, to_formatted_text
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
from rich.console import Console
from rich.markdown import Markdown

from minacode.base import (
    SELECTION_BACK,
    SELECTION_FREE_TEXT,
    LogBlock,
    LogEdge,
    MinacodeError,
)
from minacode.image import IMAGE_MARKER, ImageInputs, ImageRef, UserInput
from minacode.render import UiPrinter
from minacode.tools.ask import AskSpec

TUI_MODAL_PENDING = object()
ViewLine = TypeVar("ViewLine")


@dataclass
class TuiModal:
    fragments_fn: Callable[[], StyleAndTextTuples]
    key_fn: Callable[[str, str], Any]
    exclusive: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None


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
    # about one cell per frame, so this rate is what `CommandLoop.QUEUE_SWEEP_CELLS_PER_SEC` follows.
    ANIMATION_INTERVAL: ClassVar[float] = 1 / 30
    # Idle refresh: no animation runs on the idle screen, only the 0.2s index and MCP spinners.
    IDLE_REFRESH_INTERVAL: ClassVar[float] = 0.2

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
        editor_context_fn: Callable[[], str] | None = None,
        images: ImageInputs | None = None,
        image_cwd: str = "",
        history: FileHistory | None = None,
        completer: Completer | None = None,
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
        self.activity_fragments_fn: Callable[[], StyleAndTextTuples] = activity_fragments_fn or list
        self.input_hint_fn = input_hint_fn or (lambda: "")
        self.quick_hints_fn: Callable[[], tuple[str, ...]] = quick_hints_fn or (lambda: ())
        self.editor_context_fn = editor_context_fn or (lambda: "")
        self.images = images if images is not None else ImageInputs(cwd=image_cwd)
        self.input_images: tuple[ImageRef, ...] = ()
        self._last_input_text = ""
        self._changing_input = False
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
        self.input_prompt = UiPrinter.PROMPT_PREFIX
        self._input_pending: threading.Event | None = None
        self._input_result: str = ""
        self.status_label: str = ""
        self.modal: TuiModal | None = None
        self.modal_lock = threading.Lock()
        self.input_window: Window | None = None
        self.activity_window: Window | None = None
        self.modal_window: Window | None = None
        self.exclusive_modal_window: Window | None = None
        self.status_window: Window | None = None

    def request_input(self, prompt: str) -> str:
        """Called from the agent thread to get a line of user input inline (approval prompts,
        Ask tool, etc.). Blocks until the TUI thread's widget submits. If the app has exited
        before submission, returns "" so the caller unwinds cleanly."""
        # A tool approval must not replace an already-visible selector. Wait for that selector to
        # close, then reuse the shared input row.
        with self.modal_lock:
            pass
        event = threading.Event()
        previous_mode, previous_prompt = self.input_mode, self.input_prompt
        previous_document: Document | None = None
        previous_images = self.input_images
        self._input_pending = event
        self._input_result = ""

        def switch(document: Document, mode: str, prompt_text: str, done: threading.Event) -> None:
            nonlocal previous_document
            if previous_document is None:
                previous_document = self.input_buffer.document
            images = previous_images if mode == previous_mode else ()
            self._reset_input(UserInput(document.text, images), cursor_position=document.cursor_position)
            self._set_mode(mode, prompt_text)
            done.set()

        switched = threading.Event()
        self._schedule(switch, Document(""), "approval", prompt, switched)
        switched.wait()
        try:
            event.wait()
        finally:
            self._input_pending = None
            restored = threading.Event()
            self._schedule(switch, previous_document or Document(""), previous_mode, previous_prompt, restored)
            restored.wait()
        return self._input_result

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
        self.input_prompt = prompt
        self.quick_hint_focus = -1
        if mode not in {"chat", "running"}:
            self.input_error = ""
        self.invalidate()

    def invalidate(self) -> None:
        if self.app is not None:
            self.app.invalidate()

    def invalidate_frame(self) -> None:
        """Ask for a redraw from a source that fires far faster than the eye needs.

        Model output arrives token by token. While the running region is on screen the animation
        ticker already redraws at the frame rate, so redrawing per token only makes the cadence
        swing with the model's pace; anywhere else there is no ticker, so redraw normally.
        """
        if self.input_mode != "running":
            self.invalidate()

    def write_to_scrollback(self, callback: Callable[[], None]) -> None:
        """Print above the live application and wait until the terminal has accepted it.

        `run_in_terminal` owns the erase/write/redraw sequence, while `create_app_session` routes
        nested prompt-toolkit printers to this application's output. Waiting is intentional: the
        caller must not start tool output until the promoted response is permanent scrollback.
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
        if self.input_mode == "chat" and not buffer.text.strip() and 0 <= self.quick_hint_focus < len(self.quick_hints()):
            self._reset_input(self.quick_hints()[self.quick_hint_focus])
            self.quick_hint_focus = -1
        text = buffer.text
        if self.input_mode == "approval" and self._input_pending is not None:
            self._input_result = text
            self._input_pending.set()
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
        except MinacodeError as error:
            self.input_error = str(error)
            self.invalidate()
            return None
        self.input_error = ""
        return value

    def _append_history(self, value: UserInput) -> None:
        if self.history is not None:
            self.history.append_string(value.original_text())

    def _recognize_input(self) -> UserInput:
        value = self.images.recognize(self.input_buffer.text, self.input_images)
        if str(value) != self.input_buffer.text or value.images != self.input_images:
            self._reset_input(value, cursor_position=len(value))
        return value

    def _reset_input(self, value: str | UserInput, *, cursor_position: int | None = None) -> None:
        user_input = value if isinstance(value, UserInput) else UserInput(value)
        self._changing_input = True
        try:
            self.input_images = user_input.images
            self._last_input_text = str(user_input)
            position = len(user_input) if cursor_position is None else cursor_position
            self.input_buffer.reset(Document(str(user_input), cursor_position=position))
        finally:
            self._changing_input = False

    def quick_hints(self) -> tuple[str, ...]:
        return self.quick_hints_fn()

    def quick_hint_fragments(self) -> StyleAndTextTuples:
        hints = self.quick_hints()
        if not hints:
            return []
        parts: StyleAndTextTuples = []
        for index, hint in enumerate(hints):
            if index:
                parts.append(("class:quickhint.sep", " │ "))
            style = "class:quickhint.focused" if index == self.quick_hint_focus else "class:quickhint"
            parts.append((style, f" {hint} "))
        return parts

    def cycle_quick_hint_focus(self, reverse: bool = False) -> None:
        count = len(self.quick_hints())
        if not count:
            return
        focus = self.quick_hint_focus + (-1 if reverse else 1)
        if focus >= count or focus < -1:
            focus = count - 1 if reverse else -1
        self.quick_hint_focus = focus
        self.invalidate()

    def tab_or_complete(self, buffer: Buffer, *, reverse: bool) -> None:
        # On an empty idle prompt Tab/Shift-Tab cycle the offered quick inputs; anywhere else they complete.
        if self.input_mode == "chat" and not buffer.text and buffer.complete_state is None and self.quick_hints():
            self.cycle_quick_hint_focus(reverse=reverse)
            return
        self.complete_input(buffer, reverse=reverse)

    def placeholder_text(self) -> str:
        if self.input_mode == "chat" and self.quick_hints():
            return "" if self.quick_hint_focus >= 0 else "Tab cycles suggestions \u00b7 Enter submits"
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
        self.input_error = ""
        delta = _edit_delta(old, text)
        self._sync_input_images(old, delta)
        self._last_input_text = text
        if delta.inserted and delta.inserted[-1].isspace() and self.input_mode in {"chat", "running"}:
            self._recognize_input()

    def _sync_input_images(self, old: str, delta: _EditDelta) -> None:
        """Drop image markers that an edit removed from the input text."""
        removed = delta.removed.count(IMAGE_MARKER)
        if not removed:
            return
        first = old[: delta.prefix].count(IMAGE_MARKER)
        self.input_images = self.input_images[:first] + self.input_images[first + removed :]

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

            def activate() -> None:
                self.modal = modal
                target = self.exclusive_modal_window if exclusive else self.modal_window
                assert target is not None
                app.layout.focus(target)
                if exclusive:
                    self._use_alternate_screen(True)
                app.invalidate()

            self._schedule(activate)
            modal.done.wait()
            return modal.result

    def close_modal(self, result: Any = None) -> None:
        modal = self.modal
        if modal is None:
            return
        modal.result = result
        self.modal = None
        if self.app is not None and self.input_window is not None:
            self.app.layout.focus(self.input_window)
        if modal.exclusive:
            self._use_alternate_screen(False)
        self.invalidate()
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

    def input_error_fragments(self) -> StyleAndTextTuples:
        error = self.input_error
        if not error and self.input_images and self.input_mode in {"chat", "running"} and self.images.support() is False:
            error = "Image input is disabled for the active provider/model"
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
                    input_error,
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

        bindings.add("enter", filter=~modal, eager=True)(lambda event: event.current_buffer.validate_and_handle())
        bindings.add("escape", "enter", filter=~modal, eager=True)(lambda event: event.current_buffer.insert_text("\n"))
        for key, reverse in (("tab", False), ("s-tab", True)):
            bindings.add(key, filter=~modal)(lambda event, reverse=reverse: self.tab_or_complete(event.current_buffer, reverse=reverse))

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
            #   * approval mode → cancel this specific prompt (empty reply back to the agent).
            #   * idle chat → clear the current input silently.
            #   * agent running → discard a draft, or interrupt the turn when the input is empty.
            # Exit remains reserved for Ctrl-D on an empty chat input or the /exit slash command.
            if self.modal is not None:
                result = self.modal.key_fn("c-c", event.data)
                self.close_modal(None if result is TUI_MODAL_PENDING else result)
                return
            if self.input_mode == "approval" and self._input_pending is not None:
                self._input_result = ""
                self._input_pending.set()
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
            self.input_buffer.reset(Document(""))

        bindings.add("c-u", filter=~modal & edits_input, eager=True)(clear_input)

        def ctrl_d(event):  # pragma: no cover — interactive path
            if self.input_mode == "approval" and self._input_pending is not None:
                self._input_result = self.input_buffer.text
                self._input_pending.set()
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
        fd, path = tempfile.mkstemp(prefix="minacode-input-", suffix=".md")
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

    def run(self, style: Style | None = None) -> None:  # pragma: no cover — interactive
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
        self.app = app
        self.ready.clear()

        def start() -> None:
            # pre_run already runs inside the application's loop, and the task it starts is
            # cancelled with the rest of the application's background tasks on exit.
            app.create_background_task(self.animate())
            self.ready.set()

        try:
            with patch_stdout():
                app.run(pre_run=start)
        finally:
            self.ready.set()
            self.app = None
            # If the agent thread is still parked in request_input at exit, unblock it so its
            # frame unwinds instead of leaking a thread.
            if self._input_pending is not None:
                self._input_result = ""
                self._input_pending.set()
            if self.modal is not None:
                self.close_modal(None)


@dataclass
class TabbedViewState:
    titles: tuple[str, ...]
    tab: int = 0
    scroll: int = 0

    def switch(self, delta: int) -> None:
        self.tab = (self.tab + delta) % len(self.titles)
        self.scroll = 0

    def scroll_by(self, delta: int) -> None:
        self.scroll = max(0, self.scroll + delta)

    def visible(self, lines: list[ViewLine], height: int) -> list[ViewLine]:
        self.scroll = min(self.scroll, max(0, len(lines) - height))
        return lines[self.scroll : self.scroll + height]


@dataclass
class DiffViewState:
    REFRESH: ClassVar[object] = object()

    class Mode(Enum):
        LIST = auto()
        FILE = auto()

    view: TabbedViewState
    mode: Mode = Mode.LIST
    file: int = 0

    def reset(self) -> None:
        self.mode = self.Mode.LIST
        self.file = 0
        self.view.scroll = 0

    def switch_tab(self, delta: int) -> None:
        self.view.switch(delta)
        self.reset()

    def move_file(self, delta: int, count: int) -> None:
        if count:
            self.file = (self.file + delta) % count

    def clamp_file(self, count: int) -> None:
        self.file = self.file % count if count else 0

    def open_file(self, count: int) -> None:
        if self.mode is self.Mode.LIST and count:
            self.mode = self.Mode.FILE
            self.view.scroll = 0

    def close_file(self) -> None:
        if self.mode is self.Mode.FILE:
            self.mode = self.Mode.LIST
            self.view.scroll = 0

    def handle_key(self, key: str, file_count: int, viewport: int) -> Any:
        if key in {"q", "c-c"}:
            return None
        if key == "escape":
            if self.mode is self.Mode.LIST:
                return None
            self.close_file()
        elif key in {"down", "j", "up", "k"}:
            delta = 1 if key in {"down", "j"} else -1
            if self.mode is self.Mode.LIST and file_count:
                self.move_file(delta, file_count)
            elif self.mode is self.Mode.FILE:
                self.view.scroll_by(delta)
        elif key in {"h", "l", "tab"}:
            self.switch_tab(1 if key in {"l", "tab"} else -1)
        elif key == "right" and self.mode is self.Mode.LIST:
            self.switch_tab(1)
        elif key == "left":
            if self.mode is self.Mode.FILE:
                self.close_file()
            else:
                self.switch_tab(-1)
        elif key == "enter" and self.mode is self.Mode.LIST and file_count:
            self.open_file(file_count)
        elif self.mode is self.Mode.FILE and key in {"pagedown", "pageup", "c-d", "c-u"}:
            distance = max(1, viewport if key in {"pagedown", "pageup"} else viewport // 2)
            self.view.scroll_by(distance if key in {"pagedown", "c-d"} else -distance)
        elif key in {"g", "G"}:  # less-style: g→top, G→bottom
            if self.mode is self.Mode.LIST and file_count:
                self.file = 0 if key == "g" else file_count - 1
            elif self.mode is self.Mode.FILE:
                self.view.scroll = 0 if key == "g" else 10**9  # clamped to the last page on render
        elif key == "r":
            self.reset()
            return self.REFRESH
        return TUI_MODAL_PENDING


@dataclass
class ChoiceViewState:
    FREE_TEXT: ClassVar[str] = "\x00free_text"

    choices: tuple[str, ...]
    labels: dict[str, str]
    disabled: set[str]
    query: str = ""
    selected: int = 0
    searching: bool = False

    def visible(self) -> tuple[str, ...]:
        if not self.query:
            return self.choices
        needle = self.query.lower()
        visible: list[str] = []
        header = ""
        section: list[str] = []
        for choice in self.choices:
            if choice in self.disabled:
                if section:
                    visible.extend(([header] if header else []) + section)
                header, section = choice, []
            elif needle in (choice + " " + self.labels.get(choice, choice)).lower():
                section.append(choice)
        if section:
            visible.extend(([header] if header else []) + section)
        return tuple(visible)

    def enabled(self) -> tuple[str, ...]:
        return tuple(choice for choice in self.visible() if choice not in self.disabled)

    def clamp(self, options: tuple[str, ...] | None = None) -> tuple[str, ...]:
        options = options if options is not None else self.enabled()
        self.selected = min(max(self.selected, 0), len(options) - 1) if options else 0
        return options

    def move(self, delta: int) -> None:
        options = self.enabled()
        if options:
            self.selected = min(max(self.selected + delta, 0), len(options) - 1)

    def set_query(self, query: str) -> None:
        self.query = query
        self.selected = 0

    def selected_choice(self) -> str | None:
        options = self.clamp()
        return options[self.selected] if options else None

    def fragments(self, title: str, preview_fn: Callable[[str], str] | None = None) -> StyleAndTextTuples:
        visible = self.visible()
        options = self.clamp()
        suffix = (" /" + self.query) if self.query else ""
        if self.query and not self.searching:
            suffix += " (filtered)"
        parts: StyleAndTextTuples = [
            ("class:choice.title", title + suffix + "\n"),
            ("class:choice.disabled", "  j/k move, / search, Esc/q back/cancel\n"),
        ]
        if self.query and not options:
            return [*parts, ("class:choice.disabled", "  no matches\n")]
        number = 0
        for choice in visible:
            label = self.labels.get(choice, choice)
            if choice in self.disabled:
                parts.append(("class:choice.disabled", "  " + label + "\n"))
                continue
            number += 1
            selected = number - 1 == self.selected
            if selected:
                parts.append(("[SetCursorPosition]", ""))
            style = "class:choice.selected" if selected else ""
            prefix = ("> " if selected else "  ") + f"{number:2d}. "
            if match := UiPrinter.MCP_STATUS_RE.search(label):
                parts.append((style, prefix + label[: match.start()]))
                marker_style = (style + " class:choice.status." + match.group(1)).strip()
                parts.append((marker_style, "●"))
                parts.append((style, label[match.start() + 1 :] + "\n"))
            else:
                parts.append((style, prefix + label + "\n"))
        if preview_fn and options:
            preview = preview_fn(options[self.selected]).replace("\\n", "\n")
            if preview:
                parts.append(("class:choice.disabled", "  ──────────────────────────────────\n"))
                parts.extend(("class:choice.preview", "  │ " + line + "\n") for line in preview.splitlines())
        if self.searching:
            parts.append(("", "/" + self.query))
        return parts

    def handle_key(self, key: str, data: str = "") -> Any:
        if self.searching and key not in {"enter", "escape", "backspace", "c-h"}:
            text = data if key == "any" else key
            if len(text) == 1 and text not in "\r\n":
                self.set_query(self.query + text)
        elif key in {"j", "down"} and not self.searching:
            self.move(1)
        elif key in {"k", "up"} and not self.searching:
            self.move(-1)
        elif key in {"g", "G"} and not self.searching:  # less-style: g→first, G→last
            self.move(-len(self.enabled()) if key == "g" else len(self.enabled()))
        elif key == "/":
            self.searching = True
            self.set_query("")
        elif key in {"backspace", "c-h"} and self.searching:
            self.set_query(self.query[:-1])
        elif key == "escape":
            if self.searching:
                self.searching = False
            elif self.query:
                self.set_query("")
            else:
                return SELECTION_BACK
        elif key == "q" and not self.searching:
            return SELECTION_BACK
        elif key == "enter":
            if self.searching:
                self.searching = False
            elif (choice := self.selected_choice()) is not None:
                return SELECTION_FREE_TEXT if choice == self.FREE_TEXT else choice
        elif key == "c-c":
            return KeyboardInterrupt()
        elif key.isdigit() and not self.searching:
            number = int(key)
            options = self.enabled()
            if 1 <= number <= len(options):
                self.selected = number - 1
        return TUI_MODAL_PENDING


# Batch-level Ask modal results on top of ChoiceViewState's own SELECTION_* returns: ASK_DONE means
# every question was answered; (ASK_FREE_TEXT, index) means the question at `index` dropped to the
# shared input row and the modal should reopen for the rest.
ASK_DONE = object()
ASK_FREE_TEXT = object()


@dataclass
class AskViewState:
    """The Ask tool's multi-question modal: one ChoiceViewState page per question, options on the
    left and a rich markdown preview of the selected option on the right (below the options on
    narrow terminals). Enter advances to the next question and submits on the last; Tab cycles
    pages; `n` edits a note appended to the answer; Esc cancels the whole batch. A question
    without choices is a single "Type freely..." page that reports ASK_FREE_TEXT so the caller
    drops to the shared input row."""

    specs: list[AskSpec]
    pages: list[ChoiceViewState]
    active: int = 0
    picked: list[str | None] = field(default_factory=list)
    notes: dict[int, str] = field(default_factory=dict)
    notes_mode: bool = False
    note_buffer: str = ""

    def __post_init__(self) -> None:
        if not self.picked:
            self.picked = [None] * len(self.specs)

    @classmethod
    def build(cls, specs: list[AskSpec]) -> AskViewState:
        """One page per question; the free-text escape hatch is always offered (mirroring the old
        choice_application(free_text=True)) and a recommended choice is pre-selected and marked."""
        pages: list[ChoiceViewState] = []
        for spec in specs:
            choices = list(spec.choices) if spec.choices else []
            labels: dict[str, str] = {}
            current = ""
            if spec.recommended is not None and spec.choices and 0 <= spec.recommended < len(spec.choices):
                current = spec.choices[spec.recommended]
                labels[current] = current + " (recommended)"
            choices.append(ChoiceViewState.FREE_TEXT)
            labels[ChoiceViewState.FREE_TEXT] = "Type freely..."
            pages.append(
                ChoiceViewState(
                    tuple(choices),
                    labels,
                    set(),
                    selected=choices.index(current) if current else 0,
                )
            )
        return cls(specs, pages)

    def preview_text(self) -> str:
        """The selected option's preview markdown, or '' when it has none."""
        spec = self.specs[self.active]
        if not spec.previews or not spec.choices:
            return ""
        choice = self.pages[self.active].selected_choice()
        if choice is None or choice not in spec.choices:
            return ""
        index = spec.choices.index(choice)
        return spec.previews[index] if index < len(spec.previews) else ""

    def fragments(self, width: int, max_height: int) -> StyleAndTextTuples:
        """Render the active question. Options sit left with the selected option's rich markdown
        preview right when the terminal is wide enough (>=100) and a preview exists; otherwise the
        preview stacks below the options. The caller caps max_height to the terminal's rows minus
        the reserved chrome (status bar, input row, gaps)."""
        page = self.pages[self.active]
        parts: StyleAndTextTuples = [
            ("class:choice.title", f"({self.active + 1}/{len(self.specs)}) {self.specs[self.active].question}\n"),
            ("", "\n"),
        ]
        if self.notes_mode:
            parts.append(("class:choice.disabled", "notes: " + self.note_buffer + "\n"))
        elif note := self.notes.get(self.active):
            parts.append(("class:choice.disabled", "notes: " + note + "\n"))
        if not page.enabled():
            parts.append(("class:choice.disabled", "  no matches\n"))
            parts.append(("", "\n"))
            parts.extend(self._footer())
            return parts
        option_rows = self._option_rows(page)
        preview = self.preview_text()
        side_by_side = width >= 100 and bool(preview)
        if side_by_side:
            label_widths = [get_cwidth(page.labels.get(choice, choice)) for choice in page.visible()]
            # The +9 covers the number prefix ("> " + "N. ", 6 cells) plus a 3-cell visible gutter
            # between the longest option row and the preview column.
            left_width = min(max(label_widths) + 9, width * 2 // 5)
            right_width = max(10, width - left_width - 2)
            preview_rows = self._preview_rows(preview, right_width)
            body = self._join_rows(option_rows, preview_rows, left_width)
        else:
            body = list(option_rows)
            if preview:
                preview_rows = self._preview_rows(preview, max(10, width - 4))
                body.append([("class:choice.disabled", "  " + "─" * max(10, width - 4))])
                body.extend([("class:choice.preview", "  │ "), *row] for row in preview_rows)
        # Title and footer now each sit on their own blank line (2 extra rows in the budget).
        budget = max_height - 4 - (1 if self.notes_mode or self.notes.get(self.active) else 0)
        if page.searching:
            budget -= 1
        if len(body) > budget:
            overflow = len(body) - budget + 1
            body = body[: budget - 1] + [[("class:choice.disabled", f"  … {overflow} more lines")]]
        for row in body:
            parts.extend((*row, ("", "\n")))
        if page.searching:
            parts.append(("", "/" + page.query))
        parts.append(("", "\n"))
        parts.extend(self._footer())
        return parts

    @staticmethod
    def _option_rows(page: ChoiceViewState) -> list[StyleAndTextTuples]:
        """The page's option rows (one fragment list per row, no trailing newline), styled like
        ChoiceViewState.fragments' option block."""
        rows: list[StyleAndTextTuples] = []
        number = 0
        for choice in page.visible():
            label = page.labels.get(choice, choice)
            if choice in page.disabled:
                rows.append([("class:choice.disabled", "  " + label)])
                continue
            number += 1
            selected = number - 1 == page.selected
            row: StyleAndTextTuples = []
            if selected:
                row.append(("[SetCursorPosition]", ""))
            prefix = ("> " if selected else "  ") + f"{number:2d}. "
            row.append(("class:choice.selected" if selected else "", prefix + label))
            rows.append(row)
        return rows

    @staticmethod
    def _preview_rows(markdown_text: str, panel_width: int) -> list[StyleAndTextTuples]:
        """Render markdown to ANSI with Rich (same capture trick as UiPrinter.emit_markdown) and
        split it into one (style, text) tuple list per line, styles carried across newlines.
        Preview snippets are ASCII layouts, diffs, and tables whose newlines are structural, so
        each source line gets a hard line break (Markdown folds in-paragraph newlines to spaces)."""
        hard_breaks = "\n".join(line.rstrip() + "  " for line in markdown_text.split("\n"))
        console = Console(force_terminal=True, color_system="truecolor", no_color=False, width=max(10, panel_width))
        with console.capture() as capture:
            console.print(Markdown(hard_breaks, hyperlinks=False))
        cleaned = UiPrinter.strip_unknown_escapes(UiPrinter.strip_trailing_pad(capture.get()))
        return AskViewState._ansi_lines(cleaned)

    @staticmethod
    def _ansi_lines(text: str) -> list[StyleAndTextTuples]:
        """Split an ANSI string into one (style, text) tuple list per line. prompt_toolkit's ANSI
        parser emits a fragment per character (it walks SGR state per char), so adjacent fragments
        with the same style are merged back into runs; SGR state across a newline carries exactly
        as Rich emitted it."""
        lines: list[StyleAndTextTuples] = []
        current: StyleAndTextTuples = []
        for fragment in to_formatted_text(ANSI(text)):
            style = fragment[0]
            chunk = fragment[1]
            assert isinstance(chunk, str)  # ANSI fragments are always (style, str)
            pieces = chunk.split("\n")
            for index, piece in enumerate(pieces):
                if index:
                    lines.append(current)
                    current = []
                if current and current[-1][0] == style:
                    text = current[-1][1]
                    assert isinstance(text, str)  # ANSI fragments are always (style, str)
                    current[-1] = (style, text + piece)
                else:
                    current.append((style, piece))
        lines.append(current)
        return lines

    @staticmethod
    def _join_rows(left: list[StyleAndTextTuples], right: list[StyleAndTextTuples], left_width: int) -> list[StyleAndTextTuples]:
        """Merge the option rows and the preview rows column-wise: each left row is padded to
        left_width (by visible width) and the preview row continues on the same line; rows with no
        counterpart render alone."""
        rows: list[StyleAndTextTuples] = []
        for index in range(max(len(left), len(right))):
            left_row = left[index] if index < len(left) else []
            right_row = right[index] if index < len(right) else []
            used = sum(get_cwidth(fragment[1]) for fragment in left_row if isinstance(fragment[1], str) and fragment[1])
            row: StyleAndTextTuples = list(left_row)
            row.append(("", " " * max(0, left_width - used)))
            row.extend(right_row)
            rows.append(row)
        return rows

    def _footer(self) -> StyleAndTextTuples:
        return [
            (
                "class:choice.disabled",
                f"↑/↓ or j/k move · Enter select/next · Tab switch · n notes · / search · Esc cancel · ({self.active + 1}/{len(self.specs)})\n",
            )
        ]

    def handle_key(self, key: str, data: str = "") -> Any:
        """The active page's keys plus the batch-level ones: Tab cycles pages, `n` edits a note,
        Enter advances (submitting on the last page), Esc cancels the whole batch."""
        if self.notes_mode:
            if key == "escape":
                self.notes_mode = False
                self.note_buffer = ""
                return TUI_MODAL_PENDING
            if key == "enter":
                self.notes[self.active] = self.note_buffer
                self.notes_mode = False
                self.note_buffer = ""
                return TUI_MODAL_PENDING
            if key in {"backspace", "c-h"}:
                self.note_buffer = self.note_buffer[:-1]
                return TUI_MODAL_PENDING
            text = data if key == "any" else key
            if len(text) == 1 and text not in "\r\n":
                self.note_buffer += text
            return TUI_MODAL_PENDING
        page = self.pages[self.active]
        if key in {"tab", "s-tab"}:
            self.active = (self.active + (1 if key == "tab" else -1)) % len(self.specs)
            return TUI_MODAL_PENDING
        if (key == "n" or (key == "any" and data == "n")) and not page.searching:
            self.notes_mode = True
            self.note_buffer = self.notes.get(self.active, "")
            return TUI_MODAL_PENDING
        result = page.handle_key(key, data)
        if result is TUI_MODAL_PENDING:
            return TUI_MODAL_PENDING
        if result is SELECTION_BACK:
            return SELECTION_BACK  # the page unwound its own search/query layers; batch cancelled
        if result is SELECTION_FREE_TEXT:
            return (ASK_FREE_TEXT, self.active)
        if isinstance(result, str):
            self.picked[self.active] = result
            if self.active == len(self.specs) - 1:
                return ASK_DONE
            self.active += 1
            return TUI_MODAL_PENDING
        return result  # KeyboardInterrupt and anything else pass through
