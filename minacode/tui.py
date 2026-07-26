"""minacode prompt-toolkit application and interactive view state."""

from __future__ import annotations

import contextlib
import os
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, ClassVar, TypeVar

from prompt_toolkit import search as pt_search
from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions, is_done
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
from prompt_toolkit.widgets import SearchToolbar

from minacode.base import (
    SELECTION_BACK,
    SELECTION_FREE_TEXT,
    MinacodeError,
)
from minacode.engine import LogBlock, LogEdge
from minacode.image import IMAGE_MARKER, ImageInputs, ImageRef, UserInput

try:
    import pygments
    from pygments.lexers import get_lexer_by_name, get_lexer_for_filename
    from pygments.styles import get_style_by_name
    from pygments.token import Token
except ImportError:  # pragma: no cover - optional highlighting dependency
    pygments = Token = None
    get_lexer_by_name = get_lexer_for_filename = get_style_by_name = None

from minacode.render import UiPrinter

TUI_MODAL_PENDING = object()
ViewLine = TypeVar("ViewLine")


@dataclass
class TuiModal:
    fragments_fn: Callable[[], list[tuple[str, str]]]
    key_fn: Callable[[str, str], Any]
    exclusive: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None


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
            style, text, *rest = fragment
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

    MODAL_KEYS: ClassVar[tuple[str, ...]] = tuple("j k h l g G up down left right tab enter escape q r pagedown pageup c-d c-u c-o backspace c-h /".split())

    def __init__(
        self,
        *,
        on_chat_submit: Callable[[UserInput], None] | None = None,
        on_running_submit: Callable[[UserInput], None] | None = None,
        on_exit_request: Callable[[], None] | None = None,
        on_force_exit: Callable[[], None] | None = None,
        on_interrupt: Callable[[], None] | None = None,
        on_input_cancel: Callable[[], None] | None = None,
        on_retry: Callable[[], None] | None = None,
        on_recall: Callable[[], str | UserInput] | None = None,
        on_expand_output: Callable[[], None] | None = None,
        status_fragments_fn: Callable[[], list[tuple[str, str]]] | None = None,
        activity_fragments_fn: Callable[[], list[tuple[str, str]]] | None = None,
        input_hint_fn: Callable[[], str] | None = None,
        editor_context_fn: Callable[[], str] | None = None,
        images: ImageInputs | None = None,
        image_cwd: str = "",
        history: FileHistory | None = None,
        completer: Completer | None = None,
    ) -> None:
        self.on_chat_submit = on_chat_submit or (lambda _text: None)
        self.on_running_submit = on_running_submit or (lambda _text: None)
        self.on_exit_request = on_exit_request or (lambda: None)
        self.on_force_exit = on_force_exit or (lambda: None)
        self.on_interrupt = on_interrupt or (lambda: None)
        self.on_input_cancel = on_input_cancel or (lambda: None)
        self.on_retry = on_retry or (lambda: None)
        self.on_recall = on_recall or (lambda: "")
        self.on_expand_output = on_expand_output or (lambda: None)
        self.status_fragments_fn = status_fragments_fn or list
        self.activity_fragments_fn = activity_fragments_fn or list
        self.input_hint_fn = input_hint_fn or (lambda: "")
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
        self.input_buffer.on_text_changed += self._sync_input_images
        self.search_toolbar = SearchToolbar()
        self.app: Application | None = None
        self.ready = threading.Event()
        self.input_mode = "chat"  # chat | dispatch | running | approval
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
        self.invalidate()

    def invalidate(self) -> None:
        if self.app is not None:
            self.app.invalidate()

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

    def _sync_input_images(self, buffer: Buffer) -> None:
        text = buffer.text
        if self._changing_input:
            self._last_input_text = text
            return
        old = self._last_input_text
        if old == text:
            return
        self.input_error = ""
        prefix = 0
        while prefix < min(len(old), len(text)) and old[prefix] == text[prefix]:
            prefix += 1
        suffix = 0
        while suffix < len(old) - prefix and suffix < len(text) - prefix and old[-suffix - 1] == text[-suffix - 1]:
            suffix += 1
        removed_end = len(old) - suffix
        first = old[:prefix].count(IMAGE_MARKER)
        removed = old[prefix:removed_end].count(IMAGE_MARKER)
        if removed:
            self.input_images = self.input_images[:first] + self.input_images[first + removed :]
        self._last_input_text = text
        inserted_end = len(text) - suffix
        inserted = text[prefix:inserted_end]
        if inserted and inserted[-1].isspace() and self.input_mode in {"chat", "running"}:
            self._recognize_input()

    def show_modal(
        self,
        fragments_fn: Callable[[], list[tuple[str, str]]],
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
            result = subprocess.run(command, capture_output=True, text=True, timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            return True
        return result.returncode != 0 or result.stdout.strip() != "0"

    def modal_fragments(self) -> list[tuple[str, str]]:
        return self.modal.fragments_fn() if self.modal is not None else []

    def dispatch_modal_key(self, key: str, data: str = "") -> None:
        if self.modal is None:
            return
        result = self.modal.key_fn(key, data)
        if result is not TUI_MODAL_PENDING:
            self.close_modal(result)
        else:
            self.invalidate()

    def status_fragments(self) -> list[tuple[str, str]]:
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

    def input_error_fragments(self) -> list[tuple[str, str]]:
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
        self.input_window = Window(
            BufferControl(
                buffer=self.input_buffer,
                input_processors=[
                    HighlightIncrementalSearchProcessor(),
                    ImageLabelProcessor(lambda: self.input_images),
                    BeforeInput(self.status_fragments),
                    CallbackPlaceholder(self.input_hint_fn),
                ],
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
        normal_region = ConditionalContainer(
            HSplit(
                [
                    running_gap_above,
                    activity,
                    running_gap_below,
                    input_error,
                    self.input_window,
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
            bindings.add(key, filter=~modal)(lambda event, reverse=reverse: self.complete_input(event.current_buffer, reverse=reverse))

        @bindings.add(Keys.BracketedPaste, filter=~modal)
        def _paste(event):
            event.current_buffer.insert_text(event.data.replace("\r\n", "\n").replace("\r", "\n"))
            if self.input_mode in {"chat", "running"}:
                self._recognize_input()

        @bindings.add("c-r", filter=~modal, eager=True)
        def _history_search(event):
            direction = pt_search.SearchDirection.BACKWARD
            if event.app.layout.current_control is self.search_toolbar.control:
                pt_search.do_incremental_search(direction, count=event.arg)
            else:
                pt_search.start_search(direction=direction)

        bindings.add("c-o", filter=~modal, eager=True)(lambda _event: self.on_expand_output())

        # Ctrl-P mirrors Up here: readline treats them as synonyms, and both recall the latest
        # queued follow-up (or move the cursor up / walk history) while a turn is working.
        @bindings.add("up", filter=running, eager=True)
        @bindings.add("c-p", filter=running, eager=True)
        def _recall(event):
            if self.input_buffer.text:
                self.input_buffer.cursor_up()
                return
            text = self.on_recall()
            if text:
                self._reset_input(text, cursor_position=len(text))
            else:
                event.current_buffer.auto_up(count=event.arg)

        # Ctrl-X Ctrl-E (readline `edit-and-execute-command`) and Ctrl-G hand the current input to
        # $VISUAL/$EDITOR (fallback vim) for editing, matching Claude Code's editor bindings. The
        # `c-x c-e` chord means a lone Ctrl-X waits for the second key instead of firing eagerly.
        # In-flight resend has no key; it is the `/resend` command typed in the running input.
        edits_input = Condition(lambda: self.input_mode in {"chat", "running", "approval"})

        @bindings.add("c-x", "c-e", filter=~modal & edits_input)
        @bindings.add("c-g", filter=~modal & edits_input)
        def _edit_in_editor(event):  # pragma: no cover — interactive path
            self.edit_input_in_editor()

        @bindings.add("c-c", eager=True)
        @bindings.add("<sigint>", eager=True)
        def _ctrl_c(event):  # pragma: no cover — interactive path
            # Never quit on Ctrl-C. Instead:
            #   * approval mode → cancel this specific prompt (empty reply back to the agent).
            #   * idle chat → cancel and clear the current input.
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
                self.on_input_cancel()
                return
            if self.input_mode in {"dispatch", "running"}:
                # A draft absorbs the first press, the way it already does at the idle prompt. The
                # queue hint only renders on an empty buffer, so "Ctrl-C interrupts" is shown
                # exactly when the next press interrupts.
                if self.input_buffer.text:
                    self.input_buffer.reset(Document(""))
                    return
                self.on_interrupt()

        @bindings.add("c-u", filter=~modal & edits_input, eager=True)
        def _clear_input(event):  # pragma: no cover — interactive path
            # The readline convention for discarding the line, and the one key that means the same
            # thing in every editor here. Ctrl-C also clears, but while the agent runs it spends a
            # press that would otherwise interrupt; this one never competes with stopping the turn.
            self.input_buffer.reset(Document(""))

        @bindings.add("c-d", filter=~modal, eager=True)
        def _ctrl_d(event):  # pragma: no cover — interactive path
            if self.input_mode == "approval" and self._input_pending is not None:
                self._input_result = self.input_buffer.text
                self._input_pending.set()
            elif self.input_buffer.text and self.input_mode in {"chat", "running"}:
                self.input_buffer.delete()
            elif self.input_mode == "chat":
                self.on_exit_request()
                event.app.exit()

        @bindings.add(Keys.ControlBackslash, eager=True)
        def _force_exit(event):  # pragma: no cover — interactive emergency path
            self.on_force_exit()
            event.app.exit()

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
                completed = subprocess.run([*self.editor_command(), path])
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

    def run(self, style: Style | None = None) -> None:  # pragma: no cover — interactive
        app = Application(
            layout=self.build_layout(),
            key_bindings=self.make_bindings(),
            full_screen=False,
            mouse_support=False,
            refresh_interval=0.2,
            style=style,
            erase_when_done=True,
        )
        # A persistent primary-screen renderer needs CPR after a terminal resize; otherwise its
        # stale cursor coordinates can leave the transient footer in tmux scrollback. Keep the
        # legacy behavior of silently degrading on terminals that do not answer the probe.
        app.renderer.cpr_not_supported_callback = lambda: None
        self.app = app
        self.ready.clear()
        try:
            with patch_stdout():
                app.run(pre_run=self.ready.set)
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

    def fragments(self, title: str, preview_fn: Callable[[str], str] | None = None) -> list[tuple[str, str]]:
        visible = self.visible()
        options = self.clamp()
        suffix = (" /" + self.query) if self.query else ""
        if self.query and not self.searching:
            suffix += " (filtered)"
        parts: list[tuple[str, str]] = [
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
