"""minacode TUI: terminal rendering, command loop, and entry point."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, ClassVar

from prompt_toolkit import print_formatted_text, search as pt_search
from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions, is_done
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import ConditionalContainer, Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import BeforeInput, HighlightIncrementalSearchProcessor, Processor, Transformation
from prompt_toolkit.output import create_output
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import SearchToolbar
from openai import OpenAI
from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.rule import Rule
from rich.text import Text as RichText

from minacode.base import (
    DISMISSED,
    HTTP_USER_AGENT,
    MODEL_REQUEST_RETRIES,
    REASONING_CHOICES,
    SELECTION_BACK,
    SELECTION_FREE_TEXT,
    ConfigError,
    Json,
    MinacodeError,
    ProviderConfig,
    Text,
    ToolCall,
    ToolError,
    __version__,
)
from minacode.engine import Agent, ContextManager, LogBlock, LogEdge, LogLine, LogRole, ModelClient, ToolDisplay, TurnBox, UpdateChecker
from minacode.session import Session, SessionSnapshotCodec, SessionSnapshotStore, ToolResultRecord
from minacode.tools import AskSpec, CodeIndex, TOOL_REGISTRY

try:
    import pygments
    from pygments.lexers import get_lexer_by_name, get_lexer_for_filename
    from pygments.styles import get_style_by_name
    from pygments.token import Token
except ImportError:  # pragma: no cover - optional highlighting dependency
    pygments = Token = None
    get_lexer_by_name = get_lexer_for_filename = get_style_by_name = None


class CommandCompleter(Completer):
    # fmt: on
    # fmt: off
    SET_HANDLERS: ClassVar[dict[str, tuple[str, str, Callable[[str], Any] | None]]] = {
        "provider.temperature": ("provider", "temperature", lambda v: None if v == "off" else float(v)),
        "provider.max_tokens": ("provider", "max_tokens", lambda v: max(0, int(v))),
        "provider.timeout": ("provider", "timeout", lambda v: max(1, int(v))),
        "runtime.max_agent_steps": ("settings", "max_steps", lambda v: max(1, int(v))),
        "runtime.max_context_tokens": ("settings", "max_context_tokens", lambda v: max(1, int(v))),
        "runtime.max_parallel_tools": ("settings", "max_parallel_tools", lambda v: max(1, int(v))),
        "runtime.shell_timeout": ("settings", "shell_timeout", lambda v: max(1, int(v))),
        "runtime.bash_wait_timeout": ("settings", "bash_wait_timeout", lambda v: max(0, int(v))),
    }
    SET_KEYS: ClassVar[tuple[str, ...]] = tuple(SET_HANDLERS)
    # fmt: on
    # fmt: off
    SET_VALUES: ClassVar[dict[str, tuple[str, ...]]] = {
        "provider.temperature": ("off",),
    }
    # fmt: on

    def __init__(
        self,
        providers: Callable[[], tuple[str, ...]] = tuple,
        models: Callable[[], tuple[str, ...]] = tuple,
        mcp_servers: Callable[[], tuple[str, ...]] = tuple,
        mcp_connected_servers: Callable[[], tuple[str, ...]] = tuple,
        mcp_tools: Callable[[str], tuple[str, ...]] = lambda _server: (),
        skills: Callable[[], tuple[str, ...]] = tuple,
    ):
        self.providers = providers
        self.models = models
        self.mcp_servers = mcp_servers
        self.mcp_connected_servers = mcp_connected_servers
        self.mcp_tools = mcp_tools
        self.skills = skills

    def get_completions(self, document, _complete_event):
        text = document.text_before_cursor
        if text.startswith("/set "):
            tail = text[len("/set ") :]
            if " " not in tail:
                yield from self.matches(self.SET_KEYS, tail)
                return
            key, _, value = tail.partition(" ")
            yield from self.matches(self.SET_VALUES.get(key, ()), value)
            return
        for command, values in (
            ("/model ", self.models),
            ("/provider ", self.providers),
            ("/reason ", lambda: REASONING_CHOICES),
            ("/strict ", lambda: ("on", "off")),
        ):
            if text.startswith(command):
                yield from self.matches(values(), text[len(command) :])
                return
        if text.startswith("/mcp "):
            tail = text[len("/mcp ") :]
            if " " not in tail:
                yield from self.matches(("connect", "disconnect", "tools"), tail)
                return
            sub, _, value = tail.partition(" ")
            if sub == "connect":
                completed, _, prefix = value.rpartition(" ")
                selected = set(completed.split())
                yield from self.matches((name for name in self.mcp_servers() if name not in selected), prefix)
                return
            if sub == "disconnect":
                yield from self.matches(self.mcp_servers(), value)
                return
            if sub == "tools":
                yield from self.matches(self.mcp_connected_servers(), value)
                return

        at_match = re.search(r"@([A-Za-z0-9_.-]*)$", text)
        if at_match:
            server_part, dot, tool_part = at_match.group(1).partition(".")
            if dot:
                yield from self.matches(self.mcp_tools(server_part), tool_part)
            else:
                yield from self.matches(self.mcp_servers(), server_part)
            return

        skill_match = re.search(r"(?<![A-Za-z0-9_])\$([A-Za-z0-9_-]*)$", text)
        if skill_match:
            yield from self.matches(self.skills(), skill_match.group(1))
            return

        if text.startswith("/") and " " not in text:
            yield from self.matches(CommandLoop.COMMANDS, text)

    @staticmethod
    def matches(values, prefix: str):
        return (Completion(value, start_position=-len(prefix)) for value in values if value.startswith(prefix))


class Theme:
    DARK: ClassVar[dict[str, str]] = {
        "diff.added.bg": "bg:#003b00",
        "diff.added.fg": "fg:default",
        "diff.removed.bg": "bg:#520000",
        "diff.removed.fg": "fg:default",
        "syntax.assign": "fg:#79c0ff",
        "syntax.string": "fg:#a5d6ff",
        "syntax.number": "fg:#d2a8ff",
        "syntax.ident": "fg:#a5d6ff",
        "syntax.builtin": "fg:#79c0ff",
        "syntax.default_hex": "e6edf3",
        "status.base": "#e6edf3",
        "status.sep": "#4b5563",
        "status.provider": "#e6edf3",
        "status.reason": "#a5b4fc",
        "status.mcp": "#93c5fd",
        "status.ctx": "#facc15",
        "status.update": "#fb923c",
        "status.index": "#94a3b8",
        "status.warn": "#fb7185",
        "status.runtime": "#c084fc",
        "user.log": "#e0a96d",
        "pygments": "github-dark",
    }

    LIGHT: ClassVar[dict[str, str]] = {
        "diff.added.bg": "bg:#d1f0d1",
        "diff.added.fg": "fg:#003b00",
        "diff.removed.bg": "bg:#f5c8c8",
        "diff.removed.fg": "fg:#520000",
        "syntax.assign": "fg:#005cc5",
        "syntax.string": "fg:#032f62",
        "syntax.number": "fg:#6f42c1",
        "syntax.ident": "fg:#032f62",
        "syntax.builtin": "fg:#005cc5",
        "syntax.default_hex": "24292e",
        "status.base": "#24292e",
        "status.sep": "#9ca3af",
        "status.provider": "#24292e",
        "status.reason": "#5b21b6",
        "status.mcp": "#1e40af",
        "status.ctx": "#a16207",
        "status.update": "#9a3412",
        "status.index": "#475569",
        "status.warn": "#b91c1c",
        "status.runtime": "#6b21a8",
        "user.log": "#9a5b2e",
        "pygments": "default",
    }

    _mode: ClassVar[str] = "dark"
    _pygments_cache: ClassVar[dict[str, Any]] = {}

    @classmethod
    def set_mode(cls, mode: str) -> None:
        cls._mode = "light" if mode == "light" else "dark"

    @classmethod
    def style(cls, key: str) -> str:
        return (cls.LIGHT if cls._mode == "light" else cls.DARK)[key]

    @classmethod
    def detect(cls) -> str:
        # COLORFGBG is "fg;bg" (rxvt/urxvt/Konsole) or "fg;;bg" (iTerm2). Only the standard
        # white entries are reliably light; index 8 is bright black and must remain dark.
        fgbg = os.environ.get("COLORFGBG", "")
        if ";" in fgbg:
            with contextlib.suppress(ValueError):
                bg = int(fgbg.rsplit(";", 1)[1])
                return "light" if bg in {7, 15} else "dark"
        return "dark"

    @classmethod
    def resolve(cls, configured: str) -> str:
        configured = (configured or "auto").strip().lower()
        return configured if configured in ("light", "dark") else cls.detect()

    @classmethod
    def pygments_style(cls) -> Any:
        if pygments is None:
            return None
        name = cls.style("pygments")
        if name not in cls._pygments_cache:
            try:
                cls._pygments_cache[name] = get_style_by_name(name)  # type: ignore[possibly-unbound]
            except Exception:
                cls._pygments_cache[name] = None
        return cls._pygments_cache[name]


TUI_MODAL_PENDING = object()


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

    def apply_transformation(self, ti) -> Transformation:
        text = self.text_fn()
        buffer = ti.buffer_control.buffer
        if not text or buffer is None or buffer.text or ti.lineno != ti.document.line_count - 1:
            return Transformation(ti.fragments)
        return Transformation([*ti.fragments, ("class:queue.hint", text)])


class TuiApp:
    """One primary-screen application for live activity, input, selectors, and status.

    The agent owns the main thread; prompt-toolkit owns the TUI thread. `request_input` bridges
    blocking approvals, while completed output is printed above the app into terminal scrollback.
    """

    MODAL_KEYS: ClassVar[tuple[str, ...]] = tuple("j k h l g G up down left right tab enter escape q r pagedown pageup c-d c-u c-o backspace c-h /".split())

    def __init__(
        self,
        *,
        on_chat_submit: Callable[[str], None] | None = None,
        on_running_submit: Callable[[str], None] | None = None,
        on_exit_request: Callable[[], None] | None = None,
        on_force_exit: Callable[[], None] | None = None,
        on_interrupt: Callable[[], None] | None = None,
        on_input_cancel: Callable[[], None] | None = None,
        on_retry: Callable[[], None] | None = None,
        on_recall: Callable[[], str] | None = None,
        on_expand_output: Callable[[], None] | None = None,
        status_fragments_fn: Callable[[], list[tuple[str, str]]] | None = None,
        activity_fragments_fn: Callable[[], list[tuple[str, str]]] | None = None,
        input_hint_fn: Callable[[], str] | None = None,
        editor_context_fn: Callable[[], str] | None = None,
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
        self.input_buffer = Buffer(
            history=history,
            completer=completer,
            complete_while_typing=False,
            enable_history_search=True,
            multiline=True,
            accept_handler=self._accept,
        )
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
        self._input_pending = event
        self._input_result = ""

        def switch(document: Document, mode: str, prompt_text: str, done: threading.Event) -> None:
            nonlocal previous_document
            if previous_document is None:
                previous_document = self.input_buffer.document
            self.input_buffer.reset(document)
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
            app.loop.call_soon_threadsafe(callback, *args)
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
                buffer.append_to_history()
                buffer.reset()
                self.on_running_submit(text)
                return True
            return False
        if self.input_mode == "chat":
            if not text.strip():
                return False
            buffer.append_to_history()
            buffer.reset()
            self.set_dispatching()
            self.on_chat_submit(text)
            return True
        return False

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
                app.layout.focus(target or self.modal_window)
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
        command = ["tmux", "show-options", "-v"]
        if pane := os.environ.get("TMUX_PANE"):
            command.extend(["-t", pane])
        command.append("alternate-screen")
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            return True
        return result.returncode != 0 or result.stdout.strip().lower() != "off"

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
        bindings.add(Keys.BracketedPaste, filter=~modal)(lambda event: event.current_buffer.insert_text(event.data.replace("\r\n", "\n").replace("\r", "\n")))

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
                self.input_buffer.reset(Document(text, cursor_position=len(text)))
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
        original = self.input_buffer.text
        composed, marker = self._compose_editor_text(original, self.editor_context_fn())
        edited = await run_in_terminal(lambda: self._edit_text_in_editor(composed), in_executor=True)
        if edited is None:
            return
        edited = self._strip_editor_context(edited, marker)
        if edited != original:
            self.input_buffer.reset(Document(edited, cursor_position=len(edited)))
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


class UiPrinter:
    MESSAGE_ROLE_STYLES: ClassVar[dict[str, str]] = {"user": "cyan bold", "assistant": "magenta bold"}
    PROMPT_PREFIX: ClassVar[str] = "> "
    USER_LOG_PREFIX: ClassVar[str] = "• "
    MCP_STATUS_RE: ClassVar[re.Pattern[str]] = re.compile(r"● (connected|connecting|disconnected|disconnecting|error|skipped)")
    MCP_STATUS_ANSI: ClassVar[dict[str, str]] = {
        "connected": "\x1b[32m",
        "connecting": "\x1b[32m",
        "disconnected": "\x1b[33m",
        "disconnecting": "\x1b[33m",
        "error": "\x1b[31m",
        "skipped": "\x1b[90m",
    }

    @classmethod
    def user_log_style(cls) -> str:
        return Theme.style("user.log")

    TOOL_ARG_TOKEN: ClassVar[re.Pattern] = re.compile(
        r"""\s+|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[A-Za-z_][\w.-]*=|(?:tr|job)\.\d+|\d+(?::\d+)?|[;,]|[^\s;,]+"""
    )

    def __init__(self, output_fn=print):
        self.output_fn = output_fn
        self.color = output_fn is print and sys.stdout.isatty()

    def emit(self, text: str | LogBlock = "") -> None:
        if not self.color:
            self.output_fn(str(text))
            return
        segments = self.log_segments(text) if isinstance(text, LogBlock) else self.segments(text)
        print_formatted_text(FormattedText(segments), end="", flush=True)

    # Rich right-pads every rendered line with spaces up to the console width so backgrounds and
    # padding can fill the row. Uncolored padding gets baked into scrollback and turns into wrap
    # zigzags on a narrower terminal, so we strip it — but padding that carries a background color
    # (syntax-highlighted code blocks, /diff previews) must be preserved so the block still reads
    # as a solid band. We track the SGR bg state per token and only strip whitespace rendered with
    # bg off.
    SGR_RE: ClassVar[re.Pattern[str]] = re.compile(r"\x1b\[([0-9;]*)m")
    # OSC / APC / DCS / SOS / PM sequences are terminal control strings that prompt_toolkit's ANSI
    # parser doesn't recognize. When they slip through Rich's output (OSC 8 hyperlinks were the
    # historical culprit, iTerm image escapes / Kitty graphics / shell-integration marks are
    # potential future ones), pt eats the ESC framing but leaks the payload as visible garbage
    # (e.g. `8;id=…;https://…;;` for OSC 8). Strip these up front so pt only ever sees CSI escapes.
    # The trade is that any legitimate uses of these (clickable hyperlinks, inline images) never
    # reach the terminal — but they weren't working through pt anyway; better clean than garbled.
    NON_CSI_ESCAPE_RE: ClassVar[re.Pattern[str]] = re.compile(r"\x1b[\]_PX^][^\x07\x1b]*(?:\x07|\x1b\\)")

    @classmethod
    def strip_unknown_escapes(cls, text: str) -> str:
        return cls.NON_CSI_ESCAPE_RE.sub("", text)

    @classmethod
    def strip_trailing_pad(cls, text: str) -> str:
        return "\n".join(cls._strip_line_pad(line) for line in text.split("\n"))

    @classmethod
    def _strip_line_pad(cls, line: str) -> str:
        tokens: list[tuple[str, str]] = []  # ("sgr"|"text", payload)
        bg_states: list[bool] = []  # bg active while each token renders
        bg, idx = False, 0
        for m in cls.SGR_RE.finditer(line):
            if m.start() > idx:
                tokens.append(("text", line[idx : m.start()]))
                bg_states.append(bg)
            tokens.append(("sgr", m.group(0)))
            bg_states.append(bg)
            for param in (m.group(1) or "0").split(";"):
                n = int(param) if param else 0
                if n == 0 or n == 49:
                    bg = False
                elif 40 <= n <= 47 or 100 <= n <= 107 or n == 48:
                    bg = True
            idx = m.end()
        if idx < len(line):
            tokens.append(("text", line[idx:]))
            bg_states.append(bg)
        seen_content = False
        for i in range(len(tokens) - 1, -1, -1):
            kind, payload = tokens[i]
            if kind == "sgr" or seen_content:
                continue
            if bg_states[i]:
                if payload.strip():
                    seen_content = True
                continue
            stripped = payload.rstrip()
            if stripped != payload:
                tokens[i] = ("text", stripped)
            if stripped:
                seen_content = True
        return "".join(payload for _, payload in tokens)

    def emit_answer(self, text: str, *, role: str = "", rule: bool = True, indent: int = 0) -> None:
        if not self.color:
            if role == "user":
                text, role = "\n" + self.USER_LOG_PREFIX + text, ""
            elif role == "assistant":
                role = ""
            self.output_fn(self.indent_message(text, role, indent))
            return
        console = Console(force_terminal=True, color_system="truecolor", no_color=False, width=shutil.get_terminal_size().columns)
        with console.capture() as capture:
            self.render_message(console, text, role, rule, indent)
        cleaned = self.strip_unknown_escapes(self.strip_trailing_pad(capture.get()))
        print_formatted_text(ANSI(cleaned), end="", flush=True)

    @staticmethod
    def indent_message(text: str, role: str = "", indent: int = 0) -> str:
        body = "\n".join(LogBlock.margin(indent) + line for line in text.splitlines() or [""])
        return f"{LogBlock.margin(indent)}{role}:\n{body}" if role else body

    @classmethod
    def colorize_mcp_status(cls, text: str) -> str:
        return cls.MCP_STATUS_RE.sub(lambda match: cls.MCP_STATUS_ANSI[match.group(1)] + "●\x1b[39m " + match.group(1), text)

    def render_message(self, console: Console, text: str, role: str, rule: bool, indent: int) -> None:
        error = text.startswith(("Error:", "ConfigError:", "Unknown command:"))
        styled_text = self.colorize_mcp_status(text) if role != "user" else text
        if rule and not error:
            console.print(Rule(style="bright_black", characters="─"))
        margin = LogBlock.margin(indent)
        if role == "user":
            console.print("")
            console.print(Padding(RichText(UiPrinter.USER_LOG_PREFIX + text, style=self.user_log_style()), (0, 0, 0, len(margin))))
        elif role == "assistant":
            content = RichText(styled_text, style="red") if error else Markdown(styled_text, hyperlinks=False)
            console.print(Padding(content, (0, 0, 0, len(margin))))
        else:
            if role:
                label = RichText(role + ":", style=self.MESSAGE_ROLE_STYLES.get(role, "bright_black"))
                console.print(Padding(label, (0, 0, 0, len(margin))))
            content = RichText(styled_text, style="red") if error else Markdown(styled_text, hyperlinks=False)
            console.print(Padding(content, (0, 0, 0, len(margin))))

    def emit_markdown(self, text: str) -> None:
        # Render markdown to an ANSI string and emit via prompt_toolkit. Printing Rich output directly
        # while the TUI is running can interleave raw escapes with its renderer; capturing first and
        # emitting as ANSI keeps all terminal output inside the shared application.
        if not self.color:
            self.emit(text)
            return
        console = Console(force_terminal=True, color_system="truecolor", no_color=False, width=shutil.get_terminal_size().columns)
        with console.capture() as capture:
            console.print(Markdown(text, hyperlinks=False))
        cleaned = self.strip_unknown_escapes(self.strip_trailing_pad(capture.get()))
        print_formatted_text(ANSI(cleaned), end="", flush=True)

    @staticmethod
    def tab_segments(titles: tuple[str, ...], active: int) -> list[tuple[str, str]]:
        parts: list[tuple[str, str]] = []
        for index, title in enumerate(titles):
            parts.append(("class:tab.active" if index == active else "class:tab.inactive", f" {title} "))
            if index < len(titles) - 1:
                parts.append(("class:choice.disabled", " │ "))
        return parts

    def segments(self, text: str) -> list[tuple[str, str]]:
        if text.startswith(("goal:", "check:", "plan:", "known:")):
            return self.memory_segments(text)
        if text.startswith(self.USER_LOG_PREFIX):
            prefix, content = self.USER_LOG_PREFIX, text[len(self.USER_LOG_PREFIX) :]
            return [(self.user_log_style(), prefix + content + "\n")]
        if text.startswith("+ "):
            return [("ansibrightblack", "+ "), ("fg:default", text[2:] + "\n")]
        if text.startswith("[done in "):
            return [("ansibrightblack", text + "\n")]
        if text.startswith("minacode "):
            return [("ansicyan", text + "\n")]
        if text.startswith("Error:") or text.startswith("ConfigError:") or text.startswith("Unknown command:"):
            return [("ansired", text + "\n")]
        return [("fg:default", line + "\n") for line in text.splitlines() or [""]]

    LOG_STYLES: ClassVar[dict[LogRole, tuple[str, str]]] = {
        LogRole.TOOL: ("ansigreen", "fg:default"),
        LogRole.AUTO: ("ansiblue", "fg:default"),
        LogRole.META: ("ansibrightblack", "ansibrightblack"),
        LogRole.OUTPUT: ("ansibrightblack", "ansibrightblack"),
        LogRole.ERROR: ("ansired", "fg:default"),
        LogRole.MUTED: ("ansibrightblack", "ansibrightblack"),
        LogRole.DIFF: ("fg:default", "fg:default"),
    }

    def log_segments(self, block: LogBlock) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        width = max(1, shutil.get_terminal_size((120, 20)).columns - 1)
        entries = list(block.walk())
        index = 0
        while index < len(entries):
            line, level = entries[index]
            if line.role is LogRole.DIFF:
                end = index + 1
                while end < len(entries) and entries[end][0].role is LogRole.DIFF and entries[end][1] == level:
                    end += 1
                diff_lines = [item for item, _level in entries[index:end]]
                # The bg band lives inside diff_segments; size it to the width remaining after this
                # log tree's own margin+edge, so the padding does not overflow into a phantom wrap row.
                sample_prefix = [("", block.margin(level)), *self.edge_segments(diff_lines[0].edge)]
                sample_prefix_width = sum(get_cwidth(text) for _style, text in sample_prefix)
                diff_row_width = max(1, width - sample_prefix_width)
                diff_text = "\n".join(item.text for item in diff_lines)
                highlighted = self.segment_lines(self.diff_segments(diff_text, diff_row_width))
                for item, rendered in zip(diff_lines, highlighted):
                    prefix = [("", block.margin(level)), *self.edge_segments(item.edge)]
                    rendered = self.remove_line_ending(rendered)
                    for row in Text.wrap_styled(prefix, prefix, rendered, width):
                        segments.extend([*row, ("", "\n")])
                index = end
                continue
            label_style, text_style = self.LOG_STYLES[line.role]
            prefix = [("", block.margin(level)), *self.edge_segments(line.edge)]
            if line.label:
                prefix.append((label_style, line.label))
            content: list[tuple[str, str]] = []
            if line.text:
                separator = "  " if line.edge is LogEdge.NONE and line.label else " " if line.label else ""
                prefix.append((text_style, separator))
                content.extend(self.syntax_segments(line.text, line.syntax, text_style))
            if line.meta:
                content.append(("ansired" if line.role is LogRole.ERROR else "ansibrightblack", line.meta))
            continuation = [("", block.margin(level) + " " * get_cwidth(line.text_prefix()))]
            for row in Text.wrap_styled(prefix, continuation, content, width):
                segments.extend([*row, ("", "\n")])
            index += 1
        return segments

    @staticmethod
    def edge_segments(edge: LogEdge) -> list[tuple[str, str]]:
        return [] if edge is LogEdge.NONE else [("ansibrightblack", edge.value + " ")]

    @staticmethod
    def remove_line_ending(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
        result = list(segments)
        if result and result[-1][1].endswith("\n"):
            style, text = result[-1]
            result[-1] = (style, text[:-1])
            if not result[-1][1]:
                result.pop()
        return result

    @classmethod
    def syntax_segments(cls, text: str, lexer_name: str, fallback_style: str) -> list[tuple[str, str]]:
        if lexer_name == "tool-args":
            return cls.tool_arg_segments(text, fallback_style)
        if pygments is None or not lexer_name:
            return [(fallback_style, text)]
        try:
            lexer = get_lexer_by_name(lexer_name, stripnl=False, ensurenl=False)
            return [(cls.pygments_style(token_type), value) for token_type, value in lexer.get_tokens(text) if value]
        except Exception:
            return [(fallback_style, text)]

    @classmethod
    def tool_arg_segments(cls, text: str, fallback_style: str) -> list[tuple[str, str]]:
        segments = []
        for match in cls.TOOL_ARG_TOKEN.finditer(text):
            token = match.group(0)
            if token.isspace():
                style = fallback_style
            elif token.endswith("="):
                style = Theme.style("syntax.assign")
            elif token.startswith(('"', "'")):
                style = Theme.style("syntax.string")
            elif re.fullmatch(r"(?:tr|job)\.\d+|\d+(?::\d+)?", token):
                style = Theme.style("syntax.number")
            elif token in {";", ","}:
                style = "ansibrightblack"
            else:
                style = Theme.style("syntax.ident")
            segments.append((style, token))
        return segments or [(fallback_style, text)]

    def memory_segments(self, text: str) -> list[tuple[str, str]]:
        segments = []
        for line in text.splitlines() or [""]:
            if line.startswith(("goal:", "check:")):
                segments.append(("ansimagenta", line))
            elif line in {"summary:", "plan:", "known:"}:
                segments.append(("ansicyan", line))
            elif line.lstrip().startswith("- [x]"):
                segments.append(("ansigreen", line))
            elif line.lstrip().startswith("- [~]"):
                segments.append(("ansiyellow", line))
            elif line.lstrip().startswith("- [-]"):
                segments.append(("ansired", line))
            elif line.lstrip().startswith("+ "):
                segments.append(("ansigreen", line))
            else:
                segments.append(("fg:default", line))
            segments.append(("", "\n"))
        return segments

    @classmethod
    def pygments_style(cls, token_type: Any) -> str:
        style = Theme.pygments_style()
        if style is None:
            return "fg:default"
        if token_type in Token.Text.Whitespace:
            return "fg:default"
        if token_type in Token.Name.Builtin:
            return Theme.style("syntax.builtin")
        definition = style.style_for_token(token_type)
        color = definition.get("color")
        default_hex = Theme.style("syntax.default_hex")
        parts = ["fg:default" if not color or color.lower() == default_hex else f"fg:#{color}"]
        parts.extend(attribute for attribute in ("bold", "italic", "underline") if definition.get(attribute))
        return " ".join(parts)

    def _diff_tokenize_lines(self, code_text: str, path: str | None) -> list[list[tuple[str, str]]] | None:
        """Tokenize a whole block of code and return highlighted segments per line.

        Pygments lexers are designed to work on whole files; splitting by diff
        lines and lexing each one independently breaks multiline strings and
        indentation-sensitive languages.  We therefore lex the assembled code
        block once and split the resulting token stream back into lines.
        """
        if pygments is None or not path:
            return None
        try:
            lexer = get_lexer_for_filename(path, stripnl=False)
        except Exception:
            return None
        try:
            tokens = lexer.get_tokens(code_text)
        except Exception:
            return None

        lines: list[list[tuple[str, str]]] = [[]]
        for token_type, value in tokens:
            style = self.pygments_style(token_type)
            parts = value.split("\n")
            for i, part in enumerate(parts):
                if i > 0:
                    lines.append([])
                if part:
                    lines[-1].append((style, part))
        return lines

    # Width taken by the line-number gutter emitted inside diff_segments (`NNNN NNNN | `).
    DIFF_GUTTER_WIDTH: ClassVar[int] = 12

    def diff_segments(self, text: str, row_width: int | None = None) -> list[tuple[str, str]]:
        return self._diff_segments(text, row_width=row_width, live=False)

    def diff_segments_live(self, text: str, row_width: int | None = None) -> list[tuple[str, str]]:
        """Same as diff_segments, but pads the bg band to the current pane width. Only for live
        live renderers that repaint on resize (the `/diff` viewer). Scrollback callers must
        NOT use this — baked-in wide padding wraps on a later pane shrink and drops the bg color on
        the wrapped continuation, which looks broken."""
        return self._diff_segments(text, row_width=row_width, live=True)

    def _diff_segments(self, text: str, *, row_width: int | None, live: bool) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        old_line: int | None = None
        new_line: int | None = None
        lines = text.splitlines()
        natural_changed_width = max(
            (get_cwidth(line) for line in lines if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))),
            default=1,
        )
        if row_width is None:
            row_width = shutil.get_terminal_size((120, 20)).columns - 3
        available_changed_width = max(1, row_width - self.DIFF_GUTTER_WIDTH)
        # Scrollback path: cap the bg band at the widest actual changed line, and drop bg entirely
        # if that would already exceed the pane — a later resize can't wrap what wasn't padded. Live
        # path (the /diff viewer): fill edge-to-edge with the current pane width; the viewer repaints
        # on resize, so wide padding stays fresh.
        if live:
            changed_width: int | None = available_changed_width
        else:
            changed_width = natural_changed_width if natural_changed_width <= available_changed_width else None

        # Determine the target file path from the diff header.  The `+++` line
        # names the resulting file; for created files `---` is /dev/null.
        file_path: str | None = None
        for header in lines:
            if header.startswith("+++"):
                candidate = header[4:].strip()
                if candidate != "/dev/null":
                    file_path = candidate
                break

        # Collect lines that belong to the new file version: context lines and
        # added lines.  These are lexed together so the highlighted diff is
        # syntactically coherent. Removed lines stay neutral on a red background so
        # the "before" state does not interfere with lexing the "after" state.
        new_code_lines: list[str] = []
        new_code_indices: list[int] = []
        for i, line in enumerate(lines):
            # Skip the unified-diff file headers / hunk markers (the trailing space avoids matching a
            # real added line whose content starts with "+++"); feed only actual code to the lexer.
            if line.startswith(("+++ ", "--- ", "@@ ")):
                continue
            if line.startswith(("+", " ")):
                new_code_lines.append(line[1:])
                new_code_indices.append(i)

        highlighted: list[list[tuple[str, str]]] | None = None
        if new_code_lines:
            highlighted = self._diff_tokenize_lines("\n".join(new_code_lines), file_path)

        hl_by_index: dict[int, list[tuple[str, str]]] = {}
        if highlighted is not None:
            for hl_index, line_index in enumerate(new_code_indices):
                if hl_index < len(highlighted):
                    hl_by_index[line_index] = highlighted[hl_index]

        def hunk_start(part: str, prefix: str) -> int | None:
            if not part.startswith(prefix):
                return None
            try:
                return int(part[1:].split(",", 1)[0])
            except ValueError:
                return None

        def number(old: int | None, new: int | None) -> None:
            old_text = "" if old is None else str(old)
            new_text = "" if new is None else str(new)
            segments.append(("ansibrightblack", f"{old_text:>4} {new_text:>4} | "))

        def append_hl(prefix: str, prefix_style: str, content_hl: list[tuple[str, str]], suffix: str, background: str = "") -> None:
            def styled(style: str) -> str:
                return (style + " " + background).strip()

            segments.append((styled(prefix_style), prefix))
            for style, piece in content_hl:
                segments.append((styled(style), piece))
            width = get_cwidth(prefix) + sum(get_cwidth(piece) for _style, piece in content_hl)
            padding = " " * max(0, changed_width - width) if background and changed_width is not None else ""
            segments.append((background if padding else "", padding + suffix))

        for index, line in enumerate(lines):
            suffix = "\n" if index < len(lines) - 1 else ""
            if line.startswith("@@"):
                parts = line.split()
                if len(parts) >= 3:
                    old_line = hunk_start(parts[1], "-")
                    new_line = hunk_start(parts[2], "+")
                number(None, None)
                segments.append(("ansicyan", line + suffix))
            elif line.startswith(("---", "+++")):
                number(None, None)
                segments.append(("ansibrightblack", line + suffix))
            elif line.startswith("+"):
                number(None, new_line)
                content_hl = hl_by_index.get(index) or [(Theme.style("diff.added.fg"), line[1:])]
                append_hl("+", "ansigreen", content_hl, suffix, Theme.style("diff.added.bg"))
                new_line = None if new_line is None else new_line + 1
            elif line.startswith("-"):
                number(old_line, None)
                append_hl("-", "ansired", [(Theme.style("diff.removed.fg"), line[1:])], suffix, Theme.style("diff.removed.bg"))
                old_line = None if old_line is None else old_line + 1
            elif line.startswith(" "):
                number(old_line, new_line)
                content_hl = hl_by_index.get(index) or [("fg:default", line[1:])]
                append_hl(" ", "fg:default", content_hl, suffix)
                old_line = None if old_line is None else old_line + 1
                new_line = None if new_line is None else new_line + 1
            else:
                number(None, None)
                segments.append(("fg:default", line + suffix))
        return segments

    @staticmethod
    def segment_lines(segments: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
        lines: list[list[tuple[str, str]]] = [[]]
        for style, text in segments:
            parts = text.split("\n")
            for index, part in enumerate(parts):
                if index > 0:
                    lines[-1].append((style, "\n"))
                    lines.append([])
                if part:
                    lines[-1].append((style, part))
        if lines and not lines[-1]:
            lines.pop()
        return lines


class BashLivePreview:
    HEIGHT: ClassVar[int] = 6
    MAX_CHARS: ClassVar[int] = 8000
    # Heartbeat tick so the elapsed timer advances even while a command produces no output
    # (e.g. quiet long-runners or `... | tail` that buffers until EOF), so the terminal never
    # looks frozen during a blocking command.
    TICK: ClassVar[float] = 0.3

    def __init__(self):
        self.output = create_output(sys.stderr)
        self.active = False
        self.rendered_lines = 0
        self.rendered_rows: list[list[tuple[str, str]]] = []
        self.text = ""
        self.started_at = 0.0
        self.lock = threading.Lock()
        self.timer: threading.Thread | None = None

    def start(self) -> None:
        if not sys.stderr.isatty():
            return
        with self.lock:
            self.active, self.rendered_lines, self.rendered_rows, self.text = True, 0, [], ""
            self.started_at = time.monotonic()
            self.render()
        self.timer = threading.Thread(target=self.tick, daemon=True)
        self.timer.start()

    def tick(self) -> None:
        while True:
            time.sleep(self.TICK)
            with self.lock:
                if not self.active:
                    return
                self.render()

    def update(self, text: str) -> None:
        with self.lock:
            if not self.active:
                return
            self.text = (self.text + text)[-self.MAX_CHARS :]
            self.render()

    def finish(self) -> None:
        with self.lock:
            if not self.active:
                return
            self.active = False
        timer = self.timer
        if timer is not None:
            timer.join()
        with self.lock:
            self.rendered_lines, self.rendered_rows, self.text = 0, [], ""

    def render(self) -> None:
        if not self.active:
            return
        rows: list[list[tuple[str, str]]] = [[("ansibrightblack", line)] for line in self.frame_lines()]
        if rows == self.rendered_rows:
            return
        previous = self.rendered_lines
        if self.rendered_lines:
            self.output.write_raw(f"\x1b[{self.rendered_lines}A")
        for row in rows:
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            print_formatted_text(FormattedText(row), output=self.output, end="", flush=True)
            self.output.write_raw("\n")
        for _ in range(max(0, previous - len(rows))):
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            self.output.write_raw("\n")
        if previous > len(rows):
            self.output.write_raw(f"\x1b[{previous - len(rows)}A")
        self.output.flush()
        self.rendered_lines = len(rows)
        self.rendered_rows = rows

    def frame_lines(self) -> list[str]:
        width = max(20, shutil.get_terminal_size((120, 20)).columns)
        body = [line.expandtabs(4) for line in self.text.replace("\r", "\n").splitlines()[-self.HEIGHT :]]
        label = Text.elapsed_since(self.started_at, precise=True)
        # `limit` leaves a column of slack so a full-width line cannot auto-wrap and desync the
        # cursor-up math in render().
        limit = max(1, width - get_cwidth(LogBlock.prefix(2, LogEdge.CONTINUE)) - 1)

        def clip(line: str) -> str:
            return Text.clip_width(line, limit)

        # Always emit a status row so the frame is visible even before any output arrives.
        status = f"output · {label}" if body else f"running… {label}"
        lines = [LogLine(status, role=LogRole.META, edge=LogEdge.BRANCH)]
        lines.extend(LogLine("", clip(line), LogRole.OUTPUT, LogEdge.CONTINUE) for line in body)
        return str(LogBlock.hierarchy(None, lines)).splitlines()


class StatusBar:
    INTERVAL: ClassVar[float] = 0.2
    RETRY_NOTICE_DURATION: ClassVar[float] = 2.0
    INDEX_SPINNER: ClassVar[tuple[str, ...]] = ("~", "/", "-", "\\", "|")
    ROLE_KEYS: ClassVar[tuple[str, ...]] = ("provider", "reason", "mcp", "ctx", "update", "index", "warn", "runtime")

    @classmethod
    def role_style(cls, role: str) -> str:
        return Theme.style("status." + role) if role in cls.ROLE_KEYS else Theme.style("status.base")

    def __init__(self, session: Session):
        self.session = session
        self.started_at = 0.0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.rendered = False
        self.output = create_output(sys.stderr)
        self.seen_retry_count = session.state.model_retry_count
        self.retry_notice_until = 0.0

    def start(self, *, reset: bool = True) -> None:
        if self.thread is not None or not sys.stderr.isatty():
            return
        self.begin(reset=reset)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def begin(self, *, reset: bool = True) -> None:
        if reset or not self.started_at:
            self.started_at = time.monotonic()

    def stop(self) -> None:
        if self.thread is None:
            return
        self.stop_event.set()
        self.thread.join()
        self.thread = None
        self.clear()

    def is_running(self) -> bool:
        return self.thread is not None

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            print_formatted_text(FormattedText(self.display_fragments(active=True)), output=self.output, end="", flush=True)
            self.rendered = True
            self.stop_event.wait(self.INTERVAL)

    def model_elapsed(self) -> float:
        return max(0.0, time.monotonic() - started) if (started := self.session.state.current_model_call_started_at) > 0 else 0.0

    def clear(self) -> None:
        if self.rendered:
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            self.output.flush()
            self.rendered = False

    def display_fragments(self, *, active: bool) -> list[tuple[str, str]]:
        if not active:
            return self.fragments(sweep=False, show_elapsed=False)
        return self.fragments(sweep=True, show_elapsed=True)

    def retry_notice_active(self) -> bool:
        now = time.monotonic()
        count = self.session.state.model_retry_count
        if count != self.seen_retry_count:
            self.seen_retry_count = count
            self.retry_notice_until = now + self.RETRY_NOTICE_DURATION
        return self.retry_notice_until > now

    def model_attempt_status(self) -> str:
        attempt = self.session.state.current_model_attempt
        return f"attempt {attempt}/{MODEL_REQUEST_RETRIES + 1}" if attempt > 1 else ""

    def retry_status(self) -> str:
        if not self.retry_notice_active():
            return ""
        attempt = self.session.state.current_model_attempt
        text = f"retrying {attempt}/{MODEL_REQUEST_RETRIES + 1}" if attempt > 1 else "retrying"
        reason = self.session.state.model_retry_reason
        return text + (" · " + reason if reason else "")

    def fragments(self, *, sweep: bool, show_elapsed: bool) -> list[tuple[str, str]]:
        entries = self.entries(show_elapsed=show_elapsed)
        text = " | ".join(text for text, _ in entries)
        columns = shutil.get_terminal_size((120, 20)).columns
        if get_cwidth(text) >= columns:
            text = Text.clip_width(text, columns - 1)
            return self.sweep_fragments(text) if sweep else [(Theme.style("status.base"), text)]
        return self.sweep_fragments(text) if sweep else self.styled_fragments(entries)

    def entries(self, *, show_elapsed: bool) -> list[tuple[str, str]]:
        provider = self.session.config.provider
        model = provider.model.rsplit("/", 1)[-1] or "(no model)"
        reason = provider.reasoning
        parts = [(self.session.config.active_provider + "/" + model, "provider"), (reason, "reason")]

        mcp_status = self.mcp_status()
        if mcp_status:
            parts.append((mcp_status, "mcp"))
        skill_count = len(self.session.skills.skills) if self.session.skills else 0
        if skill_count:
            parts.append((f"skills {skill_count}", "mcp"))
        running_jobs = len(self.session.running_jobs())
        if running_jobs:
            parts.append((f"jobs {running_jobs}", "warn"))
        parts.append(("ctx " + str(self.session.state.context_percent) + "%", "ctx"))
        update_status = self.update_status()
        if update_status:
            parts.append((update_status, "update"))
        index_status = self.index_status()
        if index_status:
            parts.append(("index" + index_status, "index"))
        if self.session.settings.yolo:
            parts.append(("yolo", "warn"))
        if show_elapsed:
            parts.extend(
                [
                    ("step " + str(self.session.state.turn_step) + "/" + str(self.session.settings.max_steps), "runtime"),
                    ("tools " + str(self.session.state.turn_tool_calls), "runtime"),
                ]
            )
            if retry_status := self.retry_status():
                parts.append((retry_status, "warn"))
            elif attempt_status := self.model_attempt_status():
                parts.append((attempt_status, "warn"))
            elif self.model_elapsed() >= self.stress_after():
                parts.append(("/resend to re-request", "warn"))
        return parts

    def styled_fragments(self, entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
        fragments: list[tuple[str, str]] = []
        for index, (text, role) in enumerate(entries):
            if index:
                fragments.append((Theme.style("status.sep"), " | "))
            fragments.append((self.role_style(role), text))
        return fragments or [("", "")]

    def sweep_fragments(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return [("", "")]
        width = max(1, len(text) - 1)
        sweep = (time.monotonic() * 0.55) % 1.0
        model_elapsed = self.model_elapsed()
        heat = min(1.0, max(0.0, model_elapsed - self.stress_after()) / max(30.0, self.session.config.provider.timeout - self.stress_after()))
        fragments = []
        for index, char in enumerate(text):
            ratio = index / width
            red = round(75 + (180 - 75) * ratio)
            green = round(180 + (130 - 180) * ratio)
            blue = 235
            red = round(red + (240 - red) * heat)
            green = round(green * (1 - 0.65 * heat))
            blue = round(blue * (1 - 0.75 * heat))
            intensity = max(0.0, 1.0 - abs(ratio - sweep) * 5.0) ** 2
            red = round(red + (230 - red) * intensity)
            green = round(green + (245 - green) * intensity)
            blue = round(blue + (255 - blue) * intensity)
            fragments.append((f"#{red:02x}{green:02x}{blue:02x}", char))
        return fragments

    def index_status(self) -> str:
        if self.session.state.code_index_error:
            return CodeIndex.label("error")
        if self.session.state.code_index_refreshing:
            notice = self.session.state.code_index_notice or "syncing"
            return self.INDEX_SPINNER[int(time.monotonic() / self.INTERVAL) % len(self.INDEX_SPINNER)] if notice in {"syncing", "updating"} else notice
        return CodeIndex.label(self.session.state.code_index_status)

    def update_status(self) -> str:
        update = self.session.update
        if update.checking:
            return "update..."
        return "update " + update.latest if update.newer_than(__version__) else ""

    def mcp_status(self) -> str:
        if self.session.mcp is None:
            return ""
        configs = self.session.mcp.parse_configs()
        if not configs:
            return ""
        status = self.session.mcp.discovery_status
        if status == "discovering":
            spinner = self.INDEX_SPINNER[int(time.monotonic() / self.INTERVAL) % len(self.INDEX_SPINNER)]
            loaded, total = self.session.mcp.discovery_progress()
            return f"mcp {loaded}/{total}{spinner}"
        if status == "error":
            return "mcp err"
        if status != "ready":
            return ""
        # "!" flags that the tools index overflowed the cap and some tools are hidden.
        return f"mcp {len(self.session.mcp.tools)}{'!' if self.session.mcp.index_truncated else ''}"

    def stress_after(self) -> float:
        return max(30.0, self.session.config.provider.timeout * 0.5)


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

    def visible(self, lines: list[Any], height: int) -> list[Any]:
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


class CommandLoop:
    QUEUE_EMPTY_HINT = "Enter queues follow-up · Ctrl-C interrupts"
    QUEUE_PENDING_HINT = "↑ recalls queued · Ctrl-C interrupts"
    IDLE_HINTS = (
        "Ctrl-X Ctrl-E opens $EDITOR",
        "Type / for commands",
        "Ctrl-U clears the line",
    )
    TRANSCRIPT_DIFF_LINES: ClassVar[int] = 40
    EDITOR_CONTEXT_MAX_LINES: ClassVar[int] = 200
    # fmt: off
    COMMAND_HANDLERS: ClassVar[dict[str, str]] = {
        "/help": "help", "/status": "status", "/ps": "ps_command", "/diff": "diff_command",
        "/skills": "skills_command", "/config": "config",
        "/compact": "compact", "/index": "index", "/provider": "provider", "/model": "model",
        "/reason": "reason", "/set": "set_value", "/yolo": "yolo", "/strict": "strict",
        "/mcp": "mcp_command", "/resend": "resend_command",
    }
    COMMANDS: ClassVar[tuple[str, ...]] = tuple(COMMAND_HANDLERS) + ("/exit", "/quit")
    # fmt: on

    # Commands safe to run from the follow-up input while the agent works: read-only
    # views plus /yolo, whose single atomic flag flip the agent simply reads at the next approval.
    QUEUE_RUN_COMMANDS: ClassVar[frozenset[str]] = frozenset({"/help", "/status", "/skills", "/ps", "/mcp", "/diff", "/yolo", "/resend"})
    MODEL_CONFIGURED_LABEL = "---- Configured models ----"
    MODEL_DISCOVERED_LABEL = "---- Discovered models ----"
    MODEL_LABELS = frozenset((MODEL_CONFIGURED_LABEL, MODEL_DISCOVERED_LABEL))
    MCP_COMMANDS: ClassVar[dict[str, tuple[int, int, str]]] = {
        "connect": (1, sys.maxsize, "Usage: /mcp connect <server> [server ...]"),
        "disconnect": (1, 1, "Usage: /mcp disconnect <server>"),
        "tools": (0, 1, "Usage: /mcp tools [server]"),
    }
    MCP_HELP = "Try /mcp, /mcp connect <server> [server ...], /mcp disconnect <server>, or /mcp tools [server]"

    HELP = """Commands:
  /help              Show this help.
  /status            Show runtime status.
  /ps                Show active background jobs.
  /diff              Show latest edits and overall session diff.
  /skills            List installed skills (load with Skill(name) or reference inline with $name).
  /config            Show active config.
  /compact           Compact context now.
  /resend            Resend the in-flight model request (type it while a turn is working).
  /index [force]      Sync or rebuild code symbol index.
  /provider [NAME]   Select or show the active provider.
  /model [MODEL]     Select or set the active model.
  /reason [EFFORT]   Select or set reasoning effort.
  /set KEY VALUE     Set provider.* and runtime.*.
  /yolo              Toggle tool confirmations.
  /strict            Toggle strict tool-call schemas (OpenAI / DeepSeek).
  /mcp               Manage MCP server connections.
  /exit, /quit       Exit.
Mentions:
  @server[.tool]     Point the agent at an MCP server/tool in your message (tab-completes).
  $skill             Reference a skill in your message to load its instructions for that turn (tab-completes).
CLI:
  -c, --last, --latest       Resume the latest session in the current project.
  --resume [UID]             Resume a saved session; defaults to latest (last also works).
Tools:
  Read, InspectCode, Search, Edit, Bash, Job, Recall, Note, Ask, MCP, Skill.
  Skill(name) loads a skill's full instructions on demand (see the SKILLS section / $skill).
"""

    DIFF_MAX_BYTES: ClassVar[int] = 50_000
    DIFF_MAX_LINES: ClassVar[int] = 1_200

    @classmethod
    def bounded_diff(cls, text: str) -> tuple[str, bool]:
        if len(text.encode("utf-8")) <= cls.DIFF_MAX_BYTES and text.count("\n") <= cls.DIFF_MAX_LINES:
            return text, False
        clipped: list[str] = []
        length = 0
        for line in text.splitlines():
            line_bytes = len(line.encode("utf-8")) + 1
            if length + line_bytes > cls.DIFF_MAX_BYTES or len(clipped) >= cls.DIFF_MAX_LINES:
                break
            clipped.append(line)
            length += line_bytes
        return "\n".join(clipped), True

    @staticmethod
    def diff_counts(text: str) -> tuple[int, int]:
        added = removed = 0
        old_remaining = new_remaining = 0
        hunk_header = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
        for line in text.splitlines():
            if match := hunk_header.match(line):
                old_remaining = int(match.group(1) or 1)
                new_remaining = int(match.group(2) or 1)
            elif line.startswith("+") and new_remaining:
                added += 1
                new_remaining -= 1
            elif line.startswith("-") and old_remaining:
                removed += 1
                old_remaining -= 1
            elif line.startswith(" "):
                old_remaining = max(0, old_remaining - 1)
                new_remaining = max(0, new_remaining - 1)
        return added, removed

    def __init__(self, agent: Agent, input_fn=input, output_fn=print):
        self.agent = agent
        self.session = agent.session
        # A single idle-input tip, chosen once per session so the empty prompt stays stable
        # (no per-render flicker) while still surfacing different shortcuts across sessions.
        self._idle_hint = random.choice(self.IDLE_HINTS)
        self.input_fn = input_fn
        self.ui = UiPrinter(output_fn)
        self.status_bar = StatusBar(self.session)
        self.live_preview = BashLivePreview()
        self.live_status_paused = False
        self.background_output_lock = threading.Lock()
        self.background_output_open = True
        self.interactive_input = input_fn is input and sys.stdin.isatty()
        # Set by run_tui() while the full-TUI shell is active; tool_input reroutes through it so
        # approval prompts land in the same input widget the user is already typing in.
        self.tui: TuiApp | None = None
        if self.interactive_input:
            history_path = self.session.data_path("history.txt")
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            self.input_history = FileHistory(history_path)
        else:
            self.input_history = None
        self.input_completer = CommandCompleter(
            providers=lambda: tuple(sorted(self.session.config.providers)),
            models=lambda: self.session.config.provider.available_models,
            mcp_servers=lambda: tuple(config.name for config in self.session.mcp.parse_configs()),
            mcp_connected_servers=lambda: tuple(config.name for config in self.session.mcp.parse_configs() if self.session.mcp.connected(config.name)),
            mcp_tools=lambda server: tuple(tool.name for tool in self.session.mcp.tools.get(server, [])),
            skills=lambda: tuple(skill.name for skill in self.session.skills.all()) if self.session.skills else (),
        )
        self.agent.output_fn = self.agent_output
        self.agent.on_queue_flush = self.flush_queued_to_log
        self.agent.tools.output_fn = self.tool_output
        self.agent.tools.input_fn = self.tool_input
        self.agent.tools.live_start = self.tool_live_start
        self.agent.tools.live_output = self.tool_live_output
        self.agent.tools.question_fn = self.question_interaction

    def flush_queued_to_log(self, texts: list[str]) -> None:
        # Move flushed queued messages from the live activity region into terminal scrollback.
        texts = [text for text in texts if text.strip()]
        if not texts:
            return
        fragments: list[tuple[str, str]] = [("", "\n")]
        for index, text in enumerate(texts):
            if index:
                fragments.append(("", "\n"))
            fragments.extend([("class:prompt", UiPrinter.USER_LOG_PREFIX), (UiPrinter.user_log_style(), text), ("", "\n")])
        fragments.append(("", "\n"))
        print_formatted_text(FormattedText(fragments), style=self.style(), end="", flush=True)

    # Breathing green dot shown on the working divider while a model request is in flight — it sits
    # just before the "working (…)" label and vanishes as soon as the response returns (non-streaming
    # client, so response return is the analogue of "first token arrives"). Palette dim → bright green.
    WAITING_PULSE_STYLES: ClassVar[tuple[str, ...]] = (
        "fg:#0a3d0a",
        "fg:#146114",
        "fg:#1f8a1f",
        "fg:#2dbf2d bold",
        "fg:#43e043 bold",
        "fg:#7bff7b bold",
    )
    WAITING_PULSE_PERIOD: ClassVar[float] = 1.6

    def waiting_pulse_fragments(self) -> list[tuple[str, str]]:
        if self.session.state.current_model_call_started_at <= 0:
            return []
        # Triangular breath: 0 → 1 → 0 over WAITING_PULSE_PERIOD seconds, mapped onto the palette.
        phase = (time.monotonic() % self.WAITING_PULSE_PERIOD) / self.WAITING_PULSE_PERIOD
        intensity = 1.0 - abs(2.0 * phase - 1.0)
        idx = min(len(self.WAITING_PULSE_STYLES) - 1, int(intensity * len(self.WAITING_PULSE_STYLES)))
        return [(self.WAITING_PULSE_STYLES[idx], "● ")]

    QUEUE_SWEEP_CELLS_PER_SEC: ClassVar[float] = 34.0
    # A comet: a bright head with a fading tail, by distance from the head. Beyond the tail the dash
    # falls back to the dim rule. The divider is only ever drawn while working, so there is no idle look.
    GLOW_STYLES: ClassVar[tuple[str, ...]] = (
        "class:divider.glow0",
        "class:divider.glow1",
        "class:divider.glow2",
        "class:divider.glow3",
        "class:divider.glow4",
    )

    def sweep_divider_fragments(self, label: str, width: int | None = None, prefix: list[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
        prefix = prefix or []
        prefix_len = sum(len(text) for _style, text in prefix)
        cols = shutil.get_terminal_size((80, 20)).columns
        width = width if width is not None else max(20, min(52, cols - 2))
        body_len = prefix_len + len(label) + 2  # prefix + " label "
        lead = 3
        trail = max(3, width - lead - body_len)
        dash_count = lead + trail
        # The comet head bounces over the horizontal rule only. The label stays stable and readable
        # while the glow appears to pass through the dash track on either side.
        span = max(1, dash_count - 1)
        phase = time.monotonic() * self.QUEUE_SWEEP_CELLS_PER_SEC % (2 * span)
        head = phase if phase <= span else 2 * span - phase

        def dashes(offset: int, count: int) -> list[tuple[str, str]]:
            fragments = []
            for i in range(count):
                distance = round(abs(offset + i - head))
                fragments.append((self.GLOW_STYLES[distance] if distance < len(self.GLOW_STYLES) else "class:queue.rule", "-"))
            return fragments

        return [
            *dashes(0, lead),
            ("class:queue.rule", " "),
            *prefix,
            ("class:divider.working", label),
            ("class:queue.rule", " "),
            *dashes(lead, trail),
        ]

    def queue_divider_fragments(self, queued: int = 0) -> list[tuple[str, str]]:
        status = self.tui.status_label if self.tui is not None and self.tui.status_label else "working"
        if status == "working":
            retry_status = self.status_bar.retry_status()
            attempt_status = self.status_bar.model_attempt_status()
            activity = retry_status or ("working · " + attempt_status if attempt_status else "working")
            label = f"{activity} ({Text.elapsed_since(self.status_bar.started_at)})"
        else:
            label = status
        label = f"{label} [ {queued} queued ]" if queued else label
        return self.sweep_divider_fragments(label, prefix=self.waiting_pulse_fragments())

    def queue_region_fragments(self) -> list[tuple[str, str]]:
        with self.session._queue_lock:
            pending = list(self.session.pending_user_inputs)
        # The divider is a standing boundary for the whole turn: flushed messages move up into the log
        # above it, so it stays put even once the queue empties rather than vanishing.
        fragments = self.queue_divider_fragments(len(pending))
        for item in pending:
            marker = "→ " if item.inflight else "+ "
            for index, line in enumerate(item.text.splitlines()):
                fragments.extend([("", "\n"), (UiPrinter.user_log_style(), (marker if index == 0 else "  ") + line)])
        return fragments

    def tui_activity_fragments(self) -> list[tuple[str, str]]:
        with self.live_preview.lock:
            lines = self.live_preview.frame_lines() if self.live_preview.active else []
        fragments = []
        for line in lines:
            fragments.extend([("ansibrightblack", line), ("", "\n")])
        if lines:
            fragments.append(("", "\n"))
        fragments.extend(self.queue_region_fragments())
        return fragments

    def tui_input_hint(self) -> str:
        if self.tui is None:
            return ""
        if self.tui.input_mode == "running":
            with self.session._queue_lock:
                has_pending = any(not item.inflight for item in self.session.pending_user_inputs)
            return self.QUEUE_PENDING_HINT if has_pending else self.QUEUE_EMPTY_HINT
        if self.tui.input_mode == "chat":
            return self._idle_hint
        return ""

    def editor_context(self) -> str:
        """The agent's most recent reply, restated as read-only reference inside the external
        editor (Ctrl-X Ctrl-E / Ctrl-G): the full-screen editor hides the terminal scrollback
        the reply is printed into. Long replies keep only their most recent lines so the
        editor's temp file stays small."""
        for message in reversed(self.session.messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                lines = content.strip().splitlines()
                if len(lines) > self.EDITOR_CONTEXT_MAX_LINES:
                    drop = len(lines) - self.EDITOR_CONTEXT_MAX_LINES
                    lines = ["# [... earlier lines of the reply omitted ...]"] + lines[drop:]
                return "\n".join(lines)
        return ""

    def run_queued_command(self, text: str) -> None:
        """Dispatch a read-only slash command while an agent turn is running."""
        name = text.partition(" ")[0]
        if name not in self.QUEUE_RUN_COMMANDS:
            self.emit(f"{name} is unavailable while the agent is working; press Ctrl-C to run it.")
            return
        if name == "/mcp":
            sub = text.partition(" ")[2].split()
            if sub and sub[0] != "tools":
                self.emit("Only read-only /mcp (status, tools) is available while the agent is working.")
                return
        self.command(text)

    def take_pending_inputs(self) -> list[str]:
        """Remove and return queued inputs that are not currently being flushed."""
        with self.session._queue_lock:
            texts = [item.text for item in self.session.pending_user_inputs if not item.inflight]
            self.session.pending_user_inputs = [item for item in self.session.pending_user_inputs if item.inflight]
        return texts

    def recall_pending_input(self, on_inflight: Callable[[], None]) -> str:
        """Move the newest queued input back to the editor, retrying if it was already claimed."""
        with self.session._queue_lock:
            item = next(reversed(self.session.pending_user_inputs), None)
            if item is None:
                return ""
            self.session.pending_user_inputs.remove(item)
            was_inflight = item.inflight
            if was_inflight:
                for pending_item in self.session.pending_user_inputs:
                    pending_item.inflight = False
        if was_inflight:
            on_inflight()
        self.session.save_snapshot()
        return item.text

    def run(self) -> int:
        # Interactive terminals use the full TUI; injected/non-TTY callers use the simple REPL.
        if self.interactive_input:
            return self.run_tui()
        self.start_session()
        while True:
            try:
                entered = self.take_pending_inputs()
                user_input = self.read_input(initial_text="\n".join(entered))
            except EOFError:
                self.emit(TurnBox.SEPARATOR)
                self.save_and_emit_resume()
                return 0
            except KeyboardInterrupt:
                self.emit("Cancelled")
                continue
            if not user_input.strip():
                continue
            handled, exit_now = self.command(user_input.strip())
            if exit_now:
                return 0
            if handled:
                continue
            self.emit("")
            started = time.monotonic()
            try:
                self.status_bar.start()
                try:
                    answer = self.agent.run(user_input)
                except KeyboardInterrupt:
                    self.emit("Cancelled")
                    continue
                except MinacodeError as error:
                    answer = f"Error: {error}"
            finally:
                CodeIndex(self.session).update_pending_async()
                self.status_bar.stop()
            elapsed = time.monotonic() - started
            self.ui.emit_answer(answer)
            self.emit(f"[done in {int(elapsed // 60)}m{elapsed % 60:.0f}s]")
            self.session.save_snapshot()

    def start_session(self) -> None:
        """Initialize output and background services shared by both command-loop frontends."""
        self.emit(f"minacode {__version__}. /help for commands.")
        UpdateChecker(self.session).start()
        if self.session.update.newer_than(__version__):
            self.emit(f"update available: {__version__} -> {self.session.update.latest}. upgrade with `uv tool upgrade minacode`.")
        SessionSnapshotStore.clean_expired(self.session)
        self.render_resumed_session()
        CodeIndex(self.session).refresh_existing_async()
        # Discover auto_connect servers in the background so an unreachable one cannot block the
        # prompt for the discovery timeout; the tools index picks them up as they connect.
        threading.Thread(target=self.session.mcp.discover_auto, name="mcp-discover", daemon=True).start()

    def run_tui(self) -> int:
        return TuiRuntime(self).run()

    def render_resumed_session(self) -> None:
        # Transcript reconstruction owns historical call/result matching and ordering invariants.
        if not self.session.resumed:
            return
        self.session.resumed = False
        # The percent is derived, not persisted, so a resumed session carries a full history with a
        # zeroed reading. Recompute it now or the status bar reports 0% until the first turn.
        self.agent.context.update_current_percent(self.agent.SYSTEM_PROMPT)
        messages = [message for message in self.session.messages if not SessionSnapshotCodec.is_internal_message(message) and message.get("role") != "tool"]
        if not messages:
            return
        self.emit(f"Restored session: {self.session.uid}")
        diffs = {diff.key: diff.diff for diff in self.session.turn_diffs if diff.key and diff.diff}
        tool_record_index = 0
        for index, turn in enumerate(TurnBox.group(messages)):
            if index:
                self.emit("")
            for message in turn.messages:
                tool_record_index = self.render_transcript_message(message, tool_record_index, diffs)
        self.render_remaining_tool_records(tool_record_index, diffs)

    def render_transcript_message(self, message: Json, tool_record_index: int = 0, diffs: dict[str, str] | None = None) -> int:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "").strip()
        raw_calls = message.get("tool_calls")
        has_tool_calls = isinstance(raw_calls, list) and bool(raw_calls)
        if role == "assistant" and content:
            self.ui.emit_answer(content, role=role, rule=False, indent=TurnBox.CONTENT_LEVEL if has_tool_calls else TurnBox.ROOT_LEVEL)
        if role == "assistant":
            return self.render_transcript_tool_calls(message, tool_record_index, diffs or {})
        if role == "user" and content:
            self.ui.emit_answer(content, role=role, rule=False)
        return tool_record_index

    def render_transcript_tool_calls(self, message: Json, tool_record_index: int, diffs: dict[str, str]) -> int:
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            return tool_record_index
        for raw in raw_calls:
            call = self.transcript_tool_call(raw)
            if call is None:
                continue
            record, tool_record_index = self.transcript_tool_record(call, tool_record_index)
            self.emit_transcript_tool(call, record.key if record else "", diffs)
        return tool_record_index

    def render_remaining_tool_records(self, tool_record_index: int, diffs: dict[str, str]) -> None:
        for record in self.session.tool_records[tool_record_index:]:
            call = ToolCall(id="", name=record.name, args=record.args)
            self.emit_transcript_tool(call, record.key, diffs)

    def emit_transcript_tool(self, call: ToolCall, key: str, diffs: dict[str, str]) -> None:
        """An Edit shows the diff it made, the way it did when the edit ran live. Live, that preview
        comes from the approval block; here the stored diff text is the same string, so replaying it
        needs no reconstruction."""
        preview = diffs.get(key, "") if call.name == "Edit" else ""
        if not preview:
            self.emit(self.agent.tools.finish_display(call, key, "", failed=False))
            return
        # The preview block carries the call line, so the result collapses to its trailing marker
        # underneath it — the same nesting the live approval block produces.
        self.emit(self.transcript_edit_preview(call, preview))
        self.emit(self.agent.tools.finish_display(call, key, "", failed=False, d=ToolDisplay(nested_display=True)))

    def transcript_edit_preview(self, call: ToolCall, preview: str) -> LogBlock:
        tools = self.agent.tools
        lines = preview.rstrip().splitlines()
        # A long replay would bury the prompt under diffs, so each one is trimmed to a readable
        # window; `/diff` still holds the full text.
        hidden = max(0, len(lines) - self.TRANSCRIPT_DIFF_LINES)
        if hidden:
            lines = lines[: self.TRANSCRIPT_DIFF_LINES]
        children = [LogLine("preview", role=LogRole.META, edge=LogEdge.BRANCH)]
        children.extend(LogLine("", line, LogRole.DIFF, LogEdge.CONTINUE) for line in lines)
        if hidden:
            children.append(LogLine("", f"… {hidden} more lines, see /diff", LogRole.META, LogEdge.CONTINUE))
        return LogBlock.hierarchy(tools.log_root(tools.short_call(call), LogRole.AUTO, "", call), children)

    @staticmethod
    def transcript_tool_call(raw: Any) -> ToolCall | None:
        if not isinstance(raw, dict):
            return None
        function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
        name = str(function.get("name") or "")
        if not name:
            return None
        arguments = function.get("arguments")
        try:
            # strict=False tolerates literal newlines in argument strings (e.g. multi-line
            # git commit messages) that would otherwise be rejected as invalid JSON.
            payload = json.loads(arguments, strict=False) if isinstance(arguments, str) else (arguments or {})
        except json.JSONDecodeError:
            payload = {}
        try:
            args = ModelClient.tool_payload(name, payload)
        except ToolError:
            # A malformed historical call (e.g. tool args that fail validation) must not crash
            # the resume; render it without parsed args.
            args = [payload] if payload else []
        return ToolCall(id=str(raw.get("id") or ""), name=name, args=args)

    def transcript_tool_record(self, call: ToolCall, tool_record_index: int) -> tuple[ToolResultRecord | None, int]:
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is not None and not tool_class.STORES_RESULT:
            return None, tool_record_index
        records = self.session.tool_records
        while tool_record_index < len(records):
            record = records[tool_record_index]
            tool_record_index += 1
            if record.name == call.name:
                return record, tool_record_index
        return None, tool_record_index

    def save_and_emit_resume(self) -> None:
        uid = self.session.save_snapshot()
        if uid:
            self.emit(f"Resume with:\nminacode --resume {uid}")

    def style(self) -> Style:
        return Style.from_dict(
            {
                "prompt": "ansicyan bold",
                "queue.rule": "ansibrightblack",
                "queue.hint": "ansibrightblack",
                "divider.working": "ansimagenta bold",
                # Comet gradient: bright head fading through cyan into the dim rule.
                "divider.glow0": "ansibrightcyan bold",
                "divider.glow1": "ansicyan bold",
                "divider.glow2": "ansicyan",
                "divider.glow3": "ansibrightblack",
                "divider.glow4": "ansibrightblack",
                "approval": "ansiyellow",
                "approval.wait": "ansimagenta",
                "choice.title": "ansicyan bold",
                "choice.selected": "reverse",
                "choice.disabled": "ansibrightblack",
                "choice.preview": "ansigreen italic",
                "choice.status.connected": "ansigreen bold",
                "choice.status.connecting": "ansigreen bold",
                "choice.status.disconnected": "ansiyellow bold",
                "choice.status.disconnecting": "ansiyellow bold",
                "choice.status.error": "ansired bold",
                "choice.status.skipped": "ansibrightblack",
                "tab.active": "bold reverse ansicyan",
                "tab.inactive": "ansicyan",
                "completion-menu": "noreverse bg:default",
                "completion-menu.completion": "noreverse bg:default fg:default",
                "completion-menu.completion.current": "noreverse bg:default fg:ansicyan bold",
                "completion-menu.meta.completion": "noreverse bg:default fg:ansibrightblack",
                "completion-menu.meta.completion.current": "noreverse bg:default fg:ansicyan",
                "bottom-toolbar": "noreverse bg:default fg:default",
                "bottom-toolbar.text": "noreverse bg:default fg:default",
                "search-toolbar": "noreverse bg:default fg:default",
                "search-toolbar.prompt": "ansicyan",
                "search-toolbar.text": "fg:default",
            }
        )

    def read_input(
        self,
        prompt_text: str = UiPrinter.PROMPT_PREFIX,
        *,
        initial_text: str = "",
    ) -> str:
        """Read from the injected/non-TTY input path; interactive terminals use TuiApp."""
        return initial_text or self.input_fn(prompt_text)

    def emit(self, text: str | LogBlock = "") -> None:
        self.ui.emit(text)

    def emit_background(self, text: str) -> None:
        """Emit from a daemon worker only while this loop still owns terminal output."""
        with self.background_output_lock:
            if self.background_output_open:
                self.emit(text)

    def close_background_output(self, final_output: Callable[[], None] | None = None) -> None:
        with self.background_output_lock:
            self.background_output_open = False
            if final_output is not None:
                final_output()

    def with_status_paused(self, action):
        # Only quiet the standalone status-bar thread used by the simple/non-TTY path. The full TUI
        # renders status and output together, so it never needs this terminal-level coordination.
        was_running = self.status_bar.is_running()
        if was_running:
            self.status_bar.stop()
        try:
            return action()
        finally:
            if was_running:
                self.status_bar.start(reset=False)

    def tool_output(self, text: str | LogBlock = "") -> None:
        self.with_status_paused(lambda: self.emit(text))

    def agent_output(self, text: str = "") -> None:
        self.with_status_paused(lambda: self.emit_agent_output(text))

    def tool_input(self, prompt: str = "") -> str:
        # When the TUI is running, route agent approvals through TuiApp's
        # own input widget so the user answers inline in the persistent shell instead of a
        # separate pt Application (which would fail because pt does not nest).
        if self.tui is not None:
            return self.tui.request_input(prompt)

        return self.with_status_paused(lambda: self.input_fn(prompt))

    def emit_agent_output(self, text: str) -> None:
        if self.ui.color and text.strip():
            self.emit()
            self.ui.emit_answer(text, rule=False, indent=TurnBox.CONTENT_LEVEL)
            self.emit()
            return
        self.ui.emit_answer(text, rule=False, indent=TurnBox.CONTENT_LEVEL)

    def _begin_cli_preview(self) -> None:
        """Pause the status bar if running and start the CLI Bash live-preview line."""
        self.live_status_paused = self.status_bar.is_running()
        if self.live_status_paused:
            self.status_bar.stop()
        self.live_preview.start()

    def tool_live_start(self) -> None:
        if not self.ui.color:
            return
        if self.tui is not None:
            with self.live_preview.lock:
                self.live_preview.active = True
                self.live_preview.text = ""
                self.live_preview.started_at = time.monotonic()
            self.tui.invalidate()
            return
        self._begin_cli_preview()

    def tool_live_output(self, _stream: str, text: str) -> None:
        if not self.ui.color:
            return
        if self.tui is not None:
            with self.live_preview.lock:
                if text:
                    self.live_preview.active = True
                    self.live_preview.text = (self.live_preview.text + text)[-self.live_preview.MAX_CHARS :]
                else:
                    self.live_preview.active = False
                    self.live_preview.text = ""
            self.tui.invalidate()
            return
        if text:
            if not self.live_preview.active:
                self._begin_cli_preview()
            self.live_preview.update(text)
            return
        if self.live_preview.active:
            self.live_preview.finish()
        if self.live_status_paused:
            self.status_bar.start(reset=False)
            self.live_status_paused = False

    def command(self, text: str) -> tuple[bool, bool]:
        if text in {"/exit", "/quit", "exit", "quit"}:
            self.save_and_emit_resume()
            return True, True
        if not text.startswith("/"):
            return False, False
        name, _, args = text.partition(" ")
        method_name = self.COMMAND_HANDLERS.get(name)
        handler = getattr(self, method_name, None) if method_name else None
        output = handler(args.strip()) if handler else f"Unknown command: {name}"
        # A None result means the handler already rendered its own UI (e.g. /diff's viewer).
        if output is not None:
            (self.ui.emit_answer if name in {"/status", "/ps", "/mcp", "/skills", "/diff"} else self.emit)(output)
        return True, False

    def resend_command(self, _args: str) -> str | None:
        """Resend the in-flight model request. Available only in the running queue-input region:
        typed while a turn works, it re-requests the current model call (same path as on_retry)."""
        if self.tui is None or self.tui.input_mode != "running":
            return "/resend re-requests the current model request — type it while a turn is working."
        if self.session.state.current_model_call_started_at <= 0:
            return "Nothing to resend right now; /resend works while the model is generating."
        self.tui.on_retry()
        return None

    def mcp_command(self, args: str) -> str | None:
        mcp = self.session.mcp
        if mcp is None:
            return "MCP not configured"

        parts = args.split()
        if not parts:
            if self.tui is not None and self.tui.input_mode != "running":
                return self.mcp_manager()
            return mcp.render_server_status()

        sub = parts[0]
        rest = parts[1:]
        command = self.MCP_COMMANDS.get(sub)
        if command is None:
            return f"Unknown /mcp subcommand: {sub}. {self.MCP_HELP}"
        min_args, max_args, usage = command
        if not min_args <= len(rest) <= max_args:
            return usage

        if sub == "connect":
            return mcp.connect_servers(rest, interactive=self.interactive_input, notify=self.emit)
        if sub == "disconnect":
            return mcp.disconnect_server(rest[0])
        if sub == "tools":
            return mcp.render_tool_listing(rest[0] if rest else None)
        raise AssertionError("unreachable MCP subcommand")

    def mcp_manager(self) -> None:
        mcp = self.session.mcp
        tui = self.tui
        if mcp is None or tui is None:
            return
        configs = tuple(mcp.parse_configs())
        if not configs:
            self.ui.emit_answer(mcp.render_server_status())
            return

        state = ChoiceViewState(tuple(config.name for config in configs), {}, set())
        transitions: dict[str, str] = {}
        errors: dict[str, str] = {}
        state_lock = threading.Lock()
        modal_open = threading.Event()
        modal_open.set()

        def server_labels() -> dict[str, str]:
            with state_lock:
                changing = dict(transitions)
                failed = dict(errors)
            server_rows = []
            for config in configs:
                if transition := changing.get(config.name):
                    status = mcp.STATUS_MARKER + " " + transition
                elif config.name in failed:
                    status = mcp.STATUS_MARKER + " error"
                elif issue := mcp.server_issue(config.name):
                    status = mcp.STATUS_MARKER + " " + issue[0]
                elif mcp.connected(config.name):
                    status = mcp.STATUS_MARKER + " connected"
                else:
                    status = mcp.STATUS_MARKER + " disconnected"
                mode = "auto" if config.auto_connect else "manual"
                count = len(mcp.tools.get(config.name, []))
                server_rows.append((config.name, status, mode, count))
            name_width = max(len(name) for name, *_rest in server_rows)
            status_width = max(len(mcp.STATUS_MARKER + " disconnecting"), *(len(status) for _name, status, _mode, _count in server_rows))
            return {name: f"{name:<{name_width}}  {status:<{status_width}}  {mode:<6}  {count:>3} tools" for name, status, mode, count in server_rows}

        def preview(name: str) -> str:
            with state_lock:
                if message := errors.get(name):
                    return message
            if issue := mcp.server_issue(name):
                return issue[1]
            return ""

        def fragments() -> list[tuple[str, str]]:
            state.labels = server_labels()
            return state.fragments("MCP servers · Enter toggles connection", preview)

        def toggle(name: str, connect: bool) -> None:
            try:
                if connect:
                    result = mcp.connect_server(name, interactive=True, notify=self.emit)
                else:
                    result = mcp.disconnect_server(name)
            except Exception as error:  # Keep background failures visible in the selector.
                result = f"MCP server error: {name}: {error}"

            succeeded = mcp.connected(name) == connect
            with state_lock:
                transitions.pop(name, None)
                if succeeded:
                    errors.pop(name, None)
                else:
                    errors[name] = result
            if modal_open.is_set():
                tui.invalidate()
            else:
                self.emit_background(result)

        def handle_key(key: str, data: str = "") -> Any:
            result = state.handle_key(key, data)
            if not isinstance(result, str):
                return result
            with state_lock:
                if result in transitions:
                    return TUI_MODAL_PENDING
                connect = not mcp.connected(result)
                errors.pop(result, None)
                transitions[result] = "connecting" if connect else "disconnecting"
            threading.Thread(target=toggle, args=(result, connect), name="mcp-toggle-" + result, daemon=True).start()
            return TUI_MODAL_PENDING

        try:
            tui.show_modal(fragments, handle_key)
        finally:
            modal_open.clear()

    def select_choice(
        self,
        title: str,
        choices: tuple[str, ...],
        *,
        labels: dict[str, str] | None = None,
        current: str = "",
        disabled: set[str] | frozenset[str] = frozenset(),
    ) -> str | object | None:
        labels = labels or {}
        if not choices or not self.interactive_input:
            return None
        enabled = tuple(choice for choice in choices if choice not in disabled)
        if len(enabled) == 1:
            return enabled[0]
        try:
            return self.choice_application(title, choices, labels, current, set(disabled))
        except (EOFError, KeyboardInterrupt):
            self.emit("Cancelled")
            return None

    def choice_application(
        self,
        title: str,
        choices: tuple[str, ...],
        labels: dict[str, str],
        current: str,
        disabled: set[str],
        *,
        preview_fn: Callable[[str], str] | None = None,
        free_text: bool = False,
    ) -> str | object | None:
        if free_text and self.interactive_input:
            choices = (*choices, ChoiceViewState.FREE_TEXT)
            labels = {**labels, ChoiceViewState.FREE_TEXT: "Type freely..."}
        state = ChoiceViewState(choices, labels, disabled)
        options = state.enabled()
        state.selected = options.index(current) if current in options else 0
        if self.tui is None:
            return None
        result = self.tui.show_modal(lambda: state.fragments(title, preview_fn), state.handle_key)
        if isinstance(result, KeyboardInterrupt):
            raise result
        return result

    def question_application(self, spec: AskSpec, position: str = "") -> str:
        """Ask via the shared choice selector, with dynamic previews and a free-text fallback."""
        choices = spec.choices
        # Prefix the position (e.g. "(1/3) ...") into the question text so it renders as plain
        # markdown — no separate styled line, hence no ANSI escapes to mangle.
        prompt = f"({position}) {spec.question}" if position else spec.question
        if not choices:
            return self.tui.request_input("\n" + prompt) if self.tui is not None else self.read_input("\n" + prompt)
        if not self.interactive_input:
            return self.read_input("\n" + prompt)

        # Blank separator line before each question so multi-question prompts don't run together.
        if self.ui.color:
            self.emit("")
            self.ui.emit_markdown(prompt)
        else:
            self.emit("\n" + prompt + "\n")

        # An optional recommended choice is pre-selected (via current) and marked (via labels),
        # reusing the selector's existing machinery.
        labels, current = {}, ""
        if spec.recommended is not None and 0 <= spec.recommended < len(choices):
            current = choices[spec.recommended]
            labels = {current: current + " (recommended)"}
        previews = spec.previews
        preview_map = {c: previews[i] for i, c in enumerate(choices) if previews and i < len(previews) and previews[i]}
        result = self.choice_application(
            "Select:",
            tuple(choices),
            labels,
            current,
            set(),
            preview_fn=lambda choice: preview_map.get(choice, ""),
            free_text=True,
        )
        if result is SELECTION_FREE_TEXT:
            # The question was already rendered before the choice selector; do not repeat a long
            # raw prompt when the user switches to free text.
            self.emit("")
            return self.tui.request_input("> ") if self.tui is not None else self.read_input("> ")
        if isinstance(result, str):
            return result
        return DISMISSED  # SELECTION_BACK (Esc) — user declined to answer

    def question_interaction(self, spec: AskSpec, position: str = "") -> str:
        """Entry point for Ask; the final tool log renders the returned answer."""
        return self.question_application(spec, position)

    def select_reasoning(self) -> str | object | None:
        current = self.session.config.provider.reasoning
        labels = {"off": "off - disable reasoning"}
        labels[current] = labels.get(current, current) + " (current)"
        return self.select_choice("Reasoning effort", REASONING_CHOICES, labels=labels, current=current)

    def help(self, args: str) -> str:
        return self.HELP.rstrip()

    def status(self, args: str) -> str:
        def progress_bar(value: int, total: int, width: int = 20) -> str:
            ratio = min(1.0, max(0.0, value / total)) if total else 0.0
            eighths = int(ratio * width * 8 + 0.5)
            full, partial = divmod(eighths, 8)
            partials = "▏▎▍▌▋▊▉"
            return "[" + "█" * full + (partials[partial - 1] if partial else "") + "░" * (width - full - bool(partial)) + "]"

        def token_count(value: int) -> str:
            if value >= 1_000_000:
                return f"{value / 1_000_000:.1f}M"
            if value >= 1_000:
                return f"{value / 1_000:.1f}K"
            return str(value)

        usage = self.session.usage
        provider = self.session.config.provider
        context_tokens = self.agent.context.update_current_tokens(self.agent.SYSTEM_PROMPT)
        context_budget = self.agent.context.request_token_budget()
        index = CodeIndex(self.session)
        index_status, index_message = index.status(check=False)
        index.update_pending_async()
        if self.session.state.code_index_refreshing:
            index_status, index_message = self.session.state.code_index_notice or "syncing", ""
        elif self.session.state.code_index_error:
            index_status, index_message = "error", self.session.state.code_index_error
        if index_status in {"missing", "unavailable", "error"} and "run /index" not in index_message:
            index_message = (index_message + "; " if index_message else "") + "run /index"
        elif index_status == "stale" and "run /index" not in index_message:
            index_message = (index_message + "; " if index_message else "") + "run /index or wait for auto update"
        cache_ratio = (usage.cached_prompt_tokens * 100 / usage.prompt_tokens) if usage.prompt_tokens else 0
        last_cache_ratio = (usage.last_cached_prompt_tokens * 100 / usage.last_prompt_tokens) if usage.last_prompt_tokens else 0
        connected_mcp = sum(self.session.mcp.connected(config.name) for config in self.session.mcp.parse_configs()) if self.session.mcp else 0
        # fmt: off
        rows = [
            ("workspace", "`" + self.session.cwd + "`"),
            ("session", "`" + self.session.uid + "`"),
            ("model", f"`{self.session.config.active_provider}/{provider.model or '(empty)'}`; api `{provider.resolved_api()} ({provider.api})`; reasoning `{provider.reasoning} ({provider.resolved_chat_reasoning()})`"),
            ("context", f"`{progress_bar(context_tokens, context_budget)}` `~{token_count(context_tokens)} / {token_count(context_budget)} ({self.session.state.context_percent}%)`; history `{len(self.session.messages)}`; turn `{self.session.state.turn_messages}`; tools `{len(self.session.tool_results)}`; mcp `{connected_mcp}`; skills `{len(self.session.skills.skills) if self.session.skills else 0}`; known `{len(self.session.state.known)}`; compactions `{self.session.state.compaction_count}`"),
            ("cache", f"`{progress_bar(usage.last_cached_prompt_tokens, usage.last_prompt_tokens)}` last `{token_count(usage.last_cached_prompt_tokens)} / {token_count(usage.last_prompt_tokens)} ({last_cache_ratio:.1f}%)`; session `{token_count(usage.cached_prompt_tokens)} / {token_count(usage.prompt_tokens)} ({cache_ratio:.1f}%)`"),
            ("goal", self.session.state.goal or "(empty)"),
            ("usage", f"calls `{usage.calls}`; total `{usage.total_tokens}`"),
            ("runtime", f"yolo `{'on' if self.session.settings.yolo else 'off'}`; max steps `{self.session.settings.max_steps}`"),
            ("index", CodeIndex.status_line(index_status, index_message)),
            ("jobs", f"running `{len(self.session.running_jobs())}`; total `{len(self.session.jobs)}`"),
            ("update", UpdateChecker(self.session).status_line().removeprefix("update: ")),
        ]
        # fmt: on
        return "\n".join(
            [
                "| status | value |",
                "| --- | --- |",
                *(f"| {name} | {Text.clean(str(value)).replace(chr(10), ' ').replace('|', chr(92) + '|')} |" for name, value in rows),
            ]
        )

    def skills_command(self, args: str) -> str:
        library = self.session.skills
        skills = library.all() if library else []
        if not skills:
            return "No skills installed. Add `<name>/SKILL.md` under `.minacode/skills/` (project) or `~/.minacode/skills/` (user)."
        table = ContextManager.md_table(
            ["skill", "source", "description"],
            [(f"`{skill.name}`", skill.source, skill.description or "(no description)") for skill in skills],
        )
        return "\n".join([f"### Skills · {len(skills)}", "", "Load with `Skill(name)` or reference inline with `$name`.", "", table])

    def ps_command(self, args: str) -> str:
        if args.strip():
            return "Usage: /ps"
        running = self.session.running_jobs()
        if not running:
            total = len(self.session.jobs)
            return f"No active jobs ({total} total)."
        rows = [(job.id, job.status, f"{job.elapsed():.1f}s", job.command[:80]) for job in running]
        table = ContextManager.md_table(["id", "status", "elapsed", "command"], rows)
        return f"### Active jobs · {len(running)}\n\n{table}"

    def bash_output_viewer(self) -> None:
        """Browse recent completed Bash previews without copying them into scrollback."""
        if self.tui is None:
            return
        records = []
        for record in reversed(self.session.tool_records):
            if record.name != "Bash":
                continue
            preview = self.agent.tools.bash_result_preview(record.output)
            if preview:
                records.append((record, preview))
            if len(records) == 10:
                break
        if not records:
            return
        width = max(20, shutil.get_terminal_size((120, 20)).columns - 12)
        labels = {}
        calls = {}
        for index, (record, _preview) in enumerate(records):
            call = self.agent.tools.short_call(ToolCall("", "Bash", record.args))
            choice = str(index)
            calls[choice] = call
            labels[choice] = Text.clip_width(f"{record.key}  {call}", width)
        choices = tuple(labels)
        state = ChoiceViewState(choices, labels, set())
        opened: str | None = None

        def rule(label: str) -> list[tuple[str, str]]:
            cols = shutil.get_terminal_size((80, 20)).columns
            rule_width = max(20, min(72, cols - 2))
            lead = "──── "
            trail = " " + "─" * max(3, rule_width - get_cwidth(lead + label) - 1)
            return [("", "\n"), ("class:choice.disabled", lead + label + trail + "\n")]

        def fragments() -> list[tuple[str, str]]:
            if opened is None:
                list_fragments = state.fragments("")
                return [*rule(f"Bash outputs · latest {len(records)}"), *list_fragments[1:]]
            record, preview = records[int(opened)]
            detail_width = max(20, shutil.get_terminal_size((120, 20)).columns - 6)
            parts = [*rule(f"Bash output · {record.key}"), ("ansibrightblack", f"  {Text.clip_width(calls[opened], detail_width)}\n\n")]
            parts.extend(("ansibrightblack", f"  {Text.clip_width(line, detail_width)}\n") for line in preview.splitlines())
            parts.append(("class:choice.disabled", "\n  Esc / ← back · Ctrl-O / q closes\n"))
            return parts

        def handle_key(key: str, data: str) -> Any:
            nonlocal opened
            if key in {"c-o", "q"}:
                return None
            if opened is not None:
                if key in {"escape", "left", "h"}:
                    opened = None
                return TUI_MODAL_PENDING
            result = state.handle_key(key, data)
            if result is SELECTION_BACK:
                return None
            if isinstance(result, str):
                opened = result
            return TUI_MODAL_PENDING

        self.tui.show_modal(fragments, handle_key)

    def diff_command(self, args: str) -> str | None:
        if args.strip():
            return "Usage: /diff"
        if self.interactive_input and self.ui.color and (self.tui is None or self.tui.alternate_screen_available()):
            self.diff_viewer()
            return None
        latest = self.agent.session.latest_round_diff_sections()
        session = self.agent.session.session_diff_sections()
        groups: list[tuple[str, list[tuple[str, str, str]]]] = []
        if latest is not None and latest[1]:
            round, sections = latest
            groups.append((f"Latest · Round {round}", sections))
        if session:
            groups.append(("Session", session))
        if not groups:
            return "No changes"
        lines: list[str] = []
        for title, sections in groups:
            lines.append("### " + title)
            for _status, path, diff in sections:
                lines.append(f"#### {path}")
                bounded, truncated = CommandLoop.bounded_diff(diff)
                lines.append(f"```diff\n{bounded}\n```")
                if truncated:
                    lines.append("\n*Diff truncated. Full edit output is stored in the session.*")
        return "\n".join(lines)

    def diff_viewer(self) -> None:
        """Interactive diff viewer. First shows a file list; open a file to see its diff.

        List mode: ↑/↓ or j/k move, h/l or ←/→ switches tabs, Enter opens the selected file,
        r refreshes, q/Esc closes.
        Diff mode: ↑/↓ scroll one line, Ctrl-U/Ctrl-D half a page, PgUp/PgDn a page,
        Esc/← returns to list, r refreshes, q closes.
        """
        state = DiffViewState(TabbedViewState(("Latest", "Session")))

        def build_model() -> list[list[tuple[str, str, str]]]:
            latest = self.agent.session.latest_round_diff_sections()
            return [latest[1] if latest is not None else [], self.agent.session.session_diff_sections()]

        model = build_model()

        def viewport() -> int:
            return max(3, shutil.get_terminal_size().lines - 7)

        def active_sections() -> list[tuple[str, str, str]]:
            return model[state.view.tab]

        def list_fragments(parts: list[tuple[str, str]], sections: list[tuple[str, str, str]]) -> None:
            parts.append(("", "\n"))
            counts = [CommandLoop.diff_counts(diff) for _status, _path, diff in sections]
            added_width = max(len(str(added)) for added, _removed in counts)
            removed_width = max(len(str(removed)) for _added, removed in counts)
            for index, ((_status, path, _diff), (added, removed)) in enumerate(zip(sections, counts)):
                selected = index == state.file
                marker = "> " if selected else "  "
                style = "ansicyan" if selected else "class:choice.disabled"
                parts.extend(
                    [
                        (style, marker),
                        ("ansigreen", f"+{added:>{added_width}}"),
                        ("", " "),
                        ("ansired", f"-{removed:>{removed_width}}"),
                        (style, f" {path}\n"),
                    ]
                )
            parts.append(("", "\n"))

        def file_fragments(parts: list[tuple[str, str]], sections: list[tuple[str, str, str]]) -> None:
            state.clamp_file(len(sections))
            status, path, diff = sections[state.file]
            parts.append(("", "\n"))
            parts.append(("ansicyan", f"  {status.title()} · {path}\n"))
            lines = self.ui.segment_lines(self.ui.diff_segments_live(diff))
            visible = state.view.visible(lines, viewport())
            for line in visible:
                parts.extend(line)
            if not visible or not visible[-1] or not visible[-1][-1][1].endswith("\n"):
                parts.append(("", "\n"))

        def fragments():
            parts: list[tuple[str, str]] = [("", "\n")]
            parts.extend(self.ui.tab_segments(state.view.titles, state.view.tab))
            parts.append(("", "\n"))

            sections = active_sections()
            if not sections:
                parts.append(("class:choice.disabled", "  No diffs\n"))
            elif state.mode is DiffViewState.Mode.LIST:
                list_fragments(parts, sections)
            else:
                file_fragments(parts, sections)
            mode_hint = "list" if state.mode is DiffViewState.Mode.LIST else "diff"
            if state.mode is DiffViewState.Mode.LIST:
                hint = "↑/↓ or j/k move · ←/→ or h/l tab · Enter open · r refresh · Esc/q close"
            else:
                hint = "↑/↓ scroll · Ctrl-U/D half-page · PgUp/PgDn page · Esc/← back · r refresh · q close"
            position = f"{state.file + 1 if sections else 0}/{len(sections)}"
            parts.append(("class:choice.disabled", f"\n  [{mode_hint}] {hint} [{position}]\n"))
            return parts

        if self.tui is None:
            return

        def modal_key(key: str, _data: str) -> Any:
            nonlocal model
            result = state.handle_key(key, len(active_sections()), viewport())
            if result is DiffViewState.REFRESH:
                model = build_model()
                return TUI_MODAL_PENDING
            return result

        self.tui.show_modal(fragments, modal_key, exclusive=True)

    def config(self, args: str) -> str:
        provider = self.session.config.provider
        return "\n".join(
            [
                f"provider.active: {self.session.config.active_provider}",
                f"provider.available: {', '.join(sorted(self.session.config.providers))}",
                f"provider.url: {provider.url or '(empty)'}",
                f"provider.key: {'(set)' if provider.key else '(empty)'}",
                f"provider.model: {provider.model or '(empty)'}",
                f"provider.api: {provider.api}",
                f"provider.resolved_api: {provider.resolved_api()}",
                f"provider.prompt_cache_key: {provider.prompt_cache_key}",
                f"provider.available_models: {', '.join(provider.available_models) or '(empty)'}",
                f"provider.reasoning: {provider.reasoning}",
                f"provider.resolved_chat_reasoning: {provider.resolved_chat_reasoning()}",
                f"provider.chat_reasoning: {provider.chat_reasoning}",
                f"provider.temperature: {provider.temperature if provider.temperature is not None else '(off)'}",
                f"provider.max_tokens: {provider.max_tokens or ('(resolved ' + str(provider.resolved_max_tokens() or 'server default') + ')')}",
                f"provider.strict_tools: {provider.strict_tools} (active {provider.resolved_strict_tools()})",
                f"provider.extra_body: {json.dumps(provider.extra_body, ensure_ascii=False, sort_keys=True) if provider.extra_body else '(off)'}",
                f"provider.timeout: {provider.timeout}",
                f"paths.data_dir: {self.session.data_path()}",
                f"runtime.shell_timeout: {self.session.settings.shell_timeout}",
                f"runtime.max_agent_steps: {self.session.settings.max_steps}",
                f"runtime.max_context_tokens: {self.session.settings.max_context_tokens}",
                f"runtime.max_parallel_tools: {self.session.settings.max_parallel_tools}",
                f"runtime.session_retention_days: {self.session.settings.session_retention_days}",
                f"runtime.yolo: {'on' if self.session.settings.yolo else 'off'}",
            ]
        )

    def compact(self, args: str) -> str:
        if args.strip():
            return "Usage: /compact"
        before = len(self.session.messages)
        compacted, keep = self.agent.context.compaction_parts()
        if not compacted:
            return "No prior conversation to compact"
        fallback = False
        self.status_bar.begin()
        if self.tui is not None:
            self.tui.set_running("compacting context")
        else:
            self.status_bar.start(reset=False)
        try:
            data = self.agent.model.compact(self.agent.context.compaction_input(compacted))
        except KeyboardInterrupt:
            return "Cancelled"
        except Exception:
            self.agent.context.apply_compaction(None, keep, fallback_note="Previous context was deterministically trimmed.", compacted=compacted)
            fallback = True
            data = None
        finally:
            if self.tui is not None:
                self.tui.set_dispatching()
            else:
                self.status_bar.stop()
        if data is not None:
            self.agent.context.apply_compaction(data, keep, compacted=compacted)
        self.agent.context.update_current_percent(self.agent.SYSTEM_PROMPT)
        # Compaction rewrites the history in place. Persist it now: leaving the session without
        # running another turn would otherwise resume from the log's pre-compaction state.
        self.session.save_snapshot()
        fallback_note = " (fallback)" if fallback else ""
        return (
            f"Compacted context: messages {before} -> {len(self.session.messages)}, "
            f"prior summary inserted, ctx {self.session.state.context_percent}%{fallback_note}"
        )

    def index(self, args: str) -> str:
        value = args.strip()
        if value not in {"", "force"}:
            return "Usage: /index [force]"
        try:
            self.status_bar.start()
            return CodeIndex(self.session).sync(force=value == "force")
        finally:
            self.status_bar.stop()

    def provider(self, args: str) -> str:
        parts = args.split()
        if len(parts) > 1:
            return "Usage: /provider [NAME]"
        if parts:
            return self.set_provider(parts[0])
        choices = tuple(sorted(self.session.config.providers))
        summary = "provider: " + self.session.config.active_provider + "\nproviders: " + ", ".join(choices)
        current = self.session.config.active_provider
        choice = self.select_choice("Provider", choices, labels={current: current + " (current)"}, current=current)
        if not isinstance(choice, str):
            return "No change" if choice is SELECTION_BACK else summary
        provider_result = self.set_provider(choice)
        model_result = self.model("")
        return provider_result + ("\n" + model_result if model_result else "")

    def set_provider(self, name: str) -> str:
        if name not in self.session.config.providers:
            return "Unknown provider: " + name
        self.session.config.active_provider = name
        return "Set provider = " + name

    def model(self, args: str) -> str:
        parts = args.split()
        if len(parts) > 1:
            return "Usage: /model [MODEL]"
        if parts:
            result = self.set_model(parts[0])
            return "No change" if result is SELECTION_BACK else str(result)
        provider = self.session.config.provider
        configured = tuple(dict.fromkeys(provider.available_models))
        show_loading = self.tui is not None and bool(provider.url and provider.key)
        if show_loading:
            self.tui.set_dispatching("Loading models...")
        try:
            remote = tuple(model for model in self.remote_models(provider) if model not in configured)
        finally:
            if show_loading:
                self.tui.set_dispatching()
        choices: list[str] = []
        if configured:
            choices.extend((self.MODEL_CONFIGURED_LABEL, *configured))
        if remote:
            choices.extend((self.MODEL_DISCOVERED_LABEL, *remote))
        choice_values = tuple(choices)
        if not choice_values:
            return "Current provider.model is " + (self.session.config.provider.model or "(empty)")
        while True:
            current = self.session.config.provider.model
            labels = {label: label for label in self.MODEL_LABELS if label in choice_values}
            labels.update({current: current + " (current)"} if current in choice_values else {})
            choice = self.select_choice("Model", choice_values, labels=labels, current=current, disabled=self.MODEL_LABELS)
            if choice is SELECTION_BACK:
                return "No change"
            if not isinstance(choice, str):
                return "Current provider.model is " + (self.session.config.provider.model or "(empty)")
            if choice in self.MODEL_LABELS:
                continue
            result = self.set_model(choice, back_to_model=True)
            if result is SELECTION_BACK:
                continue
            return str(result)

    def remote_models(self, provider: ProviderConfig) -> tuple[str, ...]:
        if not provider.url or not provider.key:
            return ()
        try:
            page = OpenAI(
                api_key=provider.key,
                base_url=provider.base_url(),
                timeout=min(provider.timeout, 10),
                max_retries=0,
                default_headers={"User-Agent": HTTP_USER_AGENT},
            ).models.list()
        except Exception:
            return ()
        names = []
        for item in getattr(page, "data", page) or []:
            name = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
            if isinstance(name, str) and name:
                names.append(name)
        return tuple(sorted(dict.fromkeys(names)))

    def set_model(self, model: str, *, back_to_model: bool = False) -> str | object:
        reasoning = self.select_reasoning()
        if reasoning is SELECTION_BACK:
            return SELECTION_BACK if back_to_model else "No change"
        provider = self.session.config.provider
        provider.model = model
        lines = ["Set provider.model = " + model]
        if isinstance(reasoning, str):
            provider.reasoning = reasoning
            lines.append("Set provider.reasoning = " + reasoning)
        return "\n".join(lines)

    def reason(self, args: str) -> str:
        value = args.strip()
        if value:
            if value not in REASONING_CHOICES:
                return "Usage: /reason " + "|".join(REASONING_CHOICES)
            self.session.config.provider.reasoning = value
            return "Set provider.reasoning = " + value
        choice = self.select_reasoning()
        if isinstance(choice, str):
            self.session.config.provider.reasoning = choice
            return "Set provider.reasoning = " + choice
        return "No change"

    def yolo(self, args: str) -> str:
        self.session.settings.yolo = not self.session.settings.yolo
        return "yolo: " + ("on" if self.session.settings.yolo else "off")

    def strict(self, args: str) -> str:
        if args:
            return "Usage: /strict"
        provider = self.session.config.provider
        provider.strict_tools = not provider.strict_tools
        state = "on" if provider.strict_tools else "off"
        if provider.strict_tools and not provider.resolved_strict_tools():
            return f"strict_tools: {state} (inactive: {provider.host() or 'this provider'} does not support strict tool calling)"
        return f"strict_tools: {state}"

    def set_value(self, args: str) -> str:
        key, _, value = args.partition(" ")
        if not key or not value:
            return "Usage: /set KEY VALUE"
        handler = CommandCompleter.SET_HANDLERS.get(key)
        if handler is None:
            return "Unknown config key: " + key
        target_name, attr, coerce = handler
        obj = self.session.config.provider if target_name == "provider" else self.session.settings
        try:
            if coerce is not None:
                value = coerce(value)
            setattr(obj, attr, value)
        except (ConfigError, ValueError):
            return "Invalid value for " + key
        return "Set " + key


class TuiRuntime:
    """Own the interactive session timeline while CommandLoop owns session behavior."""

    def __init__(self, command_loop: CommandLoop):
        self.loop = command_loop
        self.pending: queue.Queue[str] = queue.Queue()
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

    def _request_model_retry(self, status_label: str) -> None:
        state = self.loop.session.state
        if state.current_model_call_started_at <= 0 or state.manual_model_retry_requested:
            return
        state.manual_model_retry_requested = True
        state.model_retry_count += 1
        self.tui.status_label = status_label
        self.tui.invalidate()
        self._interrupt_active(self.loop.agent.model.cancel)

    def submit_running(self, text: str) -> None:
        text = Text.clean(text.strip())
        if not text:
            return
        if "\n" not in text and text.startswith("/"):
            threading.Thread(target=self.loop.run_queued_command, args=(text,), daemon=True).start()
        else:
            self.loop.session.enqueue_user_input(text)
            self.loop.session.save_snapshot()
        self.tui.invalidate()

    def recall(self) -> str:
        return self.loop.recall_pending_input(lambda: self._request_model_retry("revising queued input"))

    def expand_output(self) -> None:
        threading.Thread(target=self.loop.bash_output_viewer, name="bash-output", daemon=True).start()

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
            on_input_cancel=lambda: self.loop.emit("Cancelled"),
            on_retry=lambda: self._request_model_retry("working"),
            on_recall=self.recall,
            on_expand_output=self.expand_output,
            status_fragments_fn=lambda: self.loop.status_bar.display_fragments(active=self.tui.input_mode == "running"),
            activity_fragments_fn=self.loop.tui_activity_fragments,
            input_hint_fn=self.loop.tui_input_hint,
            editor_context_fn=self.loop.editor_context,
            history=self.loop.input_history,
            completer=self.loop.input_completer,
        )

    def submit_next(self, entered: list[str]) -> None:
        if not entered:
            return
        self.pending.put(entered[0])
        for text in entered[1:]:
            self.loop.session.enqueue_user_input(text)

    def reset_turn(self) -> None:
        self.tui.set_idle()
        self.cancel_pending.clear()
        self.main_busy.clear()

    def dispatch(self, user_input: str) -> bool:
        """Dispatch one input. Return true when it was fully handled as a command."""
        self.loop.ui.emit_answer(user_input, role="user", rule=False)
        try:
            handled, exit_now = self.loop.command(user_input.strip())
        except (KeyboardInterrupt, MinacodeError) as error:
            self.loop.emit("Cancelled" if isinstance(error, KeyboardInterrupt) else f"Error: {error}")
            self.reset_turn()
            return True
        if exit_now:
            self.stop.set()
            self.main_busy.clear()
            self.tui.exit()
            return True
        if handled:
            self.reset_turn()
            return True
        return False

    def run_agent_turn(self, user_input: str) -> None:
        self.loop.emit("")
        self.loop.status_bar.begin()
        self.tui.set_running("working")
        started = time.monotonic()
        try:
            answer = self.loop.agent.run(user_input)
        except KeyboardInterrupt:
            self.submit_next(self.loop.take_pending_inputs())
            self.loop.emit("Cancelled")
            return
        except MinacodeError as error:
            answer = f"Error: {error}"
        finally:
            self.reset_turn()
            self.loop.session.state.manual_model_retry_requested = False
            CodeIndex(self.loop.session).update_pending_async()
        elapsed = time.monotonic() - started
        self.loop.ui.emit_answer(answer)
        self.loop.emit(f"[done in {int(elapsed // 60)}m{elapsed % 60:.0f}s]")
        self.loop.session.save_snapshot()
        self.submit_next(self.loop.take_pending_inputs())

    def run_agent_loop(self) -> None:
        while not self.stop.is_set():
            try:
                user_input = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            self.main_busy.set()
            if self.cancel_pending.is_set():
                self.loop.emit("Cancelled")
                self.reset_turn()
                continue
            if not self.dispatch(user_input):
                self.run_agent_turn(user_input)

    def run_tui_app(self) -> None:
        try:
            self.tui.run(style=self.loop.style())
        except BaseException as error:
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
            self.loop.start_session()
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
