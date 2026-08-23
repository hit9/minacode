"""TuiApp behavior: layout, input modes, key bindings, modals, and approval prompts."""

import asyncio
import multiprocessing
import os
import signal
import threading
import time
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Size
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput
from tui_harness import ResizableOutput, loop, rendered_screen_text, run_interactive_tui, session, wait_until

import minacode.tui.app as tui_module
from minacode.base import (
    SESSION_EVENT_KEY,
    LogBlock,
    LogEdge,
)
from minacode.cli import CommandCompleter, CommandLoop, TuiRuntime, hints
from minacode.cli.commands import select_choice
from minacode.cli.hints import HintPicker
from minacode.cli.update import UpdateChecker
from minacode.config import (
    Config,
)
from minacode.engine import Agent
from minacode.mentions import FilePick, active_mention
from minacode.prompts import LIVE_FOLLOWUP_PREFIX
from minacode.session import Session, SessionSnapshotStore
from minacode.tools import CodeIndex
from minacode.tui import TUI_MODAL_PENDING, CallbackPlaceholder, TuiApp








































































class _StubJob:
    def __init__(self, status):
        self.status = status
















ACTIONS = [("Approve", ""), ("View order", "v"), ("Worker config", "c"), ("Refuse", "n")]


def _approval_app():
    app = TuiApp()
    app._input_pending = threading.Event()
    app.input_mode = "approval"
    assert app.set_approval_form(ACTIONS) is True
    return app


def _active(app, key):
    return [binding for binding in reversed(app.make_bindings().bindings) if binding.keys == (key,) and binding.filter()]














































def quick_hint_app(hints=("run the tests", "show the diff", "commit")):
    submitted = []
    app = TuiApp(on_chat_submit=submitted.append, quick_hints_fn=lambda: hints)
    app.set_idle()
    return app, submitted














































































