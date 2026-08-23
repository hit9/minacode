"""TuiRuntime behavior: command dispatch, the follow-up queue, streamed response promotion,
resume, and session housekeeping at startup."""

import asyncio
import os
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from tui_harness import ResizableOutput, loop, run_interactive_tui, session, wait_until

import minacode.cli as loop_module
import minacode.render as render_module
import minacode.tui.app as tui_module
from minacode.base import (
    MalformedToolCallError,
    MinacodeError,
    ToolCall,
    TurnBox,
)
from minacode.cli import QUEUE_SAFE_COMMANDS, CommandLoop, TuiRuntime
from minacode.cli.runtime import RESUME_STATUS_LABEL
from minacode.cli.update import UpdateChecker
from minacode.config import (
    Config,
)
from minacode.engine import Agent
from minacode.prompts import LIVE_FOLLOWUP_PREFIX
from minacode.session import Session, SessionSnapshotStore
from minacode.tools import CodeIndex
from minacode.tui import TuiApp


def history_file(path, entries, line="x" * 200):
    """Write a prompt_toolkit history file with `entries` numbered entries."""
    with open(path, "wb") as file:
        file.writelines(f"\n# 2026-01-01 00:00:{index:02d}\n+{index}-{line}\n".encode() for index in range(entries))
    return path


class TextRecordingOutput(ResizableOutput):
    def __init__(self, rows=24, columns=80):
        super().__init__(rows, columns)
        self.writes = []
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            self.writes.append(data)

    def write_raw(self, data):
        with self.lock:
            self.writes.append(data)

    def text(self):
        with self.lock:
            return "".join(self.writes)






































































































