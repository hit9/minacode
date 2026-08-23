"""CommandLoop surfaces around a turn: the input queue, slash commands, skills, transcript
rendering, and status output."""

import itertools
import json
import os
import time
import tomllib
from types import SimpleNamespace

import pytest
from agent_harness import call, queue, session
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.utils import get_cwidth

import minacode.cli as loop_module
import minacode.cli.commands as commands_mod
import minacode.cli.modals as modals_mod
from minacode.base import (
    DISMISSED,
    SELECTION_BACK,
    SESSION_EVENT_KEY,
    LogBlock,
    LogLine,
    MinacodeError,
    Text,
    ToolError,
    TurnBox,
    __version__,
)
from minacode.cli import CommandLoop
from minacode.cli.commands import (
    name_command,
    session_label_fn,
    session_preview,
    session_rows,
    session_table,
    sessions_command,
    skills_command,
    status,
)
from minacode.cli.modals import choice_application, question_interaction, select_choice
from minacode.config import (
    Config,
)
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.prompts import SYSTEM_PROMPT
from minacode.render import StatusBar, UiPrinter
from minacode.runner import ToolRunner
from minacode.session import Session, SessionEntry, SessionSnapshotStore, ToolResultRecord
from minacode.skill import SkillLibrary
from minacode.tools import AskSpec, CodeIndex, SkillTool, Tool
from minacode.tui import ASK_DONE, ASK_FREE_TEXT, TuiApp


def _write_skill(root, name, description, body, *, scripts=None):
    folder = os.path.join(root, ".minacode", "skills", name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "SKILL.md"), "w", encoding="utf-8") as handle:
        handle.write(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n")
    for script_name, script_body in (scripts or {}).items():
        script_dir = os.path.join(folder, "scripts")
        os.makedirs(script_dir, exist_ok=True)
        with open(os.path.join(script_dir, script_name), "w", encoding="utf-8") as handle:
            handle.write(script_body)
    return folder


def queued_texts(s):
    return [item.text for item in s.pending_user_inputs]












































































































































































