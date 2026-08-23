"""loop ask (split from tests/test_loop_commands.py)."""
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

def test_choice_application_expands_escaped_preview_newlines(tmp_path):
    output = []
    loop = CommandLoop(Agent(session(tmp_path), output_fn=output.append), input_fn=lambda prompt="": "", output_fn=output.append)
    loop.interactive_input = True
    rendered = []

    class Modal:
        def show_modal(self, fragments_fn, key_fn, exclusive=False):
            rendered.extend(fragments_fn())
            return key_fn("enter", "")

    loop.tui = Modal()

    result = choice_application(
        loop,
        "Select:",
        ("A", "B"),
        {},
        "",
        set(),
        preview_fn=lambda choice: "one\\ntwo" if choice == "A" else "",
    )

    assert result == "A"
    previews = [text for style, text in rendered if style == "class:choice.preview"]
    assert previews == ["  │ one\n", "  │ two\n"]
    assert all("\\n" not in text for _, text in rendered)

def test_ask_free_text_prompt_has_no_control_newline(tmp_path):
    """A free-text page drops out of the modal to the shared input row; the answer flows into
    the batch and the modal reopens (ASK_DONE ends it)."""
    output = []
    loop = CommandLoop(Agent(session(tmp_path), output_fn=output.append), input_fn=lambda prompt="": "", output_fn=output.append)
    loop.interactive_input = True
    prompts = []
    results = iter([(ASK_FREE_TEXT, 0), ASK_DONE])
    loop.tui = SimpleNamespace(
        request_input=lambda prompt: prompts.append(prompt) or "typed answer",
        show_modal=lambda fragments_fn, key_fn: next(results),
    )

    assert question_interaction(loop, [AskSpec("Pick?", choices=["A"], previews=["preview"])]) == ["typed answer"]
    assert prompts == ["\nPick?"]  # one shared-input prompt, the question spelled out again

def test_ask_free_text_empty_answer_is_kept(tmp_path):
    """An explicitly empty free-text answer is a legal answer: the batch must return [""] and
    never fall back to the question text (which is only the placeholder for unanswered pages)."""
    output = []
    loop = CommandLoop(Agent(session(tmp_path), output_fn=output.append), input_fn=lambda prompt="": "", output_fn=output.append)
    loop.interactive_input = True
    results = iter([(ASK_FREE_TEXT, 0), ASK_DONE])
    loop.tui = SimpleNamespace(
        request_input=lambda prompt: "",
        show_modal=lambda fragments_fn, key_fn: next(results),
    )

    assert question_interaction(loop, [AskSpec("Pick?")]) == [""]

def test_ask_free_text_on_last_question_submits_without_reentering_modal(tmp_path):
    """A free-text answer to the final question completes the batch right after the shared input
    row; the modal must not reopen for it (a second show_modal would fail the call-count assert)."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    loop.interactive_input = True
    calls = []

    def show_modal(fragments_fn, key_fn):
        calls.append(1)
        key_fn("enter")  # page 1: accept the preselected option, advance to page 2
        key_fn("2")  # page 2: move onto "Type freely..." (digits only move the cursor)
        return key_fn("enter")  # ...and select it -> drops to the shared input row

    loop.tui = SimpleNamespace(request_input=lambda prompt: "typed", show_modal=show_modal)

    assert question_interaction(loop, [AskSpec("One?", choices=["A"]), AskSpec("Two?", choices=["B"])]) == ["A", "typed"]
    assert len(calls) == 1

def test_ask_without_choices_uses_shared_tui_input(tmp_path):
    """A question without choices is a single Type-freely page; Enter drops to the shared row."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "fallback", output_fn=lambda text: None)
    loop.interactive_input = True
    prompts = []
    results = iter([(ASK_FREE_TEXT, 0), ASK_DONE])
    loop.tui = SimpleNamespace(
        request_input=lambda prompt: prompts.append(prompt) or "typed answer",
        show_modal=lambda fragments_fn, key_fn: next(results),
    )

    assert question_interaction(loop, [AskSpec("Explain the issue")]) == ["typed answer"]
    assert prompts == ["\nExplain the issue"]

def test_ask_headless_keeps_plain_per_question_prompts(tmp_path):
    """Without a TUI the batch falls back to one read_input per question, in order."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "fallback", output_fn=lambda text: None)
    prompts = []
    loop.read_input = lambda prompt: prompts.append(prompt) or "answer"

    assert question_interaction(loop, [AskSpec("One?"), AskSpec("Two?", choices=["A"])]) == ["answer", "answer"]
    assert prompts == ["\nOne?", "\nTwo?"]

def test_ask_choice_is_not_echoed_before_final_tool_log(tmp_path, monkeypatch):
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), output_fn=lambda text: None)
    emitted = []
    loop.emit = lambda text="", indent=0: emitted.append(text)
    monkeypatch.setattr(modals_mod, "question_interaction", lambda _loop, specs: ["B"])

    assert modals_mod.question_interaction(loop, [AskSpec("Which?", choices=["A", "B"])]) == ["B"]
    assert emitted == []

def test_ask_notes_flow_into_the_answer(tmp_path):
    """A note entered on a page (`n`, text, Enter) is appended to that question's answer."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "fallback", output_fn=lambda text: None)
    loop.interactive_input = True

    def show_modal(fragments_fn, key_fn):
        key_fn("n")
        for ch in "keep the header":
            key_fn("any", ch)
        key_fn("enter")  # save the note
        return key_fn("enter")  # pick the recommended "A" and submit the batch

    loop.tui = SimpleNamespace(show_modal=show_modal)

    assert question_interaction(loop, [AskSpec("Q?", choices=["A"], recommended=0)]) == ["A\n\nUser notes: keep the header"]

def test_ask_escape_cancels_the_whole_batch(tmp_path):
    """Esc on any page cancels every question with the DISMISSED marker."""
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), input_fn=lambda prompt: "fallback", output_fn=lambda text: None)
    loop.interactive_input = True
    loop.tui = SimpleNamespace(show_modal=lambda fragments_fn, key_fn: SELECTION_BACK)

    result = question_interaction(loop, [AskSpec("One?"), AskSpec("Two?")])
    assert result == [DISMISSED, DISMISSED]

def test_elapsed_since_uses_whole_seconds(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 104.9)
    assert Text.elapsed_since(100.0) == "4s"

    monkeypatch.setattr(time, "monotonic", lambda: 162.9)
    assert Text.elapsed_since(100.0) == "1m02s"

def test_bash_live_start_pauses_standalone_status(tmp_path):
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda text: None), output_fn=lambda text: None)
    loop.ui.color = True
    loop.live_preview.start = lambda: setattr(loop.live_preview, "active", True)
    loop.status_bar.thread = object()
    loop.status_bar.stop = lambda: setattr(loop.status_bar, "thread", None)
    loop.status_bar.start = lambda **_kwargs: setattr(loop.status_bar, "thread", object())

    loop.tool_live_start()
    assert loop.live_status_paused is True
    assert loop.status_bar.thread is None

    loop.tool_live_output("", "")
    assert loop.live_status_paused is False
    assert loop.status_bar.thread is not None

def test_command_loop_indents_intermediate_and_final_messages(tmp_path):
    output = []
    loop = CommandLoop(Agent(session(tmp_path), output_fn=output.append), output_fn=output.append)

    loop.emit_agent_output("First line.\nSecond line.")
    loop.ui.emit_answer("Done.\nFinal detail.")

    assert output == ["  First line.\n  Second line.", "Done.\nFinal detail."]

def test_colored_assistant_and_tool_blocks_each_start_with_one_blank_line(tmp_path):
    loop = CommandLoop(Agent(session(tmp_path), output_fn=lambda _text: None), output_fn=lambda _text: None)
    loop.ui.color = True
    events = []
    loop.emit = lambda text="", indent=0: events.append(text)
    loop.ui.emit_answer = lambda text, **_kwargs: events.append(text)
    first = LogBlock.hierarchy(LogLine("Bash", "first"), [])
    first_result = LogBlock.hierarchy(None, [LogLine("stored", "tr.1")])
    second = LogBlock.hierarchy(LogLine("Bash", "second"), [])

    loop.emit_agent_output("Working on it.")
    loop.tool_output(first)
    loop.tool_output(first_result)
    loop.tool_output(second)

    assert events == ["", "Working on it.", "", first, first_result, "", second]
