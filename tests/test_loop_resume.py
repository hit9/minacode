"""loop resume (split from tests/test_loop_commands.py)."""
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

def test_empty_exit_does_not_print_resume_command(tmp_path):
    s = session(tmp_path)
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    handled, exit_now = loop.command("/exit")

    assert (handled, exit_now) == (True, True)
    assert output == []
    assert not os.path.exists(SessionSnapshotStore.session_path(s.config.data_dir, s.cwd, s.uid))

def test_resumed_session_does_not_render_tool_results(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    arguments = json.dumps({"files": [{"path": "a.py", "ranges": [[0, 1]]}]})
    s.messages.extend(
        [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "need tool",
                "tool_calls": [{"id": "tc.1", "type": "function", "function": {"name": "Read", "arguments": arguments}}],
            },
            {"role": "tool", "tool_call_id": "tc.1", "content": "raw tool result"},
            {"role": "system", "content": f"[Session resumed: uid={s.uid}]"},
        ]
    )
    s.tool_records.append(ToolResultRecord("tr.1", "Read", [{"path": "a.py", "ranges": [[0, 1]]}], "raw tool result", "a.py 0:1"))
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    text = "\n".join(output)
    assert s.resumed is False
    assert f"Restored session: {s.uid}" in text
    assert "• hello" in text
    assert "  need tool" in text
    assert "user:" not in text and "assistant:" not in text
    assert "Read  a.py 0:1 → tr.1" in text
    assert "tool:" not in text
    assert "raw tool result" not in text

def test_resumed_session_matches_retried_tool_results_by_call_id(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    arguments = json.dumps({"files": [{"path": "a.py", "ranges": [[0, 1]]}]})
    failed_call = {"id": "failed", "type": "function", "function": {"name": "Read", "arguments": arguments}}
    successful_call = {"id": "successful", "type": "function", "function": {"name": "Read", "arguments": arguments}}
    s.transcript_messages.extend(
        [
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": None, "tool_calls": [failed_call]},
            {"role": "tool", "tool_call_id": "failed", "result_key": "", "status": "failed"},
            {"role": "assistant", "content": None, "tool_calls": [successful_call]},
            {"role": "tool", "tool_call_id": "successful", "result_key": "tr.1", "status": "ok"},
        ]
    )
    s.tool_records.append(ToolResultRecord("tr.1", "Read", [{"path": "a.py", "ranges": [[0, 1]]}], "data", "a.py 0:1"))
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    text = "\n".join(output)
    failed = text.index("[failed]")
    stored = text.index("tr.1")
    assert failed < stored
    assert text.count("tr.1") == 1

def test_resumed_session_warns_when_older_version_wrote_after_transcript(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    s.transcript_incomplete = True
    s.transcript_messages.append({"role": "user", "content": "visible"})
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    assert output[0] == f"Restored session: {s.uid}"
    assert output[1] == "Warning: this transcript may omit turns written by an older minacode version."

def test_resumed_session_hides_internal_checkpoint_and_resume_events(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    s.messages.extend(
        [
            {"role": "user", "content": "visible request"},
            {
                "role": "user",
                "content": "hidden compaction checkpoint",
                SESSION_EVENT_KEY: "compaction_checkpoint",
            },
            {
                "role": "user",
                "content": "hidden working-state checkpoint",
                SESSION_EVENT_KEY: "state_checkpoint",
            },
            {
                "role": "user",
                "content": '<session_event type="resumed" at="2026-07-31T08:00:00+08:00" />',
                SESSION_EVENT_KEY: "resumed",
            },
        ]
    )
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    text = "\n".join(output)
    assert f"Restored session: {s.uid}" in text
    assert "visible request" in text
    assert "hidden compaction checkpoint" not in text
    assert "hidden working-state checkpoint" not in text
    assert "<session_event" not in text

def test_resumed_session_with_only_internal_events_still_confirms_restore(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    s.messages.extend(
        [
            {"role": "user", "content": "hidden checkpoint", SESSION_EVENT_KEY: "compaction_checkpoint"},
            {"role": "user", "content": "hidden lifecycle event", SESSION_EVENT_KEY: "resumed"},
        ]
    )
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    assert output == [f"Restored session: {s.uid}"]

def test_resumed_session_renders_saved_tool_records_without_matching_tool_calls(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    s.messages.extend(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "compacted answer\nfinal detail"},
        ]
    )
    s.tool_records.append(ToolResultRecord("tr.1", "Bash", ["wc -l minacode.py"], "999 minacode.py", "wc -l minacode.py"))
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    text = "\n".join(output)
    assert f"Restored session: {s.uid}" in text
    assert "  compacted answer\n  final detail" in text  # the answer sits in the content column
    assert "user:" not in text and "assistant:" not in text
    assert "  Bash  wc -l minacode.py\n    └ stored tr.1" in text
    assert "999 minacode.py" not in text

def test_resumed_session_separates_turn_boxes(tmp_path):
    s = session(tmp_path)
    s.resumed = True
    s.messages.extend(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "two"},
        ]
    )
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), output_fn=output.append)

    loop.render_resumed_session()

    # Both answers sit in the content column, where the live turn printed them; the user's `• `
    # bullet hangs in that same two-space margin, so every line of text starts at column 2.
    assert output[1:] == ["\n• first", "  one", "", "\n• second", "  two"]

def test_turn_box_groups_followup_users_until_final_assistant():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "working", "tool_calls": [{"id": "one"}]},
        {"role": "user", "content": "follow-up"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
    ]

    boxes = TurnBox.group(messages)

    assert [len(box.messages) for box in boxes] == [4, 1]

def test_turn_box_groups_tool_results_with_calling_assistant():
    # Tool results (role="tool") are kept in the same TurnBox as the
    # assistant that issued the tool_calls, not split prematurely.
    messages = [
        {"role": "user", "content": "read a.py"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "tr.1", "function": {"name": "Read"}}]},
        {"role": "tool", "tool_call_id": "tr.1", "content": "# file content"},
        {"role": "assistant", "content": "done"},
    ]
    boxes = TurnBox.group(messages)
    assert len(boxes) == 1
    assert len(boxes[0].messages) == 4
    roles = [m["role"] for m in boxes[0].messages]
    assert roles == ["user", "assistant", "tool", "assistant"]

def test_eof_exit_prints_resume_command(tmp_path):
    s = session(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), input_fn=lambda prompt="": (_ for _ in ()).throw(EOFError()), output_fn=output.append)

    assert loop.run() == 0

    # The session took its name from the opening message; the pasted line still carries the uid.
    assert output[-1] == f"Resume 'hello' with:\nminacode --resume {s.uid}"
    assert os.path.exists(SessionSnapshotStore.session_path(s.config.data_dir, s.cwd, s.uid))

@pytest.mark.parametrize(("interrupt_phase", "expected_cancelled"), [("input", 0), ("request", 1)])
def test_simple_repl_ctrl_c_output_matches_interrupted_phase(tmp_path, monkeypatch, interrupt_phase, expected_cancelled):
    output = []
    reads = iter([KeyboardInterrupt(), EOFError()] if interrupt_phase == "input" else ["question", EOFError()])

    def read_input(_prompt=""):
        value = next(reads)
        if isinstance(value, BaseException):
            raise value
        return value

    agent = Agent(session(tmp_path), output_fn=output.append)
    if interrupt_phase == "request":
        agent.run = lambda _input: (_ for _ in ()).throw(KeyboardInterrupt())
    command_loop = CommandLoop(agent, input_fn=read_input, output_fn=output.append)
    monkeypatch.setattr(loop_module.UpdateChecker, "start", lambda _checker: None)
    monkeypatch.setattr(CodeIndex, "status", lambda _index: False)
    monkeypatch.setattr(CodeIndex, "update_pending_async", lambda _index: None)

    assert command_loop.run() == 0

    assert [line.strip() for line in output].count("Cancelled") == expected_cancelled

@pytest.mark.parametrize("raised", [None, "error"])
def test_simple_repl_publishes_the_final_answer_exactly_once(tmp_path, monkeypatch, raised):
    """The engine publishes a completed turn's answer through output_fn, so the REPL must not
    print the return value on top of it; an error raised before that publish still prints here,
    because nothing else put it in the scrollback."""
    output = []
    reads = iter(["question", EOFError()])

    def read_input(_prompt=""):
        value = next(reads)
        if isinstance(value, BaseException):
            raise value
        return value

    agent = Agent(session(tmp_path), output_fn=output.append)

    def run(_user_input):
        if raised:
            raise MinacodeError("provider is down")
        agent.output_fn("The answer.")  # what Agent.run does before it returns
        return "The answer."

    agent.run = run
    command_loop = CommandLoop(agent, input_fn=read_input, output_fn=output.append)
    monkeypatch.setattr(loop_module.UpdateChecker, "start", lambda _checker: None)
    monkeypatch.setattr(CodeIndex, "status", lambda _index: False)
    monkeypatch.setattr(CodeIndex, "update_pending_async", lambda _index: None)

    assert command_loop.run() == 0

    printed = [str(item) for item in output]
    expected = "Error: provider is down" if raised else "The answer."
    assert sum(expected in line for line in printed) == 1

def test_select_choice_noninteractive_does_not_prompt(tmp_path):
    output = []
    loop = CommandLoop(Agent(session(tmp_path), output_fn=output.append), input_fn=lambda prompt="": "1", output_fn=output.append)

    assert select_choice(loop, "Pick", ("a", "b"), labels={"a": "A"}, current="a") is None
    assert output == []
