"""tool runner (split from tests/test_agent_turn.py)."""
import json
import threading
import time
from types import SimpleNamespace
import pytest
from agent_harness import call, queue, session
import minacode.engine as engine_module
from minacode.base import (
    SESSION_EVENT_KEY,
    LogBlock,
    MalformedToolCallError,
    ModelError,
    ToolCall,
)
from minacode.config import (
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    Config,
    ProviderConfig,
)
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.model import ModelClient
from minacode.prompts import FAILED_TURN_MARKER, INTERRUPT_MARKER, LIVE_FOLLOWUP_PREFIX, SYSTEM_PROMPT
from minacode.runner import ToolRunner
from minacode.session import Session, SessionSnapshotCodec
from minacode.skill import SkillLibrary
from minacode.tools import BashTool, ReadTool, Tool

def test_tool_runner_refusal_stops_batch_and_invalid_args_are_not_stored(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "skip it", output_fn=lambda text: None)
    runner.run([call("Bash", [":"]), call("Edit", ["second.txt", [{"op": "create", "content": "second"}]])])

    assert s.tool_records == []
    assert len(s.tool_errors) == 1
    assert "skip it" in s.tool_errors[0].error
    assert not (tmp_path / "second.txt").exists()

    outputs = []
    bad = session(tmp_path)
    ToolRunner(bad, ContextManager(bad), output_fn=lambda text: outputs.append(str(text))).run([call("Bash", [])])
    assert bad.tool_records == []
    assert len(bad.tool_errors) == 1
    assert outputs and "· rejected:" in outputs[0]  # argument errors collapse to a quiet line

def test_rejected_and_failed_calls_collapse_a_multiline_display_to_one_line(tmp_path):
    # A tool's display is whatever its short_args produced, and Note's is the whole rendered note so
    # a successful call can print it. A rejection is meant to be a quiet one-liner and a failure
    # leads with a red tag, so neither may inherit those lines: a rejected Note used to dim its
    # entire body and bury the reason at the end of the last line.
    from minacode.base import LogRole
    from minacode.runner import ToolDisplay

    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)
    note = call("Note", [{"replace_plan": [f"Task {index}" for index in range(1, 11)]}])
    display = ToolDisplay()
    display.display = runner.short_call(note)
    assert len(display.display.splitlines()) > 1, "precondition: Note renders a multi-line display"

    rejected = list(runner.reject_display(note, "ToolError: Note fields is only valid for view", d=display).walk())
    assert [item.role for item, _ in rejected] == [LogRole.MUTED]
    assert len(rejected[0][0].text.splitlines()) == 1
    assert rejected[0][0].text.endswith("· rejected: Note fields is only valid for view")

    failed = list(runner.finish_display(note, "", "Note: disk is full", failed=True, d=display).walk())
    assert all(len((item.text or "").splitlines()) == 1 for item, _ in failed)

    # A successful Note is the one case that should keep every line: printing the note is the point.
    assert len(runner.finish_display(note, "", "ok", failed=False, d=display).splitlines()) > 1

def test_tool_runner_refuses_without_reason_on_n(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "n", output_fn=lambda text: None)

    runner.run([call("Bash", [":"])])

    assert s.tool_errors[0].error == "Cancelled: user refused tool call"

def test_tool_runner_refuses_with_direct_reason_input(tmp_path):
    s = session(tmp_path)
    runner = ToolRunner(s, ContextManager(s), input_fn=lambda prompt: "not now", output_fn=lambda text: None)

    runner.run([call("Bash", [":"])])

    assert s.tool_records == []
    assert len(s.tool_errors) == 1
    assert "not now" in s.tool_errors[0].error

def test_recall_tool_runner_does_not_create_new_result_keys(tmp_path):
    s = session(tmp_path)
    key = s.store_tool_result("Read", ["a.txt"], "result")
    runner = ToolRunner(s, ContextManager(s), output_fn=lambda text: None)

    runner.run([call("Recall", [key])])
    assert [record.key for record in s.tool_records] == [key]
