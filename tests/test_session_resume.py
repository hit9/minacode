"""session resume (split from tests/test_session_persistence.py)."""
import itertools
import json
import os
import time
from typing import ClassVar
import pytest
from minacode.base import SESSION_EVENT_KEY, MinacodeError
from minacode.cli import CommandLoop
from minacode.cli.commands import compact, provider, set_model
from minacode.config import (
    Config,
    ProviderConfig,
    RuntimeSettings,
)
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.model import ModelClient
from minacode.prompts import LIVE_FOLLOWUP_PREFIX
from minacode.session import HistorySegment, Session, SessionSnapshotCodec, SessionSnapshotStore, TurnDiff
from test_session_persistence import _resumed_transcript, log_path, read_jsonl, session_with_data_dir

def test_resume_replays_full_transcript_after_model_context_and_retained_records_are_pruned(tmp_path):
    s = session_with_data_dir(tmp_path)
    call_message = {
        "role": "assistant",
        "content": "Updating the old file.",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "Edit", "arguments": '{"path": "x.py"}'}}],
    }
    visible = [
        {"role": "user", "content": "old request that must remain visible"},
        call_message,
        {"role": "tool", "tool_call_id": "c1", "content": "tool tr.1 Edit x.py\noutput:\nupdated"},
        {"role": "assistant", "content": "old answer that must remain visible"},
    ]
    s.messages.extend(visible)
    s.transcript_messages.extend(visible)
    s.store_tool_result("Edit", ["x.py"], "updated")
    s.store_turn_diff("tr.1", 1, "x.py", "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-old\n+new\n")
    s.save_snapshot()

    s.messages[:] = [{"role": "user", "content": "compacted model context", SESSION_EVENT_KEY: "compaction_checkpoint"}]
    s.tool_records.clear()
    s.tool_results.clear()
    s.turn_diffs.clear()
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    output = []
    CommandLoop(Agent(restored, output_fn=output.append), output_fn=output.append).render_resumed_session()
    text = "\n".join(str(item) for item in output)

    assert "old request that must remain visible" in text
    assert "Updating the old file." in text
    assert "old answer that must remain visible" in text
    assert "stored tr.1" in text
    assert "-old" in text and "+new" in text
    assert "compacted model context" not in text

def test_compact_command_persists_the_compacted_history(tmp_path):
    """/compact rewrites the history in place; without a save, leaving the session would resume
    from the pre-compaction log."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "older request"})
    for i in range(12):
        s.messages.append({"role": "assistant", "content": f"step {i}"})
    s.messages.append({"role": "user", "content": "current request"})
    s.messages.append({"role": "assistant", "content": "working on it"})
    s.save_snapshot()
    before = len(s.messages)

    loop = CommandLoop(Agent(s, output_fn=lambda _text: None), output_fn=lambda _text: None)
    loop.agent.model.compact = lambda _context, *_args, **_kwargs: {"summary": "a compacted summary"}
    result = compact(loop, "")

    assert "Compacted context" in result
    assert len(s.messages) < before

    # The compacted history is on disk, not just in memory.
    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    persisted = restored.messages[:-1]  # load appends one new resume event
    assert persisted == s.messages
    assert any("a compacted summary" in str(m.get("content") or "") for m in persisted)
    # /compact also captures the evicted conversation as a recallable segment, and persists it.
    assert [segment.key for segment in s.history] == ["seg.1"]
    assert s.history[0].title == "older request"
    assert "older request" in s.history[0].text
    assert [segment.key for segment in restored.history] == ["seg.1"]
    assert "older request" in restored.history[0].text

def test_history_segments_persist_and_restore(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.history.append(HistorySegment(key="seg.1", title="earlier task", text="user:\nfind the bug\n\nassistant:\nfixed it"))
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))

    assert len(restored.history) == 1
    segment = restored.history[0]
    assert segment.key == "seg.1"
    assert segment.title == "earlier task"
    assert "find the bug" in segment.text

def test_history_delta_appends_new_segments(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.history.append(HistorySegment(key="seg.1", title="first", text="one"))
    s.save_snapshot()
    s.history.append(HistorySegment(key="seg.2", title="second", text="two"))
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    assert [segment.key for segment in restored.history] == ["seg.1", "seg.2"]

    # The second save appended only seg.2 (digest-delta), not a full rewrite.
    lines = read_jsonl(log_path(s))
    assert any("history" in line and [seg["key"] for seg in line["history"]] == ["seg.2"] for line in lines)
    assert not any("history_replace" in line for line in lines)

def test_history_delta_rewrites_when_saved_segments_change(tmp_path):
    """History is append-only in practice, so the digest-delta normally appends. If the saved segments
    ever disagree with the current ones (a reordered or trimmed history), the save must fall back to a
    full history_replace so the log still reconstructs the current set."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.history.append(HistorySegment(key="seg.1", title="first", text="one"))
    s.history.append(HistorySegment(key="seg.2", title="second", text="two"))
    s.save_snapshot()

    # Mutate the saved history out of band: the prefix digest no longer matches the last save.
    s.history[0], s.history[1] = s.history[1], s.history[0]
    s.save_snapshot()

    lines = read_jsonl(log_path(s))
    assert any("history_replace" in line and [seg["key"] for seg in line["history_replace"]] == ["seg.2", "seg.1"] for line in lines)

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    assert [segment.key for segment in restored.history] == ["seg.2", "seg.1"]

def test_resume_recomputes_the_context_percent(tmp_path):
    """`context_percent` is derived rather than persisted, so a resumed session would report an
    empty context until its first turn."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "x" * 40000})
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    assert restored.state.context_percent == 0

    loop = CommandLoop(Agent(restored, output_fn=lambda _text: None), output_fn=lambda _text: None)
    loop.render_resumed_session()

    assert restored.state.context_percent > 0

def test_resumed_transcript_replays_the_edit_diff(tmp_path):
    """A resumed session shows what each Edit changed, not just that an Edit ran."""
    text = _resumed_transcript(tmp_path, "--- x.py\n+++ x.py\n@@ -1 +1 @@\n-a\n+b\n")

    assert "preview" in text
    assert "-a" in text and "+b" in text
    assert "stored tr.1" in text
    # The preview block carries the call line, so it is not repeated by the result line.
    assert text.count("Edit") == 1

def test_resumed_transcript_trims_long_diffs(tmp_path):
    diff = "--- x.py\n+++ x.py\n" + "\n".join(f"+line {i}" for i in range(40))
    text = _resumed_transcript(tmp_path, diff, lines_cap=10)

    assert "+line 7" in text
    assert "+line 30" not in text
    assert "more lines, see /diff" in text

def test_resumed_transcript_without_a_stored_diff_shows_the_call_only(tmp_path):
    """Edits whose diff has been evicted still render as a plain call line."""
    text = _resumed_transcript(tmp_path, "")

    assert "preview" not in text
    assert "Edit" in text

def _bash_raw_call(arguments: str) -> dict:
    return {"id": "c1", "type": "function", "function": {"name": "Bash", "arguments": arguments}}

def test_resumed_transcript_hides_the_live_followup_marker(tmp_path):
    """The marker stays in history because it was sent, but the scrollback shows the user's own
    words: a resumed session must not read back runtime instructions the user never typed."""
    s = session_with_data_dir(tmp_path)
    agent = Agent(s, output_fn=lambda _text: None)
    command_loop = CommandLoop(agent, output_fn=lambda _text: None)
    rendered = []
    command_loop.ui.emit_answer = lambda text, **kwargs: rendered.append(text)

    marked = {"role": "user", "content": LIVE_FOLLOWUP_PREFIX + "also update the tests"}
    command_loop.render_transcript_message(marked)
    command_loop.render_transcript_message({"role": "user", "content": "plain request"})

    assert rendered == ["also update the tests", "plain request"]

def test_transcript_tool_call_parses_multiline_arguments():
    """Argument strings with literal newlines (invalid strict JSON) still parse, so the
    Bash command survives instead of being dropped to {}."""
    raw = _bash_raw_call('{"command": "printf \'line one\nline two\'"}')
    call = CommandLoop.transcript_tool_call(raw)
    assert call is not None
    assert call.args == ["printf 'line one\nline two'"]

def test_transcript_tool_call_does_not_crash_on_unparseable_args():
    """A historical Bash call whose payload fails validation must render, not raise."""
    raw = _bash_raw_call("{not valid json at all")
    call = CommandLoop.transcript_tool_call(raw)  # must not raise ToolError
    assert call is not None
    assert call.name == "Bash"

def test_chat_tool_calls_parse_multiline_commit_message():
    """The live chat path recovers args from a multi-line Bash command too."""

    class _Fn:
        name = "Bash"
        arguments = '{"command": "printf \'subject\n\nbody line\'"}'

    class _Raw:
        id = "x1"
        function = _Fn()

    class _Msg:
        tool_calls: ClassVar[list] = [_Raw()]

    s = Session(cwd="/tmp")
    calls = ModelClient(s).tool_calls(_Msg())
    assert len(calls) == 1
    assert calls[0].args == ["printf 'subject\n\nbody line'"]
