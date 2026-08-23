"""session snapshot (split from tests/test_session_persistence.py)."""
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
from test_session_persistence import log_path, read_jsonl, read_lines, session_with_data_dir

def test_transcript_diff_preview_is_bounded(tmp_path):
    s = session_with_data_dir(tmp_path)
    key = s.store_tool_result("Edit", ["x.py"], "done")
    s.store_turn_diff(key, 1, "x.py", "+line\n" * 20_000)
    s.save_snapshot()

    preview = read_jsonl(log_path(s))[0]["transcript_turn_diffs"][0]["diff"]
    assert len(preview) < TurnDiff.TRANSCRIPT_CHAR_LIMIT + 100
    assert preview.endswith("see /diff for the retained session diff")

def test_loading_legacy_snapshot_migrates_surviving_history_before_later_compaction(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.messages.extend([{"role": "user", "content": "legacy request"}, {"role": "assistant", "content": "legacy answer"}])
    key = s.store_tool_result("Bash", ["pwd"], str(tmp_path))
    s.store_turn_diff(key, 1, "x.py", "-old\n+new\n")
    s.save_snapshot()

    lines = read_lines(log_path(s))
    for line in lines[1:]:
        line.pop("transcript_messages", None)
        line.pop("active_transcript_messages", None)
        line.pop("transcript_sync", None)
        line.pop("transcript_tool_records", None)
        line.pop("transcript_turn_diffs", None)
    with open(log_path(s), "w", encoding="utf-8") as file:
        for line in lines:
            file.write(json.dumps(line) + "\n")

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    assert [message["content"] for message in restored.transcript_messages] == ["legacy request", "legacy answer"]
    assert [record.key for record in restored.transcript_tool_records] == [key]
    assert [diff.key for diff in restored.transcript_turn_diffs] == [key]

    restored.messages[:] = [{"role": "user", "content": "new compacted context", SESSION_EVENT_KEY: "compaction_checkpoint"}]
    restored.save_snapshot()
    migrated = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    assert [message["content"] for message in migrated.transcript_messages] == ["legacy request", "legacy answer"]

def test_active_transcript_is_replaced_separately_then_committed_once(tmp_path):
    s = session_with_data_dir(tmp_path)
    user = {"role": "user", "content": "working request"}
    s._active_transcript_messages = [user]
    s.save_snapshot()

    first = read_jsonl(log_path(s))[0]
    assert first["transcript_messages"] == []
    assert first["active_transcript_messages"] == [user]

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    assert restored.transcript_messages == [user]
    restored.save_snapshot()

    merged, _, _ = SessionSnapshotStore.read_merged(log_path(s))
    assert merged["transcript_messages"] == [user]
    assert merged["active_transcript_messages"] == []
    loaded_again = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    assert loaded_again.transcript_messages == [user]

def test_transcript_checkpoint_does_not_hash_the_saved_prefix(tmp_path, monkeypatch):
    s = session_with_data_dir(tmp_path)
    s.transcript_messages.extend({"role": "user", "content": str(index)} for index in range(2_000))
    s.save_snapshot()
    saved_prefix = s.transcript_messages
    original_digest = SessionSnapshotCodec.digest

    def guarded_digest(value):
        assert value is not saved_prefix
        assert not isinstance(value, list) or len(value) < 2_000
        return original_digest(value)

    monkeypatch.setattr(SessionSnapshotCodec, "digest", guarded_digest)
    s.transcript_messages.append({"role": "assistant", "content": "new"})
    s.save_snapshot()

    assert read_jsonl(log_path(s))[-1]["transcript_messages"] == [{"role": "assistant", "content": "new"}]

def test_transcript_projection_strips_provider_state_and_keeps_semantic_tool_result():
    assistant = {
        "role": "assistant",
        "content": "checking",
        "_responses_output": [{"type": "reasoning", "encrypted_content": "opaque"}],
        "reasoning_content": "hidden",
        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}],
    }
    tool = {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "tool tr.7 Read file.py\noutput:\nmatching file text\nstatus: failed\nstill a successful Read",
    }

    assert SessionSnapshotCodec.transcript_message(assistant) == {
        "role": "assistant",
        "content": "checking",
        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}],
    }
    assert SessionSnapshotCodec.transcript_message(tool) == {
        "role": "tool",
        "tool_call_id": "call-1",
        "result_key": "tr.7",
        "status": "ok",
    }
    assert SessionSnapshotCodec.transcript_message(
        {"role": "tool", "tool_call_id": "call-2", "content": "tool - Read missing.py\nstatus: failed\noutput:\nmissing"}
    ) == {"role": "tool", "tool_call_id": "call-2", "result_key": "", "status": "failed"}

    assistant["tool_calls"][0]["function"]["arguments"] = "x" * (SessionSnapshotCodec.TRANSCRIPT_TOOL_ARGUMENT_CHAR_LIMIT + 1)
    projected = SessionSnapshotCodec.transcript_message(assistant)
    assert projected is not None
    assert projected["tool_calls"][0]["function"]["arguments"] == "{}"
    assert projected["tool_calls"][0]["arguments_truncated"] is True

def test_old_version_write_after_transcript_sync_is_detected(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "first"})
    s.transcript_messages.append({"role": "user", "content": "first"})
    s.save_snapshot()

    SessionSnapshotStore.write_jsonl(
        log_path(s),
        {"messages": [{"role": "assistant", "content": "written by old version"}]},
        mode="a",
    )

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    assert restored.transcript_incomplete is True
    assert [message["content"] for message in restored.transcript_messages] == ["first"]

def test_delta_omits_messages_when_nothing_new(tmp_path):
    """Delta line omits the messages key when no new messages."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hi"})
    s.save_snapshot()  # init

    # No new messages
    s.save_snapshot()  # delta

    lines = read_jsonl(log_path(s))
    delta = lines[1]
    assert "messages" not in delta

def test_delta_omits_tool_records_when_nothing_new(tmp_path):
    """Delta line omits tool_records when no new tool calls."""
    s = session_with_data_dir(tmp_path)
    s.store_tool_result("Bash", ["pwd"], "/home")
    s.save_snapshot()  # init

    s.messages.append({"role": "user", "content": "more"})
    s.save_snapshot()  # delta

    lines = read_jsonl(log_path(s))
    delta = lines[1]
    assert "messages" in delta
    assert "tool_records" not in delta  # No new tool calls since init
    assert "tool_results" not in delta

def test_delta_omits_unchanged_turn_diffs_without_serializing_payload(tmp_path, monkeypatch):
    s = session_with_data_dir(tmp_path)
    s.store_turn_diff("tr.1", 1, "large.py", "-old\n+new\n", before="old\n" * 1000, after="new\n" * 1000, round=1)
    s.save_snapshot()  # init

    def fail_turn_diff(_diff, _blobs):
        raise AssertionError("unchanged turn diffs should not be serialized")

    monkeypatch.setattr(SessionSnapshotCodec, "turn_diff", fail_turn_diff)
    s.messages.append({"role": "user", "content": "next"})
    s.save_snapshot()  # delta

    lines = read_jsonl(log_path(s))
    assert "turn_diffs" not in lines[1]
    assert "turn_diffs_replace" not in lines[1]

def test_file_snapshots_are_stored_once_by_content_hash(tmp_path):
    """Editing a file repeatedly makes each version appear twice — one edit's `after` is the next
    edit's `before`. The log stores each version once and references it by hash."""
    s = session_with_data_dir(tmp_path)
    versions = [f"v{i}\n" for i in range(4)]
    for turn, (before, after) in enumerate(itertools.pairwise(versions), start=1):
        s.store_turn_diff(f"tr.{turn}", turn, "x.py", f"-{before}+{after}", before=before, after=after, round=turn)
        s.save_snapshot()

    lines = read_lines(log_path(s))
    blobs = [line for line in lines if "blob" in line]

    assert sorted(line["text"] for line in blobs) == versions
    assert len({line["blob"] for line in blobs}) == len(blobs)  # each hash written once
    entry = [line for line in lines if "turn_diffs" in line][-1]["turn_diffs"][0]
    assert entry["before_blob"] and entry["after_blob"]
    assert "before" not in entry and "after" not in entry

def test_turn_diff_snapshots_survive_a_roundtrip(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.store_turn_diff("tr.1", 1, "x.py", "-old\n+new\n", before="old\n", after="new\n", round=1)
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))

    assert [(d.key, d.path, d.before, d.after) for d in restored.turn_diffs] == [("tr.1", "x.py", "old\n", "new\n")]
