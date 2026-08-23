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


def session_with_data_dir(tmp_path):
    """Session targeting tmp_path as data_dir (avoids touching ~/.minacode)."""
    return Session(
        cwd=str(tmp_path),
        config=Config(data_dir=str(tmp_path)),
    )


def log_path(s):
    """Path of a session's log inside its project shard."""
    return SessionSnapshotStore.session_path(s.config.data_dir, s.cwd, s.uid)


def project_dir(s):
    return SessionSnapshotStore.project_dir(s.config.data_dir, s.cwd)


def read_jsonl(path) -> list[dict]:
    """Snapshot and delta lines, with the header line dropped."""
    return read_lines(path)[1:]


def visible_contents(messages):
    return [message["content"] for message in messages if not SessionSnapshotCodec.is_internal_message(message)]


def read_lines(path) -> list[dict]:
    """Every JSON line, header included."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_first_save_writes_init_line(tmp_path):
    """First save writes a single init line with full snapshot data."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.store_tool_result("Read", ["foo.py"], "# content")
    s.save_snapshot()

    lines = read_jsonl(log_path(s))
    assert len(lines) == 1
    init = lines[0]
    assert init["uid"] == s.uid
    assert init["messages"] == [{"role": "user", "content": "hello"}]
    assert init["tool_counter"] == 1
    assert init["tool_records"][0]["output"] == "# content"
    assert "usage" in init
    assert "state" in init
    # Runtime/config and derivable data are NOT stored in the snapshot
    assert "config" not in init
    assert "settings" not in init
    assert "tool_results" not in init






































































































































def _resumed_transcript(tmp_path, diff_text, *, lines_cap=None):
    """Save a session holding one Edit call plus its diff, resume it, and capture the replay."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "change it"})
    s.messages.append(
        {
            "role": "assistant",
            "content": "Updating.",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "Edit", "arguments": '{"path": "x.py"}'}}],
        }
    )
    s.store_tool_result("Edit", ["x.py"], '<Edit path="x.py"/>')
    s.store_turn_diff("tr.1", 1, "x.py", diff_text, before="a\n", after="b\n", round=1)
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    output = []
    loop = CommandLoop(Agent(restored, output_fn=output.append), output_fn=output.append)
    if lines_cap is not None:
        loop.TRANSCRIPT_DIFF_LINES = lines_cap
    loop.render_resumed_session()
    return "\n".join(str(item) for item in output)




















# ---------------------------------------------------------------------------
# Transcript replay resilience (regression: multi-line tool arguments such as
# a Bash command with embedded newlines must not crash --resume rendering)
# ---------------------------------------------------------------------------




































