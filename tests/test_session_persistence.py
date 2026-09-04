import asyncio
import json
import threading

import pytest

from wizolt.cli import CommandLoop
from wizolt.config import (
    Config,
)
from wizolt.engine import Agent
from wizolt.session import Session, SessionSnapshotCodec, SessionSnapshotStore
from wizolt.session.store import SnapshotWritePlan


def session_with_data_dir(tmp_path):
    """Session targeting tmp_path as data_dir (avoids touching ~/.wizolt)."""
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


async def test_first_save_writes_init_line(tmp_path):
    """First save writes a single init line with full snapshot data."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.store_tool_result("Read", ["foo.py"], "# content")
    await s.save_snapshot()

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






































































































































async def _resumed_transcript(tmp_path, diff_text, *, lines_cap=None):
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
    await s.save_snapshot()

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






































# --- the write boundary ------------------------------------------------------------------------


async def test_a_blocked_snapshot_write_does_not_stop_the_loop(tmp_path, monkeypatch):
    """The write is a worker's, so a slow disk costs the reader a save, not a prompt."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    release = threading.Event()
    real_execute = SnapshotWritePlan.execute

    def slow_execute(plan):
        release.wait(5)
        return real_execute(plan)

    monkeypatch.setattr(SnapshotWritePlan, "execute", slow_execute)
    beats = 0

    async def heartbeat():
        nonlocal beats
        while True:
            beats += 1
            await asyncio.sleep(0.001)

    pulse = asyncio.ensure_future(heartbeat())
    save = asyncio.ensure_future(s.save_snapshot())
    await asyncio.sleep(0.05)
    assert beats > 5
    release.set()
    assert await save == s.uid
    pulse.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pulse


async def test_concurrent_saves_write_in_capture_order_and_stay_resumable(tmp_path):
    """Two saves in flight are serialized by the session's own gate, so the second computes its
    delta from what the first actually wrote rather than from whatever is current when it lands."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "first"})
    first = asyncio.ensure_future(s.save_snapshot())
    s.messages.append({"role": "user", "content": "second"})
    second = asyncio.ensure_future(s.save_snapshot())

    assert await asyncio.gather(first, second) == [s.uid, s.uid]

    restored = Session.load_snapshot(s.uid, config=s.config)
    assert visible_contents(restored.messages) == ["first", "second"]


async def test_input_queued_during_a_save_lands_in_the_next_delta(tmp_path, monkeypatch):
    """The plan is frozen before the worker starts, so a keystroke that arrives mid-write is newer
    than the captured marker: it is neither half-written into this record nor lost."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    entered, release = threading.Event(), threading.Event()
    real_execute = SnapshotWritePlan.execute

    def slow_execute(plan):
        entered.set()
        release.wait(5)
        return real_execute(plan)

    monkeypatch.setattr(SnapshotWritePlan, "execute", slow_execute)

    save = asyncio.ensure_future(s.save_snapshot())
    await asyncio.to_thread(entered.wait, 5)
    s.enqueue_user_input("typed while saving")
    release.set()
    await save

    monkeypatch.undo()
    assert [item.text for item in Session.load_snapshot(s.uid, config=s.config).pending_user_inputs] == []
    await s.save_snapshot()
    assert [item.text for item in Session.load_snapshot(s.uid, config=s.config).pending_user_inputs] == ["typed while saving"]


async def test_cancelling_a_save_commits_the_marker_it_captured(tmp_path, monkeypatch):
    """The bytes reached disk, so the marker has to advance: leaving it behind would make the next
    save append the same records a second time."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real_execute = SnapshotWritePlan.execute

    def slow_execute(plan):
        entered.set()
        release.wait(5)
        try:
            return real_execute(plan)
        finally:
            finished.set()

    monkeypatch.setattr(SnapshotWritePlan, "execute", slow_execute)

    save = asyncio.ensure_future(s.save_snapshot())
    await asyncio.to_thread(entered.wait, 5)
    save.cancel()
    await asyncio.sleep(0.05)
    assert not finished.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await save

    assert finished.is_set()
    assert s._snapshot_saved  # the captured marker was installed even though cancellation won
    monkeypatch.undo()
    s.messages.append({"role": "user", "content": "after"})
    await s.save_snapshot()
    assert visible_contents(Session.load_snapshot(s.uid, config=s.config).messages) == ["hello", "after"]


async def test_a_failed_write_leaves_the_markers_alone_and_the_next_save_retries(tmp_path, monkeypatch):
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})

    def boom(_plan):
        raise OSError("disk went away")

    monkeypatch.setattr(SnapshotWritePlan, "execute", boom)

    with pytest.raises(OSError, match="disk went away"):
        await s.save_snapshot()

    assert s._snapshot_saved == {}
    monkeypatch.undo()

    await s.save_snapshot()

    assert visible_contents(Session.load_snapshot(s.uid, config=s.config).messages) == ["hello"]


def test_the_save_gate_is_rebound_for_a_later_loop(tmp_path):
    """A `Session` outlives loops; a lock created on a closed one is not a lock.

    Synchronous on purpose: the contract is behavior across two separate `asyncio.run` calls."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})

    asyncio.run(s.save_snapshot())
    first = s._snapshot_gate
    s.messages.append({"role": "user", "content": "second run"})
    asyncio.run(s.save_snapshot())

    assert s._snapshot_gate is not first
    assert visible_contents(Session.load_snapshot(s.uid, config=s.config).messages) == ["hello", "second run"]
