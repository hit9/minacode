import itertools
import json
import os
import time
from typing import ClassVar

import pytest

from minacode.base import SESSION_EVENT_KEY, MinacodeError
from minacode.cli import CommandLoop
from minacode.cli.commands import compact
from minacode.config import (
    Config,
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


def test_pending_user_inputs_persist_and_restore(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.enqueue_user_input("queued one")
    s.enqueue_user_input("queued two")

    s.save_snapshot()

    lines = read_jsonl(log_path(s))
    assert lines[0]["pending_user_inputs"] == ["queued one", "queued two"]
    restored = Session.load_snapshot(s.uid, config=s.config)
    assert [item.text for item in restored.pending_user_inputs] == ["queued one", "queued two"]
    assert all(not item.inflight for item in restored.pending_user_inputs)


def test_pending_user_input_delta_replaces_queue_state(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "active"})
    s.save_snapshot()
    s.enqueue_user_input("queued")
    s.save_snapshot()
    s.pending_user_inputs.clear()
    s.save_snapshot()

    lines = read_jsonl(log_path(s))
    assert lines[1]["pending_user_inputs"] == ["queued"]
    assert lines[2]["pending_user_inputs"] == []
    restored = Session.load_snapshot(s.uid, config=s.config)
    assert restored.pending_user_inputs == []


def test_latest_pointer_created_on_first_save(tmp_path):
    """First save creates the latest pointer file."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    latest_path = os.path.join(project_dir(s), "latest")
    assert os.path.exists(latest_path)
    with open(latest_path) as file:
        assert file.read().strip() == s.uid


def test_second_save_writes_delta_with_only_new_data(tmp_path):
    """Second save appends a delta line containing only new messages and tool records."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "first"})
    s.store_tool_result("Read", ["a.py"], "# a")
    s.save_snapshot()  # init

    s.messages.append({"role": "assistant", "content": "reply"})
    s.store_tool_result("Search", ["pat"], "result")
    s.save_snapshot()  # delta

    lines = read_jsonl(log_path(s))
    assert len(lines) == 2
    delta = lines[1]
    # Only new data in delta
    assert delta["messages"] == [{"role": "assistant", "content": "reply"}]
    assert [record["key"] for record in delta["tool_records"]] == ["tr.2"]
    assert delta["tool_records"][0]["output"] == "result"
    assert delta["tool_counter"] == 2


def test_transcript_appends_when_model_messages_are_replaced(tmp_path):
    s = session_with_data_dir(tmp_path)
    first = {"role": "user", "content": "original request"}
    s.messages.append(first)
    s.transcript_messages.append(first)
    s.save_snapshot()

    s.messages[:] = [{"role": "user", "content": "compacted context", SESSION_EVENT_KEY: "compaction_checkpoint"}]
    s.transcript_messages.append({"role": "assistant", "content": "original answer"})
    s.save_snapshot()

    delta = read_jsonl(log_path(s))[1]
    assert delta["messages_replace"] == s.messages
    assert delta["transcript_messages"] == [{"role": "assistant", "content": "original answer"}]

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    assert restored.messages[0]["content"] == "compacted context"
    assert [message["content"] for message in restored.transcript_messages] == ["original request", "original answer"]


def test_transcript_tool_metadata_does_not_duplicate_retained_output_or_file_snapshots(tmp_path):
    s = session_with_data_dir(tmp_path)
    key = s.store_tool_result("Edit", ["x.py"], "full retained output")
    s.store_turn_diff(key, 1, "x.py", "-old\n+new\n", before="old\n", after="new\n")
    s.save_snapshot()

    snapshot = next(line for line in read_jsonl(log_path(s)) if "uid" in line)
    assert "transcript_tool_records" not in snapshot
    assert snapshot["transcript_turn_diffs"] == [{"key": key, "turn": 1, "path": "x.py", "diff": "-old\n+new\n", "round": 0}]
    assert "before_blob" not in snapshot["transcript_turn_diffs"][0]


def test_materialized_tool_output_survives_the_asset_collector(tmp_path):
    """The reported failure: the asset collector kept only image refs, so the file a truncated tool
    result was materialized to was deleted on the very next save -- while the marker in the
    conversation went on advertising its path, sending the model hunting for a file that was gone."""
    s = session_with_data_dir(tmp_path)
    large = "\n".join(f"line {index}" for index in range(20000))
    key = s.store_tool_result("Bash", ["big"], large)
    marker = ContextManager(s).bound_output(large, key)
    path = os.path.join(s.images.assets_dir(), key + ".txt")
    assert f'file="{path}"' in marker

    s.messages.append({"role": "user", "content": marker})
    s.save_snapshot()
    s.save_snapshot()  # the collector runs on every save, not only the first

    with open(path, encoding="utf-8") as file:
        assert file.read() == large


def test_owns_asset_covers_the_assets_directory_and_nothing_else(tmp_path):
    """The predicate that waives the out-of-workspace approval, so its edges are a permission
    boundary: a sibling whose name merely starts the same way is not inside it, and neither is a
    parent reached back through it."""
    s = session_with_data_dir(tmp_path)
    assets = s.images.assets_dir()
    os.makedirs(assets, exist_ok=True)
    parent = os.path.dirname(assets)

    assert s.owns_asset(os.path.join(assets, "tr.1.txt")) is True
    assert s.owns_asset(os.path.join(assets, "nested", "deep.txt")) is True
    assert s.owns_asset(assets) is True
    assert s.owns_asset(assets + "-sibling/secret.txt") is False  # prefix match is not containment
    assert s.owns_asset(os.path.join(assets, "..", "other.jsonl")) is False  # normalized, not textual
    assert s.owns_asset(parent) is False
    assert s.owns_asset(str(tmp_path / "elsewhere.txt")) is False


def test_materialized_tool_output_expires_with_its_tool_result(tmp_path):
    """It is retained by its result, not forever: once the result is pruned the file is collected,
    so a long session cannot accumulate an asset for every large output it ever saw."""
    s = session_with_data_dir(tmp_path)
    large = "\n".join(f"line {index}" for index in range(20000))
    key = s.store_tool_result("Bash", ["big"], large)
    ContextManager(s).bound_output(large, key)
    path = os.path.join(s.images.assets_dir(), key + ".txt")
    assert os.path.exists(path)

    s.tool_results.pop(key)
    s.save_snapshot()

    assert not os.path.exists(path)


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


def test_oversized_snapshots_are_dropped_before_reaching_the_log(tmp_path):
    """Snapshots over the size limit are still discarded, and leave no blob behind."""
    s = session_with_data_dir(tmp_path)
    huge = "x" * (TurnDiff.SNAPSHOT_CHAR_LIMIT + 1)
    s.store_turn_diff("tr.1", 1, "big.py", "-o\n+n\n", before=huge, after=huge, round=1)
    s.save_snapshot()

    lines = read_lines(log_path(s))
    entry = [line for line in lines if "turn_diffs" in line][-1]["turn_diffs"][0]

    assert not [line for line in lines if "blob" in line]
    assert (entry["before_blob"], entry["after_blob"]) == ("", "")
    assert Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path)).turn_diffs[0].before == ""


def test_rewriting_the_retained_window_does_not_rewrite_snapshots(tmp_path):
    """Once the 100-entry cap starts evicting, every save rewrites the whole window. It must
    rewrite references only — the snapshots are already in the log."""
    s = session_with_data_dir(tmp_path)
    big = "x" * 100_000
    for i in range(100):
        s.store_turn_diff(f"tr.{i}", i, "a.py", "-o\n+n\n", before=big, after=big + str(i), round=i)
    s.save_snapshot()
    size_before = os.path.getsize(log_path(s))

    s.store_turn_diff("tr.100", 100, "a.py", "-o\n+n\n", before=big, after=big + "100", round=100)
    s.save_snapshot()

    lines = read_lines(log_path(s))
    assert "turn_diffs_replace" in lines[-1]  # the window was rewritten in full
    assert len(lines[-1]["turn_diffs_replace"]) == 100
    # One new snapshot (~100KB), not 100 of them (~10MB).
    assert os.path.getsize(log_path(s)) - size_before < 400_000


def test_resumed_session_does_not_rewrite_existing_blobs(tmp_path):
    """A resumed session knows which snapshots its log already holds."""
    s = session_with_data_dir(tmp_path)
    s.store_turn_diff("tr.1", 1, "x.py", "-old\n+new\n", before="old\n", after="new\n", round=1)
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    restored.store_turn_diff("tr.2", 2, "x.py", "-new\n+newer\n", before="new\n", after="newer\n", round=2)
    restored.save_snapshot()

    blobs = [line["text"] for line in read_lines(log_path(s)) if "blob" in line]
    assert sorted(blobs) == ["new\n", "newer\n", "old\n"]  # "new\n" not stored a second time


def test_load_merges_init_and_deltas(tmp_path):
    """load_snapshot reads and merges all lines, returning the full session state."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "q1"})
    s.store_tool_result("Read", ["f.py"], "# f")
    s.save_snapshot()  # init

    s.messages.append({"role": "assistant", "content": "a1"})
    s.store_tool_result("Search", ["pat"], "found")
    s.save_snapshot()  # delta

    s.messages.append({"role": "user", "content": "q2"})
    s.save_snapshot()  # delta (no new tool results)

    s2 = Session.load_snapshot(s.uid, config=s.config)
    # All messages across all lines
    assert [m["content"] for m in s2.messages[:3]] == ["q1", "a1", "q2"]
    # Fourth message is a durable user-role resume event.
    assert s2.messages[3]["role"] == "user"
    assert s2.messages[3][SESSION_EVENT_KEY] == "resumed"
    assert s2.messages[3]["content"].startswith('<session_event type="resumed" at="')
    # All tool results
    assert s2.tool_results["tr.1"] == "# f"
    assert s2.tool_results["tr.2"] == "found"
    assert s2.tool_counter == 2


def test_load_preserves_uid(tmp_path):
    """load_snapshot preserves the original uid."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    s2 = Session.load_snapshot(s.uid, config=s.config)
    assert s2.uid == s.uid


def test_load_with_latest_alias(tmp_path):
    """load_snapshot with uid='latest' resolves this project's latest pointer."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    s2 = Session.load_snapshot("latest", config=s.config, cwd=str(tmp_path))
    assert s2.uid == s.uid


def test_load_with_last_alias(tmp_path):
    """load_snapshot with uid='last' resolves this project's latest pointer."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    s2 = Session.load_snapshot("last", config=s.config, cwd=str(tmp_path))
    assert s2.uid == s.uid


def test_latest_uid_ignores_newer_sessions_from_other_projects(tmp_path):
    data_dir = tmp_path / "data"
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    config = Config(data_dir=str(data_dir))

    project_session = Session(cwd=str(project), config=config)
    project_session.messages.append({"role": "user", "content": "project"})
    project_session.save_snapshot()

    other_session = Session(cwd=str(other), config=config)
    other_session.messages.append({"role": "user", "content": "other"})
    other_session.save_snapshot()
    os.utime(log_path(project_session), (1, 1))
    os.utime(log_path(other_session), (2, 2))

    assert SessionSnapshotStore.latest_uid(str(data_dir), str(project)) == project_session.uid
    assert SessionSnapshotStore.latest_uid(str(data_dir), str(other)) == other_session.uid


def test_latest_uid_returns_empty_without_project_session(tmp_path):
    assert SessionSnapshotStore.latest_uid(str(tmp_path / "missing"), str(tmp_path)) == ""


def test_sessions_are_sharded_per_project(tmp_path):
    """Two projects sharing a data_dir get separate directories, so listing or deleting one
    project's history never touches the other's."""
    data_dir, project, other = tmp_path / "data", tmp_path / "project", tmp_path / "other"
    project.mkdir()
    other.mkdir()
    config = Config(data_dir=str(data_dir))

    first = Session(cwd=str(project), config=config)
    first.messages.append({"role": "user", "content": "project"})
    first.save_snapshot()
    second = Session(cwd=str(other), config=config)
    second.messages.append({"role": "user", "content": "other"})
    second.save_snapshot()

    assert project_dir(first) != project_dir(second)
    assert sorted(os.listdir(project_dir(first))) == sorted(["latest", first.uid + ".jsonl", first.uid + ".meta.json"])
    assert sorted(os.listdir(project_dir(second))) == sorted(["latest", second.uid + ".jsonl", second.uid + ".meta.json"])


def test_project_slug_separates_same_named_directories(tmp_path):
    """Two checkouts named alike hash to different shards."""
    left, right = tmp_path / "a" / "repo", tmp_path / "b" / "repo"
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    slugs = [SessionSnapshotStore.project_slug(str(path)) for path in (left, right)]

    assert all(slug.startswith("repo-") for slug in slugs)
    assert slugs[0] != slugs[1]


def test_load_finds_a_session_by_uid_from_any_directory(tmp_path):
    """An explicit UID resolves regardless of which project it belongs to."""
    data_dir, project = tmp_path / "data", tmp_path / "project"
    project.mkdir()
    config = Config(data_dir=str(data_dir))
    s = Session(cwd=str(project), config=config)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    loaded = Session.load_snapshot(s.uid, config=config, cwd=str(tmp_path))

    assert loaded.uid == s.uid
    assert loaded.cwd == str(project)


def test_latest_never_crosses_into_another_project(tmp_path):
    """The regression the shard layout closes: a newer session elsewhere must not be resumable
    as this project's latest."""
    data_dir, project, other = tmp_path / "data", tmp_path / "project", tmp_path / "other"
    project.mkdir()
    other.mkdir()
    config = Config(data_dir=str(data_dir))
    elsewhere = Session(cwd=str(other), config=config)
    elsewhere.messages.append({"role": "user", "content": "other"})
    elsewhere.save_snapshot()

    with pytest.raises(MinacodeError, match="No previous session for this project"):
        Session.load_snapshot("latest", config=config, cwd=str(project))


def test_latest_falls_back_to_newest_log_when_pointer_is_missing(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()
    os.unlink(os.path.join(project_dir(s), "latest"))

    assert SessionSnapshotStore.latest_uid(str(tmp_path), str(tmp_path)) == s.uid


def test_header_line_precedes_the_snapshot(tmp_path):
    """Line 1 is a bounded header, so project queries never parse the conversation behind it."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    header = read_lines(log_path(s))[0]

    assert header == {"v": SessionSnapshotStore.FORMAT_VERSION, "uid": s.uid, "cwd": s.cwd, "created_at": header["created_at"]}
    assert header["created_at"] == s.created_at
    assert header["created_at"][-6:-5] in {"+", "-"}
    assert "messages" not in header


def test_load_rejects_an_unknown_format_version(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()
    lines = read_lines(log_path(s))
    lines[0]["v"] = 99
    with open(log_path(s), "w") as file:
        file.write("\n".join(json.dumps(line) for line in lines) + "\n")

    with pytest.raises(MinacodeError, match="Unsupported session format v99"):
        Session.load_snapshot(s.uid, config=s.config)


def test_load_appends_local_time_resume_event(tmp_path, monkeypatch):
    """Resume is durable user-role context with a local wall time and explicit offset."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    monkeypatch.setattr("minacode.session.local_timestamp", lambda value=None: "2026-07-30T15:04:05+08:00")
    s2 = Session.load_snapshot(s.uid, config=s.config)
    assert len(s2.messages) == 2  # hello + resume marker
    assert s2.messages[-1] == {
        "role": "user",
        "content": '<session_event type="resumed" at="2026-07-30T15:04:05+08:00" />',
        SESSION_EVENT_KEY: "resumed",
    }


def test_save_after_load_produces_a_delta(tmp_path):
    """Save after load appends a delta (not re-init)."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()  # init (line 1)

    s2 = Session.load_snapshot(s.uid, config=s.config)
    # s2 now has messages = [hello, resume_marker]
    s2.messages.append({"role": "assistant", "content": "post-resume"})
    s2.save_snapshot()  # delta (line 2)

    lines = read_jsonl(log_path(s))
    assert len(lines) == 2
    delta = lines[1]
    # Both the lifecycle event and subsequent assistant reply are new append-only history.
    assert delta["messages"][0][SESSION_EVENT_KEY] == "resumed"
    assert delta["messages"][1] == {"role": "assistant", "content": "post-resume"}


def test_repeated_resume_preserves_history(tmp_path):
    """Repeated resume/save cycles keep appending new messages to the same history."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "m1"})
    s.save_snapshot()

    expected = ["m1"]
    for role, content in (("assistant", "a1"), ("user", "m2"), ("assistant", "a2")):
        s = Session.load_snapshot(s.uid, config=s.config)
        assert visible_contents(s.messages) == expected
        s.messages.append({"role": role, "content": content})
        s.save_snapshot()
        expected.append(content)

    loaded = Session.load_snapshot(s.uid, config=s.config)
    assert visible_contents(loaded.messages) == expected
    assert sum(message.get(SESSION_EVENT_KEY) == "resumed" for message in loaded.messages) == 4


def test_resume_marker_is_never_persisted(tmp_path):
    """Resume markers are runtime-only, including when a message rewrite forces replace."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "m1"})
    s.save_snapshot()

    resumed = Session.load_snapshot(s.uid, config=s.config)
    resumed.messages = [
        {"role": "system", "content": f"[Session resumed: uid={s.uid}]"},
        {"role": "user", "content": "rewritten"},
    ]
    resumed.save_snapshot()

    lines = read_jsonl(log_path(s))
    assert lines[-1]["messages_replace"] == [{"role": "user", "content": "rewritten"}]
    assert "[Session resumed:" not in json.dumps(lines)


def test_load_discards_persisted_resume_markers(tmp_path):
    """Older or malformed snapshots may contain resume markers; load should not keep them."""
    s = session_with_data_dir(tmp_path)
    path = log_path(s)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    marker = {"role": "system", "content": f"[Session resumed: uid={s.uid}]"}
    SessionSnapshotStore.write_jsonl(path, SessionSnapshotStore.header(s), mode="w")
    SessionSnapshotStore.write_jsonl(
        path,
        {
            "uid": s.uid,
            "cwd": str(tmp_path),
            "messages": [{"role": "user", "content": "m1"}, marker, {"role": "assistant", "content": "a1"}],
        },
        mode="a",
    )

    loaded = Session.load_snapshot(s.uid, config=s.config)

    assert visible_contents(loaded.messages) == ["m1", "a1"]
    assert sum(1 for m in loaded.messages if SessionSnapshotCodec.is_internal_message(m)) == 1


def test_old_context_layout_migrates_with_one_full_state_checkpoint(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.context_layout_version = 1
    s.state.goal = "ship cache layout"
    s.state.plan = [{"status": "doing", "text": "migrate"}]
    s.state.known = ["old snapshots have no layout field"]
    s.state.check = "one checkpoint"
    s.messages.append({"role": "user", "content": "original request"})
    s.save_snapshot()

    migrated = Session.load_snapshot(s.uid, config=s.config)
    checkpoints = [message for message in migrated.messages if message.get(SESSION_EVENT_KEY) == "state_checkpoint"]
    assert migrated.context_layout_version == 2
    assert len(checkpoints) == 1
    assert "Goal: ship cache layout" in checkpoints[0]["content"]
    assert "- doing: migrate" in checkpoints[0]["content"]
    assert "Check: one checkpoint" in checkpoints[0]["content"]

    migrated.save_snapshot()
    resumed_again = Session.load_snapshot(s.uid, config=s.config)
    assert sum(message.get(SESSION_EVENT_KEY) == "state_checkpoint" for message in resumed_again.messages) == 1


def test_real_legacy_snapshot_without_layout_field_converts_numeric_local_time(tmp_path, monkeypatch):
    s = session_with_data_dir(tmp_path)
    path = log_path(s)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    legacy_created_at = 1_700_000_000.0
    SessionSnapshotStore.write_jsonl(
        path,
        {"v": SessionSnapshotStore.FORMAT_VERSION, "uid": s.uid, "cwd": s.cwd, "created_at": legacy_created_at},
        mode="w",
    )
    SessionSnapshotStore.write_jsonl(
        path,
        {
            "uid": s.uid,
            "cwd": s.cwd,
            "messages": [{"role": "user", "content": "legacy request"}],
            "state": {
                "goal": "migrate a real legacy record",
                "plan": [{"status": "doing", "text": "load"}],
                "known": ["context_layout_version is absent"],
                "check": "local timestamp converted",
            },
        },
        mode="a",
    )
    timestamp_calls = []

    def timestamp(value=None):
        timestamp_calls.append(value)
        return "2023-11-14T17:13:20-05:00" if value is not None else "2026-07-30T15:04:05+08:00"

    monkeypatch.setattr("minacode.session.local_timestamp", timestamp)

    loaded = Session.load_snapshot(s.uid, config=s.config)

    assert timestamp_calls == [legacy_created_at, None]
    assert loaded.created_at == "2023-11-14T17:13:20-05:00"
    assert loaded.context_layout_version == 2
    checkpoint = next(message for message in loaded.messages if message.get(SESSION_EVENT_KEY) == "state_checkpoint")
    assert "Goal: migrate a real legacy record" in checkpoint["content"]
    assert checkpoint["content"].startswith("--- Working State Checkpoint ---")


def test_empty_session_first_save_is_skipped(tmp_path):
    """A session with no recoverable content is not persisted."""
    s = session_with_data_dir(tmp_path)
    assert s.save_snapshot() == ""

    assert not os.path.exists(project_dir(s))


def test_tool_results_roundtrip(tmp_path):
    """Tool results survive save/load."""
    s = session_with_data_dir(tmp_path)
    s.store_tool_result("Bash", ["echo hi"], "hi")
    s.store_tool_result("Read", ["f.py"], "code")
    s.save_snapshot()

    s2 = Session.load_snapshot(s.uid, config=s.config)
    assert s2.tool_results["tr.1"] == "hi"
    assert s2.tool_results["tr.2"] == "code"


def test_tool_records_roundtrip(tmp_path):
    """Tool records survive save/load."""
    s = session_with_data_dir(tmp_path)
    s.store_tool_result("Bash", ["pwd"], "/tmp")
    s.store_tool_result("Search", ["x"], "match")
    s.save_snapshot()

    s2 = Session.load_snapshot(s.uid, config=s.config)
    assert len(s2.tool_records) == 2
    assert s2.tool_records[0].key == "tr.1"
    assert s2.tool_records[0].name == "Bash"
    assert s2.tool_records[0].output == "/tmp"
    assert s2.tool_records[1].key == "tr.2"
    assert s2.tool_records[1].name == "Search"


def test_tool_errors_roundtrip(tmp_path):
    """Tool errors survive save/load."""
    s = session_with_data_dir(tmp_path)
    s.record_tool_error("tr.1", "Bash", ["bad"], "command not found")
    s.save_snapshot()

    s2 = Session.load_snapshot(s.uid, config=s.config)
    assert len(s2.tool_errors) == 1
    assert s2.tool_errors[0].key == "tr.1"
    assert s2.tool_errors[0].error == "command not found"


def test_usage_roundtrip_with_prompt_and_completion_tokens(tmp_path):
    """All usage fields (including prompt_tokens/completion_tokens) survive save/load."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.usage.calls = 3
    s.usage.prompt_tokens = 100
    s.usage.completion_tokens = 50
    s.usage.total_tokens = 150
    s.usage.cached_prompt_tokens = 20
    s.usage.cache_write_prompt_tokens = 12
    s.usage.last_cached_prompt_tokens = 5
    s.usage.last_cache_write_prompt_tokens = 7
    s.save_snapshot()

    s2 = Session.load_snapshot(s.uid, config=s.config)
    assert s2.usage.calls == 3
    assert s2.usage.prompt_tokens == 100
    assert s2.usage.completion_tokens == 50
    assert s2.usage.total_tokens == 150
    assert s2.usage.cached_prompt_tokens == 20
    assert s2.usage.cache_write_prompt_tokens == 12
    assert s2.usage.last_cached_prompt_tokens == 5
    assert s2.usage.last_cache_write_prompt_tokens == 7


def test_agent_state_roundtrip(tmp_path):
    """Agent state (goal, plan, known, check, summary) survives save/load."""
    s = session_with_data_dir(tmp_path)
    s.state.goal = "fix bug"
    s.state.plan = ["step 1", "step 2"]
    s.state.known = ["file at src/a.py"]
    s.state.check = "assert x == 1"
    s.state.summary = "working on it"
    s.state.round_count = 7
    s.save_snapshot()

    s2 = Session.load_snapshot(s.uid, config=s.config)
    assert s2.state.goal == "fix bug"
    assert [vars(item) for item in s2.state.plan] == [{"status": "todo", "text": "step 1"}, {"status": "todo", "text": "step 2"}]
    assert s2.state.known == ["file at src/a.py"]
    assert s2.state.check == "assert x == 1"
    assert s2.state.summary == "working on it"
    assert s2.state.round_count == 7


def test_multiple_deltas_accumulate_correctly(tmp_path):
    """Multiple delta saves accumulate data correctly."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "m1"})
    s.save_snapshot()  # init
    s.messages.append({"role": "assistant", "content": "a1"})
    s.save_snapshot()  # delta 1
    s.messages.append({"role": "user", "content": "m2"})
    s.save_snapshot()  # delta 2
    s.messages.append({"role": "assistant", "content": "a2"})
    s.save_snapshot()  # delta 3

    s2 = Session.load_snapshot(s.uid, config=s.config)
    assert visible_contents(s2.messages) == ["m1", "a1", "m2", "a2"]


def test_multiple_deltas_with_tool_calls(tmp_path):
    """Tool calls across multiple deltas accumulate correctly."""
    s = session_with_data_dir(tmp_path)
    s.store_tool_result("Read", ["a.py"], "# a")
    s.save_snapshot()  # init: tr.1
    s.store_tool_result("Search", ["pat"], "hit")
    s.save_snapshot()  # delta 1: tr.2
    s.store_tool_result("Bash", ["pwd"], "/tmp")
    s.save_snapshot()  # delta 2: tr.3

    s2 = Session.load_snapshot(s.uid, config=s.config)
    assert s2.tool_results["tr.1"] == "# a"
    assert s2.tool_results["tr.2"] == "hit"
    assert s2.tool_results["tr.3"] == "/tmp"
    assert s2.tool_counter == 3
    assert len(s2.tool_records) == 3


def test_load_missing_snapshot_raises_error(tmp_path):
    """Loading a non-existent session raises MinacodeError."""
    with pytest.raises(MinacodeError, match="Session snapshot not found"):
        Session.load_snapshot("nonexistent-uid", config=Config(data_dir=str(tmp_path)))


@pytest.mark.parametrize("alias", ["latest", "last"])
def test_resolve_uid_without_a_project_session(tmp_path, alias):
    """Resolving an alias in a project with no sessions raises MinacodeError."""
    with pytest.raises(MinacodeError, match="No previous session for this project"):
        SessionSnapshotStore.resolve_uid(alias, str(tmp_path), str(tmp_path))


def test_resolve_uid_passthrough_normal_uid(tmp_path):
    """Resolving a normal uid (not an alias) returns it as-is."""
    assert SessionSnapshotStore.resolve_uid("my-uid", str(tmp_path), str(tmp_path)) == "my-uid"


def test_jsonl_file_is_append_only(tmp_path):
    """Multiple saves only add lines, never rewrite the file."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()  # l1
    s.save_snapshot()  # l2
    s.save_snapshot()  # l3
    s.save_snapshot()  # l4

    lines = read_jsonl(log_path(s))
    assert len(lines) == 4
    # First line has all fields (init)
    assert "uid" in lines[0]
    assert "messages" not in lines[1]
    assert "tool_records" not in lines[2]
    assert "tool_results" not in lines[3]


def test_runtime_session_retention_defaults_to_seven_days():
    settings = RuntimeSettings.from_dict({})

    assert settings.session_retention_days == 7


def test_clean_expired_sessions_removes_old_files_and_latest(tmp_path):
    s = session_with_data_dir(tmp_path)
    old = session_with_data_dir(tmp_path)
    old.messages.append({"role": "user", "content": "old"})
    old.save_snapshot()
    old_path = log_path(old)
    stale_time = time.time() - 8 * 86400
    os.utime(old_path, (stale_time, stale_time))

    assert SessionSnapshotStore.clean_expired(s) == 1

    assert not os.path.exists(old_path)
    # The pointer named the expired session, and the emptied shard is pruned with it.
    assert not os.path.exists(project_dir(old))


def test_clean_expired_sessions_skips_current_session(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "current"})
    s.save_snapshot()
    path = log_path(s)
    stale_time = time.time() - 8 * 86400
    os.utime(path, (stale_time, stale_time))

    assert SessionSnapshotStore.clean_expired(s) == 0

    assert os.path.exists(path)


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
    loop.agent.model.compact = lambda _context, *_args: {"summary": "a compacted summary"}
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


# ---------------------------------------------------------------------------
# Transcript replay resilience (regression: multi-line tool arguments such as
# a Bash command with embedded newlines must not crash --resume rendering)
# ---------------------------------------------------------------------------


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


def test_snapshot_messages_strips_non_persistable_roles(tmp_path):
    s = Session(cwd=str(tmp_path))
    s.messages = [
        {"role": "system", "content": "[Session resumed: old-session-id]"},
        {"role": "user", "content": "hello"},
    ]
    messages = SessionSnapshotCodec.snapshot_messages(s)
    # Internal resume marker is stripped; user message is kept.
    roles = [m["role"] for m in messages]
    assert "system" not in roles
    assert "user" in roles
    assert len(messages) == 1


def test_session_name_latches_then_follows_the_goal(tmp_path):
    s = session_with_data_dir(tmp_path)
    assert s.name == ""

    s.messages.append({"role": "user", "content": "fix the fd leak in MCPFileTokenStore\nsecond line"})
    s.save_snapshot()
    # Nothing to derive from until there is a message, then the opening line names the session.
    assert (s.name, s.state.name_source) == ("fix the fd leak in MCPFileTokenStore", "input")

    s.state.goal = "close every descriptor opened by the token store"
    s.save_snapshot()
    # A goal is a better description of the same work, so it takes over from the opening line.
    assert (s.name, s.state.name_source) == ("close every descriptor opened by the token store", "goal")

    s.rename("token store cleanup")
    s.save_snapshot()
    s.state.goal = "something else entirely"
    s.save_snapshot()
    # A name the user chose is never replaced by a derived one.
    assert (s.name, s.state.name_source) == ("token store cleanup", "user")
    assert Session.load_snapshot(s.uid, config=s.config).name == "token store cleanup"


def test_session_name_does_not_change_when_goal_changes(tmp_path):
    """Once the name is derived from a goal, later goal changes do not overwrite it."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "fix the parser"})
    s.save_snapshot()
    assert (s.name, s.state.name_source) == ("fix the parser", "input")

    s.state.goal = "rewrite the tokenizer"
    s.save_snapshot()
    assert (s.name, s.state.name_source) == ("rewrite the tokenizer", "goal")

    s.state.goal = "add error recovery to the parser"
    s.save_snapshot()
    # Goal changed, but the name was already latched from the first goal — stays put.
    assert (s.name, s.state.name_source) == ("rewrite the tokenizer", "goal")
    assert Session.load_snapshot(s.uid, config=s.config).name == "rewrite the tokenizer"


def test_session_name_survives_compaction_dropping_the_opening_message(tmp_path):
    from minacode.prompts import COMPACTION_SUMMARY_TITLE

    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "add a session picker"})
    s.save_snapshot()

    s.messages = [
        {"role": "user", "content": COMPACTION_SUMMARY_TITLE + "\nearlier work"},
        {"role": "user", "content": "now also sort them by date"},
    ]
    s.save_snapshot()

    # Compaction replaces the opening message; a name derived afresh here would silently rewrite
    # what the session has been listed as since it started.
    assert s.name == "add a session picker"
    assert s.opening_text() == "now also sort them by date"


def test_listing_sessions_reads_no_logs(tmp_path, monkeypatch):
    config = Config(data_dir=str(tmp_path / "data"))
    project = tmp_path / "project"
    project.mkdir()
    first = Session(cwd=str(project), config=config)
    first.messages.append({"role": "user", "content": "older session"})
    first.save_snapshot()
    second = Session(cwd=str(project), config=config)
    second.messages.append({"role": "user", "content": "newer session"})
    second.state.round_count = 3
    second.save_snapshot()
    os.utime(SessionSnapshotStore.session_path(config.data_dir, str(project), first.uid), (1, 1))

    real_open = open

    def guard(file, *args, **kwargs):
        assert not str(file).endswith(".jsonl"), f"listing opened a session log: {file}"
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guard)
    entries = SessionSnapshotStore.list_sessions(config.data_dir, str(project))

    assert [entry.uid for entry in entries] == [second.uid, first.uid]
    assert [entry.name for entry in entries] == ["newer session", "older session"]
    assert entries[0].rounds == 3
    assert entries[0].cwd == str(project)


def test_listing_survives_a_missing_sidecar(tmp_path):
    config = Config(data_dir=str(tmp_path / "data"))
    s = Session(cwd=str(tmp_path), config=config)
    s.messages.append({"role": "user", "content": "unlabelled"})
    s.save_snapshot()
    os.unlink(SessionSnapshotStore.meta_path(config.data_dir, s.cwd, s.uid))

    entry = SessionSnapshotStore.list_sessions(config.data_dir, s.cwd)[0]

    # The log is what makes a session real; the sidecar only labels it.
    assert (entry.uid, entry.name) == (s.uid, "")
    assert entry.label() == s.uid


def test_expired_sessions_take_their_sidecar_with_them(tmp_path):
    config = Config(data_dir=str(tmp_path / "data"))
    s = Session(cwd=str(tmp_path), config=config)
    s.messages.append({"role": "user", "content": "old"})
    s.save_snapshot()
    stale = Session(cwd=str(tmp_path), config=config)
    stale.messages.append({"role": "user", "content": "stale"})
    stale.save_snapshot()
    meta = SessionSnapshotStore.meta_path(config.data_dir, stale.cwd, stale.uid)
    old = time.time() - 40 * 86400
    os.utime(SessionSnapshotStore.session_path(config.data_dir, stale.cwd, stale.uid), (old, old))
    s.settings.session_retention_days = 30

    assert SessionSnapshotStore.clean_expired(s) == 1
    assert not os.path.exists(meta)


def test_resume_accepts_a_name_or_uid_prefix(tmp_path):
    config = Config(data_dir=str(tmp_path / "data"))
    project = tmp_path / "project"
    project.mkdir()
    s = Session(cwd=str(project), config=config)
    s.messages.append({"role": "user", "content": "teach the status bar to breathe"})
    s.save_snapshot()

    for query in ("status bar", "TEACH the status", s.uid[:8]):
        assert SessionSnapshotStore.resolve_uid(query, config.data_dir, str(project)) == s.uid

    # A search from another directory still finds it: the user moved, the session did not.
    assert SessionSnapshotStore.resolve_uid("status bar", config.data_dir, str(tmp_path)) == s.uid
    assert Session.load_snapshot("status bar", config=config, cwd=str(project)).uid == s.uid


def test_ambiguous_resume_names_its_candidates(tmp_path):
    config = Config(data_dir=str(tmp_path / "data"))
    first = Session(cwd=str(tmp_path), config=config)
    first.messages.append({"role": "user", "content": "rename the sweep constants"})
    first.save_snapshot()
    second = Session(cwd=str(tmp_path), config=config)
    second.messages.append({"role": "user", "content": "rename the glow styles"})
    second.save_snapshot()

    with pytest.raises(MinacodeError) as error:
        SessionSnapshotStore.resolve_uid("rename the", config.data_dir, str(tmp_path))

    # Guessing between them would resume the wrong work silently.
    assert "2 sessions match" in str(error.value)
    assert first.uid in str(error.value) and second.uid in str(error.value)


def test_listing_survives_a_malformed_sidecar(tmp_path):
    config = Config(data_dir=str(tmp_path / "data"))
    s = Session(cwd=str(tmp_path), config=config)
    s.messages.append({"role": "user", "content": "labelled session"})
    s.save_snapshot()
    # A hand-edited or torn sidecar: valid JSON, but the turn count is not a number.
    with open(SessionSnapshotStore.meta_path(config.data_dir, s.cwd, s.uid), "w", encoding="utf-8") as file:
        json.dump({"name": "kept", "opening": "labelled session", "rounds": "many", "cwd": s.cwd}, file)

    entry = SessionSnapshotStore.list_sessions(config.data_dir, s.cwd)[0]

    # The bad turn count is dropped, not the session: one corrupt cache must not break the picker.
    assert (entry.uid, entry.name, entry.rounds) == (s.uid, "kept", 0)


def test_search_widens_only_after_a_miss(tmp_path, monkeypatch):
    config = Config(data_dir=str(tmp_path / "data"))
    here = tmp_path / "here"
    here.mkdir()
    local = Session(cwd=str(here), config=config)
    local.messages.append({"role": "user", "content": "a local session"})
    local.save_snapshot()
    calls = []
    real = SessionSnapshotStore.list_sessions

    def spy(cls, data_dir, cwd="", *, all_projects=False):
        calls.append(all_projects)
        return real(data_dir, cwd, all_projects=all_projects)

    monkeypatch.setattr(SessionSnapshotStore, "list_sessions", classmethod(spy))

    # A hit in the current project never scans the rest.
    assert SessionSnapshotStore.search_sessions("local", config.data_dir, str(here))
    assert calls == [False]

    # Only a miss widens to every project.
    calls.clear()
    assert SessionSnapshotStore.search_sessions("local", config.data_dir, str(tmp_path / "elsewhere"))
    assert calls == [False, True]


def test_history_segment_persists_effective_model(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.history.append(HistorySegment(key="seg.1", title="earlier task", text="user:\nfind the bug", model="compactor-x"))
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))

    assert restored.history[0].model == "compactor-x"


def test_history_segment_missing_model_field_restores_empty():
    """Snapshots written before the model field existed decode with an empty model, not a crash."""
    data = [
        {
            "key": "seg.1",
            "title": "old",
            "blob": "",
            "created_at": "",
            "scope": "history",
            "trigger": "auto",
            "fallback": False,
            "messages": 2,
            "summary": "",
        }
    ]
    segments = SessionSnapshotCodec.history(data, {})
    assert segments[0].model == ""
    assert segments[0].key == "seg.1"
