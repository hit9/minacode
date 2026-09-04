"""session snapshot (split from tests/test_session_persistence.py)."""
import itertools
import json

import pytest
from test_session_persistence import log_path, read_jsonl, read_lines, session_with_data_dir

from wizolt.base import SESSION_EVENT_KEY
from wizolt.session import Session, SessionSnapshotCodec, SessionSnapshotStore, TurnDiff


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


def test_source_views_survive_a_snapshot_roundtrip(tmp_path):
    """A view the model was shown survives save/load, so a resumed assistant can still edit with it."""
    from wizolt.tools import ReadTool

    s = session_with_data_dir(tmp_path)
    path = tmp_path / "a.py"
    path.write_text("one\ntwo\nthree\n")
    out = ReadTool(s, [{"path": "a.py"}]).call()
    key = s.register_source_drafts(list(out.drafts))[0]
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    view = restored.get_source_view(key)

    assert view is not None
    assert view.path == str(path)
    assert view.total_lines == 3
    assert [line for span in view.spans for line in span.lines] == ["one\n", "two\n", "three\n"]


def test_an_edit_still_resolves_its_view_after_a_restart(tmp_path):
    """The point of persisting views: a turn interrupted after a Read must be resumable. A fresh
    process loads the session and the id the model was shown still edits the file it named, with
    the counter continuing past it rather than reissuing a live key."""
    from wizolt.tools import EditTool, ReadTool

    s = session_with_data_dir(tmp_path)
    path = tmp_path / "a.py"
    path.write_text("alpha\nbeta\ngamma\n")
    key = s.register_source_drafts(list(ReadTool(s, [{"path": "a.py"}]).call().drafts))[0]
    s.messages.append({"role": "assistant", "content": f"about to edit {key}"})
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    EditTool(restored, ["a.py", key, [{"op": "replace", "start": 2, "end": 2, "content": "BETA\n"}]]).call()

    assert path.read_text() == "alpha\nBETA\ngamma\n"
    assert restored.source_view_counter == 1
    assert restored.register_source_drafts(list(ReadTool(restored, [{"path": "a.py"}]).call().drafts)) == ["view.2"]


def test_source_view_span_text_uses_content_addressed_blobs(tmp_path):
    """Span text is stored as a content-addressed blob, deduplicating equal text across views and diffs."""
    from wizolt.tools import ReadTool

    s = session_with_data_dir(tmp_path)
    path = tmp_path / "a.py"
    path.write_text("one\ntwo\nthree\n")
    s.register_source_drafts(list(ReadTool(s, [{"path": "a.py"}]).call().drafts))
    s.store_turn_diff("tr.1", 1, "a.py", "-one\n+ONE\n", before="one\n", after="ONE\n", round=1)
    s.save_snapshot()

    lines = read_lines(log_path(s))
    blobs = sorted(line["text"] for line in lines if "blob" in line)
    assert blobs == ["ONE\n", "one\n", "one\ntwo\nthree\n"]
    entry = [line for line in lines if "source_views" in line][-1]["source_views"][0]
    assert entry["spans"][0]["blob"]
    assert "lines" not in entry["spans"][0]


def test_source_view_delta_appends_new_views_and_drops_pruned(tmp_path):
    """A second save appends newly registered views; after pruning, the delta replaces the set."""
    from wizolt.tools import ReadTool

    s = session_with_data_dir(tmp_path)
    path = tmp_path / "a.py"
    path.write_text("one\ntwo\nthree\n")
    keep = s.register_source_drafts(list(ReadTool(s, [{"path": "a.py", "ranges": [[1, 1]]}]).call().drafts))[0]
    s.save_snapshot()

    dropped = s.register_source_drafts(list(ReadTool(s, [{"path": "a.py", "ranges": [[2, 2]]}]).call().drafts))[0]
    s.save_snapshot()
    s.prune_source_views({keep})
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    assert restored.get_source_view(keep) is not None
    assert restored.get_source_view(dropped) is None


def test_source_view_counter_survives_pruning_and_restart(tmp_path):
    """Expired public ids are never reassigned to different evidence after a restart."""
    from wizolt.tools import ReadTool

    s = session_with_data_dir(tmp_path)
    path = tmp_path / "a.py"
    path.write_text("one\ntwo\nthree\n")
    first = s.register_source_drafts(list(ReadTool(s, [{"path": "a.py", "ranges": [[1, 1]]}]).call().drafts))[0]
    second = s.register_source_drafts(list(ReadTool(s, [{"path": "a.py", "ranges": [[2, 2]]}]).call().drafts))[0]
    assert (first, second) == ("view.1", "view.2")
    s.prune_source_views(set())
    s.messages.append({"role": "user", "content": "keep this session durable"})
    s.save_snapshot()

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    new = restored.register_source_drafts(list(ReadTool(restored, [{"path": "a.py", "ranges": [[3, 3]]}]).call().drafts))[0]

    assert restored.get_source_view(first) is None
    assert restored.get_source_view(second) is None
    assert new == "view.3"


def test_loading_drops_a_view_whose_span_blob_is_gone(tmp_path):
    """A view is only as good as the text behind it. When the blob its span points at is missing
    from the log, the view is dropped rather than restored empty: an id that resolves to no
    content would let an Edit validate against nothing."""
    from wizolt.tools import ReadTool

    s = session_with_data_dir(tmp_path)
    path = tmp_path / "a.py"
    path.write_text("one\n")
    key = s.register_source_drafts(list(ReadTool(s, [{"path": "a.py"}]).call().drafts))[0]
    s.save_snapshot()

    lines = [line for line in read_lines(log_path(s)) if "blob" not in line]  # drop every stored blob
    with open(log_path(s), "w") as file:
        file.write("\n".join(json.dumps(line) for line in lines) + "\n")

    restored = Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
    assert restored.get_source_view(key) is None


def _persisted_view(**overrides):
    """One encoded view, as the codec writes it, with `overrides` applied.

    Spans 1:3 and 5:5 of a five-line file: sorted, non-overlapping, and non-touching, which is
    what normalization guarantees and therefore what load is entitled to insist on.
    """
    view = {
        "key": "view.1",
        "path": "/w/a.py",
        "display_path": "a.py",
        "total_lines": 5,
        "producer": "Read",
        "round": 1,
        "step": 2,
        "spans": [{"start": 1, "blob": "b3"}, {"start": 5, "blob": "b1"}],
    }
    view.update(overrides)
    return view


_BLOBS = {"b3": "one\ntwo\nthree\n", "b1": "five\n", "empty": ""}


@pytest.mark.parametrize(
    ("overrides", "note"),
    [
        ({"key": "view"}, "malformed key"),
        ({"key": "tr.1"}, "someone else's key space"),
        ({"path": ""}, "no canonical path"),
        ({"total_lines": "four"}, "non-numeric line count"),
        ({"total_lines": -1}, "negative line count"),
        ({"round": "later"}, "non-numeric round"),
        ({"spans": ["1:3"]}, "span that is not an object"),
        ({"spans": [{"start": "one", "blob": "b3"}]}, "non-numeric span start"),
        ({"spans": [{"start": 0, "blob": "b3"}]}, "span before line 1"),
        ({"spans": [{"start": 1, "blob": "gone"}]}, "span pointing at a blob that is not there"),
        ({"spans": [{"start": 1, "blob": "empty"}]}, "span with no lines"),
        ({"spans": [{"start": 1, "blob": "b3"}, {"start": 3, "blob": "b1"}]}, "overlapping spans"),
        ({"spans": [{"start": 1, "blob": "b3"}, {"start": 4, "blob": "b1"}]}, "touching spans normalization would have merged"),
        ({"spans": [{"start": 5, "blob": "b1"}, {"start": 1, "blob": "b3"}]}, "spans out of order"),
        ({"spans": [{"start": 1, "blob": "b3"}, {"start": 5, "blob": "b3"}]}, "span running past total_lines"),
        ({"total_lines": 2}, "spans that do not fit the file they claim"),
    ],
)
def test_loading_drops_structurally_invalid_views(overrides, note):
    """Persisted views are data on disk that later authorizes writes, so load validates rather than
    repairs: anything it cannot read exactly as written is dropped, and nothing is guessed."""
    assert SessionSnapshotCodec.source_views([_persisted_view(**overrides)], _BLOBS) == [], note


def test_loading_keeps_a_well_formed_view():
    """The other half of the check above: the base fixture really is loadable, so each rejection
    above is caused by its own override rather than by the shape they all share."""
    views = SessionSnapshotCodec.source_views([_persisted_view(), "not a view", {"key": "view.2"}], _BLOBS)

    assert [view.key for view in views] == ["view.1"]
    assert [(span.start, span.lines) for span in views[0].spans] == [(1, ("one\n", "two\n", "three\n")), (5, ("five\n",))]
    assert (views[0].total_lines, views[0].producer, views[0].round, views[0].step) == (5, "Read", 1, 2)
