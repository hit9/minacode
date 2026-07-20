import json
import os
import time

import pytest

import nanocode as n


def session_with_data_dir(tmp_path):
    """Session targeting tmp_path as data_dir (avoids touching ~/.nanocode)."""
    return n.Session(
        cwd=str(tmp_path),
        config=n.Config(data_dir=str(tmp_path)),
    )


def log_path(s):
    """Path of a session's log inside its project shard."""
    return n.SessionSnapshotStore.session_path(s.config.data_dir, s.cwd, s.uid)


def project_dir(s):
    return n.SessionSnapshotStore.project_dir(s.config.data_dir, s.cwd)


def read_jsonl(path) -> list[dict]:
    """Snapshot and delta lines, with the header line dropped."""
    return read_lines(path)[1:]


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
    restored = n.Session.load_snapshot(s.uid, config=s.config)
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
    restored = n.Session.load_snapshot(s.uid, config=s.config)
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

    monkeypatch.setattr(n.SessionSnapshotCodec, "turn_diff", fail_turn_diff)
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
    for turn, (before, after) in enumerate(zip(versions, versions[1:]), start=1):
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

    restored = n.Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))

    assert [(d.key, d.path, d.before, d.after) for d in restored.turn_diffs] == [("tr.1", "x.py", "old\n", "new\n")]


def test_oversized_snapshots_are_dropped_before_reaching_the_log(tmp_path):
    """Snapshots over the size limit are still discarded, and leave no blob behind."""
    s = session_with_data_dir(tmp_path)
    huge = "x" * (n.TurnDiff.SNAPSHOT_CHAR_LIMIT // 2 + 1)
    s.store_turn_diff("tr.1", 1, "big.py", "-o\n+n\n", before=huge, after=huge, round=1)
    s.save_snapshot()

    lines = read_lines(log_path(s))
    entry = [line for line in lines if "turn_diffs" in line][-1]["turn_diffs"][0]

    assert not [line for line in lines if "blob" in line]
    assert (entry["before_blob"], entry["after_blob"]) == ("", "")
    assert n.Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path)).turn_diffs[0].before == ""


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

    restored = n.Session.load_snapshot(s.uid, config=s.config, cwd=str(tmp_path))
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

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    # All messages across all lines
    assert [m["content"] for m in s2.messages[:3]] == ["q1", "a1", "q2"]
    # Fourth message is resume marker
    assert s2.messages[3]["content"].startswith("[Session resumed:")
    # All tool results
    assert s2.tool_results["tr.1"] == "# f"
    assert s2.tool_results["tr.2"] == "found"
    assert s2.tool_counter == 2


def test_load_preserves_uid(tmp_path):
    """load_snapshot preserves the original uid."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    assert s2.uid == s.uid


def test_load_with_latest_alias(tmp_path):
    """load_snapshot with uid='latest' resolves this project's latest pointer."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    s2 = n.Session.load_snapshot("latest", config=s.config, cwd=str(tmp_path))
    assert s2.uid == s.uid


def test_load_with_last_alias(tmp_path):
    """load_snapshot with uid='last' resolves this project's latest pointer."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    s2 = n.Session.load_snapshot("last", config=s.config, cwd=str(tmp_path))
    assert s2.uid == s.uid


def test_latest_uid_ignores_newer_sessions_from_other_projects(tmp_path):
    data_dir = tmp_path / "data"
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    config = n.Config(data_dir=str(data_dir))

    project_session = n.Session(cwd=str(project), config=config)
    project_session.messages.append({"role": "user", "content": "project"})
    project_session.save_snapshot()

    other_session = n.Session(cwd=str(other), config=config)
    other_session.messages.append({"role": "user", "content": "other"})
    other_session.save_snapshot()
    os.utime(log_path(project_session), (1, 1))
    os.utime(log_path(other_session), (2, 2))

    assert n.SessionSnapshotStore.latest_uid(str(data_dir), str(project)) == project_session.uid
    assert n.SessionSnapshotStore.latest_uid(str(data_dir), str(other)) == other_session.uid


def test_latest_uid_returns_empty_without_project_session(tmp_path):
    assert n.SessionSnapshotStore.latest_uid(str(tmp_path / "missing"), str(tmp_path)) == ""


def test_sessions_are_sharded_per_project(tmp_path):
    """Two projects sharing a data_dir get separate directories, so listing or deleting one
    project's history never touches the other's."""
    data_dir, project, other = tmp_path / "data", tmp_path / "project", tmp_path / "other"
    project.mkdir()
    other.mkdir()
    config = n.Config(data_dir=str(data_dir))

    first = n.Session(cwd=str(project), config=config)
    first.messages.append({"role": "user", "content": "project"})
    first.save_snapshot()
    second = n.Session(cwd=str(other), config=config)
    second.messages.append({"role": "user", "content": "other"})
    second.save_snapshot()

    assert project_dir(first) != project_dir(second)
    assert sorted(os.listdir(project_dir(first))) == sorted(["latest", first.uid + ".jsonl"])
    assert sorted(os.listdir(project_dir(second))) == sorted(["latest", second.uid + ".jsonl"])


def test_project_slug_separates_same_named_directories(tmp_path):
    """Two checkouts named alike hash to different shards."""
    left, right = tmp_path / "a" / "repo", tmp_path / "b" / "repo"
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    slugs = [n.SessionSnapshotStore.project_slug(str(path)) for path in (left, right)]

    assert all(slug.startswith("repo-") for slug in slugs)
    assert slugs[0] != slugs[1]


def test_load_finds_a_session_by_uid_from_any_directory(tmp_path):
    """An explicit UID resolves regardless of which project it belongs to."""
    data_dir, project = tmp_path / "data", tmp_path / "project"
    project.mkdir()
    config = n.Config(data_dir=str(data_dir))
    s = n.Session(cwd=str(project), config=config)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    loaded = n.Session.load_snapshot(s.uid, config=config, cwd=str(tmp_path))

    assert loaded.uid == s.uid
    assert loaded.cwd == str(project)


def test_latest_never_crosses_into_another_project(tmp_path):
    """The regression the shard layout closes: a newer session elsewhere must not be resumable
    as this project's latest."""
    data_dir, project, other = tmp_path / "data", tmp_path / "project", tmp_path / "other"
    project.mkdir()
    other.mkdir()
    config = n.Config(data_dir=str(data_dir))
    elsewhere = n.Session(cwd=str(other), config=config)
    elsewhere.messages.append({"role": "user", "content": "other"})
    elsewhere.save_snapshot()

    with pytest.raises(n.NanocodeError, match="No previous session for this project"):
        n.Session.load_snapshot("latest", config=config, cwd=str(project))


def test_latest_falls_back_to_newest_log_when_pointer_is_missing(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()
    os.unlink(os.path.join(project_dir(s), "latest"))

    assert n.SessionSnapshotStore.latest_uid(str(tmp_path), str(tmp_path)) == s.uid


def test_header_line_precedes_the_snapshot(tmp_path):
    """Line 1 is a bounded header, so project queries never parse the conversation behind it."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    header = read_lines(log_path(s))[0]

    assert header == {"v": n.SessionSnapshotStore.FORMAT_VERSION, "uid": s.uid, "cwd": s.cwd, "created_at": header["created_at"]}
    assert "messages" not in header
    assert n.SessionSnapshotStore.read_header(log_path(s))["cwd"] == s.cwd


def test_load_rejects_an_unknown_format_version(tmp_path):
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()
    lines = read_lines(log_path(s))
    lines[0]["v"] = 99
    with open(log_path(s), "w") as file:
        file.write("\n".join(json.dumps(line) for line in lines) + "\n")

    with pytest.raises(n.NanocodeError, match="Unsupported session format v99"):
        n.Session.load_snapshot(s.uid, config=s.config)


def test_load_appends_resume_marker(tmp_path):
    """After load, the session has a resume marker message at the end."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    assert len(s2.messages) == 2  # hello + resume marker
    assert s2.messages[-1]["content"].startswith(f"[Session resumed: uid={s.uid}]")


def test_save_after_load_produces_a_delta(tmp_path):
    """Save after load appends a delta (not re-init)."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "hello"})
    s.save_snapshot()  # init (line 1)

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    # s2 now has messages = [hello, resume_marker]
    s2.messages.append({"role": "assistant", "content": "post-resume"})
    s2.save_snapshot()  # delta (line 2)

    lines = read_jsonl(log_path(s))
    assert len(lines) == 2
    delta = lines[1]
    # The delta should contain the post-resume message, NOT the resume marker
    # (resume marker was already in s2 when _snapshot_saved was set by load)
    assert delta["messages"] == [{"role": "assistant", "content": "post-resume"}]


def test_repeated_resume_preserves_history(tmp_path):
    """Repeated resume/save cycles keep appending new messages to the same history."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "m1"})
    s.save_snapshot()

    expected = ["m1"]
    for role, content in (("assistant", "a1"), ("user", "m2"), ("assistant", "a2")):
        s = n.Session.load_snapshot(s.uid, config=s.config)
        assert [m["content"] for m in n.SessionSnapshotCodec.persistable_messages(s.messages)] == expected
        s.messages.append({"role": role, "content": content})
        s.save_snapshot()
        expected.append(content)

    loaded = n.Session.load_snapshot(s.uid, config=s.config)
    assert [m["content"] for m in n.SessionSnapshotCodec.persistable_messages(loaded.messages)] == expected


def test_resume_marker_is_never_persisted(tmp_path):
    """Resume markers are runtime-only, including when a message rewrite forces replace."""
    s = session_with_data_dir(tmp_path)
    s.messages.append({"role": "user", "content": "m1"})
    s.save_snapshot()

    resumed = n.Session.load_snapshot(s.uid, config=s.config)
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
    n.SessionSnapshotStore.write_jsonl(path, n.SessionSnapshotStore.header(s), mode="w")
    n.SessionSnapshotStore.write_jsonl(
        path,
        {
            "uid": s.uid,
            "cwd": str(tmp_path),
            "messages": [{"role": "user", "content": "m1"}, marker, {"role": "assistant", "content": "a1"}],
        },
        mode="a",
    )

    loaded = n.Session.load_snapshot(s.uid, config=s.config)

    assert [m["content"] for m in n.SessionSnapshotCodec.persistable_messages(loaded.messages)] == ["m1", "a1"]
    assert sum(1 for m in loaded.messages if n.SessionSnapshotCodec.is_internal_message(m)) == 1


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

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    assert s2.tool_results["tr.1"] == "hi"
    assert s2.tool_results["tr.2"] == "code"


def test_tool_records_roundtrip(tmp_path):
    """Tool records survive save/load."""
    s = session_with_data_dir(tmp_path)
    s.store_tool_result("Bash", ["pwd"], "/tmp")
    s.store_tool_result("Search", ["x"], "match")
    s.save_snapshot()

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
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

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
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
    s.usage.last_cached_prompt_tokens = 5
    s.usage.last_total_tokens = 60
    s.save_snapshot()

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    assert s2.usage.calls == 3
    assert s2.usage.prompt_tokens == 100
    assert s2.usage.completion_tokens == 50
    assert s2.usage.total_tokens == 150
    assert s2.usage.cached_prompt_tokens == 20
    assert s2.usage.last_cached_prompt_tokens == 5
    assert s2.usage.last_total_tokens == 60


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

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
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

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    contents = [m["content"] for m in s2.messages if not m["content"].startswith("[Session resumed:")]
    assert contents == ["m1", "a1", "m2", "a2"]


def test_multiple_deltas_with_tool_calls(tmp_path):
    """Tool calls across multiple deltas accumulate correctly."""
    s = session_with_data_dir(tmp_path)
    s.store_tool_result("Read", ["a.py"], "# a")
    s.save_snapshot()  # init: tr.1
    s.store_tool_result("Search", ["pat"], "hit")
    s.save_snapshot()  # delta 1: tr.2
    s.store_tool_result("Bash", ["pwd"], "/tmp")
    s.save_snapshot()  # delta 2: tr.3

    s2 = n.Session.load_snapshot(s.uid, config=s.config)
    assert s2.tool_results["tr.1"] == "# a"
    assert s2.tool_results["tr.2"] == "hit"
    assert s2.tool_results["tr.3"] == "/tmp"
    assert s2.tool_counter == 3
    assert len(s2.tool_records) == 3


def test_load_missing_snapshot_raises_error(tmp_path):
    """Loading a non-existent session raises NanocodeError."""
    with pytest.raises(n.NanocodeError, match="Session snapshot not found"):
        n.Session.load_snapshot("nonexistent-uid", config=n.Config(data_dir=str(tmp_path)))


@pytest.mark.parametrize("alias", ["latest", "last"])
def test_resolve_uid_without_a_project_session(tmp_path, alias):
    """Resolving an alias in a project with no sessions raises NanocodeError."""
    with pytest.raises(n.NanocodeError, match="No previous session for this project"):
        n.SessionSnapshotStore.resolve_uid(alias, str(tmp_path), str(tmp_path))


def test_resolve_uid_passthrough_normal_uid(tmp_path):
    """Resolving a normal uid (not an alias) returns it as-is."""
    assert n.SessionSnapshotStore.resolve_uid("my-uid", str(tmp_path), str(tmp_path)) == "my-uid"


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
    settings = n.RuntimeSettings.from_dict({})

    assert settings.session_retention_days == 7


def test_clean_expired_sessions_removes_old_files_and_latest(tmp_path):
    s = session_with_data_dir(tmp_path)
    old = session_with_data_dir(tmp_path)
    old.messages.append({"role": "user", "content": "old"})
    old.save_snapshot()
    old_path = log_path(old)
    stale_time = time.time() - 8 * 86400
    os.utime(old_path, (stale_time, stale_time))

    assert n.SessionSnapshotStore.clean_expired(s) == 1

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

    assert n.SessionSnapshotStore.clean_expired(s) == 0

    assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Transcript replay resilience (regression: multi-line tool arguments such as
# a Bash command with embedded newlines must not crash --resume rendering)
# ---------------------------------------------------------------------------

def _bash_raw_call(arguments: str) -> dict:
    return {"id": "c1", "type": "function", "function": {"name": "Bash", "arguments": arguments}}


def test_transcript_tool_call_parses_multiline_arguments():
    """Argument strings with literal newlines (invalid strict JSON) still parse, so the
    Bash command survives instead of being dropped to {}."""
    raw = _bash_raw_call("{\"command\": \"printf 'line one\nline two'\"}")
    call = n.CommandLoop.transcript_tool_call(raw)
    assert call is not None
    assert call.args == ["printf 'line one\nline two'"]


def test_transcript_tool_call_does_not_crash_on_unparseable_args():
    """A historical Bash call whose payload fails validation must render, not raise."""
    raw = _bash_raw_call("{not valid json at all")
    call = n.CommandLoop.transcript_tool_call(raw)  # must not raise ToolError
    assert call is not None
    assert call.name == "Bash"


def test_chat_tool_calls_parse_multiline_commit_message():
    """The live chat path recovers args from a multi-line Bash command too."""
    class _Fn:
        name = "Bash"
        arguments = "{\"command\": \"printf 'subject\n\nbody line'\"}"
    class _Raw:
        id = "x1"
        function = _Fn()
    class _Msg:
        tool_calls = [_Raw()]
    s = n.Session(cwd="/tmp")
    calls = n.ModelClient(s).tool_calls(_Msg())
    assert len(calls) == 1
    assert calls[0].args == ["printf 'subject\n\nbody line'"]


def test_snapshot_messages_strips_non_persistable_roles(tmp_path):
    s = n.Session(cwd=str(tmp_path))
    s.messages = [
        {"role": "system", "content": "[Session resumed: old-session-id]"},
        {"role": "user", "content": "hello"},
    ]
    messages = n.SessionSnapshotCodec.snapshot_messages(s)
    # Internal resume marker is stripped; user message is kept.
    roles = [m["role"] for m in messages]
    assert "system" not in roles
    assert "user" in roles
    assert len(messages) == 1
